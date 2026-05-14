"""Event-feature score matrix.

Joins per-event SAE top-k activations (from `event_sae.openvla.activations`
either online or via `scripts/extract_topk.py`) with VLM-annotated event
clusters, builds event-centered temporal windows around each event, and
projects three time templates (pulse, step-up, step-down) onto the
per-feature trajectory. The per-feature score is the maximum positive
projection across templates; per-event scores are averaged within each
`(cluster, episode)` group and then across episodes to give one row per
cluster.

Output payload (single torch.save .pt):

  - `matrix`              : `(num_rows, dict_size)` float32 score matrix
  - `row_keys`            : per-row cluster metadata (`task_description`,
                            `cluster_id`, `phrase`, `phase`, episode coverage,
                            member counts, ...)
  - `row_results`         : per-row top-N feature summaries
  - `templates`           : the three time templates used
  - `selection_counts`    : join + filter accounting
  - `selected_events`     : per-event provenance after scoring
  - `source`              : input paths + manifest summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from event_sae.events.io import load_jsonl


# ---------------------------------------------------------------------------
# Templates + projections
# ---------------------------------------------------------------------------


def _normalize_template(template: torch.Tensor) -> torch.Tensor:
    template = template.to(dtype=torch.float32)
    template = template - torch.mean(template)
    norm = torch.linalg.vector_norm(template)
    if float(norm) == 0.0:
        raise ValueError("Template norm is zero after mean-centering.")
    return template / norm


def build_templates(window_size: int) -> dict[str, torch.Tensor]:
    """Build the three normalized time templates used for event scoring."""
    positions = torch.arange(-window_size, window_size + 1, dtype=torch.float32)
    pulse = 1.0 - torch.abs(positions) / float(window_size + 1)
    pulse = _normalize_template(pulse)
    step_up = torch.where(positions < 0, -torch.ones_like(positions), torch.ones_like(positions))
    step_up = _normalize_template(step_up)
    step_down = -step_up
    return {"pulse": pulse, "step_up": step_up, "step_down": step_down}


def _project_pattern_scores(
    centered_matrix: torch.Tensor,
    templates: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    centered_signal = centered_matrix - torch.mean(centered_matrix, dim=0, keepdim=True)
    return {
        name: torch.clamp(torch.matmul(centered_signal.transpose(0, 1), template), min=0.0)
        for name, template in templates.items()
    }


def _top_features(scores: torch.Tensor, top_n: int) -> tuple[list[int], list[float]]:
    k = min(top_n, int(scores.numel()))
    values, indices = torch.topk(scores, k=k, largest=True)
    return indices.tolist(), [float(x) for x in values.tolist()]


def _row_top_summary(scores: torch.Tensor, top_n: int) -> dict[str, list]:
    feature_ids, values = _top_features(scores, top_n)
    return {"feature_ids": feature_ids, "scores": values}


# ---------------------------------------------------------------------------
# Window placement + per-step action-token aggregation
# ---------------------------------------------------------------------------


def _fit_step_window(
    *,
    center_step: int,
    window_size: int,
    num_steps: int,
) -> tuple[list[int] | None, list[int], int | None]:
    """Shift a (2w+1)-step centered window to stay inside [0, num_steps)."""
    requested_steps = list(range(center_step - window_size, center_step + window_size + 1))
    if not requested_steps:
        return [], [], 0
    min_req, max_req = min(requested_steps), max(requested_steps)
    if (max_req - min_req) >= num_steps:
        return None, requested_steps, None
    shift = 0
    if max_req >= num_steps:
        shift -= max_req - (num_steps - 1)
    if min_req + shift < 0:
        shift += -(min_req + shift)
    return [s + shift for s in requested_steps], requested_steps, shift


def _mean_action_token_rows(
    forward_rows: dict[int, dict],
    *,
    dict_size: int,
    action_dim: int,
) -> torch.Tensor | None:
    """Average the last-token sparse SAE rows across the first `action_dim` forwards."""
    if len(forward_rows) < action_dim:
        return None
    step_vector = torch.zeros(dict_size, dtype=torch.float32)
    for forward_idx in sorted(forward_rows)[:action_dim]:
        row = forward_rows[forward_idx]
        step_vector.index_add_(0, row["top_feature_ids"], row["top_feature_vals"])
    return step_vector / float(action_dim)


# ---------------------------------------------------------------------------
# Cluster / event join
# ---------------------------------------------------------------------------


@dataclass
class _JoinResult:
    selected_events: list[dict]
    cluster_metadata_by_id: dict[str, dict]
    counts: dict[str, int]


def join_cluster_events(
    *,
    event_features: list[dict],
    cluster_assignments: list[dict],
    cluster_annotations: list[dict],
) -> _JoinResult:
    """Join event_features + cluster_assignments + cluster_annotations, filter
    out clusters with API/parse errors or empty phrase/phase."""
    event_by_sample_id = {}
    for record in event_features:
        sample_id = str(record["sample_id"])
        if sample_id in event_by_sample_id:
            raise ValueError(f"Duplicate sample_id in event_features.jsonl: {sample_id}")
        event_by_sample_id[sample_id] = record

    counts = {
        "valid_clusters": 0,
        "joined_events": 0,
        "skipped_api_error": 0,
        "skipped_parse_error": 0,
        "skipped_empty_phrase": 0,
        "skipped_empty_phase": 0,
        "skipped_missing_cluster_annotation": 0,
        "skipped_missing_event_features": 0,
    }

    cluster_metadata_by_id: dict[str, dict] = {}
    for annotation in cluster_annotations:
        if annotation.get("api_error") is not None:
            counts["skipped_api_error"] += 1
            continue
        if annotation.get("parse_error") is not None:
            counts["skipped_parse_error"] += 1
            continue
        phrase = str(annotation.get("phrase", "")).strip()
        if not phrase:
            counts["skipped_empty_phrase"] += 1
            continue
        phase = str(annotation.get("phase", "")).strip()
        if not phase:
            counts["skipped_empty_phase"] += 1
            continue
        cluster_id = str(annotation["cluster_id"])
        if cluster_id in cluster_metadata_by_id:
            raise ValueError(f"Duplicate cluster_id in cluster annotations: {cluster_id}")
        cluster_metadata_by_id[cluster_id] = {
            "cluster_id": cluster_id,
            "task_description": str(annotation["task_description"]),
            "phrase": phrase,
            "phase": phase,
            "episode_coverage": float(annotation.get("episode_coverage", 0.0)),
            "model": str(annotation.get("model", "")),
            "prompt_version": str(annotation.get("prompt_version", "")),
            "representative_sample_ids": list(annotation.get("representative_sample_ids", [])),
            "representative_clip_paths": list(annotation.get("representative_clip_paths", [])),
            "representative_frame_paths": list(annotation.get("representative_frame_paths", [])),
            "representative_progress_percents": list(annotation.get("representative_progress_percents", [])),
        }
    counts["valid_clusters"] = len(cluster_metadata_by_id)

    joined_events: list[dict] = []
    for assignment in cluster_assignments:
        cluster_id = str(assignment["cluster_id"])
        cluster_meta = cluster_metadata_by_id.get(cluster_id)
        if cluster_meta is None:
            counts["skipped_missing_cluster_annotation"] += 1
            continue
        sample_id = str(assignment["sample_id"])
        event = event_by_sample_id.get(sample_id)
        if event is None:
            counts["skipped_missing_event_features"] += 1
            continue
        if str(event["task_description"]) != str(assignment["task_description"]):
            raise ValueError(f"Task description mismatch for sample_id={sample_id}")
        joined_events.append(
            {
                "sample_id": sample_id,
                "task_description": str(event["task_description"]),
                "task_id": int(event["task_id"]),
                "task_episode_idx": int(event["task_episode_idx"]),
                "episode_num": int(event["episode_num"]),
                "waypoint_rank": int(event["waypoint_rank"]),
                "waypoint_step": int(event["waypoint_step"]),
                "progress_percent": float(event["progress_percent"]),
                "num_steps": int(event["num_steps"]),
                "cluster_id": cluster_id,
                "phrase": cluster_meta["phrase"],
                "phase": cluster_meta["phase"],
            }
        )
    counts["joined_events"] = len(joined_events)

    member_counts: dict[str, int] = defaultdict(int)
    episode_sets: dict[str, set[int]] = defaultdict(set)
    for event in joined_events:
        member_counts[event["cluster_id"]] += 1
        episode_sets[event["cluster_id"]].add(int(event["episode_num"]))
    for cluster_id, meta in cluster_metadata_by_id.items():
        meta["num_members"] = int(member_counts.get(cluster_id, 0))
        meta["num_episodes"] = int(len(episode_sets.get(cluster_id, set())))

    return _JoinResult(joined_events, cluster_metadata_by_id, counts)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _load_topk_activation_index(
    topk_run_dir: Path,
    *,
    selected_episode_nums: set[int],
    needed_steps_by_episode: dict[int, set[int]],
) -> tuple[dict[tuple[int, int], dict[int, dict]], dict]:
    """Walk topk shards (token_topk_sparse_v1) and collect, for each
    (episode, step) we need, the last-token sparse row per forward_idx."""
    manifest_path = topk_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "token_topk_sparse_v1":
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")

    max_episode = max(selected_episode_nums) if selected_episode_nums else -1
    last_token_row: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    for shard_meta in manifest["shards"]:
        shard_path = topk_run_dir / shard_meta["path"]
        payload = torch.load(shard_path, map_location="cpu")
        episode_num = payload["episode_num"].to(dtype=torch.int64)
        if int(episode_num.numel()) == 0:
            continue
        if int(episode_num[0]) > max_episode:
            break
        step_in_episode = payload["step_in_episode"].to(dtype=torch.int64)
        global_forward_idx = payload["global_forward_idx"].to(dtype=torch.int64)
        token_idx = payload["token_idx"].to(dtype=torch.int64)
        top_feature_ids = payload["top_feature_ids"].to(dtype=torch.int64)
        top_feature_vals = payload["top_feature_vals"].to(dtype=torch.float32)
        for row_idx in range(int(episode_num.shape[0])):
            ep = int(episode_num[row_idx])
            if ep not in selected_episode_nums:
                continue
            st = int(step_in_episode[row_idx])
            if st not in needed_steps_by_episode[ep]:
                continue
            fwd = int(global_forward_idx[row_idx])
            tok = int(token_idx[row_idx])
            current = last_token_row[(ep, st)].get(fwd)
            if current is None or tok > current["token_idx"]:
                last_token_row[(ep, st)][fwd] = {
                    "token_idx": tok,
                    "top_feature_ids": top_feature_ids[row_idx].clone(),
                    "top_feature_vals": top_feature_vals[row_idx].clone(),
                }
    return last_token_row, manifest


def score_cluster_features(
    *,
    topk_run_dir: Path,
    event_features_path: Path,
    cluster_assignments_path: Path,
    cluster_annotations_path: Path,
    output_path: Path,
    window_size: int = 5,
    top_n: int = 20,
    action_dim: int = 7,
) -> dict:
    """Build the event-feature score matrix and save a single `.pt` payload.

    `topk_run_dir` must contain `manifest.json` (`token_topk_sparse_v1`) + the
    referenced shards. Either online (sbatch `mode=topk`) or offline
    (`scripts/extract_topk.py`) sources are accepted.
    """
    topk_run_dir = Path(topk_run_dir).resolve()
    event_features_path = Path(event_features_path).resolve()
    cluster_assignments_path = Path(cluster_assignments_path).resolve()
    cluster_annotations_path = Path(cluster_annotations_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    join = join_cluster_events(
        event_features=load_jsonl(event_features_path),
        cluster_assignments=load_jsonl(cluster_assignments_path),
        cluster_annotations=load_jsonl(cluster_annotations_path),
    )
    if not join.selected_events:
        raise RuntimeError("No usable clustered events after the join.")

    w = window_size
    selected_episode_nums: set[int] = set()
    needed_steps_by_episode: dict[int, set[int]] = defaultdict(set)
    usable_events: list[dict] = []
    skipped_window = 0
    for event in join.selected_events:
        episode = int(event["episode_num"])
        num_steps = int(event["num_steps"])
        window_steps, requested, shift = _fit_step_window(
            center_step=int(event["waypoint_step"]), window_size=w, num_steps=num_steps
        )
        if window_steps is None:
            skipped_window += 1
            continue
        event = dict(event)
        event.update({"window_steps": window_steps, "requested_steps": requested, "window_shift": int(shift)})
        usable_events.append(event)
        selected_episode_nums.add(episode)
        for step in window_steps:
            needed_steps_by_episode[episode].add(step)
    if not usable_events:
        raise RuntimeError("No events remained after centered-window filtering.")

    last_token_row, manifest = _load_topk_activation_index(
        topk_run_dir,
        selected_episode_nums=selected_episode_nums,
        needed_steps_by_episode=needed_steps_by_episode,
    )
    dict_size = int(manifest["dict_size"])
    templates = build_templates(w)

    episode_group_scores: dict[tuple[str, int], dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    episode_group_event_counts: dict[tuple[str, int], int] = defaultdict(int)
    selected_event_payloads: list[dict] = []
    skipped_missing_rows = 0
    event_iter = tqdm(usable_events, desc="Scoring events", unit="event") if tqdm is not None else usable_events
    for event in event_iter:
        episode = int(event["episode_num"])
        centered = torch.zeros((2 * w + 1, dict_size), dtype=torch.float32)
        ok = True
        for row_idx, step in enumerate(event["window_steps"]):
            forward_rows = last_token_row.get((episode, step))
            if not forward_rows:
                ok = False
                break
            step_vector = _mean_action_token_rows(forward_rows, dict_size=dict_size, action_dim=action_dim)
            if step_vector is None:
                ok = False
                break
            centered[row_idx] = step_vector
        if not ok:
            skipped_missing_rows += 1
            continue
        pattern_scores = _project_pattern_scores(centered, templates)
        group_key = (str(event["cluster_id"]), episode)
        for name, vec in pattern_scores.items():
            episode_group_scores[group_key][name].append(vec)
        episode_group_event_counts[group_key] += 1
        selected_event_payloads.append(
            {
                "sample_id": event["sample_id"],
                "task_description": event["task_description"],
                "task_id": event["task_id"],
                "task_episode_idx": event["task_episode_idx"],
                "episode_num": episode,
                "waypoint_rank": event["waypoint_rank"],
                "waypoint_step": event["waypoint_step"],
                "progress_percent": event["progress_percent"],
                "num_steps": event["num_steps"],
                "cluster_id": event["cluster_id"],
                "phrase": event["phrase"],
                "phase": event["phase"],
                "requested_steps": event["requested_steps"],
                "window_steps": event["window_steps"],
                "window_shift": event["window_shift"],
            }
        )
    if not selected_event_payloads:
        raise RuntimeError("No events remained after activation-window filtering.")

    row_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    row_episode_counts: dict[str, int] = defaultdict(int)
    row_event_counts: dict[str, int] = defaultdict(int)
    for (cluster_id, _ep), pattern_lists in episode_group_scores.items():
        group_means = {
            name: torch.stack(score_list, dim=0).mean(dim=0)
            for name, score_list in pattern_lists.items()
        }
        combined = torch.maximum(
            group_means["pulse"], torch.maximum(group_means["step_up"], group_means["step_down"])
        )
        row_scores[cluster_id].append(combined)
        row_episode_counts[cluster_id] += 1
        row_event_counts[cluster_id] += episode_group_event_counts[(cluster_id, _ep)]

    row_cluster_ids = sorted(
        row_scores,
        key=lambda cid: (
            str(join.cluster_metadata_by_id[cid]["task_description"]),
            str(cid),
        ),
    )
    num_rows = len(row_cluster_ids)
    matrix = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        matrix[row_idx] = torch.stack(row_scores[cluster_id], dim=0).mean(dim=0)

    row_results = []
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        meta = join.cluster_metadata_by_id[cluster_id]
        row_results.append(
            {
                "task_description": meta["task_description"],
                "cluster_id": cluster_id,
                "phrase": meta["phrase"],
                "phase": meta["phase"],
                "num_episode_groups": row_episode_counts[cluster_id],
                "num_events": row_event_counts[cluster_id],
                "episode_coverage": meta["episode_coverage"],
                "top_features": _row_top_summary(matrix[row_idx], top_n),
            }
        )

    payload = {
        "source": {
            "topk_run_dir": str(topk_run_dir),
            "event_features_path": str(event_features_path),
            "cluster_assignments_path": str(cluster_assignments_path),
            "cluster_annotations_path": str(cluster_annotations_path),
            "dict_size": dict_size,
            "topk": int(manifest["topk"]),
            "layer": manifest.get("layer"),
            "sae_path": manifest.get("sae_path"),
        },
        "window_size": window_size,
        "top_n": top_n,
        "action_dim": action_dim,
        "row_semantics": "(task_description, cluster_id, phrase, phase)",
        "score_definitions": {
            "pulse": "positive projection onto a symmetric local-peak template after time-centering",
            "step_up": "positive projection onto a low-to-high step template after time-centering",
            "step_down": "positive projection onto a high-to-low step template after time-centering",
            "combined_score": "max(pulse, step_up, step_down) per event, then averaged within (cluster, episode) and across episodes",
            "activation_row_semantics": "per env step: average of the last sparse top-k row across the first `action_dim` forwards",
        },
        "templates": {name: template.clone() for name, template in templates.items()},
        "selection_counts": {
            **join.counts,
            "skipped_window": skipped_window,
            "selected_events_before_activation_filter": len(usable_events),
            "selected_events_after_activation_filter": len(selected_event_payloads),
            "skipped_missing_action_token_rows": skipped_missing_rows,
        },
        "selected_events": selected_event_payloads,
        "row_keys": [
            {
                **join.cluster_metadata_by_id[cid],
                "num_episode_groups": row_episode_counts[cid],
                "num_events": row_event_counts[cid],
            }
            for cid in row_cluster_ids
        ],
        "matrix": matrix,
        "row_results": row_results,
    }
    torch.save(payload, output_path)
    return {
        "output_path": str(output_path),
        "num_rows": num_rows,
        "dict_size": dict_size,
        "selected_events": len(selected_event_payloads),
    }

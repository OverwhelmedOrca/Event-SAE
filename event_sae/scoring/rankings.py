"""Four feature-ranking strategies for SAE candidate selection.

Implements the rankings compared in paper Section 4.4:

- ``event_aligned`` — top-N features per cluster row from an existing
  event-feature score matrix (output of ``score_cluster_features.py``).
- ``window_mean`` — for each cluster row, mean SAE activation over the
  same event windows used by event-aligned, then top-N features.
- ``task_mean`` — for each task, mean SAE activation over every rollout
  timestep in the run, then top-N features.
- ``random_alive`` — uniform random sample from features that fire at
  least once in the run, excluding features already selected by any of
  the three informed rankings.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from event_sae.events.io import load_jsonl


# ---------------------------------------------------------------------------
# Shared shard iteration
# ---------------------------------------------------------------------------


def _load_manifest(topk_run_dir: Path) -> dict:
    manifest_path = topk_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")
    import json
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "token_topk_sparse_v1":
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")
    return manifest


def _iter_shards(topk_run_dir: Path, manifest: dict, *, desc: str | None = None):
    shards = manifest["shards"]
    it = shards
    if tqdm is not None and desc:
        it = tqdm(shards, desc=desc, unit="shard")
    for shard_meta in it:
        payload = torch.load(topk_run_dir / shard_meta["path"], map_location="cpu")
        yield payload


def _top_pairs(vec: torch.Tensor, top_n: int) -> list[dict]:
    k = min(top_n, int(vec.numel()))
    values, indices = torch.topk(vec, k=k, largest=True)
    return [{"feature_id": int(i), "score": float(v)} for i, v in zip(indices.tolist(), values.tolist())]


# ---------------------------------------------------------------------------
# Per-step action-token aggregator
# ---------------------------------------------------------------------------


def _build_step_vectors(
    topk_run_dir: Path,
    *,
    dict_size: int,
    action_dim: int,
    desc: str,
):
    """Yield (episode_num, step_in_episode, step_vector[dict_size]) tuples.

    Each step vector is the per-step mean of the last sparse top-k row across
    the first ``action_dim`` forwards at that step. Forwards are deduplicated
    by ``global_forward_idx``, keeping the highest ``token_idx`` row (the
    action token).
    """
    manifest = _load_manifest(topk_run_dir)

    last_token_row: dict[tuple[int, int], dict[int, dict]] = defaultdict(dict)
    for payload in _iter_shards(topk_run_dir, manifest, desc=desc):
        episode_num = payload["episode_num"].to(dtype=torch.int64)
        step_in_episode = payload["step_in_episode"].to(dtype=torch.int64)
        global_forward_idx = payload["global_forward_idx"].to(dtype=torch.int64)
        token_idx = payload["token_idx"].to(dtype=torch.int64)
        top_feature_ids = payload["top_feature_ids"].to(dtype=torch.int64)
        top_feature_vals = payload["top_feature_vals"].to(dtype=torch.float32)
        for row_idx in range(int(episode_num.shape[0])):
            ep, st = int(episode_num[row_idx]), int(step_in_episode[row_idx])
            fwd = int(global_forward_idx[row_idx])
            tok = int(token_idx[row_idx])
            current = last_token_row[(ep, st)].get(fwd)
            if current is None or tok > current["token_idx"]:
                last_token_row[(ep, st)][fwd] = {
                    "token_idx": tok,
                    "top_feature_ids": top_feature_ids[row_idx].clone(),
                    "top_feature_vals": top_feature_vals[row_idx].clone(),
                }

    for (ep, st), forwards in last_token_row.items():
        if len(forwards) < action_dim:
            continue
        step_vector = torch.zeros(dict_size, dtype=torch.float32)
        for fwd in sorted(forwards)[:action_dim]:
            row = forwards[fwd]
            step_vector.index_add_(0, row["top_feature_ids"], row["top_feature_vals"])
        step_vector /= float(action_dim)
        yield ep, st, step_vector


# ---------------------------------------------------------------------------
# Ranking implementations
# ---------------------------------------------------------------------------


def _load_event_aligned_matrix(scores_pt_path: Path) -> tuple[torch.Tensor, list[dict]]:
    payload = torch.load(Path(scores_pt_path).resolve(), map_location="cpu")
    return payload["matrix"].to(dtype=torch.float32), list(payload["row_keys"])


def event_aligned_top_features_per_row(scores_pt_path: Path, top_n: int) -> list[dict]:
    """Read the score matrix produced by ``score_cluster_features.py`` and
    return per-cluster-row top-N features."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        out.append(
            {
                "ranking": "event_aligned",
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": _top_pairs(matrix[row_idx], top_n),
            }
        )
    return out


def event_aligned_suite_top_k(scores_pt_path: Path, top_k: int) -> list[dict]:
    """Suite-level top-K event-aligned features: mean of the score matrix
    over rows, then top-K. Matches paper Section 5.3 aggregation."""
    matrix, _ = _load_event_aligned_matrix(scores_pt_path)
    suite_vec = matrix.mean(dim=0)
    return _top_pairs(suite_vec, top_k)


def _compute_window_mean_matrix(
    *,
    scores_pt_path: Path,
    topk_run_dir: Path,
    action_dim: int,
) -> tuple[torch.Tensor, list[dict], list[int]]:
    """Returns (matrix [num_rows, dict_size], row_keys, num_events_per_row).

    Reuses the same ``selected_events`` and ``row_keys`` saved in the score
    payload, so the events used here are identical to event-aligned —
    only the temporal weighting (template projection vs. flat mean) differs.
    """
    scores_pt_path = Path(scores_pt_path).resolve()
    topk_run_dir = Path(topk_run_dir).resolve()
    payload = torch.load(scores_pt_path, map_location="cpu")
    row_keys: list[dict] = list(payload["row_keys"])
    selected_events: list[dict] = list(payload["selected_events"])
    dict_size = int(payload["source"]["dict_size"])

    cluster_ids = [str(row["cluster_id"]) for row in row_keys]
    cluster_index = {cid: i for i, cid in enumerate(cluster_ids)}

    needed_steps: dict[tuple[int, int], list[int]] = defaultdict(list)
    for event in selected_events:
        idx = cluster_index.get(str(event["cluster_id"]))
        if idx is None:
            continue
        ep = int(event["episode_num"])
        for step in event["window_steps"]:
            needed_steps[(ep, int(step))].append(idx)
    if not needed_steps:
        raise RuntimeError("No event-window steps found in scores payload.")

    matrix = torch.zeros((len(cluster_ids), dict_size), dtype=torch.float32)
    counts = [0] * len(cluster_ids)
    for ep, st, step_vector in _build_step_vectors(
        topk_run_dir, dict_size=dict_size, action_dim=action_dim, desc="window-mean shards"
    ):
        rows_here = needed_steps.get((ep, st))
        if not rows_here:
            continue
        for idx in rows_here:
            matrix[idx] += step_vector
            counts[idx] += 1
    for idx, c in enumerate(counts):
        if c > 0:
            matrix[idx] /= float(c)
    num_events_per_row = [int(row.get("num_events", 0)) for row in row_keys]
    return matrix, row_keys, num_events_per_row


def window_mean_top_features_per_row(
    *,
    scores_pt_path: Path,
    topk_run_dir: Path,
    top_n: int,
    action_dim: int = 7,
) -> list[dict]:
    """Per cluster row, top-N features by mean SAE activation over the
    cluster's event windows. No temporal templating."""
    matrix, row_keys, _ = _compute_window_mean_matrix(
        scores_pt_path=scores_pt_path, topk_run_dir=topk_run_dir, action_dim=action_dim
    )
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        if float(matrix[row_idx].abs().sum().item()) == 0.0:
            continue
        out.append(
            {
                "ranking": "window_mean",
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": _top_pairs(matrix[row_idx], top_n),
            }
        )
    return out


def window_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    topk_run_dir: Path,
    top_k: int,
    action_dim: int = 7,
) -> list[dict]:
    """Suite-level top-K window-mean features: weighted mean of per-row
    window-mean vectors by ``num_events`` per row, then top-K."""
    matrix, _row_keys, num_events_per_row = _compute_window_mean_matrix(
        scores_pt_path=scores_pt_path, topk_run_dir=topk_run_dir, action_dim=action_dim
    )
    weights = torch.tensor([float(n) for n in num_events_per_row], dtype=torch.float32)
    if float(weights.sum().item()) <= 0:
        raise RuntimeError("Total event-weight is zero for window-mean aggregation.")
    suite_vec = (matrix * weights[:, None]).sum(dim=0) / weights.sum()
    return _top_pairs(suite_vec, top_k)


def _compute_task_mean_matrix(
    *,
    topk_run_dir: Path,
    prompt_records_path: Path,
    action_dim: int,
) -> tuple[torch.Tensor, list[str], list[int]]:
    """Returns (matrix [num_tasks, dict_size], task_descriptions_sorted,
    step_counts_per_task)."""
    topk_run_dir = Path(topk_run_dir).resolve()
    prompt_records_path = Path(prompt_records_path).resolve()
    manifest = _load_manifest(topk_run_dir)
    dict_size = int(manifest["dict_size"])

    episode_to_task: dict[int, str] = {}
    for record in load_jsonl(prompt_records_path):
        episode_to_task[int(record["episode_num"])] = str(record["task_description"])
    if not episode_to_task:
        raise ValueError(f"prompt_records is empty: {prompt_records_path}")

    task_sums: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(dict_size, dtype=torch.float32))
    task_counts: dict[str, int] = defaultdict(int)
    for ep, _st, step_vector in _build_step_vectors(
        topk_run_dir, dict_size=dict_size, action_dim=action_dim, desc="task-mean shards"
    ):
        task = episode_to_task.get(ep)
        if task is None:
            continue
        task_sums[task] += step_vector
        task_counts[task] += 1

    tasks_sorted = sorted(task_sums)
    matrix = torch.zeros((len(tasks_sorted), dict_size), dtype=torch.float32)
    counts: list[int] = []
    for i, task in enumerate(tasks_sorted):
        matrix[i] = task_sums[task] / float(task_counts[task])
        counts.append(task_counts[task])
    return matrix, tasks_sorted, counts


def task_mean_top_features_per_task(
    *,
    topk_run_dir: Path,
    prompt_records_path: Path,
    top_n: int,
    action_dim: int = 7,
) -> list[dict]:
    """Per task, top-N features by mean SAE activation across every rollout
    step in the run."""
    matrix, tasks, counts = _compute_task_mean_matrix(
        topk_run_dir=topk_run_dir, prompt_records_path=prompt_records_path, action_dim=action_dim
    )
    out: list[dict] = []
    for i, task in enumerate(tasks):
        out.append(
            {
                "ranking": "task_mean",
                "task_description": task,
                "num_steps": counts[i],
                "top_features": _top_pairs(matrix[i], top_n),
            }
        )
    return out


def task_mean_suite_top_k(
    *,
    topk_run_dir: Path,
    prompt_records_path: Path,
    top_k: int,
    action_dim: int = 7,
) -> list[dict]:
    """Suite-level top-K task-mean features: weighted mean of per-task
    means by per-task step count, then top-K."""
    matrix, _tasks, counts = _compute_task_mean_matrix(
        topk_run_dir=topk_run_dir, prompt_records_path=prompt_records_path, action_dim=action_dim
    )
    weights = torch.tensor([float(c) for c in counts], dtype=torch.float32)
    if float(weights.sum().item()) <= 0:
        raise RuntimeError("Total task-step weight is zero for task-mean aggregation.")
    suite_vec = (matrix * weights[:, None]).sum(dim=0) / weights.sum()
    return _top_pairs(suite_vec, top_k)


def alive_feature_ids(topk_run_dir: Path) -> set[int]:
    """Set of feature ids that fire at least once anywhere in the run."""
    topk_run_dir = Path(topk_run_dir).resolve()
    manifest = _load_manifest(topk_run_dir)
    alive: set[int] = set()
    for payload in _iter_shards(topk_run_dir, manifest, desc="alive scan"):
        ids = payload["top_feature_ids"].to(dtype=torch.int64).reshape(-1).tolist()
        alive.update(ids)
    return alive


def random_alive_features(
    *,
    topk_run_dir: Path,
    num_features: int,
    exclude_feature_ids: set[int],
    seed: int = 0,
) -> list[int]:
    """Uniform-random sample of ``num_features`` alive features, excluding
    any feature already selected by the informed rankings. Paper Section
    4.4 random-alive control."""
    alive = alive_feature_ids(topk_run_dir)
    candidates = sorted(alive - set(int(x) for x in exclude_feature_ids))
    if len(candidates) < num_features:
        raise RuntimeError(
            f"Not enough alive features after exclusion: have {len(candidates)}, need {num_features}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, num_features))

"""LIBERO closed-loop eval for openpi (π₀.₅), client side.

The server (a separate process running `scripts/openpi/serve_policy.py`)
hosts the JAX policy and the SAE collector if enabled. This runner
connects via `openpi_client.websocket_client_policy.WebsocketClientPolicy`,
sends per-chunk action requests with optional `_sae_collection_context`,
writes prompt / trajectory / action / chunk records + videos for the
downstream keyframe + clustering pipeline, and reports a summary.

Pared down from openpi-mech's `SAE/raw_baseline/run_eval.py` (664 →
~280 lines). All episodes are kept regardless of success.
"""

from __future__ import annotations

import collections
import contextlib
import csv
import datetime as dt
import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from event_sae.openpi.eval.config import RunConfig
from event_sae.openpi.eval.libero_utils import (
    DEFAULT_MAX_STEPS,
    LIBERO_DUMMY_ACTION,
    make_libero_env,
    observation_state,
    prepare_libero_env,
)


@dataclass
class EvalResult:
    run_dir: str
    success_rate: float
    total_episodes: int
    total_successes: int


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class _JsonlWriter:
    def __init__(self, path: Path, enabled: bool) -> None:
        self._enabled = enabled
        if enabled:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._file = path.open("w", encoding="utf-8")
        else:
            self._file = None

    def write(self, record: dict[str, Any]) -> None:
        if self._file is not None:
            self._file.write(json.dumps(_jsonable(record)) + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()


def _resolve_task_ids(configured, num_tasks: int) -> list[int]:
    """Match openpi-mech run_eval.py::_task_ids: ``configured`` is a list
    of ints or ``None`` (= all tasks)."""
    if configured is None:
        return list(range(num_tasks))
    for task_id in configured:
        if task_id < 0 or task_id >= num_tasks:
            raise ValueError(f"task_id={task_id} out of range for suite with {num_tasks} tasks")
    return list(configured)


def _make_run_dir(cfg: RunConfig) -> Path:
    root = Path(cfg.logging.root_dir).resolve()
    tag = cfg.logging.run_tag or cfg.env.task_suite_name
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / f"EVAL-{tag}-openpi-{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _sae_context(
    *,
    prompt_record: dict[str, Any],
    step_in_episode: int,
    env_t: int,
    obs: dict[str, Any],
    executed_chunk_len: int,
) -> dict[str, Any]:
    return {
        **prompt_record,
        "step_in_episode": step_in_episode,
        "chunk_start_step": step_in_episode,
        "action_chunk_start_step": step_in_episode,
        "env_t": env_t,
        "done": False,
        "reward": None,
        "execution_policy": "prefix_replan",
        "executed_chunk_len": executed_chunk_len,
        "executed_token_start": 0,
        "executed_token_end": executed_chunk_len,
        "wall_timestamp": time.time(),
        "state": observation_state(obs),
        "eef_pos": obs.get("robot0_eef_pos"),
        "eef_quat": obs.get("robot0_eef_quat"),
        "gripper_qpos": obs.get("robot0_gripper_qpos"),
    }


def _finalize_sae_episode(client, episode_num: int, keep: bool) -> dict[str, Any]:
    """Tell the server to flush (keep=True) or drop (keep=False) this
    episode's pending shards. Matches openpi-mech run_eval.py."""
    response = client.infer(
        {
            "_sae_collection_control": {
                "command": "finalize_episode",
                "episode_num": episode_num,
                "keep": keep,
            }
        }
    )
    result = response.get("sae_collection_control")
    if not isinstance(result, dict) or result.get("status") != "ok":
        raise RuntimeError(f"SAE collection finalize failed: {response!r}")
    return result


def _parse_layer_indices(raw: list[int] | str | None) -> list[int] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        return [int(p.strip()) for p in raw.split(",") if p.strip()]
    return [int(v) for v in raw]


@contextlib.contextmanager
def _libero_legacy_torch_load():
    """LIBERO init-state files are trusted local benchmark assets saved
    before PyTorch 2.6 changed ``torch.load`` to default to
    ``weights_only=True``.
    """
    import torch

    original_load = torch.load

    def _load_with_legacy_default(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_with_legacy_default
    try:
        yield
    finally:
        torch.load = original_load


def eval_libero(cfg: RunConfig, *, config_path: Path, libero_root: Path) -> EvalResult:
    prepare_libero_env(cfg.libero, libero_root=libero_root)

    import imageio
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client import websocket_client_policy
    import tqdm

    np.random.seed(cfg.env.seed)

    task_suite = benchmark.get_benchmark_dict()[cfg.env.task_suite_name]()
    task_ids = _resolve_task_ids(cfg.env.task_ids, task_suite.n_tasks)
    max_steps = cfg.env.max_steps or DEFAULT_MAX_STEPS[cfg.env.task_suite_name]

    client = websocket_client_policy.WebsocketClientPolicy(cfg.server.host, cfg.server.port)
    server_metadata = client.get_server_metadata()
    sae_metadata = server_metadata.get("sae_collection") if isinstance(server_metadata, dict) else None
    # The server advertises SAE collection when it is in dense / topk
    # mode. The *client* opts in via ``cfg.sae_collect.enabled``. Only
    # when BOTH sides agree do we coordinate (share run_dir, validate
    # layer/target, inject `_sae_collection_context`, finalize episodes).
    # A common asymmetry: intervention sweeps reuse the dense server for
    # baseline runs but want vanilla eval output — the client opts out
    # via ``--override sae_collect.enabled=false`` and we must skip the
    # coordination path even though the server still advertises.
    server_sae_enabled = isinstance(sae_metadata, dict)
    sae_collection_enabled = server_sae_enabled and bool(cfg.sae_collect.enabled)

    if cfg.sae_collect.enabled and not server_sae_enabled:
        logging.warning(
            "Config has sae_collect.enabled=true but server did not advertise SAE collection."
        )
    # Online TopK mode does not actually drop rows on keep=False (rows
    # enter the global buffer as the JAX callback fires and may be
    # flushed before finalize_episode arrives). If the user asked for
    # ``keep_failed_episodes=False`` AND the server is in TopK mode, warn
    # loudly so they know to filter downstream by success.csv instead of
    # assuming failed-episode shards are absent.
    server_mode = (sae_metadata or {}).get("mode")
    if (
        cfg.sae_collect.enabled
        and not bool(getattr(cfg.sae_collect, "keep_failed_episodes", True))
        and server_mode == "topk"
    ):
        logging.warning(
            "Online TopK mode does not support keep_failed_episodes=False. "
            "Failed-episode rows have already entered the shard buffer and "
            "cannot be retracted. Filter by success.csv downstream instead, "
            "or switch to --mode dense + offline extract_topk.py."
        )
    if sae_collection_enabled:
        if cfg.sae_collect.capture_target and sae_metadata.get("capture_target") != cfg.sae_collect.capture_target:
            raise RuntimeError(
                "SAE capture_target mismatch: "
                f"client={cfg.sae_collect.capture_target!r} server={sae_metadata.get('capture_target')!r}"
            )
        expected_layers = _parse_layer_indices(cfg.sae_collect.layer_idxs)
        if expected_layers is not None and list(sae_metadata.get("layer_indices", [])) != expected_layers:
            raise RuntimeError(
                "SAE layer_indices mismatch: "
                f"client={expected_layers!r} server={sae_metadata.get('layer_indices')!r}"
            )

    # If the server is collecting, dump everything into its run_dir so
    # activation shards + rollout records co-locate. Otherwise create a fresh
    # client-side run_dir.
    artifact_run_dir = (
        Path(sae_metadata["run_dir"]) if sae_collection_enabled else _make_run_dir(cfg)
    )
    if sae_collection_enabled:
        logging.info("Sharing server SAE run dir: %s", artifact_run_dir)
    shutil.copy2(config_path, artifact_run_dir / "config.yaml")

    prompt_w = _JsonlWriter(artifact_run_dir / "prompt_records.jsonl", cfg.logging.save_prompt_records)
    traj_w = _JsonlWriter(artifact_run_dir / "trajectory_records.jsonl", cfg.logging.save_trajectory_records)
    action_w = _JsonlWriter(artifact_run_dir / "actions.jsonl", cfg.logging.save_actions)
    chunk_w = _JsonlWriter(artifact_run_dir / "chunk_records.jsonl", cfg.logging.save_chunk_records)
    success_csv_path = artifact_run_dir / "success.csv"
    success_csv = success_csv_path.open("w", newline="", encoding="utf-8")
    success_writer = csv.DictWriter(
        success_csv,
        fieldnames=["episode_num", "task_id", "task_episode_idx", "task_description", "success", "steps"],
    )
    success_writer.writeheader()

    video_dir = artifact_run_dir / "videos"
    if cfg.logging.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Run dir: %s", artifact_run_dir)
    logging.info("Task suite: %s | task_ids=%s", cfg.env.task_suite_name, task_ids)

    total_episodes = 0
    total_successes = 0
    try:
        for task_id in tqdm.tqdm(task_ids, desc="tasks"):
            task = task_suite.get_task(task_id)
            with _libero_legacy_torch_load():
                initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = make_libero_env(task, cfg.env.resolution, cfg.env.seed)

            for episode_idx in tqdm.tqdm(range(cfg.env.num_trials_per_task), desc=f"task {task_id}", leave=False):
                episode_num = total_episodes
                done = False
                t = 0
                replay_images: list[np.ndarray] = []
                action_plan: collections.deque = collections.deque()

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx % len(initial_states)])

                prompt_record = {
                    "episode_num": episode_num,
                    "task_id": task_id,
                    "task_episode_idx": episode_idx,
                    "task_description": task_description,
                    "prompt_task_description": task_description,
                }
                # When SAE collection is on, defer prompt/chunk/traj writes
                # until _finalize_sae_episode decides keep/drop. Otherwise
                # write immediately.
                episode_chunk_records: list[dict[str, Any]] = []
                episode_step_records: list[dict[str, Any]] = []
                if not sae_collection_enabled:
                    prompt_w.write(prompt_record)

                while t < max_steps + cfg.env.num_steps_wait:
                    if t < cfg.env.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, cfg.server.resize_size, cfg.server.resize_size)
                    )
                    wrist = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist, cfg.server.resize_size, cfg.server.resize_size)
                    )
                    replay_images.append(img)

                    if not action_plan:
                        chunk_start_step = t - cfg.env.num_steps_wait
                        element: dict[str, Any] = {
                            "observation/image": img,
                            "observation/wrist_image": wrist,
                            "observation/state": observation_state(obs),
                            "prompt": str(task_description),
                        }
                        if sae_collection_enabled:
                            element["_sae_collection_context"] = _sae_context(
                                prompt_record=prompt_record,
                                step_in_episode=chunk_start_step,
                                env_t=t,
                                obs=obs,
                                executed_chunk_len=cfg.server.replan_steps,
                            )
                        action_chunk = client.infer(element)["actions"]
                        if len(action_chunk) < cfg.server.replan_steps:
                            raise RuntimeError(
                                f"Policy returned {len(action_chunk)} actions, replan_steps={cfg.server.replan_steps}"
                            )
                        chunk_arr = np.asarray(action_chunk)
                        chunk_record = {
                            **prompt_record,
                            "step_in_episode": int(chunk_start_step),
                            "chunk_start_step": int(chunk_start_step),
                            "env_t": int(t),
                            "executed_chunk_len": int(cfg.server.replan_steps),
                            "action_chunk_len": int(chunk_arr.shape[0]),
                            "action_chunk": chunk_arr,
                        }
                        if sae_collection_enabled:
                            episode_chunk_records.append(chunk_record)
                        else:
                            chunk_w.write(chunk_record)
                        for offset, act in enumerate(action_chunk[: cfg.server.replan_steps]):
                            action_plan.append((act, chunk_start_step, offset, len(action_chunk)))

                    chunk_action, chunk_step, action_offset, chunk_len = action_plan.popleft()
                    action = np.asarray(chunk_action)
                    obs, reward, done, info = env.step(action.tolist())

                    step_record = {
                        **prompt_record,
                        "step_in_episode": t - cfg.env.num_steps_wait,
                        "env_t": t,
                        "done": bool(done),
                        "reward": reward,
                        "state": observation_state(obs),
                        "eef_pos": obs.get("robot0_eef_pos"),
                        "eef_quat": obs.get("robot0_eef_quat"),
                        "gripper_qpos": obs.get("robot0_gripper_qpos"),
                        "action": action,
                        "gripper_action": float(action[-1]),
                        "chunk_start_step": int(chunk_step),
                        "action_chunk_offset": int(action_offset),
                        "token_idx": int(action_offset),
                    }
                    if sae_collection_enabled:
                        episode_step_records.append(step_record)
                    else:
                        traj_w.write(step_record)
                        action_w.write(step_record)

                    t += 1
                    if done:
                        total_successes += 1
                        break

                success_writer.writerow(
                    {
                        "episode_num": episode_num,
                        "task_id": task_id,
                        "task_episode_idx": episode_idx,
                        "task_description": task_description,
                        "success": bool(done),
                        "steps": max(0, t - cfg.env.num_steps_wait),
                    }
                )
                success_csv.flush()

                keep_episode = True
                if sae_collection_enabled and cfg.sae_collect.finalize_episodes:
                    keep_episode = bool(done) or bool(cfg.sae_collect.keep_failed_episodes)
                    finalize_result = _finalize_sae_episode(client, episode_num, keep_episode)
                    logging.info(
                        "Finalized SAE episode=%s keep=%s result=%s",
                        episode_num,
                        keep_episode,
                        finalize_result,
                    )
                if sae_collection_enabled and keep_episode:
                    prompt_w.write(prompt_record)
                    for rec in episode_chunk_records:
                        chunk_w.write(rec)
                    for rec in episode_step_records:
                        traj_w.write(rec)
                        action_w.write(rec)

                if cfg.logging.save_video and replay_images and keep_episode:
                    status = "success" if done else "failure"
                    task_slug = str(task_description).replace(" ", "_").replace("/", "_")[:60]
                    imageio.mimwrite(
                        video_dir / f"episode_{episode_num:05d}_task_{task_id:02d}_{task_slug}_{status}.mp4",
                        replay_images,
                        fps=10,
                    )
                total_episodes += 1
    finally:
        prompt_w.close()
        traj_w.close()
        action_w.close()
        chunk_w.close()
        success_csv.close()

    success_rate = float(total_successes) / max(float(total_episodes), 1.0)
    summary = {
        "run_dir": str(artifact_run_dir),
        "task_suite_name": cfg.env.task_suite_name,
        "task_ids": task_ids,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": success_rate,
    }
    with (artifact_run_dir / "results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return EvalResult(
        run_dir=str(artifact_run_dir),
        success_rate=success_rate,
        total_episodes=total_episodes,
        total_successes=total_successes,
    )

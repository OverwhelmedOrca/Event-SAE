"""Configuration dataclasses and YAML loader for openpi LIBERO eval.

Mirrors `event_sae.openvla.eval.config` where possible. openpi runs a
server-client split, so an extra ``ServerConfig`` block records the
host / port / replan that the openpi-client websocket attaches to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    task_ids: list[int] | None = None   # None = every task in the suite
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7
    resolution: int = 256
    max_steps: int | None = None


@dataclass
class LoggingConfig:
    root_dir: str = "logs/openpi"
    run_tag: str = ""
    save_video: bool = True
    save_actions: bool = True
    save_chunk_records: bool = True
    save_prompt_records: bool = True
    save_trajectory_records: bool = True


@dataclass
class LiberoConfig:
    config_path: str = "examples/libero/libero_config"
    mujoco_gl: str = "egl"


@dataclass
class SAECollectConfig:
    enabled: bool = False
    mode: str = "dense"               # "dense" or "topk"
    capture_target: str = "action_expert"  # "action_expert" or "paligemma"
    layer_idxs: str = "17"            # comma-separated for dense; single int for topk
    flush_every_rows: int = 50_000

    # episode lifecycle (matches openpi-mech SaeCollectConfig)
    finalize_episodes: bool = True    # send _sae_collection_control finalize per episode
    keep_failed_episodes: bool = True   # collection-time default: keep failed-rollout
                                        # activations (still valid training data); flip
                                        # to False only if downstream needs success-only.

    # topk-only
    sae_checkpoint: str = ""
    topk: int = 64
    rows_per_shard: int = 20_000


@dataclass
class RunConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    libero: LiberoConfig = field(default_factory=LiberoConfig)
    sae_collect: SAECollectConfig = field(default_factory=SAECollectConfig)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> RunConfig:
    import yaml

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if overrides:
        data = _deep_update(data, overrides)
    return RunConfig(
        server=ServerConfig(**data.get("server", {})),
        env=EnvConfig(**data.get("env", {})),
        logging=LoggingConfig(**data.get("logging", {})),
        libero=LiberoConfig(**data.get("libero", {})),
        sae_collect=SAECollectConfig(**data.get("sae_collect", {})),
    )


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse ``--override key.path=value`` strings into nested dicts.

    Values are parsed via ``yaml.safe_load`` (matches openpi-mech
    `run_eval.py::_parse_override`), so list / null / bool / int / float
    literals all work, e.g.
    ``env.task_ids=[0,1,2]`` → ``[0, 1, 2]``.
    """
    import yaml

    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides

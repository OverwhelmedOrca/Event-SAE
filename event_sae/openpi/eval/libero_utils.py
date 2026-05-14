"""LIBERO env helpers for openpi-side eval (client process).

Imports from the `libero` package; only safe to call after the LIBERO
config has been written and `LIBERO_CONFIG_PATH` is set in the
environment.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from event_sae.openpi.eval.config import LiberoConfig


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]

DEFAULT_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def prepare_libero_env(cfg: LiberoConfig, *, libero_root: Path) -> Path:
    """Set ``MUJOCO_GL`` and ``LIBERO_CONFIG_PATH`` and write a minimal
    LIBERO config.yaml that points at the bundled assets under
    ``third_party/libero/``."""
    if cfg.mujoco_gl:
        os.environ.setdefault("MUJOCO_GL", cfg.mujoco_gl)

    libero_pkg_root = libero_root / "libero" / "libero"
    config_dir = Path(cfg.config_path).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    config_file = config_dir / "config.yaml"
    payload = {
        "benchmark_root": str(libero_pkg_root),
        "bddl_files": str(libero_pkg_root / "bddl_files"),
        "init_states": str(libero_pkg_root / "init_files"),
        "datasets": str(libero_root / "libero" / "datasets"),
        "assets": str(libero_pkg_root / "assets"),
    }
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return config_file


def make_libero_env(task, resolution: int, seed: int):
    """Construct an ``OffScreenRenderEnv`` for the given LIBERO task."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)
    return env, task.language


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.array(quat, copy=True)
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def observation_state(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    )

"""Structured logging utilities (run dir, CSV, JSON)."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

from event_sae.openvla.eval.utils import DATE_TIME


def create_run_dir(root_dir: str, run_id: str) -> str:
    run_dir = os.path.join(root_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def get_run_id(task_suite: str, model_family: str, tag: str | None = None) -> str:
    # Append a per-process uniqueness suffix so two near-simultaneous jobs
    # never share the same EVAL directory (was a real race in slurm arrays).
    uniq = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
    base = f"EVAL-{task_suite}-{model_family}-{DATE_TIME}-{uniq}"
    if tag:
        base = f"{tag}-{base}"
    return base


def open_log_file(run_dir: str) -> str:
    log_path = os.path.join(run_dir, "stdout.log")
    return log_path


def write_csv_header(csv_path: str):
    with open(csv_path, mode="w", newline="") as f:
        csv.writer(f).writerow(["Task Description", "Task Success Rate"])


def append_csv_row(csv_path: str, task_description: str, success_rate: float):
    with open(csv_path, mode="a", newline="") as f:
        csv.writer(f).writerow([task_description, success_rate])


def save_actions_json(path: str, data: Dict[str, Dict[int, List[List[float]]]]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

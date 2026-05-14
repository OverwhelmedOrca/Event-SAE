"""Tiny JSONL helpers (load + write)."""

from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

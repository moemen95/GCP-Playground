"""Dataset loaders for the golden eval sets."""
from __future__ import annotations

import json
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parent.parent / "datasets"


def dataset_path(name: str) -> Path:
    return DATASETS_DIR / name


def _resolve(name_or_path: str | Path) -> Path:
    """Bare filenames resolve under DATASETS_DIR; anything with a directory
    component (or absolute) is used as given (relative to cwd)."""
    p = Path(name_or_path)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    return DATASETS_DIR / p


def load_jsonl(name_or_path: str | Path) -> list[dict]:
    path = _resolve(name_or_path)
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("//"):
                rows.append(json.loads(line))
    return rows


def write_jsonl(name_or_path: str | Path, rows: list[dict]) -> Path:
    path = _resolve(name_or_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path

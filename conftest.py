"""Pytest bootstrap: make ``src`` importable and force the offline stub backend
so the default suite runs with zero credentials.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
for p in (str(ROOT), str(SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault("EVAL_BACKEND", "stub")

"""Shared utilities for sequential tools and analysis scripts.

Common helpers for timestamps, git metadata, JSON I/O, and key normalization
used across the tools/ directory.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str:
    """Return short git commit hash, or 'unknown' if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


def safe_div(a: float, b: float) -> float:
    """Divide a/b, returning 0.0 if b is zero."""
    return a / b if b else 0.0


def norm_qty(value: Any) -> int:
    """Normalize a quantity value to int >= 1."""
    try:
        q = int(value)
    except Exception:
        q = 1
    return q if q >= 1 else 1


def norm_key_part(value: Any, default: str) -> str:
    """Normalize a key component to lowercase stripped string."""
    s = str(value or "").strip().lower()
    return s or default


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, returning None on any error."""
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def load_items_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load items list from a JSON file with standard schema.

    Handles both {"items": [...]} and bare [...] formats.
    Returns empty list on any error.
    """
    try:
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        return []
    if isinstance(data, dict):
        items = data.get("items")
        if isinstance(items, list):
            return [it for it in items if isinstance(it, dict)]
        return []
    if isinstance(data, list):
        return [it for it in data if isinstance(it, dict)]
    return []


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """Compute precision, recall, F1 from TP/FP/FN counts."""
    p = safe_div(tp, tp + fp)
    r = safe_div(tp, tp + fn)
    f1 = safe_div(2 * p * r, p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 6), "recall": round(r, 6), "f1": round(f1, 6)}


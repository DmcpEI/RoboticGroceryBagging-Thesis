#!/usr/bin/env python3
"""Write one planner-input JSON per b50 scene from a perception run, in the
format of external/risk_aware_mcts/benchmark/planner_input_*/ ({scene_id,
items_data}, one entry per unit keyed "<name>_<n>"): catalog planner fields by
name, unknown_object left out. Rebuilding from runs/b50_detector_v15b
reproduces the files delivered on 2026-09-08 (see --check).

  .venv/bin/python3.12 tools/eval/build_planner_inputs_b50.py \
      --pred runs/b50_detector_v18 --out runs/eval/planner_inputs_b50/our_perception_v18
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from collections import Counter  # noqa: E402

from build_planner_output import (  # noqa: E402
    PLANNER_FIELDS, estimate_attrs, load_planner_attrs, match_attrs,
)

SPILL_VULNERABLE_CATS = {"Bakery", "Produce", "Snacks"}


def build_items(items: list, attrs: dict) -> dict:
    """Perception items -> one planner entry per physical unit."""
    data, seen = {}, Counter()
    for it in items:
        name = str(it.get("name") or "").strip()
        if not name or name == "unknown_object":
            continue
        pa = match_attrs(name, attrs) or estimate_attrs(it)
        for _ in range(max(1, int(it.get("quantity") or 1))):
            seen[name] += 1
            # the planner environment has no `edible`; the delivered files carry none
            row = {f: pa.get(f) for f in PLANNER_FIELDS if f != "edible"}
            row["display_name"] = name
            row["spill_vulnerable"] = bool(
                row.get("spill_vulnerable", row.get("category") in SPILL_VULNERABLE_CATS))
            data[f"{name}_{seen[name]}"] = row
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--check", type=Path, help="compare with an existing directory instead of writing")
    a = ap.parse_args()
    attrs = load_planner_attrs()
    a.out.mkdir(parents=True, exist_ok=True)
    n = bad = 0
    for d in sorted(p for p in a.pred.iterdir() if (p / "frames/frame_000.json").is_file()):
        items = json.loads((d / "frames/frame_000.json").read_text())["items"]
        rec = {"scene_id": d.name, "items_data": build_items(items, attrs)}
        n += len(rec["items_data"])
        if a.check:
            bad += json.loads((a.check / f"{d.name}.json").read_text()) != rec
        else:
            (a.out / f"{d.name}.json").write_text(json.dumps(rec, indent=1))
    print(f"{n} items" + (f", {bad} scenes differ from {a.check}" if a.check else f" -> {a.out}"))


if __name__ == "__main__":
    main()

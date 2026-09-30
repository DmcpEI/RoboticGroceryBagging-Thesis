#!/usr/bin/env python3
"""Recompute ALL catalog attrs (group, packaging, weight_class, cold_chain,
fragile, edible, spill_risk, rigidity, orientation_sensitive, leak_risk,
is_liquid) in-place for an existing run_gemini_robotics.py output, via the
token-overlap + synonym catalog match. Identity is perceived, every other
field is looked up (Section IV-C) -- lets the offline matcher improve
without re-calling the API.

  python tools/eval/reattach_gemini_sku_attrs.py --run runs/robot_lab_prod_49_gemini_er_v1_namelookup
"""
import argparse
import json
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from run_gemini_robotics import full_catalog_attrs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    args = ap.parse_args()

    n = 0
    for fp in sorted(args.run.glob("*/frames/frame_000.json")):
        d = json.loads(fp.read_text())
        for it in d.get("items", []):
            it.update(full_catalog_attrs(it.get("name")))
        fp.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n")
        n += 1
    print(f"[OK] reattached attrs for {n} scenes under {args.run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

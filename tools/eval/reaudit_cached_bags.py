#!/usr/bin/env python3
"""Re-audit bag assignments already stored in a run file.

The safety predicates read `category`, `temperature`, `crush_score` and the
spill flags, all of which come from the catalog rather than from the planner.
When the derivation of those changes -- as it did on 2026-09-19, when `hygiene`
stopped mapping to `Cleaning` -- every cached run's violation count is stale,
and re-running the planner would mean re-calling an API for assignments that
have not changed. This replays the predicates over the stored bags instead.

  .venv/bin/python3.12 tools/eval/reaudit_cached_bags.py \
      --run runs/eval/gemini_full_pipeline_gt_b50.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.planner.perception_to_packbot_adapter import _build_extended_packbot_item  # noqa: E402
from tools.planner.planner_safety_audit import pair_violations_for_items  # noqa: E402


def props_for(name: str, attrs: dict) -> dict | None:
    a = attrs.get(name)
    if a is None:
        return None
    return _build_extended_packbot_item({"name": name, **a})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--attrs", type=Path,
                    default=ROOT / "data/robot_lab/robot_sku_attrs.json")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    run = json.loads(args.run.read_text())
    attrs = json.loads(args.attrs.read_text())

    fam = Counter()
    pairs = 0
    scenes_hit = 0
    unresolved = Counter()
    per_scene = []
    for sc in run.get("per_scene", []):
        hits = Counter()
        for bag in sc.get("bags", []):
            props = []
            for n in bag:
                p = props_for(n, attrs)
                if p is None:
                    unresolved[n] += 1
                else:
                    props.append(p)
            for i in range(len(props)):
                for j in range(i + 1, len(props)):
                    pairs += 1
                    # list order is bottom-to-top, which is what the prompt asks for
                    for f in pair_violations_for_items(props[i], props[j], ordered=True):
                        hits[f] += 1
        if hits:
            scenes_hit += 1
        fam.update(hits)
        per_scene.append({"scene": sc.get("scene"), "violations": dict(hits)})

    total = sum(fam.values())
    print(f"{args.run.name}")
    print(f"  scenes {len(run.get('per_scene', []))}   pairs audited {pairs}")
    print(f"  violations {total}   scenes affected {scenes_hit}")
    for k, v in fam.most_common():
        print(f"     {k:34s} {v}")
    if unresolved:
        print(f"  names not in the catalog: {sum(unresolved.values())} "
              f"({len(unresolved)} distinct, e.g. {list(unresolved)[:4]})")
    print(f"  PREVIOUSLY: violations {run.get('total_violations')} over "
          f"{run.get('total_pairs_audited')} pairs")
    if args.json:
        args.json.write_text(json.dumps(
            {"source": str(args.run), "total_violations": total,
             "total_pairs_audited": pairs, "scenes_affected": scenes_hit,
             "violations": dict(fam), "per_scene": per_scene}, indent=1) + "\n")
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

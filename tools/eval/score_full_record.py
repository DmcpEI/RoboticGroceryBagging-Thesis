#!/usr/bin/env python3
"""How often is the WHOLE record right, not just three fields of it?

Table III's headline counted an object correct when its group, packaging and
weight class matched. Those three predate the current planner interface, which
also reads rigidity, crush score, cold-chain and spill risk. This scores the
record the planner actually receives: an object is correct only when EVERY
characteristic matches.

Reported as a nested series -- fields added one at a time, always in the same
order -- so the cost of each additional field is visible rather than folded
into one number. Quantity needs no column: every item is expanded to one entry
per physical object, so predicting the wrong count already costs a false
positive and a false negative.

Matching is a multiset intersection per scene, the same as every other scorer
here: no boxes exist, so an object is correct if some prediction in the same
scene carries an identical record.

  python tools/eval/score_full_record.py --gt datasets_seq_GT_b50 \
      --run detector=runs/b50_detector_v15b --out runs/eval/b50_full_record.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/eval"))

from score_characteristics import value  # noqa: E402

# Order matters: each row of the output adds the next field to the one above.
# group and packaging first because they are what the schema is built on, then
# the planner-facing fields in the order Section IV introduces them.
# The characteristics a system is actually asked to perceive. crush_score and
# edible are NOT here: in this pipeline both are derived -- edible from group,
# crush_score from packaging, rigidity and group -- so putting them in the
# tuple scores the same evidence twice. Pass them with --fields to see the
# whole record the planner receives.
LADDER = ["group", "packaging", "weight_class", "rigidity",
          "cold_chain", "spill_risk"]


def records(items, fields):
    """One tuple per physical object; None anywhere means the record is
    incomplete, which is not the same as being wrong about a value."""
    out = []
    for it in items or []:
        rec = tuple(value(it, f) for f in fields)
        out += [rec] * max(1, int(it.get("quantity", 1) or 1))
    return out


def score(gt_root: Path, pred_root: Path, fields):
    tp = n_gt = n_pred = 0
    for gfp in sorted(gt_root.glob("*/frames/frame_000.json")):
        sc = gfp.parent.parent.name
        g = json.loads(gfp.read_text()).get("items", [])
        pfp = pred_root / sc / "frames/frame_000.json"
        p = json.loads(pfp.read_text()).get("items", []) if pfp.exists() else []
        G, P = Counter(records(g, fields)), Counter(records(p, fields))
        # An incomplete record cannot be credited, but the object it belongs to
        # is still there, so it stays in the denominator on both sides.
        hit = sum(n for r, n in (G & P).items() if all(v is not None for v in r))
        tp += hit
        n_gt += sum(G.values())
        n_pred += sum(P.values())
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "gt_objects": n_gt, "predicted_objects": n_pred}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--fields", help="comma-separated override for the ladder")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    runs = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.run]
    ladder = [f.strip() for f in args.fields.split(",")] if args.fields else LADDER
    report = {"gt": str(args.gt), "ladder": ladder, "runs": {}}

    width = max(len(n) for n, _ in runs) + 2
    print(f"{'fields':<34}" + "".join(f"{n:>{width}}" for n, _ in runs))
    print("-" * (34 + width * len(runs)))
    rows = []
    for i in range(1, len(ladder) + 1):
        fields = ladder[:i]
        label = "+" + fields[-1] if i > 1 else fields[0]
        cells = []
        for name, path in runs:
            r = score(args.gt, path, fields)
            report["runs"].setdefault(name, {})["+".join(fields)] = r
            cells.append(r["f1"])
        rows.append((label, cells))
        print(f"{label:<34}" + "".join(f"{c:>{width}.4f}" for c in cells))
    print("-" * (34 + width * len(runs)))
    print(f"{'ALL ' + str(len(ladder)) + ' characteristics':<34}"
          + "".join(f"{c:>{width}.4f}" for c in rows[-1][1]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\n[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

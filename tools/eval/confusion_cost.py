#!/usr/bin/env python3
"""When the detector names an object wrongly, what does the planner lose?

A name error is usually reported as a name error. That is the wrong unit for a
system whose output is a set of characteristics: two products can carry the same
group, packaging, weight class, cold-chain flag, spill risk and rigidity, and if
the detector swaps one for the other the record it hands the planner is
identical. The substitution is then free, whatever it does to an identity score.

This pairs each missed object with a surplus prediction in the SAME scene --
the same attribution the name-confusion analysis uses, and no more reliable
than that, since there are no ground-truth boxes to match against -- and asks
whether the substitute is treated the same by the solver: the same membership of
every separation set (chemical, raw meat, refrigerated, may leak, spoiled by a
leak) and the same load class (protected, neutral, heavy). Where it is not, it
reports which of these moved. Other fields (weight class, rigidity) only reach
the solver through capacity or through the load class, so they are not counted
on their own.

Reported as an attribution, not a measurement: a scene with one miss and one
surplus is assumed to be one confusion, which is usually but not always true.

  python tools/eval/confusion_cost.py --weights <best.pt> --prefix "bag of"
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "tools/eval"))
sys.path.insert(0, str(ROOT / "tools/planner"))
from build_planner_output import load_planner_attrs  # noqa: E402
import planner_safety_audit as psa  # noqa: E402


def predicates(p: dict) -> dict:
    """What the solver reads from one item, via the audit's own predicates."""
    if not p:
        return {}
    load = "protected" if psa._is_fragile(p) else "heavy" if psa._is_heavy(p) else "neutral"
    return {"chemical": psa._is_cleaning(p), "raw_meat": psa._is_raw_meat(p),
            "refrigerated": psa._is_nonambient(p), "may_leak": bool(p.get("spill_risk")),
            "spoiled_by_leak": bool(p.get("spill_vulnerable")), "load": load}


SAFETY = ("chemical", "raw_meat", "refrigerated", "may_leak", "spoiled_by_leak", "load")
KEYS = SAFETY


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "robot/robot_pc_package/weights/best.pt")
    ap.add_argument("--classes", type=Path, default=ROOT / "robot/robot_pc_package/classes.json")
    ap.add_argument("--gt", type=Path, default=ROOT / "datasets_seq_GT_b50")
    ap.add_argument("--images", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--prefix", default="", help="restrict to classes starting with this")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO
    names = json.loads(args.classes.read_text())
    sku = {n: predicates(p) for n, p in load_planner_attrs().items()}
    sig = lambda n: tuple(sku.get(n, {}).get(k) for k in KEYS)
    sel = {n for n in names if n.startswith(args.prefix)}
    model = YOLO(str(args.weights))

    pairs, same, diff, fields, safety_hits = Counter(), 0, 0, Counter(), 0
    for d in sorted(args.gt.glob("*")):
        f, img = d / "frames/frame_000.json", args.images / d.name / "color.png"
        if not (f.exists() and img.exists()):
            continue
        gt = Counter()
        for i in json.loads(f.read_text())["items"]:
            gt[i["name"]] += i.get("quantity", 1)
        if not set(gt) & sel:
            continue
        r = model(str(img), conf=args.conf, verbose=False)[0]
        got = Counter(names[int(c)] for c in
                      (r.boxes.cls.tolist() if r.boxes is not None else []))
        surplus = [k for k in got if k in sel and got[k] > gt.get(k, 0)
                   for _ in range(got[k] - gt.get(k, 0))]
        for b in sorted(set(gt) & sel):
            for _ in range(max(0, gt[b] - got[b])):
                if not surplus:
                    continue
                s = surplus.pop(0)
                pairs[(b, s)] += 1
                if sig(b) == sig(s):
                    same += 1
                    continue
                diff += 1
                moved = [k for k in KEYS if sku.get(b, {}).get(k) != sku.get(s, {}).get(k)]
                for k in moved:
                    fields[k] += 1
                if any(k in SAFETY for k in moved):
                    safety_hits += 1

    n = same + diff
    if not n:
        print("no substitutions attributed -- nothing missed, or nothing surplus to pair with")
        return 0
    print(f"{n} substitutions attributed"
          + (f" among classes starting '{args.prefix}'" if args.prefix else ""))
    print(f"  same for the solver      : {same:3d}  ({same/n:.0%})")
    print(f"  record differs           : {diff:3d}  ({diff/n:.0%})")
    print(f"  ... of which touch a SAFETY field ({', '.join(SAFETY)}): {safety_hits}")
    if fields:
        print("\nfields that move, counted over the substitutions that change anything:")
        for k, v in fields.most_common():
            print(f"   {k:14}{v:4d}" + ("   <-- safety" if k in SAFETY else ""))
    print("\ntop substitutions:")
    for (b, s), c in pairs.most_common(8):
        print(f"   {c:3d}  {b:24} -> {s}" + ("   (same record)" if sig(b) == sig(s) else ""))

    if args.json:
        args.json.write_text(json.dumps(
            dict(n=n, same=same, diff=diff, safety=safety_hits,
                 fields=dict(fields),
                 pairs={f"{b} -> {s}": c for (b, s), c in pairs.items()}), indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Localisation separated from naming, on ground-truth boxes that cost nothing.

This work has never reported a localisation number. A figure that claimed to be
one was retracted in August: the file behind it computed no box overlap at all,
and its tp/fp/fn were byte-identical to a name scorer's. So "does it find the
object" and "does it name the object" have only ever been measured together.

Removal differencing supplies the missing ground truth. The camera is fixed and
exactly one object leaves between consecutive steps, whose name the capture
recorded, so the region raised at step t and flat at t+1 is that object's box,
already labelled. `removal_diff_boxes.py` extracts them; this scores against
them.

Two quantities per IoU threshold:

  localisation  a predicted box overlaps the GT box, whatever it is called.
  naming        one of those overlapping boxes also carries the right name.

The gap between them is the part of the error that is naming rather than
finding, which is the split the retracted figure pretended to have.

**RECALL ONLY, and this is not a limitation that can be worked around.** The GT
covers the ONE object that left each step, not everything on the table, so a
predicted box with no GT box under it may be a false positive or may be a
correctly found object the GT does not cover. Precision and mAP are not
computable from this supervision and are not reported.

  python tools/eval/localisation_recall.py --weights <best.pt>
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boxes", type=Path, default=ROOT / "runs/eval/removal_diff_boxes_v2.json")
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "robot/robot_pc_package/weights/best.pt")
    ap.add_argument("--classes", type=Path, default=ROOT / "robot/robot_pc_package/classes.json")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--ious", default="0.3,0.5,0.75")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO
    names = json.loads(args.classes.read_text())
    model = YOLO(str(args.weights))
    try:
        import torch
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        print(f"model: {(ck.get('train_args') or {}).get('name', '?')}  "
              f"trained {str(ck.get('date'))[:10]}")
    except Exception as exc:
        print(f"model: could not read training metadata ({type(exc).__name__})")

    events = [e for e in json.loads(args.boxes.read_text()) if e.get("ok") and e.get("bbox_px")]
    if not events:
        raise SystemExit(f"no clean boxes in {args.boxes}")
    thresholds = [float(t) for t in args.ious.split(",")]

    # one detection pass per frame; several events never share a prev frame, but
    # caching costs nothing and keeps the count of model calls honest
    cache, rows = {}, []
    for e in events:
        img = ROOT / e["src_dir_prev"] / "color.png"
        if img not in cache:
            r = model(str(img), conf=args.conf, verbose=False)[0]
            cache[img] = ([] if r.boxes is None else
                          [(names[int(c)], list(map(float, b)))
                           for c, b in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist())])
        gt = e["bbox_px"]
        best_any = max((iou(gt, b) for _, b in cache[img]), default=0.0)
        best_named = max((iou(gt, b) for n, b in cache[img] if n == e["name"]), default=0.0)
        rows.append((e["name"], e["scene"], best_any, best_named))

    print(f"\n{len(events)} labelled boxes from {len({r[1] for r in rows})} sequences, "
          f"{len(cache)} frames, conf={args.conf}")
    print("recall only -- the GT covers one object per step, so precision is not computable\n")
    print(f"{'IoU':>6}{'localisation':>15}{'naming':>10}{'gap':>8}")
    out = {}
    for t in thresholds:
        loc = sum(1 for _, _, a, _ in rows if a >= t) / len(rows)
        nam = sum(1 for _, _, _, n in rows if n >= t) / len(rows)
        out[str(t)] = dict(localisation=loc, naming=nam, n=len(rows))
        print(f"{t:6.2f}{loc:15.3f}{nam:10.3f}{loc - nam:+8.3f}")

    t = thresholds[min(1, len(thresholds) - 1)]
    per = defaultdict(lambda: [0, 0, 0])
    for name, _, a, n in rows:
        per[name][0] += 1
        per[name][1] += a >= t
        per[name][2] += n >= t
    print(f"\nper class at IoU {t:.2f}  (classes with a clean box; "
          f"a flat object leaves no raised region and so appears here not at all)")
    print(f"{'class':28}{'n':>4}{'localised':>11}{'named':>8}")
    for name, (c, a, n) in sorted(per.items(), key=lambda kv: (kv[1][2] / kv[1][0], -kv[1][0])):
        print(f"{name:28}{c:4d}{a / c:11.2f}{n / c:8.2f}")

    if args.json:
        args.json.write_text(json.dumps(
            dict(weights=str(args.weights), conf=args.conf, n=len(rows),
                 thresholds=out,
                 per_class={k: dict(n=v[0], localised=v[1] / v[0], named=v[2] / v[0])
                            for k, v in per.items()}), indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

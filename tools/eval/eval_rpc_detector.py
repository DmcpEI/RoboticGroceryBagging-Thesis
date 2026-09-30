#!/usr/bin/env python3
"""Evaluate the 200-SKU synthetic-composite detector on real RPC val scenes (E1b).

Two metrics, deliberately:
  * name_f1 -- greedy multiset match of predicted vs GT SKU names over a
    confidence grid, the same end-to-end identity metric used for the robot
    detector, so the 43-SKU and 200-SKU results sit in the same units.
  * identity_on_gt_boxes -- each GT box takes the class of its highest-IoU
    detection. This isolates identity from localization and is therefore
    directly comparable to the retrieval index's top-1 SKU accuracy (E1),
    which was also computed on GT boxes.

Unlike the robot scorer, name matching is EXACT: RPC class names are indexed
("1_puffed_food" is a substring of "11_puffed_food"), so the fuzzy substring
matcher used there would create false positives.

  python tools/eval/eval_rpc_detector.py --weights <best.pt> \
      --classes data/rpc_synthetic_det/classes.json \
      --staged datasets_perception/rpc_val_600 --out runs/eval/rpc_detector.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = json.loads((ROOT / "data/rpc_category_names.json").read_text())


def meta_of(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


def greedy_exact(gt, pred):
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        for i, p in enumerate(pred):
            if not used[i] and p == g:
                used[i] = True
                tp += 1
                break
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gt-iou", type=float, default=0.3)
    args = ap.parse_args()

    labels = json.loads(args.classes.read_text())
    model = YOLO(str(args.weights))

    scenes = sorted(p for p in args.staged.iterdir() if p.is_dir())
    if args.limit:
        scenes = scenes[:args.limit]

    cached = []                       # (gt_names, gt_boxes, dets)
    for sd in scenes:
        gt = json.loads((sd / "gt.json").read_text())
        gt_names = [CATEGORY_NAMES[c] if isinstance(c, int) else c for c in gt["categories"]]
        gt_boxes = [(x, y, x + w, y + h) for x, y, w, h in gt["bboxes"]]
        r = model(str(sd / "frames/frame_000.png"), imgsz=args.imgsz, verbose=False)[0]
        dets = []
        if r.boxes is not None:
            for cls, conf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                       r.boxes.xyxy.tolist()):
                dets.append((labels[int(cls)], float(conf), tuple(xyxy)))
        cached.append((gt_names, gt_boxes, dets))

    grid, best = [], None
    for mc in (0.10, 0.25, 0.40, 0.55, 0.70):
        tot = [0, 0, 0]
        for gt_names, _, dets in cached:
            tp, fp, fn = greedy_exact(gt_names, [n for n, c, _ in dets if c >= mc])
            tot[0] += tp; tot[1] += fp; tot[2] += fn
        p, r, f = prf(*tot)
        g = {"min_conf": mc, "P": round(p, 3), "R": round(r, 3), "F1": round(f, 3)}
        grid.append(g)
        if best is None or f > best["F1"]:
            best = g

    # identity on GT boxes -- comparable to the retrieval index's top-1
    hit = tot_inst = covered = 0
    per_meta = defaultdict(lambda: [0, 0])
    for gt_names, gt_boxes, dets in cached:
        for gname, gbox in zip(gt_names, gt_boxes):
            tot_inst += 1
            m = meta_of(gname)
            per_meta[m][1] += 1
            bi, bv = None, args.gt_iou
            for name, conf, box in dets:
                v = iou(gbox, box)
                if v >= bv:
                    bi, bv = name, v
            if bi is not None:
                covered += 1
                if bi == gname:
                    hit += 1
                    per_meta[m][0] += 1

    res = {
        "n_scenes": len(cached),
        "n_instances": tot_inst,
        "imgsz": args.imgsz,
        "weights": str(args.weights),
        "name_f1_grid": grid,
        "name_f1_best": best,
        "identity_on_gt_boxes": {
            "top1_sku_acc": round(hit / tot_inst, 4) if tot_inst else 0.0,
            "gt_box_coverage": round(covered / tot_inst, 4) if tot_inst else 0.0,
            "iou_thresh": args.gt_iou,
            "per_meta": {m: {"acc": round(v[0] / v[1], 3), "n": v[1]}
                          for m, v in sorted(per_meta.items())},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("n_scenes", "n_instances", "name_f1_best")}, indent=1))
    print(json.dumps({k: v for k, v in res["identity_on_gt_boxes"].items() if k != "per_meta"}, indent=1))
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

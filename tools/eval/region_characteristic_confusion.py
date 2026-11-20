#!/usr/bin/env python3
"""Confusion matrices per characteristic, on the labelled regions of the
removal sequences.

The static benchmark records which products are in a scene but not where, so a
wrong value there cannot be tied to the object it replaced. The removal
sequences can: each of the 137 clean regions is one known object at a known
place. Each region takes the name of the detection overlapping it most (IoU >=
0.3, as in robot_identity_comparison.py), and the characteristics of the true
and the predicted product are read from the catalog. A region no detection
covers is "not found".

  .venv/bin/python3.12 tools/eval/region_characteristic_confusion.py \
      --weights robot/robot_pc_package/weights/best.pt \
      --classes robot/robot_pc_package/classes.json \
      --out runs/eval/region_characteristic_confusion_v18.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.planner.perception_to_packbot_adapter import compute_crush_score_v2  # noqa: E402

FIELDS = ["group", "packaging", "weight_class", "rigidity", "crush_score",
          "cold_chain", "spill_risk", "edible"]
MISS = "not found"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u else 0.0


def attrs(entry):
    out = {f: entry.get(f) for f in FIELDS if f != "crush_score"}
    out["crush_score"] = compute_crush_score_v2(entry["packaging"], entry["rigidity"], entry["name"],
                                                entry["group"], entry.get("deformation") or "")
    return {k: str(v).lower() for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--boxes", type=Path, default=ROOT / "runs/eval/removal_diff_boxes_v2.json")
    ap.add_argument("--catalog", type=Path, default=ROOT / "data/robot_lab/robot_item_catalog_perception.json")
    ap.add_argument("--gt-iou", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from ultralytics import YOLO
    det = YOLO(str(a.weights))
    labels = json.loads(a.classes.read_text())
    cat = {e["name"]: attrs(e) for e in json.loads(a.catalog.read_text())["items"]
           if e["name"] in labels}
    boxes = [b for b in json.loads(a.boxes.read_text()) if b.get("ok") and b["name"] in cat]

    pairs = []
    for b in boxes:
        r = det(str(Path(b["src_dir_prev"]) / "color.png"), verbose=False)[0]
        best, bv = None, a.gt_iou
        for cls, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
            v = iou(b["bbox_px"], xyxy)
            if v >= bv:
                best, bv = labels[int(cls)], v
        pairs.append({"scene": b["scene"], "step": b["step_prev"], "true": b["name"], "pred": best})

    n = len(pairs)
    named = sum(p["pred"] == p["true"] for p in pairs)
    found = sum(p["pred"] is not None for p in pairs)
    matrices = {}
    for f in FIELDS:
        m = defaultdict(Counter)
        for p in pairs:
            t = cat[p["true"]][f]
            m[t][MISS if p["pred"] is None else cat[p["pred"]][f]] += 1
        matrices[f] = {t: dict(c) for t, c in sorted(m.items())}
    names = defaultdict(Counter)
    for p in pairs:
        names[p["true"]][p["pred"] or MISS] += 1

    res = {"regions": n, "found": found, "named_correctly": named,
           "identity_top1": round(named / n, 4), "coverage": round(found / n, 4),
           "matrices": matrices,
           "characteristic_correct_when_found": {
               f: round(sum(cat[p["true"]][f] == cat[p["pred"]][f] for p in pairs if p["pred"]) / found, 4)
               for f in FIELDS},
           "name_confusions": {t: dict(c) for t, c in sorted(names.items()) if set(c) != {t}},
           "pairs": pairs}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"{n} regions, found {found}, named {named} ({named / n:.4f}) -> {a.out}")
    for f in FIELDS:
        print(f, res["characteristic_correct_when_found"][f], matrices[f])


if __name__ == "__main__":
    main()

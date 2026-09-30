#!/usr/bin/env python3
"""Run the 200-SKU detector on full RPC scenes and write a catalog-lookup inventory.

Unlike runs/rpc_retrieval_lookup, which names objects inside the ground-truth
boxes, nothing is given here: the detector finds and counts the objects itself,
so the inventory's quantity is perceived and can be scored. Output layout is the
one score_characteristics.py reads.

  python tools/eval/rpc_detector_inventory.py \
      --weights runs/detect/runs/detect/runs/detector/rpc_syn_v1/weights/best.pt \
      --classes data/rpc_synthetic_det/classes.json \
      --staged datasets_perception/rpc_val_600 --out runs/rpc_detector_lookup
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ["group", "packaging", "weight_class", "rigidity", "cold_chain",
          "spill_risk", "edible", "deformation"]


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--attrs", type=Path, default=ROOT / "data/rpc_sku_load_attrs.json")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--min-conf", type=float, default=0.40)
    args = ap.parse_args()

    labels = json.loads(args.classes.read_text())
    attrs = json.loads(args.attrs.read_text())          # keyed by SKU id, "79"
    model = YOLO(str(args.weights))
    scenes = sorted(p for p in args.staged.iterdir() if p.is_dir())
    n_obj = 0
    for sd in scenes:
        r = model(str(sd / "frames/frame_000.png"), imgsz=args.imgsz,
                  conf=args.min_conf, verbose=False)[0]
        items = []
        for cls in (r.boxes.cls.tolist() if r.boxes is not None else []):
            name = labels[int(cls)]
            a = attrs[name.split("_", 1)[0]]
            items.append({"name": name, "quantity": 1, **{f: a.get(f) for f in FIELDS}})
        n_obj += len(items)
        fr = args.out / sd.name / "frames"
        fr.mkdir(parents=True, exist_ok=True)
        (fr / "frame_000.json").write_text(json.dumps({"items": items}, indent=1) + "\n")
    print(f"{len(scenes)} scenes, {n_obj} detections -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

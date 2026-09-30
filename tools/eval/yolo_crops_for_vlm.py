#!/usr/bin/env python3
"""Detect with YOLO, crop each detection, for VLM re-classification (hybrid test).

Tests the YOLO-detect + VLM-classify hybrid: YOLO gives the boxes (strong detector),
the VLM re-names each crop (strong classifier). Emits one crop PNG per detection plus
a manifest carrying YOLO's own name+confidence, so the reassembler can compare
YOLO-alone vs all-VLM vs VLM-only-on-low-confidence-detections.

Usage (cluster):
  python tools/eval/yolo_crops_for_vlm.py \
      --weights runs/detect/runs/detector/syn_v4_bleach/weights/best.pt \
      --classes data/robot_lab/synthetic_det_v3/classes.json \
      --pattern 'mix_*' --pattern 'rem_*' --out data/robot_lab/yolo_crops
Then run the production crop-single VLM on <out>, then score_yolo_vlm_hybrid.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    from ultralytics import YOLO
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--pattern", action="append", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-conf", type=float, default=0.15, help="skip detections below this")
    ap.add_argument("--margin", type=float, default=0.12, help="bbox expansion fraction for the crop")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    labels = json.loads(args.classes.read_text())
    model = YOLO(str(args.weights))
    args.out.mkdir(parents=True, exist_ok=True)

    dirs = []
    for pat in args.pattern:
        dirs += [d for d in args.rgbd_root.glob(pat) if d.is_dir()]
    dirs = sorted(set(dirs))

    crops = []
    for d in dirs:
        cfp, mfp = d / "color.png", d / "meta.json"
        if not (cfp.exists() and mfp.exists()):
            continue
        gt = json.loads(mfp.read_text()).get("gt_objects", [])
        img = Image.open(cfp).convert("RGB")
        W, H = img.size
        r = model(str(cfp), imgsz=args.imgsz, verbose=False)[0]
        if r.boxes is None:
            continue
        for k, (cls, conf, xyxy) in enumerate(zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                                   r.boxes.xyxy.tolist())):
            conf = float(conf)
            if conf < args.min_conf:
                continue
            x1, y1, x2, y2 = xyxy
            mw, mh = (x2 - x1) * args.margin, (y2 - y1) * args.margin
            cx1, cy1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
            cx2, cy2 = min(W, int(x2 + mw)), min(H, int(y2 + mh))
            name = f"{d.name}__d{k}"
            img.crop((cx1, cy1, cx2, cy2)).save(args.out / f"{name}.png")
            crops.append({"crop": f"{name}.png", "src_dir": d.name,
                          "bbox_px": [cx1, cy1, cx2, cy2],
                          "yolo_name": labels[int(cls)], "yolo_conf": round(conf, 4),
                          "gt_objects": gt})
    (args.out / "crop_manifest.json").write_text(json.dumps(crops, indent=1) + "\n")
    print(f"{len(crops)} YOLO-detection crops from {len(dirs)} dirs -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

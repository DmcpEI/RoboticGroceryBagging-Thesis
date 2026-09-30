#!/usr/bin/env python3
"""E2: identity methods compared on the robot 43-SKU catalog, same protocol as RPC.

Fills the missing cell of the identity-method table. The RPC study measured
retrieval and the closed-set detector on GT boxes at 200 SKUs; this runs the
identical comparison at 43 SKUs so the two scales are directly comparable.

Protocol (mirrors tools/eval/rpc_retrieval_eval.py):
  index    263 isolated single-item photos (data/robot_lab/cropped_single_items),
           labelled by single_items_gt.csv -- the robot equivalent of RPC's
           exemplar split.
  test     the removal-differencing auto-GT boxes (runs/eval/removal_diff_boxes.json):
           real cluttered scenes, real labelled boxes, DIFFERENT captures from the
           singles, so there is no leakage between index and test.
  metrics  retrieval top-1 SKU on GT boxes; detector class on the same GT boxes
           via highest-IoU match. Both are "identity given a correct box", which
           is what makes them comparable to the RPC numbers.

  python tools/eval/robot_identity_comparison.py \
      --det-weights runs/detect/runs/detector/syn_v4_bleach/weights/best.pt \
      --classes data/robot_lab/synthetic_det_v3/classes.json \
      --out runs/eval/robot_identity_comparison.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


@torch.no_grad()
def embed(crops, proc, model, device, batch=32):
    out = []
    for i in range(0, len(crops), batch):
        px = proc(images=crops[i:i + batch], return_tensors="pt").to(device)
        h = model(**px).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main() -> int:
    from transformers import AutoImageProcessor, AutoModel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--singles", type=Path, default=ROOT / "data/robot_lab/cropped_single_items")
    ap.add_argument("--singles-gt", type=Path, default=ROOT / "data/robot_lab/single_items_gt.csv")
    ap.add_argument("--boxes", type=Path, default=ROOT / "runs/eval/removal_diff_boxes.json")
    ap.add_argument("--det-weights", type=Path)
    ap.add_argument("--classes", type=Path)
    ap.add_argument("--model", default="facebook/dinov2-large")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--gt-iou", type=float, default=0.3)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model, dtype=torch.float16).to(device).eval()

    # ---- index over isolated singles
    rows = list(csv.DictReader(args.singles_gt.open()))
    imgs, labels = [], []
    for r in rows:
        fp = args.singles / r["image"]
        if fp.exists():
            imgs.append(Image.open(fp).convert("RGB"))
            labels.append(r["object"])
    names = sorted(set(labels))
    idx_of = {n: i for i, n in enumerate(names)}
    E = torch.from_numpy(embed(imgs, proc, enc, device)).to(device)
    L = torch.tensor([idx_of[l] for l in labels], device=device)
    print(f"index: {len(imgs)} singles / {len(names)} classes", flush=True)

    boxes = [b for b in json.loads(args.boxes.read_text()) if b.get("ok")]
    in_vocab = [b for b in boxes if b["name"] in idx_of]
    skipped = sorted({b["name"] for b in boxes if b["name"] not in idx_of})

    det = None
    if args.det_weights and args.classes:
        from ultralytics import YOLO
        det = YOLO(str(args.det_weights))
        det_labels = json.loads(args.classes.read_text())

    ret_hit = det_hit = det_cov = 0
    per_class = defaultdict(lambda: [0, 0, 0])   # cls -> [ret_hit, det_hit, n]
    confusions = defaultdict(int)

    for b in in_vocab:
        img = Image.open(Path(b["src_dir_prev"]) / "color.png").convert("RGB")
        x1, y1, x2, y2 = b["bbox_px"]
        crop = img.crop((x1, y1, x2, y2))
        truth = b["name"]

        sim = None
        for deg in ((0,) if args.no_tta else (0, 90, 180, 270)):
            q = crop.rotate(deg, expand=True) if deg else crop
            s = torch.from_numpy(embed([q], proc, enc, device)).to(device) @ E.T
            sim = s if sim is None else torch.maximum(sim, s)
        topk = sim.topk(min(args.k, E.shape[0]), dim=1)
        votes = torch.zeros(1, len(names), device=device)
        votes.scatter_add_(1, L[topk.indices], topk.values.clamp(min=0))
        pred = names[int(votes.argmax())]
        per_class[truth][2] += 1
        if pred == truth:
            ret_hit += 1
            per_class[truth][0] += 1
        else:
            confusions[f"{truth} -> {pred}"] += 1

        if det is not None:
            r = det(str(Path(b["src_dir_prev"]) / "color.png"), verbose=False)[0]
            best, bv = None, args.gt_iou
            if r.boxes is not None:
                for cls, xyxy in zip(r.boxes.cls.tolist(), r.boxes.xyxy.tolist()):
                    v = iou((x1, y1, x2, y2), tuple(xyxy))
                    if v >= bv:
                        best, bv = det_labels[int(cls)], v
            if best is not None:
                det_cov += 1
                if best == truth:
                    det_hit += 1
                    per_class[truth][1] += 1

    n = len(in_vocab)
    res = {
        "n_index_singles": len(imgs),
        "n_classes": len(names),
        "n_test_boxes": n,
        "n_boxes_skipped_oov": len(boxes) - n,
        "skipped_labels": skipped,
        "retrieval_top1": round(ret_hit / n, 4) if n else 0.0,
        "detector_top1": round(det_hit / n, 4) if (n and det) else None,
        "detector_gt_box_coverage": round(det_cov / n, 4) if (n and det) else None,
        "per_class": {c: {"retrieval": round(v[0] / v[2], 3),
                           "detector": round(v[1] / v[2], 3), "n": v[2]}
                       for c, v in sorted(per_class.items())},
        "retrieval_confusions": dict(sorted(confusions.items(), key=lambda kv: -kv[1])[:15]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in
                      ("n_index_singles", "n_classes", "n_test_boxes", "n_boxes_skipped_oov",
                       "retrieval_top1", "detector_top1", "detector_gt_box_coverage")}, indent=1))
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

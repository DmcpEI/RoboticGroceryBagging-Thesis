#!/usr/bin/env python3
"""E1: score identity-by-retrieval on the staged RPC validation scenes.

RPC val gt.json carries the exact per-instance SKU (200-class int into
data/rpc_category_names.json) plus its bbox -- our earlier scorers called
meta_of() and threw the SKU index away, so everything so far was scored
against meta-categories via accept-sets. This scores exact top-1 SKU.

Condition --boxes gt isolates identity from localization (GT boxes, so any
error is the retrieval index's). Anything else is end-to-end and needs a
proposal source wired in later.

  python tools/eval/rpc_retrieval_eval.py \
      --index runs/eval/rpc_dinov2_index.npz \
      --staged datasets_perception/rpc_val_600 \
      --out runs/eval/rpc_retrieval_gtboxes.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def meta_of(name: str) -> str:
    return re.sub(r"^\d+_", "", name)


@torch.no_grad()
def embed(crops, proc, model, device, batch=64) -> np.ndarray:
    out = []
    for i in range(0, len(crops), batch):
        px = proc(images=crops[i:i + batch], return_tensors="pt").to(device)
        h = model(**px).last_hidden_state[:, 0]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 768), np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--boxes", default="gt", choices=["gt", "detector"],
                    help="gt: identity in isolation. detector: end-to-end, using the "
                         "200-SKU detector's boxes as CLASS-AGNOSTIC proposals (its "
                         "predicted classes are discarded and retrieval names each crop). "
                         "If this matches the detector's own name-F1, identity is swappable "
                         "and a new SKU needs index photos, not a retrained head.")
    ap.add_argument("--det-weights", type=Path, help="required when --boxes detector")
    ap.add_argument("--det-min-conf", type=float, default=0.25)
    ap.add_argument("--k", type=int, default=5, help="kNN vote size")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-per-sku", type=int, default=0,
                    help="subsample the index to at most N exemplars per SKU. Used to test "
                         "whether retrieval's accuracy is driven by exemplars-per-class "
                         "(RPC has ~269/SKU; the robot capture has ~6).")
    ap.add_argument("--tta-rot", action="store_true",
                    help="query-side 4x in-plane rotation, max-similarity per exemplar. "
                         "Top-down scene objects lie at arbitrary rotation, exemplars do not, "
                         "and DINOv2 is not rotation invariant.")
    args = ap.parse_args()

    from transformers import AutoImageProcessor, AutoModel

    z = np.load(args.index, allow_pickle=True)
    E = torch.from_numpy(z["emb"])           # (N, D), already L2-normalized
    L = torch.from_numpy(z["label"].astype(np.int64))
    names = [str(x) for x in z["names"]]
    model_id = str(z["model"])

    if args.max_per_sku:
        rng = np.random.default_rng(0)
        keep = []
        lab = z["label"].astype(np.int64)
        for c in np.unique(lab):
            idxs = np.flatnonzero(lab == c)
            keep.append(rng.permutation(idxs)[:args.max_per_sku])
        keep = np.concatenate(keep)
        E, L = E[keep], L[keep]
        print(f"index subsampled to {len(keep)} exemplars (<={args.max_per_sku}/SKU)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    E = E.to(device)
    L = L.to(device)
    n_cls = len(names)

    hits1 = hits5 = total = 0
    meta_hit = defaultdict(lambda: [0, 0])   # meta -> [sku_correct, n]
    meta_meta_hit = defaultdict(int)         # meta -> predicted-meta-correct
    confuse = defaultdict(int)

    det = None
    if args.boxes == "detector":
        if not args.det_weights:
            raise SystemExit("--boxes detector requires --det-weights")
        from ultralytics import YOLO
        det = YOLO(str(args.det_weights))
    e2e = [0, 0, 0]      # tp, fp, fn for the end-to-end name-F1

    scenes = sorted(p for p in args.staged.iterdir() if p.is_dir())
    if args.limit:
        scenes = scenes[:args.limit]
    for sd in scenes:
        gt = json.loads((sd / "gt.json").read_text())
        im = Image.open(sd / "frames/frame_000.png").convert("RGB")
        crops, truth = [], []
        if det is not None:
            # class-agnostic use of the detector: keep boxes, discard classes
            r = det(str(sd / "frames/frame_000.png"), imgsz=896, verbose=False)[0]
            if r.boxes is not None:
                for conf, xyxy in zip(r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
                    if conf < args.det_min_conf:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in xyxy)
                    if x2 - x1 < 8 or y2 - y1 < 8:
                        continue
                    crops.append(im.crop((x1, y1, x2, y2)))
            truth = [int(c) for c in gt["categories"]]
        else:
            for c, b in zip(gt["categories"], gt["bboxes"]):
                x, y, w, h = b
                if w < 8 or h < 8:
                    continue
                crops.append(im.crop((int(x), int(y), int(x + w), int(y + h))))
                truth.append(int(c))
        if not crops:
            e2e[2] += len(truth)
            continue

        if args.tta_rot:
            sim = None
            for deg in (0, 90, 180, 270):
                qs = [c.rotate(deg, expand=True) for c in crops] if deg else crops
                s = torch.from_numpy(embed(qs, proc, model, device)).to(device) @ E.T
                sim = s if sim is None else torch.maximum(sim, s)
        else:
            q = torch.from_numpy(embed(crops, proc, model, device)).to(device)
            sim = q @ E.T                               # (n_q, N)
        topk = sim.topk(min(args.k, E.shape[0]), dim=1)
        # kNN vote weighted by similarity
        votes = torch.zeros(len(crops), n_cls, device=device)
        votes.scatter_add_(1, L[topk.indices], topk.values.clamp(min=0))
        rank = votes.argsort(dim=1, descending=True)

        if det is not None:
            # crops no longer align 1:1 with GT instances -- score as a
            # multiset of names, the same metric eval_rpc_detector.py uses
            preds = [names[int(rank[i, 0])] for i in range(len(crops))]
            gt_names = [names[t] for t in truth]
            used = [False] * len(preds)
            tp = 0
            for g in gt_names:
                for i, p in enumerate(preds):
                    if not used[i] and p == g:
                        used[i] = True
                        tp += 1
                        break
            e2e[0] += tp; e2e[1] += len(preds) - tp; e2e[2] += len(gt_names) - tp
            continue

        for i, t in enumerate(truth):
            p1 = int(rank[i, 0])
            top5 = [int(v) for v in rank[i, :5]]
            m = meta_of(names[t])
            total += 1
            meta_hit[m][1] += 1
            if p1 == t:
                hits1 += 1
                meta_hit[m][0] += 1
            else:
                confuse[f"{names[t]} -> {names[p1]}"] += 1
            if t in top5:
                hits5 += 1
            if meta_of(names[p1]) == m:
                meta_meta_hit[m] += 1

    if det is not None:
        tp, fp, fn = e2e
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        res = {"n_scenes": len(scenes), "boxes": "detector",
               "det_weights": str(args.det_weights), "det_min_conf": args.det_min_conf,
               "knn_k": args.k, "tta_rot": args.tta_rot,
               "end_to_end_name_f1": {
                   "P": round(p, 4), "R": round(r, 4),
                   "F1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                   "tp": tp, "fp": fp, "fn": fn}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
        print(f"[OK] {args.out}")
        return 0

    res = {
        "n_scenes": len(scenes),
        "n_instances": total,
        "boxes": args.boxes,
        "knn_k": args.k,
        "top1_sku_acc": round(hits1 / total, 4) if total else 0.0,
        "top5_sku_acc": round(hits5 / total, 4) if total else 0.0,
        "meta_category_acc": round(sum(meta_meta_hit.values()) / total, 4) if total else 0.0,
        "per_meta_sku_acc": {m: {"acc": round(v[0] / v[1], 3), "n": v[1]}
                              for m, v in sorted(meta_hit.items())},
        "top_confusions": dict(sorted(confuse.items(), key=lambda kv: -kv[1])[:30]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in
                      ("n_scenes", "n_instances", "top1_sku_acc", "top5_sku_acc", "meta_category_acc")}, indent=1))
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

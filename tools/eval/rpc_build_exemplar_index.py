#!/usr/bin/env python3
"""Build a DINOv2 retrieval index over the RPC exemplar (train) split.

E1: identity-by-retrieval instead of a trained classifier head. A new SKU is
onboarded by adding its exemplar photos to the index -- no retraining, no
relabeling -- which is the property that makes catalog lookup scale past the
43-item robot set the professor called too small.

RPC train = ~53.7k single-product studio photos over 200 SKUs, 19 parquet
shards on HF. Disk on knuth is tight, so shards are streamed: download ->
crop objects -> embed on GPU -> delete the shard. Crops are also kept
(resized JPEG, ~1GB total) because the same cutouts feed the 200-SKU
synthetic-composite run later.

  python tools/eval/rpc_build_exemplar_index.py --out runs/eval/rpc_dinov2_index.npz
"""
from __future__ import annotations

import argparse
import io
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SHARD_URL = ("https://huggingface.co/datasets/benjamintli/retail-product-checkout"
             "/resolve/main/data/train-{:05d}-of-00019.parquet")
N_SHARDS = 19


def load_encoder(model_id: str, device: str):
    from transformers import AutoImageProcessor, AutoModel
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
    return proc, model


@torch.no_grad()
def embed(crops, proc, model, device, batch=64) -> np.ndarray:
    """L2-normalized CLS embeddings for a list of PIL crops."""
    out = []
    for i in range(0, len(crops), batch):
        px = proc(images=crops[i:i + batch], return_tensors="pt").to(device)
        h = model(**px).last_hidden_state[:, 0]  # CLS
        h = torch.nn.functional.normalize(h.float(), dim=-1)
        out.append(h.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, model.config.hidden_size), np.float32)


def crop_objects(row, min_px: int):
    """Yield (category_int, PIL crop) for every annotated object in a row."""
    img = row["image"]
    im = Image.open(io.BytesIO(img["bytes"])) if isinstance(img, dict) else img
    im = im.convert("RGB")
    for c, b in zip(row["objects"]["category"], row["objects"]["bbox"]):
        x, y, w, h = b
        if w < min_px or h < min_px:
            continue
        yield int(c), im.crop((int(x), int(y), int(x + w), int(y + h)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/eval/rpc_dinov2_index.npz")
    ap.add_argument("--crop-dir", type=Path, default=ROOT / "data/rpc_exemplar_all",
                    help="where resized exemplar crops are kept (reused by the 200-SKU synthetic run)")
    ap.add_argument("--shard-dir", type=Path, default=ROOT / "data/rpc")
    ap.add_argument("--model", default="facebook/dinov2-base")
    ap.add_argument("--shards", type=int, default=N_SHARDS)
    ap.add_argument("--min-px", type=int, default=16)
    ap.add_argument("--save-px", type=int, default=256, help="max side of the kept crop jpeg")
    ap.add_argument("--keep-shards", action="store_true", help="do not delete shards after processing")
    ap.add_argument("--from-crops", action="store_true",
                    help="re-embed crops already saved in --crop-dir (swap backbone without re-downloading)")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    cats = json.loads((ROOT / "data/rpc_category_names.json").read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc, model = load_encoder(args.model, device)
    args.crop_dir.mkdir(parents=True, exist_ok=True)
    args.shard_dir.mkdir(parents=True, exist_ok=True)

    all_emb, all_lab = [], []

    if args.from_crops:
        # re-embed the crops kept by an earlier run (e.g. to swap the backbone)
        # instead of re-downloading 8GB of parquet shards
        idx_of = {n: i for i, n in enumerate(cats)}
        buf, lab = [], []
        for d in sorted(args.crop_dir.iterdir()):
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                buf.append(Image.open(p).convert("RGB"))
                lab.append(idx_of[d.name])
                if len(buf) >= 256:
                    all_emb.append(embed(buf, proc, model, device))
                    all_lab.extend(lab)
                    buf, lab = [], []
            print(f"{d.name}: total {len(all_lab) + len(lab)}", flush=True)
        if buf:
            all_emb.append(embed(buf, proc, model, device))
            all_lab.extend(lab)
        args.shards = 0

    for si in range(args.shards):
        fp = args.shard_dir / f"train-{si:05d}-of-00019.parquet"
        preexisting = fp.exists()
        if not preexisting:
            subprocess.run(["curl", "-sL", "-o", str(fp), SHARD_URL.format(si)], check=True)

        # flush in chunks -- a whole shard of full-res crops does not fit in RAM
        buf_crops, buf_labels, n_shard = [], [], 0

        def flush():
            nonlocal buf_crops, buf_labels, n_shard
            if not buf_crops:
                return
            all_emb.append(embed(buf_crops, proc, model, device))
            all_lab.extend(buf_labels)
            for c, crop in zip(buf_labels, buf_crops):
                d = args.crop_dir / cats[c]
                d.mkdir(exist_ok=True)
                small = crop.copy()
                small.thumbnail((args.save_px, args.save_px))
                small.save(d / f"s{si:02d}_{n_shard:05d}.jpg", quality=88)
                n_shard += 1
            buf_crops, buf_labels = [], []

        for row in pq.read_table(fp).to_pylist():
            for c, crop in crop_objects(row, args.min_px):
                buf_labels.append(c)
                buf_crops.append(crop)
            if len(buf_crops) >= 256:
                flush()
        flush()

        if not preexisting and not args.keep_shards:
            fp.unlink()
        print(f"shard {si}: +{n_shard} crops, total {len(all_lab)}", flush=True)

    E = np.concatenate(all_emb).astype(np.float32)
    L = np.asarray(all_lab, np.int32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, emb=E, label=L, model=args.model, names=np.array(cats))
    per = np.bincount(L, minlength=len(cats))
    print(json.dumps({"n_exemplars": int(len(L)), "dim": int(E.shape[1]),
                      "n_skus_covered": int((per > 0).sum()),
                      "min_per_sku": int(per.min()), "median_per_sku": int(np.median(per))}, indent=1))
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

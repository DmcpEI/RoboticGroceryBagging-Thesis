#!/usr/bin/env python3
"""RGBA cutout library from RPC exemplar crops (E1b).

Same role as build_cutout_library.py, but RPC is RGB-only so the alpha comes
from colour separation against the exemplar's uniform studio background
instead of a depth mask. Emits the identical manifest format
(`cutout_manifest.json`: [{label, file}]) so the scene compositor is shared.

Matting: estimate the background colour from the crop border, threshold on
distance from it, close holes, keep the largest component. Crops whose mask
comes out degenerate (almost empty or almost everything) fall back to a plain
rectangular alpha, which is what a bbox-only cutout would have given anyway.

  python tools/eval/build_rpc_cutouts.py --per-sku 30 \
      --crops data/rpc_exemplar_all --out data/rpc_cutouts
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]


def matte(im: Image.Image, tol: int) -> np.ndarray:
    """RGBA array; alpha from distance to the border-estimated background."""
    rgb = np.asarray(im.convert("RGB")).astype(np.int16)
    h, w = rgb.shape[:2]
    b = 3
    border = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                             rgb[:, :b].reshape(-1, 3), rgb[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    fg = np.linalg.norm(rgb - bg, axis=-1) > tol

    def consolidate(m, ksize):
        m = ndimage.binary_closing(m, np.ones((ksize, ksize)))
        m = ndimage.binary_fill_holes(m)
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            m = lab == (int(np.argmax(sizes)) + 1)
        return m

    fg = consolidate(fg, 5)
    if fg.mean() < 0.5:
        # white-on-white packaging: the object's own pale regions read as
        # background and get eaten out of the middle of the silhouette, which
        # binary_fill_holes cannot recover because they touch the mask edge.
        fg = consolidate(fg, 15)

    frac = fg.mean()
    if frac < 0.5 or frac > 0.98:
        # RPC boxes are tight, so a real object fills most of its crop -- a
        # low fraction means the matte failed, not that the object is small.
        # Fall back to the plain bbox cutout: the leftover studio background
        # is near-white and the target checkout board is near-white too, so
        # it composites almost invisibly.
        fg = np.ones((h, w), bool)          # degenerate -> plain bbox cutout

    out = np.dstack([np.asarray(im.convert("RGB")), (fg * 255).astype(np.uint8)])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=Path, default=ROOT / "data/rpc_exemplar_all")
    ap.add_argument("--out", type=Path, default=ROOT / "data/rpc_cutouts")
    ap.add_argument("--per-sku", type=int, default=30,
                    help="cutouts sampled per SKU (53k total exemplars is far more than needed)")
    ap.add_argument("--tol", type=int, default=42, help="colour distance from background")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    man, degenerate = [], 0

    for d in sorted(p for p in args.crops.iterdir() if p.is_dir()):
        files = sorted(d.iterdir())
        random.shuffle(files)
        od = args.out / d.name
        od.mkdir(exist_ok=True)
        for i, fp in enumerate(files[:args.per_sku]):
            rgba = matte(Image.open(fp), args.tol)
            if rgba[..., 3].mean() > 254:
                degenerate += 1
            op = od / f"{i:03d}.png"
            Image.fromarray(rgba).save(op)
            man.append({"label": d.name, "file": str(op.relative_to(ROOT)),
                        "rect": bool(rgba[..., 3].mean() > 254)})
        print(f"{d.name}: {min(len(files), args.per_sku)}", flush=True)

    (args.out / "cutout_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({"n_cutouts": len(man), "n_skus": len({m['label'] for m in man}),
                      "degenerate_rect_fallback": degenerate}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

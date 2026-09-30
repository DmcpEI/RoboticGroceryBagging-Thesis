#!/usr/bin/env python3
"""Synthetic RPC checkout scenes for closed-set detector training (E1b).

Our own synthetic-composite recipe (build_synthetic_scenes.py) rerun at 200
SKUs on an external real benchmark, to answer the objection that training a
detector on a 43-object catalog is too small to be relevant. The compositing
core -- z-buffer occlusion, visible-fraction label dropping, alpha cleanup,
drop shadows -- is IMPORTED from build_synthetic_scenes, not reimplemented,
so this really is the same recipe and not a lookalike.

Two things must differ, both because RPC is RGB-only where the robot rig is
RGB-D:
  * cutout alpha comes from colour matting (build_rpc_cutouts.py), not depth;
  * the background is a synthesised near-white checkout board rather than a
    depth-derived median tabletop, and objects may be placed anywhere on it
    instead of inside a depth-measured table mask.

Object scale is the one quantity calibrated from the target domain: RPC val
boxes span p10-p90 = 0.108-0.317 of the image side (median 0.186), and a
cutout's own pixel size carries no scene scale, so each object's target size
is drawn log-uniformly from that range. Stated explicitly because it is the
only place target-domain statistics enter the training data.

  python tools/eval/build_rpc_synthetic_scenes.py --n 8000 --val 800 \
      --cutouts data/rpc_cutouts --out data/rpc_synthetic_det
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from build_synthetic_scenes import clean_alpha, paste, paste_shadow  # noqa: E402

# Measured on the 600 staged RPC val scenes. These two constants (object size
# range as a fraction of the image side, and board grey level) are the ONLY
# target-domain statistics that enter the training data -- no val image or
# label is otherwise used. Both are camera-geometry/lighting constants of the
# benchmark rig, not per-object information.
SIZE_P10, SIZE_P90 = 0.108, 0.317
BOARD_LO, BOARD_HI = 128.0, 155.0     # corner-patch grey, p10..p90 = 131..152


def board(w: int, h: int, rng: random.Random) -> np.ndarray:
    """Checkout board: base grey + smooth lighting falloff + grain."""
    base = rng.uniform(BOARD_LO, BOARD_HI)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w * rng.uniform(0.35, 0.65), h * rng.uniform(0.35, 0.65)
    r = np.sqrt(((xx - cx) / w) ** 2 + ((yy - cy) / h) ** 2)
    shade = 1.0 - rng.uniform(0.04, 0.14) * r
    img = base * shade + np.random.normal(0, 2.2, (h, w))
    return np.clip(img, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutouts", type=Path, default=ROOT / "data/rpc_cutouts")
    ap.add_argument("--out", type=Path, default=ROOT / "data/rpc_synthetic_det")
    ap.add_argument("--n", type=int, default=8000, help="train scenes")
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--size", type=int, default=896, help="square canvas side")
    ap.add_argument("--min-objs", type=int, default=4)
    ap.add_argument("--max-objs", type=int, default=20)   # matches RPC val: 4..20
    ap.add_argument("--min-visible", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    S = args.size

    man = json.loads((args.cutouts / "cutout_manifest.json").read_text())
    labels = sorted({c["label"] for c in man})
    cls_idx = {l: i for i, l in enumerate(labels)}
    # Prefer properly matted cutouts. The rect fallbacks carry their studio
    # background, which is near-white and so shows as a bright halo against
    # the grey checkout board -- an obvious synthetic tell. Use them only for
    # the handful of SKUs that have too few clean ones.
    by_label = {}
    for c in man:
        by_label.setdefault(c["label"], []).append(c)
    n_fallback_skus = 0
    for l, cs in by_label.items():
        good = [c for c in cs if not c.get("rect")]
        if len(good) >= 5:
            by_label[l] = good
        else:
            n_fallback_skus += 1
    print(f"{len(man)} cutouts / {len(labels)} classes "
          f"({n_fallback_skus} SKUs keep rect fallbacks)", flush=True)

    for split, count in (("train", args.n), ("val", args.val)):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for si in range(count):
            canvas = board(S, S, rng)
            zbuf = np.full((S, S), -1, np.int32)
            placed = []
            for oid in range(rng.randint(args.min_objs, args.max_objs)):
                lab = rng.choice(labels)
                c = rng.choice(by_label[lab])
                im = Image.open(ROOT / c["file"]).convert("RGBA")
                # arbitrary in-plane rotation: checkout objects are dropped on
                # the board at any angle (this is also what made query-side
                # rotation TTA worth +0.05 for the retrieval index)
                im = im.rotate(rng.uniform(0, 360), expand=True)
                target = S * math.exp(rng.uniform(math.log(SIZE_P10), math.log(SIZE_P90)))
                s = target / max(im.width, im.height)
                im = im.resize((max(8, int(im.width * s)), max(8, int(im.height * s))))
                rgba = clean_alpha(np.asarray(im))
                oh, ow = rgba.shape[:2]
                if oh >= S or ow >= S:
                    continue
                # Real checkout scenes are a pile dumped in the middle of the
                # board, not a uniform scatter -- centre-biased placement so
                # objects touch and occlude at a realistic rate.
                x = int(min(max(rng.gauss(S / 2, S * 0.20) - ow / 2, -ow / 8), S - ow * 7 / 8))
                y = int(min(max(rng.gauss(S / 2, S * 0.20) - oh / 2, -oh / 8), S - oh * 7 / 8))
                paste_shadow(canvas, rgba, x, y)
                paste(canvas, zbuf, oid, rgba, x, y)
                placed.append((oid, lab, rgba))

            lines = []
            for oid, lab, rgba in placed:
                own = int((zbuf == oid).sum())
                area = int((rgba[..., 3] > 40).sum())
                if area == 0 or own / area < args.min_visible:
                    continue
                ys, xs = np.where(zbuf == oid)
                if xs.size == 0:
                    continue
                bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
                lines.append(f"{cls_idx[lab]} {(bx1 + bx2) / 2 / S:.6f} {(by1 + by2) / 2 / S:.6f} "
                             f"{(bx2 - bx1) / S:.6f} {(by2 - by1) / S:.6f}")
            stem = f"rpcsyn_{split}_{si:05d}"
            Image.fromarray(canvas).save(args.out / "images" / split / f"{stem}.jpg", quality=88)
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
            if si % 500 == 0:
                print(f"{split} {si}/{count}", flush=True)
        print(f"{split}: {count} scenes", flush=True)

    (args.out / "classes.json").write_text(json.dumps(labels, indent=1) + "\n")
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(labels)}\nnames: {labels}\n")
    print(f"{len(labels)} classes -> {args.out}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

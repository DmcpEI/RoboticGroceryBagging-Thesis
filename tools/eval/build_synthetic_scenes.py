#!/usr/bin/env python3
"""Synthetic multi-object scenes for closed-set detector training (T3).

Pastes the depth-masked object cutouts (build_cutout_library.py) onto an empty
tabletop background at physically plausible scale (fixed top-down camera => the
cutout's native pixel size is already ~right; only mild jitter), with random
positions and realistic occlusion. Emits YOLO-format detection labels.

Occlusion is modelled with a z-buffer: objects are painted in order, later ones
on top; each object's VISIBLE pixel fraction (its alpha not covered by a later
object) is measured, and its label is dropped if it falls below --min-visible
(heavily buried objects stay painted as realistic occluders/distractors but are
not taught as detectable). Boxes use the visible extent.

Empty-table background = per-pixel median over many single_* frames (the lone
object sits in a different place each frame, so the median is a clean table).

Usage (cluster):
  python tools/eval/build_synthetic_scenes.py \
      --cutouts data/robot_lab/cutouts --n 1500 --val 300 \
      --out data/robot_lab/synthetic_det
Outputs YOLO layout: <out>/{images,labels}/{train,val}/*, <out>/classes.json, data.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
import sys as _sys
_sys.path.insert(0, str(ROOT / "tools/eval"))
from eval_depth_separation import fit_table_plane  # noqa: E402


def bg_dirs(rgbd_root: Path, per_object: int = 3):
    """Captures to build the empty-table background/mask from.

    Prefers REAL empty-table frames (`empty_*`, capture_rgbd_session.py --empty).
    Falls back to the historical approximation -- the median of single-object
    captures -- when none exist, since the lone object sits somewhere different
    in each frame so the middle value at each pixel is usually bare table.

    That fallback assumes the sample is spread across objects AND that the table
    has not moved. Both broke: a capture session heavy in one product baked that
    product into the "empty" table, and moving the robot moved the table, so
    mixing sessions averages two different table positions into a ghost. Real
    empty frames have neither problem, which is why they are preferred.
    """
    empt = sorted(rgbd_root.glob("empty_*/"))
    if empt:
        return empt, True
    dirs = sorted(rgbd_root.glob("single_*/"))
    if per_object > 0:
        by_obj = {}
        for d in dirs:
            by_obj.setdefault(_object_of(d.name), []).append(d)
        picked = []
        for obj in sorted(by_obj):
            poses = by_obj[obj]
            step = max(1, len(poses) // per_object)
            picked.extend(poses[::step][:per_object])
        dirs = sorted(picked, key=lambda d: d.name)
    return dirs, False


def median_depth(rgbd_root: Path, w: int, h: int, limit: int = 60, per_object: int = 3):
    """Median depth over the background captures -> clean, empty board surface."""
    ds = []
    for d in bg_dirs(rgbd_root, per_object)[0]:
        dfp = d / "depth.npy"
        if not dfp.exists():
            continue
        arr = np.load(dfp).astype(np.float32)
        if arr.shape != (h, w):
            continue
        ds.append(arr)
        if len(ds) >= limit:
            break
    return np.median(np.stack(ds), axis=0) if ds else None


def table_mask(rgbd_root: Path, w: int, h: int, per_object: int = 3) -> np.ndarray:
    """Boolean mask of the tabletop BOARD via depth (not colour).

    The board sits on the fitted table plane (~0.9m); the surrounding floor is
    farther, so a small plane-residual threshold keeps the board and excludes the
    bright floor that colour thresholding wrongly included. Median depth over the
    single_* frames gives a near-empty board; objects (above the plane) leave small
    holes that fill_holes closes.
    """
    depth = median_depth(rgbd_root, w, h, per_object=per_object)
    if depth is None:
        return np.ones((h, w), bool)
    valid = (depth > 0.1) & (depth < 5.0)
    plane = fit_table_plane(depth, valid)
    resid = np.abs(plane - depth)
    onplane = valid & (resid < 0.03)          # within 3cm of the board plane
    onplane = ndimage.binary_opening(onplane, iterations=2)
    lab, n = ndimage.label(onplane)
    if n == 0:
        return np.ones((h, w), bool)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    m = ndimage.binary_fill_holes(m)
    m = ndimage.binary_erosion(m, iterations=5)  # keep objects off the very board edge
    return m


def _object_of(dirname: str) -> str:
    """single_coffee_can_003 -> coffee_can. Groups poses of one object."""
    s = dirname[len("single_"):] if dirname.startswith("single_") else dirname
    return re.sub(r"(_n?\d+|\d+)$", "", s)


def median_background(rgbd_root: Path, w: int, h: int, limit: int = 60,
                      per_object: int = 3):
    """Empty-table RGB. Real `empty_*` frames if captured, else the median of
    object captures (see bg_dirs for why that is only an approximation)."""
    from PIL import Image
    dirs, real = bg_dirs(rgbd_root, per_object)
    imgs = []
    for d in dirs:
        cfp = d / "color.png"
        if not cfp.exists():
            continue
        im = Image.open(cfp).convert("RGB").resize((w, h))
        imgs.append(np.asarray(im))
        if len(imgs) >= limit:
            break
    if not imgs:
        return np.full((h, w, 3), 200, np.uint8)
    return np.median(np.stack(imgs), axis=0).astype(np.uint8)


def clean_alpha(rgba):
    """Erode 1px + drop tiny specks so cutout edges don't carry a table-colour halo."""
    a = rgba[..., 3] > 60
    a = ndimage.binary_erosion(a, iterations=1)
    lab, n = ndimage.label(a)
    if n > 1:  # keep only the largest blob (drop stray table-marking slivers)
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        a = lab == (int(np.argmax(sizes)) + 1)
    out = rgba.copy()
    out[..., 3] = np.where(a, rgba[..., 3], 0)
    return out


def paste_shadow(canvas, rgba, x, y, dx=6, dy=8):
    """Soft dark drop-shadow under an object for compositing realism."""
    a = (rgba[..., 3] > 60).astype(np.float32)
    a = ndimage.gaussian_filter(a, sigma=4) * 0.45
    oh, ow = a.shape
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x + dx), max(0, y + dy)
    x1, y1 = min(W, x + dx + ow), min(H, y + dy + oh)
    if x1 <= x0 or y1 <= y0:
        return
    av = a[y0 - y - dy:y1 - y - dy, x0 - x - dx:x1 - x - dx][..., None]
    canvas[y0:y1, x0:x1] = (canvas[y0:y1, x0:x1] * (1 - av)).astype(np.uint8)


def paste(canvas, zbuf, obj_id, rgba, x, y):
    """Alpha-composite rgba at (x,y); mark owned pixels in zbuf with obj_id."""
    oh, ow = rgba.shape[:2]
    H, W = canvas.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + ow), min(H, y + oh)
    if x1 <= x0 or y1 <= y0:
        return
    sub = rgba[y0 - y:y1 - y, x0 - x:x1 - x]
    a = (sub[..., 3:4] / 255.0)
    m = sub[..., 3] > 40
    canvas[y0:y1, x0:x1] = (sub[..., :3] * a + canvas[y0:y1, x0:x1] * (1 - a)).astype(np.uint8)
    zbuf[y0:y1, x0:x1][m] = obj_id


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutouts", type=Path, default=ROOT / "data/robot_lab/cutouts")
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--out", type=Path, default=ROOT / "data/robot_lab/synthetic_det")
    ap.add_argument("--n", type=int, default=1500, help="train scenes")
    ap.add_argument("--val", type=int, default=300)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--min-objs", type=int, default=4)
    ap.add_argument("--max-objs", type=int, default=14)
    ap.add_argument("--min-visible", type=float, default=0.35, help="drop label if visible frac below")
    ap.add_argument("--roi", default="0.12,0.06,0.90,0.94", help="table ROI x1,y1,x2,y2 fractions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-root", type=Path, default=None,
                    help="captures used to build the empty-table background and table "
                         "mask. Defaults to --rgbd-root. Point this at a FIXED set to "
                         "stop the background moving every time new captures are added.")
    ap.add_argument("--bg-per-object", type=int, default=3,
                    help="max poses of any one object contributing to the background "
                         "median. Stops a large single-product capture session from "
                         "dominating it and baking that product into the 'empty' table. "
                         "0 = original first-N-alphabetical behaviour.")
    ap.add_argument("--exclude-labels", default="",
                    help="comma-separated labels to drop. Empty by default. This used to "
                         "drop 'plate', whose depth cutout is unusable because a flat "
                         "object barely rises off the table; build_plate_cutouts_rgb.py "
                         "now mattes it from RGB instead, and the shipped 32-class "
                         "detector is trained with it. Pass a label here only if its "
                         "cutouts are genuinely broken.")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    man = json.loads((args.cutouts / "cutout_manifest.json").read_text())

    # The manifest stores each cutout's PATH, so a library copied with `cp -r`
    # still points at the ORIGINAL directory. Editing the copy then changes
    # nothing and the build silently reproduces the source library -- which
    # cost three seeds of training on 2026-09-17 before the identical per-class
    # scores gave it away. Two checks, because they fail differently: a path
    # outside --cutouts means the manifest was copied, and a missing file means
    # cutouts were deleted without regenerating it.
    cut_dir = args.cutouts.resolve()
    stray = [c for c in man if cut_dir not in Path(ROOT / c["file"]).resolve().parents]
    if stray:
        raise SystemExit(
            f"{len(stray)} of {len(man)} manifest entries point outside {args.cutouts} "
            f"(e.g. {stray[0]['file']}).\nThe manifest was copied from another library. "
            f"Rebuild it with build_cutout_library.py --out {args.cutouts}, or repoint "
            f"the 'file' fields, before building scenes.")
    absent = [c for c in man if not (ROOT / c["file"]).exists()]
    if absent:
        raise SystemExit(
            f"{len(absent)} of {len(man)} manifest entries name a file that does not "
            f"exist (e.g. {absent[0]['file']}).\nCutouts were removed without updating "
            f"the manifest; prune it or rebuild the library.")

    excl = {s.strip() for s in args.exclude_labels.split(",") if s.strip()}
    man = [c for c in man if c["label"] not in excl]
    if not man:
        raise SystemExit("no cutouts; run build_cutout_library.py first")
    labels = sorted({c["label"] for c in man})
    cls_idx = {l: i for i, l in enumerate(labels)}
    by_label = {}
    for c in man:
        by_label.setdefault(c["label"], []).append(c)

    bg_root = args.bg_root or args.rgbd_root
    _bgd, _real = bg_dirs(bg_root, args.bg_per_object)
    bg = median_background(bg_root, args.w, args.h, per_object=args.bg_per_object)
    tmask = table_mask(bg_root, args.w, args.h, per_object=args.bg_per_object)
    print(f"background: {'REAL empty-table frames' if _real else 'median of object captures'}"
          f" ({len(_bgd)} from {bg_root})"
          + ("" if _real else f", <= {args.bg_per_object or 'unlimited'} poses per object"))
    ys_t, xs_t = np.where(tmask)  # table pixel coords for placement

    for split, count in (("train", args.n), ("val", args.val)):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for si in range(count):
            canvas = bg.copy()
            zbuf = np.full((args.h, args.w), -1, np.int32)
            k = random.randint(args.min_objs, args.max_objs)
            placed = []  # (obj_id, label, rgba, x, y)
            for oid in range(k):
                lab = random.choice(labels)
                c = random.choice(by_label[lab])
                rgba = np.asarray(Image.open(ROOT / c["file"]).convert("RGBA"))
                # fixed top-down camera => real apparent size is ~constant; only a
                # small jitter (native cutout px already matches the scene scale).
                s = random.uniform(0.9, 1.1)
                ang = random.choice([0, 90, 180, 270]) + random.uniform(-15, 15)
                im = Image.fromarray(rgba).rotate(ang, expand=True)
                if s != 1.0:
                    im = im.resize((max(8, int(im.width * s)), max(8, int(im.height * s))))
                rgba = clean_alpha(np.asarray(im))
                oh, ow = rgba.shape[:2]
                # place so MOST of the object body sits on the board (no overhang onto floor)
                amask = rgba[..., 3] > 60
                a_area = int(amask.sum())
                x = y = None
                for _ in range(40):
                    ti = random.randrange(len(xs_t))
                    cxp, cyp = int(xs_t[ti]), int(ys_t[ti])
                    xx, yy = cxp - ow // 2, cyp - oh // 2
                    x0, y0 = max(0, xx), max(0, yy)
                    x1, y1 = min(args.w, xx + ow), min(args.h, yy + oh)
                    if x1 <= x0 or y1 <= y0 or a_area == 0:
                        continue
                    sub_a = amask[y0 - yy:y1 - yy, x0 - xx:x1 - xx]
                    on = int((sub_a & tmask[y0:y1, x0:x1]).sum())
                    if on / a_area >= 0.85:          # >=85% of the object on the board
                        x, y = xx, yy
                        break
                if x is None:
                    continue
                paste_shadow(canvas, rgba, x, y)
                paste(canvas, zbuf, oid, rgba, x, y)
                placed.append((oid, lab, rgba, x, y))
            # labels from final z-buffer visibility
            lines = []
            for oid, lab, rgba, x, y in placed:
                own = int((zbuf == oid).sum())
                alpha_area = int((rgba[..., 3] > 40).sum())
                if alpha_area == 0 or own / alpha_area < args.min_visible:
                    continue
                ys, xs = np.where(zbuf == oid)
                if xs.size == 0:
                    continue
                bx1, by1, bx2, by2 = xs.min(), ys.min(), xs.max(), ys.max()
                cx, cy = (bx1 + bx2) / 2 / args.w, (by1 + by2) / 2 / args.h
                bw, bh = (bx2 - bx1) / args.w, (by2 - by1) / args.h
                lines.append(f"{cls_idx[lab]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stem = f"syn_{split}_{si:05d}"
            Image.fromarray(canvas).save(args.out / "images" / split / f"{stem}.jpg", quality=90)
            (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        print(f"{split}: {count} scenes")

    (args.out / "classes.json").write_text(json.dumps(labels, indent=1) + "\n")
    yaml = (f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\n"
            f"nc: {len(labels)}\nnames: {labels}\n")
    (args.out / "data.yaml").write_text(yaml)
    print(f"{len(labels)} classes -> {args.out}/data.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

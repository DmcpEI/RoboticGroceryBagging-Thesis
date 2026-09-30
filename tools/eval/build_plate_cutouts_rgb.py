#!/usr/bin/env python3
"""Build plate cutouts from the RGB singles instead of from depth.

build_synthetic_scenes.py excludes `plate` by default, and the reason is real:
a plate lies almost flat, so the table-plane depth segmentation cannot separate
it from the surface. Of its two RGB-D captures only one produced a cutout, and
that cutout is a ragged blob rather than a plate.

The plate is trivially separable by colour instead -- it is the only saturated
red region on a white board -- so this cuts it from the four RGB singles. The
grippers are also red, hence picking the component nearest the frame centre.

Scale is calibrated rather than guessed: the bowl appears in both capture sets,
measures 167 px here against a known 17.5 cm in the manifest, which puts this
frame at 9.54 px/cm and the plate at 28.4 cm. The depth-derived 14.6 cm in
size_reference.json came from the same broken mask.

  python tools/eval/build_plate_cutouts_rgb.py --out data/robot_lab/cutouts
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
PLATE_W_CM = 28.4          # calibrated against the bowl, see docstring
TARGET_PX_PER_CM = 6.04    # the manifest's scale, stable to +-0.05 across 43 cutouts


def cut(path: Path) -> Image.Image | None:
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(np.float32) / 255.0
    mx, mn = a.max(2), a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    red = (a[:, :, 0] > a[:, :, 1] * 1.35) & (a[:, :, 0] > a[:, :, 2] * 1.35) & (sat > 0.35)
    lab, k = ndimage.label(red)
    cy, cx = a.shape[0] / 2, a.shape[1] / 2
    best = None
    for i in range(1, k + 1):
        ys, xs = np.where(lab == i)
        if len(ys) < 3000:
            continue
        d = ((ys.mean() - cy) ** 2 + (xs.mean() - cx) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, i)
    if best is None:
        return None
    m = lab == best[1]
    # the specular highlight desaturates a notch out of the rim; close it before
    # filling, or the hole-fill leaves a bite taken out of the circle
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, np.ones((15, 15))))
    ys, xs = np.where(m)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    rgba = np.dstack([np.asarray(im), (m * 255).astype(np.uint8)])[y0:y1 + 1, x0:x1 + 1]
    return Image.fromarray(rgba, "RGBA")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--singles", type=Path, default=ROOT / "data/robot_lab/cropped_single_items")
    ap.add_argument("--out", type=Path, default=ROOT / "data/robot_lab/cutouts")
    ap.add_argument("--names", default="plate,plate2,plate3,plate4")
    args = ap.parse_args()
    args.out = args.out.resolve()   # manifest paths are ROOT-relative

    d = args.out / "plate"
    d.mkdir(parents=True, exist_ok=True)
    for f in d.glob("*.png"):        # the depth-derived blob is wrong, not merely worse
        f.unlink()

    target = int(round(PLATE_W_CM * TARGET_PX_PER_CM))
    entries = []
    for n in args.names.split(","):
        img = cut(args.singles / f"{n}.png")
        if img is None:
            print(f"[skip] {n}: no region")
            continue
        img = img.resize((target, int(round(img.height * target / img.width))), Image.LANCZOS)
        fp = d / f"rgb_{n}.png"
        img.save(fp)
        entries.append({"label": "plate", "slug": "plate", "file": str(fp.relative_to(ROOT)),
                        "src": f"rgb_{n}", "w_cm": PLATE_W_CM,
                        "h_cm": round(PLATE_W_CM * img.height / img.width, 1),
                        "w_px": img.width, "h_px": img.height,
                        "area_px": int((np.asarray(img)[:, :, 3] > 10).sum())})
        print(f"[ok] {n} -> {fp.name}  {img.width}x{img.height}")

    mf = args.out / "cutout_manifest.json"
    man = [e for e in json.loads(mf.read_text()) if e["label"] != "plate"]
    man.extend(entries)
    mf.write_text(json.dumps(man, indent=1) + "\n")
    print(f"manifest: {len(man)} cutouts, {len({e['label'] for e in man})} labels")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

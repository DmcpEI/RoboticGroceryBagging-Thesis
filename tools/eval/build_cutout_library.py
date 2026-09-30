#!/usr/bin/env python3
"""Extract depth-masked RGBA object cutouts from the isolated RGB-D singles.

Foundation for synthetic-composite detector training (T3): each single_* RGB-D
bundle holds ONE known object on the table, so the dominant raised depth blob is
that object's mask and the directory name is its catalog label. We emit a per-shot
RGBA cutout (color + alpha from the blob mask) plus the object's real metric size
(from depth) so the compositor can paste it at a physically plausible scale.

Label = single_<name>[<digit>|_NNN] -> "<name>" (underscores -> spaces), which
matches the catalog vocabulary (coffee can, glass cleaner spray bottle, ...).

Usage (cluster, where the RGB-D data lives):
  python tools/eval/build_cutout_library.py \
      --rgbd-root data/robot_lab/rgbd --out data/robot_lab/cutouts
Outputs: <out>/<label_slug>/<shot>.png (RGBA) + <out>/cutout_manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from eval_depth_separation import fit_table_plane  # noqa: E402


# A bagged-produce capture is named for the FRUIT ("bag_mango_n1_000") while the
# catalog class is the BAG ("bag of mangoes"). The plural is irregular often
# enough (limes, mangoes, strawberries, cherries) that it has to be a table.
BAG_CLASS = {
    "banana": "bag of bananas",
    "red apple": "bag of red apples",
    "green apple": "bag of green apples",
    "orange": "bag of oranges",
    "pear": "bag of pears",
    "lime": "bag of limes",
    "mango": "bag of mangoes",
    "yellow lemon": "bag of yellow lemons",
    "strawberry": "bag of strawberries",
    "cherries": "bag of cherries",
}


def label_of(dirname: str) -> str:
    """Capture directory -> catalog class name.

    Do NOT take this from the capture's own meta.json. That field is not
    consistent across sessions: the same product appears there as "coffee can"
    and as "coffee_can", and a bagged-produce capture records the loose fruit
    ("mango") rather than the bag class. The directory name plus this table is
    the only mapping that holds for every capture, and `main` checks the result
    against the deployed class list so a new naming convention fails loudly
    instead of training a phantom class.
    """
    s, is_bag = dirname, dirname.startswith("bag_")
    for pre in ("single_", "bag_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    # strip trailing capture-index tokens: _000, _n1, or a bare digit, repeatedly
    while True:
        s2 = re.sub(r"(_n\d+|_\d+|\d+)$", "", s)
        if s2 == s:
            break
        s = s2
    s = s.replace("_", " ").strip()
    return BAG_CLASS[s] if is_bag and s in BAG_CLASS else s


def object_mask(depth, intr, height_thresh=0.02):
    """Largest raised blob above the table plane -> (mask, bbox, width_cm, height_cm)."""
    valid = (depth > 0.1) & (depth < 5.0)
    plane = fit_table_plane(depth, valid)
    hm = np.where(valid, plane - depth, 0.0)
    raised = valid & (hm > height_thresh)
    raised = ndimage.binary_opening(raised, iterations=2)
    raised = ndimage.binary_closing(raised, iterations=2)
    lab, n = ndimage.label(raised)
    if n == 0:
        return None
    H, W = depth.shape
    # keep the largest blob that is NOT a frame-edge fixture (robot arm/clamp/table
    # edge enter from the border) and not absurdly tall — the isolated object sits
    # proud in the table interior, but a corner clamp can be the largest raw blob.
    best, best_area = None, 0
    for i in range(1, n + 1):
        mm = lab == i
        ys, xs = np.where(mm)
        bx1, by1, bx2, by2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        if bx1 <= 2 or by1 <= 2 or bx2 >= W - 2 or by2 >= H - 2:
            continue
        if float(np.percentile(hm[mm], 95)) > 0.40:
            continue
        a = int(mm.sum())
        if a > best_area:
            best, best_area = i, a
    if best is None:
        return None
    m = lab == best
    # Fill concave interiors: open containers (bowl/cup/plate/wine cup) raise only
    # their RIM above the table plane, so the depth blob is a hollow ring. Fill the
    # enclosed interior so the cutout is the full object silhouette, not a ring.
    m = ndimage.binary_fill_holes(m)
    ys, xs = np.where(m)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    med_d = float(np.median(depth[m]))
    w_cm = (x2 - x1) * med_d / intr["fx"] * 100.0
    h_cm = (y2 - y1) * med_d / intr["fy"] * 100.0
    return m, (x1, y1, x2, y2), w_cm, h_cm


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--out", type=Path, default=ROOT / "data/robot_lab/cutouts")
    ap.add_argument("--selfcheck", action="store_true", help="check the label mapping and exit")
    ap.add_argument("--classes", type=Path,
                    default=ROOT / "robot/robot_pc_package/classes.json",
                    help="deployed class list; every cutout label must appear in it")
    ap.add_argument("--height-thresh", type=float, default=0.02)
    ap.add_argument("--adaptive-height", action="store_true",
                    help="cut each class at a FRACTION of its own height, not a fixed 2 cm")
    ap.add_argument("--height-frac", type=float, default=0.4)
    ap.add_argument("--height-floor", type=float, default=0.006)
    ap.add_argument("--min-area-px", type=int, default=400)
    ap.add_argument("--min-solidity", type=float, default=0.45,
                    help="drop cutouts whose filled area / bbox area is below this (slivers/rings)")
    ap.add_argument("--max-aspect", type=float, default=4.0,
                    help="drop cutouts with bbox aspect ratio above this (thin fragments)")
    args = ap.parse_args()
    if args.selfcheck:
        _selfcheck()
        return 0

    # the manifest stores ROOT-relative paths, which relative_to cannot produce
    # from a relative --out
    args.out = args.out.resolve()
    args.rgbd_root = args.rgbd_root.resolve()

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = []
    per_label = {}
    unknown = {}
    n_dropped = 0
    known = set(json.loads(args.classes.read_text())) if args.classes.exists() else set()
    if not known:
        print(f"WARNING: no class list at {args.classes}; labels will not be checked")
    src_dirs = sorted(list(args.rgbd_root.glob("single_*/")) + list(args.rgbd_root.glob("bag_*/")))

    # A FIXED 2 cm cut removes a fixed slab, which is nothing off a 16 cm cracker box
    # and most of a 2.2 cm sponge. Measured over the singles, recovered mask area at
    # 8 mm against 2 cm: cracker box 1.28x, coffee can 1.30x -- the measurement floor,
    # shadow rather than object -- but plate 6.95x, kitchen sponge 4.81x, toothpaste
    # box 2.55x, gelatin dessert box 2.05x, tuna can 2.02x. Those five are the five
    # shortest classes, and three of them are the ones this project calls weak.
    # Keeping a fixed FRACTION of each object instead leaves everything above about
    # 5 cm untouched and stops slicing the short ones down to a core. Off by default:
    # every published detector was built with the fixed cut.
    thresh_for = {}
    if args.adaptive_height:
        heights = {}
        for d in src_dirs:
            if not ((d / "depth.npy").exists() and (d / "intrinsics.json").exists()):
                continue
            depth = np.load(d / "depth.npy").astype(np.float32)
            intr = json.loads((d / "intrinsics.json").read_text())
            r = object_mask(depth, intr, args.height_floor)
            if r is None:
                continue
            valid = (depth > 0.1) & (depth < 5.0)
            hm = np.where(valid, fit_table_plane(depth, valid) - depth, 0.0)
            heights.setdefault(label_of(d.name), []).append(float(np.percentile(hm[r[0]], 95)))
        for lab, hs in heights.items():
            thresh_for[lab] = min(args.height_thresh,
                                  max(args.height_floor, args.height_frac * float(np.median(hs))))
        for lab in sorted(thresh_for, key=thresh_for.get):
            if thresh_for[lab] < args.height_thresh:
                print(f"  [adaptive] {lab:28} height {np.median(heights[lab])*1000:5.1f} mm "
                      f"-> cut at {thresh_for[lab]*1000:4.1f} mm")
    for d in src_dirs:
        cfp, dfp, ifp = d / "color.png", d / "depth.npy", d / "intrinsics.json"
        if not (cfp.exists() and dfp.exists() and ifp.exists()):
            continue
        depth = np.load(dfp).astype(np.float32)
        intr = json.loads(ifp.read_text())
        res = object_mask(depth, intr, thresh_for.get(label_of(d.name), args.height_thresh))
        if res is None:
            continue
        m, (x1, y1, x2, y2), w_cm, h_cm = res
        area = int(m.sum())
        if area < args.min_area_px:
            continue
        # Reject fragment/sliver/ring cutouts (bad masks: e.g. a red table-edge
        # marking, or a flat plate depth can't see) — they inject label noise.
        # solidity = filled-object area / bbox area; solid objects (boxes/cans/
        # filled bowls) are high, thin slivers are low.
        solidity = area / max(1, (x2 - x1) * (y2 - y1))
        aspect = max(x2 - x1, y2 - y1) / max(1, min(x2 - x1, y2 - y1))
        if solidity < args.min_solidity or aspect > args.max_aspect:
            n_dropped += 1
            continue
        label = label_of(d.name)
        if known and label not in known:
            unknown[label] = unknown.get(label, 0) + 1
            continue
        slug = label.replace(" ", "_")
        color = np.asarray(Image.open(cfp).convert("RGB"))
        rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
        rgba[..., :3] = color[y1:y2, x1:x2]
        rgba[..., 3] = (m[y1:y2, x1:x2] * 255).astype(np.uint8)
        (args.out / slug).mkdir(parents=True, exist_ok=True)
        outp = args.out / slug / f"{d.name}.png"
        Image.fromarray(rgba, mode="RGBA").save(outp)
        manifest.append({"label": label, "slug": slug, "file": str(outp.relative_to(ROOT)),
                         "src": d.name, "w_cm": round(w_cm, 1), "h_cm": round(h_cm, 1),
                         "w_px": x2 - x1, "h_px": y2 - y1, "area_px": int(m.sum())})
        per_label[label] = per_label.get(label, 0) + 1

    (args.out / "cutout_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    if unknown:
        print(f"\nREFUSED {sum(unknown.values())} cutouts whose label is not a deployed class:")
        for lab, n in sorted(unknown.items(), key=lambda x: -x[1]):
            print(f"   {lab!r}: {n}")
        print("   (add it to classes.json, or to BAG_CLASS if it is a bagged-produce capture)\n")
    print(f"cutouts: {len(manifest)} from {len(per_label)} objects "
          f"({n_dropped} dropped: sliver/ring/flat) -> {args.out}")
    for lab, n in sorted(per_label.items()):
        print(f"  {lab:32s} {n}")
    return 0


def _selfcheck() -> None:
    assert label_of("single_coffee_can_012") == "coffee can"
    assert label_of("single_glass_cleaner_spray_bottle3") == "glass cleaner spray bottle"
    assert label_of("single_kitchen_sponge_023") == "kitchen sponge"
    # the case that motivated the table: directory names the fruit, class names the bag
    assert label_of("bag_mango_n1_000") == "bag of mangoes"
    assert label_of("bag_green_apple_n3_017") == "bag of green apples"
    assert label_of("bag_cherries_n5_004") == "bag of cherries"
    # "bag" only triggers the table for a bagged-produce capture
    assert label_of("single_mango_000") == "mango"
    classes = ROOT / "robot/robot_pc_package/classes.json"
    if classes.exists():
        known = set(json.loads(classes.read_text()))
        for fruit, bag in BAG_CLASS.items():
            assert bag in known, f"{bag!r} is not a deployed class"
    print("selfcheck ok")


if __name__ == "__main__":
    raise SystemExit(main())

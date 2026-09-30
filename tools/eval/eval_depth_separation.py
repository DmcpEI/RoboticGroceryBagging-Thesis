#!/usr/bin/env python3
"""Does depth separate the confusable cylinders the RGB confusion matrix flags?

The top-down RGB confusion matrix (runs/eval/loo_confusion_rt.json) says cans
interconfuse badly (tuna -> energy 0.75, soup -> coffee, chip -> coffee) because
a top-down view sees only the lid. Depth sees the *height* a top-down RGB cannot.
This tool tests, MODEL-FREE, whether physical size from depth separates them:

  table_depth - depth  > thresh   ->  raised object pixels
  connected components             ->  one blob per object
  per blob: height (table - top), footprint diameter, metric W x L

If the can blobs split cleanly in (height, diameter) space, depth is a real fix
for the cluster; if they overlap, depth alone is not enough. Pure depth + numpy,
no VLM, so the finding is independent of perception.

Two input kinds (auto-detected by dir name):
  single_*  -> one isolated object, GT name = folder suffix (clean catalog point)
  scene_* / clutter_* -> multi-object, each blob measured (label via --pred-run
                         boxes if given, else by nearest GT-count heuristic)

Usage:
  python tools/eval/eval_depth_separation.py \
      --rgbd-root data/robot_lab/rgbd \
      --gt data/robot_lab/rgbd/scene_gt.json \
      --out runs/eval/depth_separation.json --csv runs/eval/depth_separation.csv
Optional: --pred-run runs/<tag>_base  (attach predicted name to each blob by IoU)
          --plot runs/eval/depth_separation.png  (height vs diameter scatter)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def fit_table_plane(depth, valid):
    """Fit the table as a tilted plane depth = a*x + b*y + c (least squares).

    A single median table depth is wrong: the table is tilted w.r.t. the camera,
    so its near half sits centimetres "above" a flat reference and segments as a
    big false blob. Fitting a plane removes that tilt, leaving only real objects
    standing proud. Seed from the central-region median, fit on the table band,
    then refit on tight in-plane residuals to shed objects/floor. Returns the
    per-pixel plane depth (same shape as `depth`).
    """
    H, W = depth.shape
    Y, X = np.mgrid[0:H, 0:W]
    y0, y1, x0, x1 = int(H * 0.20), int(H * 0.65), int(W * 0.25), int(W * 0.75)
    cv = valid[y0:y1, x0:x1]
    td0 = float(np.median(depth[y0:y1, x0:x1][cv])) if cv.sum() > 100 else float(np.median(depth[valid]))
    band = valid & (np.abs(depth - td0) < 0.12)
    plane = np.full_like(depth, td0)
    for tol in (None, 0.02):
        if tol is not None:
            band = valid & (np.abs(plane - depth) < tol)
        if band.sum() < 200:
            break
        xs, ys = X[band].astype(float), Y[band].astype(float)
        A = np.c_[xs, ys, np.ones(xs.size)]
        coef, *_ = np.linalg.lstsq(A, depth[band], rcond=None)
        plane = coef[0] * X + coef[1] * Y + coef[2]
    return plane


def _watershed_split(mask, peak_min_dist=12):
    """Split touching objects in a binary mask via distance-transform watershed.

    Touching same-height objects merge into one connected component; their shape
    necks show up as dips in the distance transform, so each object core is a
    local maximum -> use those as watershed seeds. scipy-only (no skimage).
    """
    from scipy import ndimage

    dist = ndimage.distance_transform_edt(mask)
    if dist.max() < 3:
        return ndimage.label(mask)
    mx = ndimage.maximum_filter(dist, size=2 * peak_min_dist + 1)
    peaks = (dist == mx) & (dist > max(3.0, 0.4 * dist.max() if dist.max() < 8 else peak_min_dist * 0.5))
    markers, n = ndimage.label(peaks)
    if n <= 1:
        return ndimage.label(mask)
    # Assign every mask pixel to its NEAREST peak (Voronoi partition). The distance
    # peaks are reliable object cores; scipy's watershed_ift floods poorly here, so
    # nearest-marker is a cleaner scipy-only split for touching same-height objects.
    _, (iy, ix) = ndimage.distance_transform_edt(markers == 0, return_indices=True)
    out = markers[iy, ix]
    out[~mask] = 0
    return out, n


def blobs(depth, intr, height_thresh=0.02, min_area_frac=0.0010, split_touching=False):
    """Segment objects standing above the fitted table plane; metrics per blob."""
    from scipy import ndimage

    H, W = depth.shape
    fx, fy = intr["fx"], intr["fy"]
    valid = (depth > 0.1) & (depth < 5.0)
    plane = fit_table_plane(depth, valid)
    height_map = plane - depth                     # >0 = raised above table
    raised = valid & (height_map > height_thresh)
    raised = ndimage.binary_opening(raised, iterations=2)
    raised = ndimage.binary_closing(raised, iterations=2)
    lab, n = _watershed_split(raised) if split_touching else ndimage.label(raised)
    out = []
    for i in range(1, n + 1):
        m = lab == i
        area_px = int(m.sum())
        if area_px < min_area_frac * H * W:
            continue
        hm = height_map[m]
        top_h = float(np.percentile(hm, 95))        # tallest point = object height
        med_d = float(np.median(depth[m]))
        ys, xs = np.where(m)
        x1, x2, y1, y2 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
        # touches any frame edge = robot arm / gripper / table-edge intrusion, not an item
        if x1 <= 2 or y1 <= 2 or x2 >= W - 3 or y2 >= H - 3:
            continue
        if top_h > 0.40:          # >40cm raised = arm/background artefact, not a grocery item
            continue
        w_px, h_px = x2 - x1 + 1, y2 - y1 + 1
        # metric size at the object's own depth
        width_cm = w_px * med_d / fx * 100.0
        length_cm = h_px * med_d / fy * 100.0
        footprint_cm2 = area_px * (med_d / fx) * (med_d / fy) * 1e4
        out.append({
            "height_cm": round(top_h * 100.0, 1),
            "diam_cm": round((width_cm + length_cm) / 2.0, 1),
            "width_cm": round(width_cm, 1),
            "length_cm": round(length_cm, 1),
            "footprint_cm2": round(footprint_cm2, 1),
            "top_depth_cm": round(med_d * 100.0, 1),
            "table_depth_cm": round(float(np.median(plane)) * 100.0, 1),
            "area_px": area_px,
            "cx": float(xs.mean()), "cy": float(ys.mean()),
            "bbox_px": [x1, y1, x2, y2],
        })
    out.sort(key=lambda b: -b["area_px"])
    return out


def load_preds(pred_run, scene_id):
    fp = pred_run / scene_id / "frames" / "frame_000.json"
    if not fp.exists():
        return []
    return json.loads(fp.read_text()).get("items", [])


def attach_pred_name(blob, preds, W, H):
    """Nearest predicted box (by centroid containment / distance)."""
    best, bestd = None, 1e9
    for it in preds:
        bb = it.get("bbox_2d")
        if not bb or len(bb) != 4:
            continue
        x1, y1, x2, y2 = [v / 1000.0 * (W if k % 2 == 0 else H) for k, v in enumerate(bb)]
        pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
        d = (pcx - blob["cx"]) ** 2 + (pcy - blob["cy"]) ** 2
        if d < bestd:
            best, bestd = it, d
    return (best or {}).get("name")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--gt", type=Path, default=ROOT / "data/robot_lab/rgbd/scene_gt.json")
    ap.add_argument("--pred-run", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "runs/eval/depth_separation.json")
    ap.add_argument("--csv", type=Path, default=ROOT / "runs/eval/depth_separation.csv")
    ap.add_argument("--plot", type=Path, default=None)
    ap.add_argument("--height-thresh", type=float, default=0.02)
    ap.add_argument("--confusion", type=Path, default=None,
                    help="loo_confusion_rt.json — cross-ref RGB-confused pairs vs depth gaps")
    args = ap.parse_args()

    from PIL import Image

    gt = json.loads(args.gt.read_text()) if args.gt.exists() else {}
    rows = []
    catalog = {}  # gt object name (from single_*) -> metrics
    for d in sorted(args.rgbd_root.glob("*/")):
        name = d.name
        depth_fp, intr_fp = d / "depth.npy", d / "intrinsics.json"
        if not (depth_fp.exists() and intr_fp.exists()):
            continue
        depth = np.load(depth_fp).astype(np.float32)
        intr = json.loads(intr_fp.read_text())
        H, W = depth.shape
        bs = blobs(depth, intr, height_thresh=args.height_thresh)
        preds = load_preds(args.pred_run, name) if args.pred_run else []
        if name.startswith("single_"):
            # single_<obj> = pose 1 (upright), single_<obj>2 = pose 2 (laying).
            # Strip the trailing pose digit for the GT object name; keep both poses.
            suffix = re.sub(r"\d+$", "", name[len("single_"):])
            gt_name = suffix.replace("_", " ").strip()
            b = max(bs, key=lambda x: x["area_px"]) if bs else None
            if b:
                b = {**b, "scene": name, "gt_name": gt_name, "pred_name": None, "kind": "single"}
                catalog.setdefault(gt_name, []).append(b)
                rows.append(b)
        else:
            for b in bs:
                b = {**b, "scene": name, "gt_name": None,
                     "pred_name": attach_pred_name(b, preds, W, H) if preds else None,
                     "kind": "scene"}
                rows.append(b)

    # --- write ---
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"rows": rows, "catalog_keys": sorted(catalog)}, indent=1) + "\n")
    cols = ["scene", "kind", "gt_name", "pred_name", "height_cm", "diam_cm",
            "width_cm", "length_cm", "footprint_cm2", "top_depth_cm", "table_depth_cm", "area_px"]
    with args.csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # --- report ---
    print(f"blobs: {len(rows)} across {len(set(r['scene'] for r in rows))} frames "
          f"-> {args.out.name}, {args.csv.name}")
    if catalog:
        print("\n=== SINGLE-OBJECT CATALOG (GT-labeled; each object in 2 poses) ===")
        print(f"{'object':24s} {'pose':6s} {'h_cm':>6} {'diam':>6} {'W':>6} {'L':>6} {'foot':>7}")
        # pose label inferred from height: the taller measurement = upright
        for k in sorted(catalog, key=lambda x: -max(b['height_cm'] for b in catalog[x])):
            poses = sorted(catalog[k], key=lambda b: -b['height_cm'])
            for i, b in enumerate(poses):
                pose = "uprght" if i == 0 and len(poses) > 1 else ("laying" if len(poses) > 1 else "-")
                print(f"{k:24s} {pose:6s} {b['height_cm']:6.1f} {b['diam_cm']:6.1f} "
                      f"{b['width_cm']:6.1f} {b['length_cm']:6.1f} {b['footprint_cm2']:7.1f}")
        # can-cluster separability: upright cans only (tallest pose per can), in (h,diam)
        cans = {k: max(v, key=lambda b: b['height_cm']) for k, v in catalog.items() if "can" in k}
        if len(cans) >= 2:
            hs = [v["height_cm"] for v in cans.values()]
            ds = [v["diam_cm"] for v in cans.values()]
            print(f"\nupright cans ({len(cans)}): height {min(hs):.1f}-{max(hs):.1f}cm, "
                  f"diam {min(ds):.1f}-{max(ds):.1f}cm")
            # pairwise: nearest can in (h,diam); flag pairs closer than 2cm in BOTH dims
            names = list(cans)
            collide = []
            for a in range(len(names)):
                for c in range(a + 1, len(names)):
                    va, vc = cans[names[a]], cans[names[c]]
                    if abs(va["height_cm"] - vc["height_cm"]) < 2 and abs(va["diam_cm"] - vc["diam_cm"]) < 2:
                        collide.append((names[a], names[c]))
            print("verdict: " + ("SEPARABLE — no can pair within 2cm in both dims"
                  if not collide else f"OVERLAP pairs (depth alone insufficient): {collide}"))

    # --- cross-reference: do the actual RGB confusions land on depth-separable pairs? ---
    if args.confusion and catalog and args.confusion.exists():
        up = {k: max(v, key=lambda b: b["height_cm"]) for k, v in catalog.items()}  # upright per object
        conf = json.loads(args.confusion.read_text()).get("per_object", {})
        xref = []
        for true, info in conf.items():
            for wrong, p in info.get("perceived_as", {}).items():
                if wrong == true or p < 0.1:
                    continue
                a, b = up.get(true), up.get(wrong)
                if not (a and b):
                    verdict = "no-depth-entry (same-shape: read-text territory)"
                    dh = dd = None
                else:
                    dh, dd = abs(a["height_cm"] - b["height_cm"]), abs(a["diam_cm"] - b["diam_cm"])
                    verdict = "SEPARABLE" if (dh >= 3 or dd >= 3) else "depth-ambiguous"
                xref.append({"true": true, "wrong": wrong, "rgb_prob": p,
                             "dh_cm": dh, "dd_cm": dd, "depth": verdict})
        xref.sort(key=lambda r: -r["rgb_prob"])
        (args.out.parent / "depth_vs_confusion.json").write_text(json.dumps(xref, indent=1) + "\n")
        print("\n=== RGB CONFUSION vs DEPTH SEPARABILITY (depth-entry pairs) ===")
        sep = [r for r in xref if r["depth"] == "SEPARABLE"]
        amb = [r for r in xref if r["depth"] == "depth-ambiguous"]
        for r in xref:
            g = f"Dh={r['dh_cm']:4.1f} Dd={r['dd_cm']:4.1f}" if r["dh_cm"] is not None else "    no depth entry "
            print(f"  {r['true']:18s}-> {r['wrong']:20s} p={r['rgb_prob']:.2f}  {g}  {r['depth']}")
        withdepth = sep + amb
        if withdepth:
            print(f"-> depth separates {len(sep)}/{len(withdepth)} RGB-confused pairs that have a depth entry "
                  f"(-> runs/eval/depth_vs_confusion.json)")

    print("\n=== SCENE BLOBS ===")
    for r in [x for x in rows if x["kind"] == "scene"]:
        print(f"{r['scene']:18s} h={r['height_cm']:5.1f} d={r['diam_cm']:5.1f} "
              f"foot={r['footprint_cm2']:6.1f}  pred={r['pred_name']}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(7, 5))
            for r in rows:
                lbl = r["gt_name"] or r["pred_name"] or "?"
                mk = "o" if r["kind"] == "single" else "x"
                ax.scatter(r["diam_cm"], r["height_cm"], marker=mk, s=60)
                ax.annotate(lbl, (r["diam_cm"], r["height_cm"]), fontsize=7)
            ax.set_xlabel("diameter (cm)"); ax.set_ylabel("height (cm)")
            ax.set_title("Depth size separation (o=single GT, x=scene blob)")
            ax.grid(alpha=0.3)
            fig.tight_layout(); fig.savefig(args.plot, dpi=130)
            print(f"\nplot -> {args.plot}")
        except Exception as e:  # noqa: BLE001
            print(f"plot skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

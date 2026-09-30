#!/usr/bin/env python3
"""Auto-label object boxes from removal sequences via depth differencing.

Free supervision for the proposal-quality audit (Tier-1). In a removal sequence
the camera is fixed and exactly one object leaves between consecutive steps, and
meta.json records its NAME in `removed`. So the region that was raised above the
table at step t and is flat at step t+1 IS that object's footprint box, already
labelled. No hand annotation of names needed -- only a visual sanity check of the
boxes.

Method (per consecutive pair t -> t+1, both fixed top-down aligned RGB-D):
  1. fit the table plane at each step (reuse eval_depth_separation)
  2. height_map = plane - depth ; raised = height_map > thresh
  3. removed_region = raised(t) AND NOT raised(t+1)   (object present then gone)
  4. clean, take the largest connected component -> bbox = the removed object
  5. tag with meta[t+1]["removed"] as the GT name

Quality is reported so a human can filter: `dominance` = top-blob area / all
changed area (low = neighbours shifted / messy diff), plus a plausible-size gate.
Only events with a non-empty `removed` name are emitted.

Outputs:
  runs/eval/removal_diff_boxes.json  -- [{scene, step_prev, step_curr, name,
                                          bbox_px, area_px, dominance, height_cm,
                                          footprint_cm2, ok}]
  runs/eval/removal_diff_overlays/<scene>_<curr>.png  (with --overlays)

Usage:
  .venv/bin/python3.12 tools/eval/removal_diff_boxes.py --overlays
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from eval_depth_separation import fit_table_plane  # noqa: E402


def load_bundle(d: Path):
    depth = np.load(d / "depth.npy").astype(np.float32)
    intr = json.loads((d / "intrinsics.json").read_text())
    meta = json.loads((d / "meta.json").read_text())
    return depth, intr, meta


def raised_mask(depth, thresh=0.02):
    valid = (depth > 0.1) & (depth < 5.0)
    plane = fit_table_plane(depth, valid)
    height_map = plane - depth
    raised = valid & (height_map > thresh)
    raised = ndimage.binary_opening(raised, iterations=2)
    raised = ndimage.binary_closing(raised, iterations=2)
    return raised, height_map


def group_sequences(root: Path):
    """Group step bundles by scene prefix; return {scene: [(step_idx, dir), ...]}."""
    seqs = defaultdict(list)
    for d in sorted(root.glob("*/")):
        m = re.match(r"^(.*?)_(?:step)?(\d+)$", d.name)
        if not m:
            continue
        prefix, idx = m.group(1), int(m.group(2))
        # only multi-step removal/clearing sequences (rem_*, mix_full_basket, ...)
        if not (d / "meta.json").exists():
            continue
        seqs[prefix].append((idx, d))
    # keep only prefixes that actually form a >=2-step sequence with removals
    return {k: sorted(v) for k, v in seqs.items() if len(v) >= 2}


def extract(root: Path, thresh: float, min_area_frac: float):
    events = []
    seqs = group_sequences(root)
    for scene, steps in sorted(seqs.items()):
        # verify this is a removal-style sequence (some step carries `removed`)
        metas = [json.loads((d / "meta.json").read_text()) for _, d in steps]
        if not any(mm.get("removed") for mm in metas):
            continue
        for (i_prev, d_prev), (i_cur, d_cur) in zip(steps, steps[1:]):
            m_cur = json.loads((d_cur / "meta.json").read_text())
            name = (m_cur.get("removed") or "").strip()
            if not name:
                continue
            dp, intr, _ = load_bundle(d_prev)
            dc, _, _ = load_bundle(d_cur)
            if dp.shape != dc.shape:
                continue
            H, W = dp.shape
            fx, fy = intr["fx"], intr["fy"]
            rp, hmp = raised_mask(dp, thresh)
            rc, _ = raised_mask(dc, thresh)
            removed = rp & ~rc
            removed = ndimage.binary_opening(removed, iterations=2)
            lab, n = ndimage.label(removed)
            if n == 0:
                events.append({"scene": scene, "step_prev": i_prev, "step_curr": i_cur,
                               "name": name, "bbox_px": None, "ok": False,
                               "reason": "no_change_blob"})
                continue
            sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
            total = float(sizes.sum())
            top = int(np.argmax(sizes)) + 1
            area_px = int(sizes[top - 1])
            dominance = area_px / total if total > 0 else 0.0
            m = lab == top
            ys, xs = np.where(m)
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            med_d = float(np.median(dp[m]))
            top_h = float(np.percentile(hmp[m], 95))
            footprint_cm2 = area_px * (med_d / fx) * (med_d / fy) * 1e4
            ok = (area_px >= min_area_frac * H * W) and (dominance >= 0.55) \
                and (top_h < 0.40) and not (x1 <= 2 or y1 <= 2 or x2 >= W - 3 or y2 >= H - 3)
            events.append({
                "scene": scene, "step_prev": i_prev, "step_curr": i_cur, "name": name,
                "bbox_px": [x1, y1, x2, y2], "area_px": area_px,
                "dominance": round(dominance, 3), "height_cm": round(top_h * 100, 1),
                "footprint_cm2": round(footprint_cm2, 1), "n_change_blobs": n,
                "src_dir_prev": str(d_prev.relative_to(ROOT)),
                "src_dir_curr": str(d_cur.relative_to(ROOT)), "ok": bool(ok),
            })
    return events


def draw_overlays(events, root: Path, out_dir: Path):
    from PIL import Image, ImageDraw
    out_dir.mkdir(parents=True, exist_ok=True)
    for e in events:
        if not e.get("bbox_px"):
            continue
        cur = ROOT / e["src_dir_curr"]
        prev = ROOT / e["src_dir_prev"]
        img = Image.open(prev / "color.png").convert("RGB")  # object still present
        dr = ImageDraw.Draw(img)
        x1, y1, x2, y2 = e["bbox_px"]
        col = (0, 220, 0) if e["ok"] else (255, 90, 0)
        dr.rectangle([x1, y1, x2, y2], outline=col, width=3)
        dr.text((x1 + 2, max(0, y1 - 12)), f'{e["name"]} d={e["dominance"]}', fill=col)
        img.save(out_dir / f'{e["scene"]}_{e["step_curr"]:02d}.png')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/eval/removal_diff_boxes.json")
    ap.add_argument("--overlays", action="store_true")
    ap.add_argument("--overlay-dir", type=Path, default=ROOT / "runs/eval/removal_diff_overlays")
    ap.add_argument("--height-thresh", type=float, default=0.02)
    ap.add_argument("--min-area-frac", type=float, default=0.0010)
    # mix_full_basket is a removal-style sequence too, and it was captured
    # 2026-06-18 on the camera mount that was replaced. Its intrinsics are
    # identical to the current rig's, so nothing downstream can detect the
    # mixture -- only the capture date separates them. Restricting to the
    # sequences shot on the deployed mount is the guard.
    ap.add_argument("--scenes", default="rem2_,rem3_",
                    help="comma-separated scene-name prefixes to include; "
                         "empty string includes every sequence found")
    args = ap.parse_args()

    events = extract(args.rgbd_root, args.height_thresh, args.min_area_frac)
    keep = tuple(x for x in args.scenes.split(",") if x)
    if keep:
        dropped = sorted({e["scene"] for e in events if not e["scene"].startswith(keep)})
        events = [e for e in events if e["scene"].startswith(keep)]
        for sc in dropped:
            print(f"  [skip] {sc}: not in --scenes {args.scenes}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(events, indent=1) + "\n")

    n = len(events)
    ok = sum(1 for e in events if e["ok"])
    boxed = sum(1 for e in events if e.get("bbox_px"))
    scenes = len({e["scene"] for e in events})
    print(f"removal events: {n} across {scenes} sequences | boxed {boxed} | clean(ok) {ok}")
    print(f"auto-GT boxes -> {args.out}")
    if args.overlays:
        draw_overlays(events, args.rgbd_root, args.overlay_dir)
        print(f"overlays -> {args.overlay_dir}")
    # quick per-scene ok summary
    per = defaultdict(lambda: [0, 0])
    for e in events:
        per[e["scene"]][0] += 1
        per[e["scene"]][1] += int(e["ok"])
    for s, (tot, o) in sorted(per.items()):
        print(f"  {s:24s} {o}/{tot} clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

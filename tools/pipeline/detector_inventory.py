#!/usr/bin/env python3
"""Assemble a closed-set DETECTOR run into the production planner inventory.

The synthetic-trained YOLO detector is the primary robot-lab perception. This turns
its per-scene detections into the same deliverable format as depth_crop_inventory.py:
each kept item carries name + name_confidence + bbox_2d + the planner attributes (from
the catalog by name, else estimated — never null). Detections below --min-conf but
above --unknown-conf are surfaced as graspable `unknown_object` (needs_rescan) rather
than dropped, mirroring the depth-crop unknown-graspable policy.

Usage. The shipped detector is syn_v15b, 32 product classes, and the only copy of it
in this repo is robot/robot_pc_package/weights/best.pt. An
earlier version of this line named syn_v6_plate and claimed the copy at the repo root
was byte-identical; by then that file held syn_v7_bags, which scores 0.800 on b50
against v15b's 0.886. It has been deleted. Check what a checkpoint is before trusting
a path: torch.load(w)["train_args"]["name"].
  python tools/pipeline/detector_inventory.py \
      --weights robot/robot_pc_package/weights/best.pt \
      --classes robot/robot_pc_package/classes.json \
      --pattern 'mix_*' --min-conf 0.4 --out runs/eval/detector_inventory_mixed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from build_planner_output import load_planner_attrs, match_attrs, estimate_attrs, PLANNER_FIELDS  # noqa: E402


def main() -> int:
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--pattern", action="append", required=True)
    ap.add_argument("--min-conf", type=float, default=0.4, help="named-item confidence threshold")
    ap.add_argument("--unknown-conf", type=float, default=0.15,
                    help="detections in [unknown-conf, min-conf) become graspable unknown_object")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--depth-crosscheck", action="store_true",
                    help="fail-closed depth cross-check: raised table blobs (depth.npy + "
                         "intrinsics.json next to color.png) not claimed by any detection are "
                         "appended as graspable unknown_object (needs_rescan, provenance "
                         "depth_blob). Catches silent RGB omissions (validated 5/5 on the "
                         "bleach-miss frames); CPU-only.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seq-layout", action="store_true",
                    help="write <out>/<scene>/frames/frame_000.json, the layout "
                         "evaluate_sequences.py reads, instead of <out>/<scene>.json")
    args = ap.parse_args()

    labels = json.loads(args.classes.read_text())
    planner = load_planner_attrs()
    model = YOLO(str(args.weights))

    dirs = []
    for pat in args.pattern:
        dirs += [d for d in args.rgbd_root.glob(pat) if d.is_dir()]
    dirs = sorted(set(dirs))

    depth_blob_crosscheck = None
    if args.depth_crosscheck:
        sys.path.insert(0, str(ROOT / "tools/live"))
        from live_backends import depth_blob_crosscheck

    args.out.mkdir(parents=True, exist_ok=True)
    n_scenes = n_named = n_unknown = 0
    for d in dirs:
        cfp = d / "color.png"
        if not cfp.exists():
            continue
        r = model(str(cfp), imgsz=args.imgsz, verbose=False)[0]
        items = []
        if r.boxes is not None:
            for cls, conf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(),
                                       r.boxes.xyxy.tolist()):
                conf = float(conf)
                box = [round(v, 1) for v in xyxy]
                if conf >= args.min_conf:
                    name = labels[int(cls)]
                    pa = match_attrs(name, planner) or estimate_attrs({"name": name})
                    row = {"name": name, "name_confidence": round(conf, 4), "bbox_2d": box}
                    row.update({f: pa.get(f) for f in PLANNER_FIELDS})
                    items.append(row)
                    n_named += 1
                elif conf >= args.unknown_conf:
                    pa = estimate_attrs({"name": "unknown_object"})
                    row = {"name": "unknown_object", "name_confidence": round(conf, 4),
                           "bbox_2d": box, "graspable": True, "needs_rescan": True,
                           "vlm_name": labels[int(cls)]}  # detector's best guess, provenance
                    row.update({f: pa.get(f) for f in PLANNER_FIELDS})
                    items.append(row)
                    n_unknown += 1
        if depth_blob_crosscheck is not None:
            items, added = depth_blob_crosscheck(items, d, planner)
            n_unknown += len(added)
        if args.seq_layout:
            fp = args.out / d.name / "frames" / "frame_000.json"
            fp.parent.mkdir(parents=True, exist_ok=True)
        else:
            fp = args.out / f"{d.name}.json"
        fp.write_text(json.dumps({"items": items}, indent=1, ensure_ascii=False) + "\n")
        n_scenes += 1

    print(f"detector inventory: {n_scenes} scenes, {n_named} named + {n_unknown} unknown_graspable "
          f"(min_conf={args.min_conf}, depth_crosscheck={args.depth_crosscheck}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

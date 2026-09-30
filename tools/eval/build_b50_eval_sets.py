#!/usr/bin/env python3
"""Turn the b50 captures into the two roots every scorer in this repo expects.

The b50 bundles carry their own ground truth: --scenes capture writes the
placed object list into meta.json, so there is nothing to annotate. This makes

  datasets_perception/b50/<scene>/frames/frame_000.png   what a model reads
  datasets_seq_GT_b50/<scene>/scene.json                 what it is scored against

Characteristics of a ground-truth object are the catalog row for its name, the
same convention as datasets_seq_GT_robot_49: the name is the only thing a
person annotates, everything else follows from it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("group", "packaging", "rigidity", "fragile", "cold_chain",
          "edible", "spill_risk", "weight_class")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgbd-root", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--prefix", default="b50_")
    ap.add_argument("--catalog", type=Path,
                    default=ROOT / "data/robot_lab/robot_sku_attrs.json")
    ap.add_argument("--images-out", type=Path, default=ROOT / "datasets_perception/b50")
    ap.add_argument("--gt-out", type=Path, default=ROOT / "datasets_seq_GT_b50")
    args = ap.parse_args()

    catalog = json.loads(args.catalog.read_text())
    n_scene = n_item = n_obj = 0
    unknown = set()

    for d in sorted(args.rgbd_root.glob(f"{args.prefix}*")):
        color, meta_fp = d / "color.png", d / "meta.json"
        if not (color.exists() and meta_fp.exists()):
            continue
        gt = json.loads(meta_fp.read_text()).get("gt_objects") or []
        if not gt:
            continue

        fr = args.images_out / d.name / "frames"
        fr.mkdir(parents=True, exist_ok=True)
        link = fr / "frame_000.png"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(os.path.relpath(color.resolve(), fr), link)

        items = []
        for name, q in sorted(collections.Counter(gt).items()):
            row = catalog.get(name)
            if row is None:
                unknown.add(name)
                continue
            items.append({"name": name,
                          **{f: row[f] for f in FIELDS if f in row},
                          "quantity": q})
            n_item += 1
            n_obj += q

        # Both files, because the scorer reads per-frame ground truth and the
        # scene-level soft checks read scene.json. A b50 capture is one frame.
        blob = json.dumps({"items": items}, indent=1) + "\n"
        sd = args.gt_out / d.name
        (sd / "frames").mkdir(parents=True, exist_ok=True)
        (sd / "scene.json").write_text(blob)
        (sd / "frames" / "frame_000.json").write_text(blob)
        n_scene += 1

    print(f"scenes {n_scene}  items {n_item}  objects {n_obj}")
    print(f"images -> {args.images_out}")
    print(f"gt     -> {args.gt_out}")
    if unknown:
        print(f"NOT IN CATALOG (skipped): {sorted(unknown)}")
    return 1 if unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())

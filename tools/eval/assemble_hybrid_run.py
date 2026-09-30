#!/usr/bin/env python3
"""Reassemble the hybrid: detector boxes, vision-language model names.

yolo_crops_for_vlm.py cuts one image per detection and records which scene it
came from; the model is then run over those crops one at a time. This puts the
crops back together into one inventory per scene, in the layout the scorers
read, so the hybrid is measured exactly as the other three systems are.

A crop the model returns nothing for keeps the detector's own name, which is
the deployable behaviour: the box is real, something is there, and discarding
it would credit the hybrid for a recall failure it did not have.

--own-chars keeps the characteristics the model predicted for the crop (run it
with VMT_EMIT_FULL_RECORD=1) instead of reading them from the catalog. A crop the
model returned nothing for then carries no characteristics.

  python tools/eval/assemble_hybrid_run.py \
      --manifest data/robot_lab/yolo_crops_b50/crop_manifest.json \
      --vlm runs/b50_yolo_crops_vlm --out runs/b50_hybrid
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from run_gemini_robotics import full_catalog_attrs  # noqa: E402

OWN_FIELDS = ["group", "packaging", "weight_class", "rigidity",
              "cold_chain", "spill_risk", "edible"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--vlm", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--own-chars", action="store_true")
    args = ap.parse_args()

    by_scene = defaultdict(list)
    n_vlm = n_fallback = 0
    for c in json.loads(args.manifest.read_text()):
        sid = c["crop"][:-4]
        fp = args.vlm / sid / "frames/frame_000.json"
        name, own = None, {}
        if fp.exists():
            items = json.loads(fp.read_text()).get("items") or []
            if items:
                name, own = items[0].get("name"), items[0]
        if name:
            n_vlm += 1
        else:
            name, n_fallback = c["yolo_name"], n_fallback + 1
        row = {"name": name, "quantity": 1, "bbox_2d": c["bbox_px"],
               "yolo_name": c["yolo_name"], "yolo_conf": c["yolo_conf"]}
        if args.own_chars:
            row.update({f: own.get(f) for f in OWN_FIELDS})
        else:
            row.update(full_catalog_attrs(name))
        by_scene[c["src_dir"]].append(row)

    for scene, items in by_scene.items():
        fr = args.out / scene / "frames"
        fr.mkdir(parents=True, exist_ok=True)
        (fr / "frame_000.json").write_text(
            json.dumps({"items": items}, indent=1, ensure_ascii=False) + "\n")

    print(f"{len(by_scene)} scenes, {n_vlm} names from the model, "
          f"{n_fallback} kept from the detector -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

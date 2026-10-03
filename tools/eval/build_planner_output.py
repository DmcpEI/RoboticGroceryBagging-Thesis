#!/usr/bin/env python3
"""Build the planner-attribute perception output for the colleague (Jacopo).

Takes a perception run (Layer-1 items with name/group/packaging/bbox_2d/
name_confidence) and attaches the 7 PLANNER attributes per item by matching the
predicted name to the planner catalog:
  est_weight_g, est_volume_cc, crush_score, category, temperature,
  spill_risk, spill_vulnerable.

Output (per scene): the predicted inventory each item carrying perception fields
(name, group, packaging, quantity, name_confidence, bbox_2d) + the planner
attributes (or planner_matched=false when the name isn't in the catalog), plus
the scene GT for reference.

Usage (cluster):
  python tools/eval/build_planner_output.py \
      --pred runs/loo_planner_base \
      --run-manifest datasets_perception/loo_planner_manifest.json \
      --loo-manifest data/robot_lab/loo_scenes/manifest.csv \
      --out runs/eval/jacopo_planner_output
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
PLANNER_CATALOG = ROOT / "data/robot_lab/robot_item_catalog_planner.json"
# est_weight_g/est_volume_cc/crush_score/category/temperature/spill_risk/spill_vulnerable
# come from the YCB-sourced planner catalog (match_attrs/estimate_attrs below).
# orientation_sensitive used to be copied straight off the perception item here.
# Settled Decision 14 deleted the field; the copy outlived it and wrote a
# constant False into every row. Removed 2026-08-31.
PLANNER_FIELDS = ["est_weight_g", "est_volume_cc", "crush_score", "category",
                  "temperature", "spill_risk", "spill_vulnerable", "edible"]
SPILL_VULNERABLE_CATS = {"Bakery", "Produce", "Snacks"}


def estimate_attrs(item):
    """Deterministic planner attrs for items NOT in the catalog (never null).

    Uses the Layer-3 adapter (weight_class/packaging/group proxies) for
    weight/volume/crush/category/temperature; spill from the standard rules
    (spill_risk = leak_risk|is_liquid; spill_vulnerable = category in
    {Bakery,Produce,Snacks}).
    """
    from tools.planner.perception_to_packbot_adapter import _build_packbot_item
    pb = _build_packbot_item(item)
    cat = pb["category"]
    return {
        "est_weight_g": pb["est_weight_g"],
        "est_volume_cc": pb["est_volume_cc"],
        "crush_score": pb["crush_score"],
        "category": cat,
        "temperature": pb["temperature"],
        "spill_risk": bool(item.get("spill_risk", item.get("leak_risk") or item.get("is_liquid"))),
        "spill_vulnerable": cat in SPILL_VULNERABLE_CATS,
        "edible": pb["_source"]["edible"],
    }


def load_planner_attrs():
    """name -> {planner attrs} from the planner catalog (collapsed per product)."""
    d = json.loads(PLANNER_CATALOG.read_text())
    idmap, idata = d["item_id_map"], d["items_data"]
    by = {}
    for iid, m in idmap.items():
        by.setdefault(m["source_name"], {f: idata[iid].get(f) for f in PLANNER_FIELDS})
    return by


def norm(s):
    return str(s or "").strip().lower()


def match_attrs(name, table):
    n = norm(name)
    if n in {norm(k): k for k in table}:  # exact (case-insensitive)
        return table[{norm(k): k for k in table}[n]]
    for k in table:  # tolerant: first-word / substring
        kn = norm(k)
        if n and kn and (n in kn or kn in n or n.split()[0] == kn.split()[0]):
            return table[k]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--run-manifest", type=Path, required=True)
    ap.add_argument("--loo-manifest", type=Path, default=ROOT / "data/robot_lab/loo_scenes/manifest.csv")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    attrs = load_planner_attrs()
    run = json.loads(args.run_manifest.read_text())
    sid2stem = {s["scene_id"]: Path(s["source_image"]).stem for s in run["scenes"]}
    loo_gt = {Path(r["image"]).stem: [g for g in r["gt_objects"].split(";") if g]
              for r in csv.DictReader(args.loo_manifest.open())} if args.loo_manifest.exists() else {}

    (args.out).mkdir(parents=True, exist_ok=True)
    scenes_out = {}
    unmatched = Counter()
    n_items = n_match = 0
    for sid, stem in sorted(sid2stem.items()):
        fp = args.pred / sid / "frames" / "frame_000.json"
        items_in = json.loads(fp.read_text()).get("items", []) if fp.exists() else []
        items_out = []
        for it in items_in:
            name = it.get("name")
            uncertain = name == "unknown_object" or bool(it.get("needs_rescan"))
            n_items += 1
            row = {
                "name": name,
                "name_confidence": it.get("name_confidence"),
                "bbox_2d": it.get("bbox_2d"),
            }
            if uncertain:
                # A genuinely unidentified detection: this signal (why the perception
                # layer isn't confident) must survive into the planner-facing output,
                # not get silently overwritten by a confident-looking category guess.
                row["needs_rescan"] = True
                row["graspable"] = it.get("graspable", True)
                if it.get("vlm_name"):
                    row["vlm_name"] = it["vlm_name"]
                unmatched[name] += 1
                pa = estimate_attrs(it)  # planner fields still never null (safe fallback)
            else:
                pa = match_attrs(name, attrs)
                n_match += pa is not None
                if pa is None:
                    unmatched[name] += 1
                    pa = estimate_attrs(it)  # not in catalog -> estimate, never null
            row.update({f: pa.get(f) for f in PLANNER_FIELDS})
            items_out.append(row)
        scenes_out[stem] = {"items": items_out}
        (args.out / f"{stem}.json").write_text(json.dumps(scenes_out[stem], indent=1, ensure_ascii=False) + "\n")

    (args.out / "all_scenes.json").write_text(
        json.dumps({"scenes": scenes_out}, indent=1, ensure_ascii=False) + "\n")
    print(f"{len(scenes_out)} scenes, {n_items} items, catalog-matched {n_match}/{n_items} "
          f"({100*n_match/max(n_items,1):.0f}%); rest ESTIMATED (never null) -> {args.out}")
    if unmatched:
        print("not in catalog -> attrs estimated from group/packaging:", dict(unmatched))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

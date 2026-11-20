#!/usr/bin/env python3
"""Which characteristics does the planner need? Corrupt one, replan, audit.

For each characteristic in turn, the inventory of every b50 scene is corrupted
in that characteristic only, CP-SAT plans bags from the corrupted inventory,
and the plan is audited against the TRUE inventory: rule violations, bags over
capacity, and bags used. A characteristic whose corruption changes nothing is
not used by this planner; one whose corruption causes violations is.

Two corruptions:
  unknown   the value the adapter assumes when the field is absent
            (group other, packaging other, weight medium, rigidity semi,
            cold False, spill False, edible True, quantity 1, grams from
            weight_class, volume from packaging)
  permuted  each product takes this characteristic from another product of
            the catalog, drawn at random (one draw per seed, same for every
            scene), so the error is systematic per product, as a wrong
            catalog row or a consistent misperception would be

Units are planned one per bag slot; weight and volume come from the catalog's
grams unless --weights proxy, which reproduces the frontier of the defended version (390 bags).

  .venv/bin/python3.12 tools/eval/characteristic_perturbation.py \
      --out runs/eval/characteristic_perturbation_b50.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.planner.extended_packbot_cp import solve_bagging_extended  # noqa: E402
from tools.planner.perception_to_packbot_adapter import _build_extended_packbot_item  # noqa: E402
from tools.planner.planner_safety_audit import (  # noqa: E402
    MAX_BAG_VOLUME_CC, MAX_BAG_WEIGHT_G, pair_violations_for_items,
)

GT = ROOT / "datasets_seq_GT_b50"
CATALOG = ROOT / "data/robot_lab/robot_item_catalog_perception.json"

UNKNOWN = {"group": "other", "packaging": "other", "weight_class": "medium",
           "rigidity": "semi", "cold_chain": False, "spill_risk": False,
           "edible": True}
CATEGORICAL = list(UNKNOWN)  # characteristics carried in the record
# planner fields set directly, not derived from a record field
DIRECT = ["crush_score", "weight_g", "volume_cc"]


def load_catalog():
    cat = json.loads(CATALOG.read_text())
    return {e["name"]: e for e in cat["items"]}


def load_scenes():
    scenes = {}
    for d in sorted(GT.iterdir()):
        f = d / "frames" / "frame_000.json"
        if f.is_file():
            scenes[d.name] = json.loads(f.read_text())["items"]
    return scenes


def units(records, catalog, weights):
    """Expand records into one planner item per unit, in listing order."""
    out = []
    for rec in records:
        r = dict(rec)
        if weights == "catalog":
            c = catalog[r["name"]]
            r["measured_weight_g"] = c["measured_weight_g"]
            r["true_volume_cc"] = c["true_volume_cc"]
        for _ in range(int(r.get("quantity", 1) or 1)):
            out.append(r)
    return out


def plan_items(unit_records):
    items = {f"u{i:03d}": _build_extended_packbot_item(r) for i, r in enumerate(unit_records)}
    return items, list(items)


def audit(true_items, order, solution):
    pos = {k: i for i, k in enumerate(order)}
    fam = Counter()
    over = 0
    for bag in solution:
        ids = sorted(bag["items"], key=pos.get)
        w = sum(true_items[i]["est_weight_g"] for i in ids)
        v = sum(true_items[i]["est_volume_cc"] for i in ids)
        over += (w > MAX_BAG_WEIGHT_G) or (v > MAX_BAG_VOLUME_CC)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                for f in pair_violations_for_items(true_items[ids[a]], true_items[ids[b]], ordered=True):
                    fam[f] += 1
    return fam, over


def corrupt_records(records, field, mode, mapping):
    out = []
    for rec in records:
        r = dict(rec)
        if field == "quantity":
            r["quantity"] = 1
        elif field == "weight_g":
            r.pop("measured_weight_g", None)
            if mode == "permuted":
                r["measured_weight_g"] = mapping[r["name"]]
        elif field == "volume_cc":
            r.pop("true_volume_cc", None)
            if mode == "permuted":
                r["true_volume_cc"] = mapping[r["name"]]
        elif field in UNKNOWN:
            r[field] = UNKNOWN[field] if mode == "unknown" else mapping[r["name"]]
        out.append(r)
    return out


def run_scene(args):
    scene_id, records, catalog, weights, field, mode, mapping = args
    true_units = units(records, catalog, weights)
    true_items, order = plan_items(true_units)

    if field is None:
        bad_items = true_items
    elif field == "crush_score":
        bad_items = deepcopy(true_items)
        for k, it in bad_items.items():
            it["crush_score"] = 5 if mode == "unknown" else mapping[it["display_name"]]
    else:
        # corrupt the record (after grams are attached), then rebuild
        bad_records = corrupt_records(true_units, field, mode, mapping)
        bad_items = {k: _build_extended_packbot_item(r) for k, r in zip(order, bad_records)}

    # quantity: units dropped from the corrupted record are never planned
    left_out = 0
    if field == "quantity":
        kept, seen = [], Counter()
        for k, r in zip(order, true_units):
            seen[r["name"]] += 1
            if seen[r["name"]] == 1:
                kept.append(k)
        left_out = len(order) - len(kept)
        bad_items = {k: bad_items[k] for k in kept}
        plan_order = kept
    else:
        plan_order = order

    res = solve_bagging_extended(bad_items, plan_order, enable_spill=True)
    if res.get("status") not in ("OPTIMAL", "FEASIBLE"):
        return {"scene": scene_id, "status": res.get("status"), "bags": None}
    fam, over = audit(true_items, order, res["solution"])
    return {"scene": scene_id, "status": res["status"], "bags": int(res["total_bags_used"]),
            "violations": sum(fam.values()), "families": dict(fam),
            "over_capacity_bags": over, "left_out": left_out,
            "scene_has_violation": sum(fam.values()) > 0}


def product_mapping(field, catalog, products, rng):
    """Each product takes `field` from another product, drawn at random."""
    src = list(products)
    rng.shuffle(src)
    def val(name):
        c = catalog[name]
        if field == "weight_g":
            return c["measured_weight_g"]
        if field == "volume_cc":
            return c["true_volume_cc"]
        if field == "crush_score":
            return CRUSH[name]
        return c[field]
    return {p: val(s) for p, s in zip(products, src)}


CRUSH: dict = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weights", choices=["catalog", "proxy"], default="catalog")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fields", default=None, help="comma list; default all")
    a = ap.parse_args()

    catalog = load_catalog()
    scenes = load_scenes()
    products = sorted({r["name"] for recs in scenes.values() for r in recs})
    for p in products:  # true crush per product, as the adapter derives it
        c = dict(catalog[p])
        CRUSH[p] = _build_extended_packbot_item(c)["crush_score"]

    fields = a.fields.split(",") if a.fields else CATEGORICAL + ["quantity"] + DIRECT
    jobs = [("baseline", None, "none", None)]
    for f in fields:
        jobs.append((f, f, "unknown", None))
        if f == "quantity":
            continue
        for s in range(a.seeds):
            jobs.append((f, f, f"permuted", product_mapping(f, catalog, products, random.Random(s))))

    results = []
    with ProcessPoolExecutor(a.workers) as ex:
        for label, field, mode, mapping in jobs:
            m = "permuted" if mode == "permuted" else mode
            rows = list(ex.map(run_scene, [(sid, recs, catalog, a.weights, field, m, mapping)
                                          for sid, recs in scenes.items()]))
            failed = [r["scene"] for r in rows if r["bags"] is None]
            ok = [r for r in rows if r["bags"] is not None]
            fam = Counter()
            for r in ok:
                fam.update(r["families"])
            agg = {"characteristic": label, "mode": mode,
                   "bags": sum(r["bags"] for r in ok),
                   "violations": sum(r["violations"] for r in ok),
                   "scenes_with_violation": sum(r["scene_has_violation"] for r in ok),
                   "over_capacity_bags": sum(r["over_capacity_bags"] for r in ok),
                   "left_out": sum(r["left_out"] for r in ok),
                   "families": dict(fam), "failed_scenes": failed}
            results.append(agg)
            print(f"{label:14s} {mode:9s} bags {agg['bags']:4d}  viol {agg['violations']:4d}  "
                  f"scenes {agg['scenes_with_violation']:3d}  over {agg['over_capacity_bags']:3d}  "
                  f"left {agg['left_out']:3d}  {dict(fam)}  failed {len(failed)}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"weights": a.weights, "seeds": a.seeds,
                                 "scenes": len(scenes), "results": results}, indent=1))
    print(f"[OK] {a.out}")


if __name__ == "__main__":
    main()

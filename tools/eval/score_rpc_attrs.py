#!/usr/bin/env python3
"""Score the RPC zero-shot run on packaging/weight_class/cold_chain/fragile/
spill_risk/edible, extending score_rpc_zeroshot.py's group-level design to
the full attribute set now that the 200-SKU ledger is filled.

RPC gt.json only gives per-instance meta-category (not exact SKU identity),
so exact per-instance attribute ground truth isn't available -- same
constraint the group scorer already works around. This applies the identical
fix: for each meta-category, build an accept-set per attribute from the
distinct true values across that meta's SKUs in the ledger (e.g. "milk"
accepts packaging in {carton, bottle}), then greedy-match predictions against
it, same as the group scorer already does.

  python tools/eval/score_rpc_attrs.py \
      --pred runs/rpc_val_600_32b_base_noft \
      --staged datasets_perception/rpc_val_600 \
      --ledger <ledger json: {idx: {meta, group, packaging, weight_class,
                cold_chain, fragile, spill_risk, edible}}> \
      --out runs/eval/rpc_zeroshot_attrs.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORY_NAMES = json.loads((ROOT / "data/rpc_category_names.json").read_text())

ATTRS = ["group", "packaging", "weight_class", "cold_chain", "fragile", "spill_risk", "edible"]


def sku_of(cat) -> str:
    """gt.json `categories` are 200-class ints -> the exact SKU name."""
    return CATEGORY_NAMES[cat] if isinstance(cat, int) else cat


def meta_of(cat) -> str:
    return re.sub(r"^\d+_", "", sku_of(cat))


def build_accept_sets(ledger: dict) -> dict:
    """attr -> {meta: set(acceptable values)}"""
    out = {a: defaultdict(set) for a in ATTRS}
    for row in ledger.values():
        m = row["meta"]
        for a in ATTRS:
            out[a][m].add(row[a])
    return out


def build_exact_sets(ledger: dict) -> dict:
    """attr -> {sku_index: {the one true value}}.

    RPC gt.json carries the exact 200-class SKU, so per-instance attribute
    ground truth is available directly from the ledger -- accept-sets were
    only ever needed because the earlier scorers discarded the SKU index.
    Same shape as build_accept_sets so the matcher is unchanged; the sets
    are just singletons keyed by SKU instead of unions keyed by meta.
    """
    out = {a: {} for a in ATTRS}
    for idx, row in ledger.items():
        for a in ATTRS:
            out[a][int(idx)] = {row[a]}
    return out


def norm_val(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.lower() in ("true", "false"):
        return v.lower() == "true"
    return v


def greedy_accept(gt_metas, pred_vals, accept_sets):
    used = [False] * len(pred_vals)
    tp = 0
    for m in gt_metas:
        accept = accept_sets.get(m, set())
        for i, v in enumerate(pred_vals):
            if not used[i] and v in accept:
                used[i] = True
                tp += 1
                break
    return tp, len(pred_vals) - tp, len(gt_metas) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--staged", type=Path, required=True)
    ap.add_argument("--ledger", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--match", choices=["accept", "exact"], default="exact",
                    help="exact: one true value per instance from the SKU index (default). "
                         "accept: legacy per-meta-category union, kept to reproduce earlier numbers.")
    args = ap.parse_args()

    ledger = json.loads(args.ledger.read_text())
    accept_sets = build_accept_sets(ledger) if args.match == "accept" else build_exact_sets(ledger)

    totals = {a: [0, 0, 0] for a in ATTRS}
    per_meta = {a: defaultdict(lambda: [0, 0]) for a in ATTRS}
    n_scored = 0

    for sd in sorted(args.staged.iterdir()):
        if not sd.is_dir():
            continue
        gt_fp = sd / "gt.json"
        pred_fp = args.pred / sd.name / "frames" / "frame_000.json"
        if not (gt_fp.exists() and pred_fp.exists()):
            continue
        cats = json.loads(gt_fp.read_text())["categories"]
        gt_reports = [meta_of(c) for c in cats]        # always report per meta-category
        # match key: the exact SKU index ("79_alcohol" -> 79) or the meta name
        gt_metas = ([int(sku_of(c).split("_", 1)[0]) for c in cats]
                    if args.match == "exact" else gt_reports)
        items = json.loads(pred_fp.read_text()).get("items", [])

        for a in ATTRS:
            pred_vals = []
            for it in items:
                q = it.get("quantity", 1)
                try:
                    q = max(1, int(q))
                except (TypeError, ValueError):
                    q = 1
                pred_vals.extend([norm_val(it.get(a))] * q)
            tp, fp_, fn = greedy_accept(gt_metas, pred_vals, accept_sets[a])
            totals[a][0] += tp; totals[a][1] += fp_; totals[a][2] += fn

            used = [False] * len(pred_vals)
            for m, rep in zip(gt_metas, gt_reports):
                accept = accept_sets[a].get(m, set())
                hit = False
                for i, v in enumerate(pred_vals):
                    if not used[i] and v in accept:
                        used[i] = True
                        hit = True
                        break
                r = per_meta[a][rep]
                r[0 if hit else 1] += 1
        n_scored += 1

    result = {"n_scenes_scored": n_scored, "attrs": {}}
    for a in ATTRS:
        p, r, f1 = prf(*totals[a])
        result["attrs"][a] = {
            "P": round(p, 3), "R": round(r, 3), "F1": round(f1, 3),
            "tp": totals[a][0], "fp": totals[a][1], "fn": totals[a][2],
            "per_meta_recall": {
                m: {"recall": round(v[0] / (v[0] + v[1]), 3), "n": v[0] + v[1]}
                for m, v in sorted(per_meta[a].items())
            },
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    summary = {a: {k: result["attrs"][a][k] for k in ("P", "R", "F1")} for a in ATTRS}
    print(json.dumps({"n_scenes_scored": n_scored, "summary": summary}, indent=1))
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

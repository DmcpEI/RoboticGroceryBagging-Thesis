#!/usr/bin/env python3
"""Per-characteristic recovery, reported as macro-F1.

One number per characteristic, so Table V can put the Robot and the Retail set
side by side under the same metric. Accuracy is not reported anywhere: most of
these characteristics are heavily imbalanced booleans, and accuracy on those
hides a total loss of the positive class behind a moving third decimal.

Protocol. Both sides are multisets of values, one entry per object, compared
per scene with the greedy matcher the rest of the paper uses; a scene's
unmatched ground-truth entries are false negatives for their own class and
unmatched predictions are false positives for theirs. F1 is computed per class
and averaged over the classes that occur in the ground truth, so a class that
never occurs cannot inflate the score and one that occurs twice cannot be
ignored.

Two characteristics need a word. `crush` carries no annotation on either set:
it is derived, so its row measures how well a derived quantity survives
perception error, not how well anyone labeled it. `quantity` is a property of
an item rather than of an object, so it is scored at item level: a ground-truth
item is matched to the predictions that resolve to the same product, and counts
correct when the two multiplicities agree.

  python tools/eval/score_characteristics.py --gt datasets_seq_GT_b50 \
      --pred runs/b50_detector_v15b --label "Detector + catalog lookup"
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.planner.perception_to_packbot_adapter import compute_crush_score_v2  # noqa: E402
sys.path.insert(0, str(ROOT / "tools/eval"))
from run_gemini_robotics import full_catalog_match  # noqa: E402

# Table II's schema, in its order, plus the identity the characteristics are
# read from. Identity is scored HERE rather than taken from the identification
# table so that it is the same statistic as the rows it is compared against:
# a macro average over products, not a pooled micro F1. Comparing a macro
# characteristic against a micro identity is what made an earlier version of
# this table look like every characteristic beat identity.
VALUE_FIELDS = ["name", "group", "packaging", "weight_class", "rigidity",
                "crush_score", "cold_chain", "spill_risk", "edible"]


def norm(s) -> str:
    return str(s or "").strip().lower()


def crush_of(it: dict) -> int:
    return compute_crush_score_v2(str(it.get("packaging") or ""),
                                  str(it.get("rigidity") or ""),
                                  str(it.get("name") or ""),
                                  str(it.get("group") or ""),
                                  str(it.get("deformation") or ""))


def value(it: dict, field: str, resolve=False):
    if field == "crush_score":
        v = crush_of(it)
    elif field == "name" and resolve:
        # Score identity as the product the pipeline resolves the name to, since
        # that is what the characteristics are actually read from: "cheez-it box"
        # is the cracker box. Names the Robot matcher does not know (the Retail
        # set's anonymized ids) fall back to the name itself, so the same code
        # scores both sets.
        v = full_catalog_match(it.get("name")) or it.get("name")
    else:
        v = it.get(field)
    return None if v is None else str(v).strip().lower()


def expand(items, field, resolve=False):
    """One value per physical object, so a quantity of three weighs three.

    An object the system surfaced without naming carries no characteristic at
    all, so its value is None and the matcher drops it. It is not a false
    positive for any class -- the system asserted nothing -- but the
    ground-truth object it failed to name is still a false negative, which is
    the cost of declining to answer.
    """
    out = []
    for it in items or []:
        out += [value(it, field, resolve)] * max(1, int(it.get("quantity", 1) or 1))
    return out


def confusion(gt_vals, pred_vals, tp, fp, fn):
    """Greedy match on equal values; the rest are errors of their own class."""
    pred = Counter(v for v in pred_vals if v is not None)
    for g in gt_vals:
        if g is None:
            continue
        if pred[g] > 0:
            pred[g] -= 1
            tp[g] += 1
        else:
            fn[g] += 1
    for v, n in pred.items():
        fp[v] += n


def macro_f1(tp, fp, fn):
    classes = sorted(set(tp) | set(fn))          # classes present in ground truth
    if not classes:
        return None, {}
    per = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        r = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        per[c] = {"f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
                  "precision": round(p, 4), "recall": round(r, 4),
                  "support": tp[c] + fn[c]}
    return round(sum(per[c]["f1"] for c in classes) / len(classes), 4), per


def quantity_score(gt_items, pred_items, tp, fp, fn):
    """How many of each product, not how the run happened to group them.

    The ground truth writes one item with quantity 3; the detector writes three
    items of quantity 1. Those are the same table, so both sides are collapsed
    to a count per product before the counts are compared. Names are keyed by
    the product they resolve to, the same rule the identity row uses. A lenient
    match would fold every "bag of ..." into every produce bag in the scene.
    """
    def counts(items):
        c = Counter()
        for it in items or []:
            n = value(it, "name", resolve=True)
            if n and n != "unknown_object":
                c[n] += max(1, int(it.get("quantity", 1) or 1))
        return c

    gt_c, pred_c = counts(gt_items), counts(pred_items)
    for g, gn in gt_c.items():
        pn = pred_c.get(g, 0)
        if pn == gn:
            tp[str(gn)] += 1
        else:
            fn[str(gn)] += 1
            if pn:
                fp[str(pn)] += 1
    for p, pn in pred_c.items():
        if p not in gt_c:
            fp[str(pn)] += 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    scenes = sorted(p.parent.parent.name for p in args.gt.glob("*/frames/frame_000.json"))
    tp = {f: defaultdict(int) for f in VALUE_FIELDS + ["quantity"]}
    fp = {f: defaultdict(int) for f in VALUE_FIELDS + ["quantity"]}
    fn = {f: defaultdict(int) for f in VALUE_FIELDS + ["quantity"]}
    n_missing = 0

    for sc in scenes:
        g = json.loads((args.gt / sc / "frames/frame_000.json").read_text()).get("items", [])
        pfp = args.pred / sc / "frames/frame_000.json"
        if pfp.exists():
            p = json.loads(pfp.read_text()).get("items", [])
        else:
            p, n_missing = [], n_missing + 1
        for f in VALUE_FIELDS:
            confusion(expand(g, f), expand(p, f, resolve=(f == "name")),
                      tp[f], fp[f], fn[f])
        quantity_score(g, p, tp["quantity"], fp["quantity"], fn["quantity"])

    label = args.label or str(args.pred)
    print(f"{label}   ({len(scenes)} scenes"
          + (f", {n_missing} with no prediction" if n_missing else "") + ")\n")
    print(f"{'characteristic':16s}{'macro F1':>10s}   per-class F1")
    print("-" * 78)
    report = {"label": label, "gt": str(args.gt), "pred": str(args.pred),
              "scenes": len(scenes), "scenes_missing": n_missing, "fields": {}}
    for f in ["group", "packaging", "quantity", "weight_class", "rigidity",
              "crush_score", "cold_chain", "spill_risk", "edible", "name"]:
        m, per = macro_f1(tp[f], fp[f], fn[f])
        # A characteristic that takes one value throughout cannot be scored:
        # a constant prediction earns F1 1.0 without predicting anything. Flag
        # it so the paper prints a dash rather than a perfect score.
        single = len(per) < 2
        report["fields"][f] = {"macro_f1": m, "single_class": single, "per_class": per}
        detail = "  ".join(f"{c}={per[c]['f1']:.2f}" for c in sorted(per)) if per else "no ground truth"
        if single and per:
            detail += "   [single class -- not scoreable]"
        print(f"{f:16s}{('  --  ' if m is None else f'{m:10.4f}')}   {detail[:72]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\n[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

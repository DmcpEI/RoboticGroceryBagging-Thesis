#!/usr/bin/env python3
"""What the perception gets wrong, as probabilities a planner can consume.

Three tables per run:

  per_object[X]        P(perceived | X is on the table). How often X is seen at
                       all, and what it gets called when it is.
  given_predicted[L]   P(true | the system said L). For a planner this is the
                       useful direction: a name arrived, what is it really?
  field_accuracy       per characteristic, how often the value is right.

HOW THE CONFUSIONS ARE OBTAINED, AND WHY IT IS NOT A MEASUREMENT.
The b50 captures record WHAT was placed on the table, not WHERE, so there are
no ground-truth boxes and a prediction cannot be tied to the object it came
from. Per scene we cancel every product both sides agree on; what is left is
the misses on one side and the wrong names on the other. Each missed product is
charged against the surplus names in the SAME scene, in proportion. A scene
losing one gelatin box while gaining one cracker box contributes a whole unit
to that pair; two of each contributes a quarter to each of the four pairs.

That cannot separate a substitution from an unrelated miss plus an unrelated
hallucination in the same frame. Treat the pairs as ranked candidates, and the
diagonal (how often a product is named correctly) as solid -- the diagonal
comes from the cancellation, which needs no attribution.

  python tools/eval/perception_uncertainty.py --gt datasets_seq_GT_b50 \
      --pred runs/b50_detector_v15b --label detector \
      --out delivery/jacopo_b50/perception_uncertainty_detector.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/eval"))

from score_characteristics import (  # noqa: E402
    VALUE_FIELDS, confusion, expand, macro_f1, norm, value,
)


def r(x, n=4):
    return round(float(x), n)


def dist(counter, n=3, floor=0.005):
    """Counter -> probability distribution, biggest first.

    Anything under half a percent is dropped: at these sample sizes that tail
    is attribution noise, not a confusion worth planning around. The dropped
    mass is left out rather than redistributed, so the entries stay the rates
    they were measured at and need not sum to exactly one.
    """
    tot = sum(counter.values())
    if tot <= 0:
        return {}
    return {k: round(v / tot, n) for k, v in counter.most_common() if v / tot >= floor}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    scenes = sorted(p.parent.parent.name for p in args.gt.glob("*/frames/frame_000.json"))
    if not scenes:
        raise SystemExit(f"no scenes under {args.gt}")

    gt_total, matched, spurious = Counter(), Counter(), Counter()
    # attributed[(true, predicted_instead)] -> fractional objects
    attributed = defaultdict(float)
    g_gt_total, g_matched, g_spurious = Counter(), Counter(), Counter()
    g_attributed = defaultdict(float)
    tp = {f: defaultdict(int) for f in VALUE_FIELDS}
    fp = {f: defaultdict(int) for f in VALUE_FIELDS}
    fn = {f: defaultdict(int) for f in VALUE_FIELDS}
    n_missing = 0

    def tally(gvals, pvals, total, match, extra_c, attr):
        g, p = Counter(v for v in gvals if v), Counter(v for v in pvals if v)
        total.update(g)
        match.update(g & p)
        lost, extra = g - p, p - g
        extra_c.update(extra)
        n_extra, n_lost = sum(extra.values()), sum(lost.values())
        # Normalise by the LARGER side. Dividing by the surplus alone would
        # hand every missed object a name whenever the scene has a single
        # spurious prediction, so a scene losing four objects and inventing one
        # would report all four as "detected, misnamed". At most min(lost,
        # extra) misses can actually be substitutions; the rest were not
        # detected at all, and that is what makes detection_rate mean something.
        denom = max(n_extra, n_lost)
        if denom:
            for a, na in lost.items():
                for b, nb in extra.items():
                    attr[(a, b)] += na * nb / denom
        return lost, n_extra

    lost_unattributed = Counter()
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

        lost, n_extra = tally(expand(g, "name", resolve=True),
                              expand(p, "name", resolve=True),
                              gt_total, matched, spurious, attributed)
        if not n_extra:
            lost_unattributed.update(lost)
        tally(expand(g, "group"), expand(p, "group"),
              g_gt_total, g_matched, g_spurious, g_attributed)

    # ---- per_object: P(perceived | true object) ------------------------
    per_object = {}
    for prod, n in sorted(gt_total.items()):
        seen_as = Counter({prod: matched[prod]})
        for (a, b), w in attributed.items():
            if a == prod:
                seen_as[b] += w
        detected = sum(seen_as.values())
        per_object[prod] = {
            "detection_rate": r(detected / n),
            "observations": int(n),
            "classified_as": dist(seen_as),
        }

    # ---- given_predicted: P(true | perceived label) --------------------
    def inverse(match, attr, extra_c):
        out = {}
        labels = set(match) | {b for _, b in attr}
        for lab in sorted(labels):
            src = Counter({lab: match[lab]})
            for (a, b), w in attr.items():
                if b == lab:
                    src[a] += w
            tot = sum(src.values())
            if tot <= 0:
                continue
            # surplus predictions of this label that no missing object in the
            # same scene can explain: invented, not a confusion.
            invented = max(0.0, extra_c[lab] - (tot - match[lab]))
            out[lab] = dict(dist(src),
                            _predictions=int(match[lab] + extra_c[lab]),
                            _hallucination_rate=r(invented / max(match[lab] + extra_c[lab], 1)))
        return out

    # ---- field_accuracy: every characteristic --------------------------
    fields = {}
    for f in VALUE_FIELDS:
        t = sum(tp[f].values())
        f_p, f_n = sum(fp[f].values()), sum(fn[f].values())
        m, per = macro_f1(tp[f], fp[f], fn[f])
        support = {k: tp[f][k] + fn[f][k] for k in set(tp[f]) | set(fn[f])}
        total = sum(support.values()) or 1
        base = max(support.values()) / total if support else 0.0
        fields[f] = {
            "accuracy": r(t / total),          # of the objects that ARE there,
            "precision": r(t / max(t + f_p, 1)),  # how many carry the right value
            "macro_f1": r(m),
            "classes": len(support),
            "majority_class_rate": r(base),
        }

    report = {
        "_description": {
            "run": args.label or str(args.pred),
            "scenes": len(scenes),
            "objects_in_ground_truth": int(sum(gt_total.values())),
            "per_object": "P(perceived | true object). detection_rate = fraction of "
                          "appearances where the object was reported at all (correctly "
                          "named or not); classified_as = what it was called when it was "
                          "(the entry under the object's own name is the correct share).",
            "given_predicted": "P(true | perceived label). A detection arrived carrying "
                               "label L: this is what it actually is. _predictions is how "
                               "many times L was output; _hallucination_rate is the share "
                               "of those that no missing object in the same scene can "
                               "explain, i.e. probably nothing was there.",
            "field_accuracy": "Per characteristic over every ground-truth object. "
                              "accuracy = share carrying the right value; precision = "
                              "share of asserted values that were right. Read accuracy "
                              "NEXT TO majority_class_rate: a near-constant boolean "
                              "scores high accuracy without predicting anything, which "
                              "is why macro_f1 is given too.",
            "caveat": "NO GROUND-TRUTH BOXES EXIST for this benchmark -- the captures "
                      "record what was placed, not where. The DIAGONAL (how often a "
                      "product is named correctly) is measured. The OFF-DIAGONAL "
                      "confusions are ATTRIBUTED: per scene, misses are charged against "
                      "the surplus names in that same scene, in proportion. That cannot "
                      "tell a substitution from an unrelated miss plus an unrelated "
                      "hallucination. Use the off-diagonal as ranked candidates, not as "
                      "measured rates. Rates also carry this scene set's object priors.",
            "field_accuracy_values": fields,
            "overall_detection_rate": r((sum(matched.values()) + sum(attributed.values()))
                                        / max(sum(gt_total.values()), 1)),
            # The split that says how much of the line above is measured.
            # named_correctly is the diagonal and needs no attribution.
            # attributed is inferred by charging misses to surplus names in the
            # same scene, and SATURATES on a system that over-predicts: if a
            # scene has at least as many spurious names as misses, every miss
            # is "explained" and the detection rate approaches 1 whatever the
            # system actually saw. Where attributed_share is large, read the
            # detection rate as an upper bound, not as a measurement.
            "overall_named_correctly": r(sum(matched.values())
                                         / max(sum(gt_total.values()), 1)),
            "overall_attributed": r(sum(attributed.values())
                                    / max(sum(gt_total.values()), 1)),
            "attributed_share_of_detections": r(
                sum(attributed.values())
                / max(sum(matched.values()) + sum(attributed.values()), 1)),
            "scenes_with_no_prediction": n_missing,
        },
        "per_object": per_object,
        "given_predicted": inverse(matched, attributed, spurious),
        "given_predicted_group": inverse(g_matched, g_attributed, g_spurious),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")

    lab = args.label or args.pred.name
    D = report["_description"]
    print(f"{lab}: {len(scenes)} scenes, {sum(gt_total.values())} objects")
    print(f"  detection {D['overall_detection_rate']:.3f} "
          f"= named correctly {D['overall_named_correctly']:.3f} "
          f"+ attributed {D['overall_attributed']:.3f} "
          f"({100 * D['attributed_share_of_detections']:.0f}% inferred)")
    if D["attributed_share_of_detections"] > 0.25:
        print("  NOTE: most of the detection rate is attribution, not measurement -- "
              "this system over-predicts, so misses are always explainable.")
    print(f"{'characteristic':16s}{'accuracy':>10s}{'macro F1':>10s}{'majority':>10s}")
    for f, v in fields.items():
        print(f"{f:16s}{v['accuracy']:10.4f}{v['macro_f1']:10.4f}{v['majority_class_rate']:10.4f}")
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

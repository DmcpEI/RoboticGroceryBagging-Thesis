#!/usr/bin/env python3
"""Do derivation rules written for the robot catalog hold on the retail one?
(Dissertation Sec. 4.4.2.)

The rules that derive `cold_chain` from packaging and `edible` from group were
written for the 32-product robot catalog. Here they are applied to the TRUE
group and packaging of every instance of the retail (RPC) ground truth, and the
result is compared with the true value, so the only error measured is the
rule's own.

The rules are those of the robot pipeline's Layer 2 (planner_rules_v1),
restricted to the two characteristics reported:

  edible      group in FOOD_GROUPS -> True, in NONFOOD_GROUPS -> False,
              any other group keeps its true value
  cold_chain  carton, tray, tub -> True; jar, bottle, can -> False;
              cup -> False if household, True if dairy, otherwise unchanged;
              any other packaging keeps its true value

  python tools/eval/rpc_rules_transfer.py --gt datasets_seq_GT_rpc \
      --out runs/eval/rpc_rules_on_true_values.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FOOD_GROUPS = {"produce", "bakery", "pantry", "snack", "frozen", "raw_meat",
               "seafood", "drink", "dairy"}
NONFOOD_GROUPS = {"household", "hygiene", "cleaning", "hardware", "batteries",
                  "pharmacy", "pet"}
COLD_PACKAGING = {"carton", "tray", "tub"}
AMBIENT_PACKAGING = {"jar", "bottle", "can"}


def norm(v, default="other") -> str:
    s = str(v or "").strip().lower().replace(" ", "_")
    return s or default


def edible_rule(group: str, true_value: bool) -> bool:
    if group in FOOD_GROUPS:
        return True
    if group in NONFOOD_GROUPS:
        return False
    return true_value


def cold_chain_rule(packaging: str, group: str, true_value: bool) -> bool:
    if packaging in COLD_PACKAGING:
        return True
    if packaging in AMBIENT_PACKAGING:
        return False
    if packaging == "cup":
        if group == "household":
            return False
        if group == "dairy":
            return True
    return true_value


def main() -> int:
    ap = argparse.ArgumentParser(description="Robot-catalog derivation rules on the retail ground truth.")
    ap.add_argument("--gt", type=Path, required=True,
                    help="ground truth in the sequence layout: <scene>/frames/frame_000.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    n = 0
    cold_marked = Counter()            # (packaging/group) of instances the rule marks cold
    cold_true = 0
    edible = Counter()                 # tp, fp, fn, tn of the rule against the true value
    for fp in sorted(args.gt.glob("*/frames/frame_000.json")):
        for it in json.loads(fp.read_text()).get("items", []):
            q = max(1, int(it.get("quantity") or 1))
            grp, pkg = norm(it.get("group")), norm(it.get("packaging"))
            t_cold, t_edible = bool(it.get("cold_chain")), bool(it.get("edible"))
            r_cold = cold_chain_rule(pkg, grp, t_cold)
            r_edible = edible_rule(grp, t_edible)
            n += q
            cold_true += q * t_cold
            if r_cold and not t_cold:
                cold_marked[f"{pkg}/{grp}"] += q
            edible["tp" if r_edible and t_edible else "fp" if r_edible
                   else "fn" if t_edible else "tn"] += q

    tp, fp_, fn = edible["tp"], edible["fp"], edible["fn"]
    result = {
        "what": "robot-catalog derivation rules applied to the TRUE group and packaging "
                "of every retail instance (tools/eval/rpc_rules_transfer.py)",
        "n_instances": n,
        "cold_chain": {"true_positive_instances": cold_true,
                       "marked_by_rules": sum(cold_marked.values()),
                       "by_packaging_group": dict(cold_marked.most_common())},
        "edible_from_group": {**{k: edible[k] for k in ("tp", "fp", "fn", "tn")},
                              "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                              "precision": round(tp / (tp + fp_), 3) if tp + fp_ else None},
    }
    print(json.dumps(result, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Audit the real-robot bags with the offline solver's safety rules.

analyse_robot_runs.py counts violations as the ERM planner defines them (load on
a protected or leaking item). This replays the Chapter 4 audit predicates
(planner_safety_audit, the same ones reaudit_cached_bags.py uses) over the bags
each trial actually filled, in placement order, so the robot trials and the
offline evaluation are measured alike.

  .venv/bin/python3.12 tools/eval/robot_runs_ch4_audit.py \
      --runs runs/eval/robot_real_runs_old5000.json --json runs/eval/robot_runs_ch4_audit.json
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/eval"), str(ROOT / "tools/planner")]
from build_planner_output import load_planner_attrs  # noqa: E402
import planner_safety_audit as psa  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", type=Path, default=ROOT / "runs/eval/robot_real_runs_old5000.json")
ap.add_argument("--json", type=Path)
args = ap.parse_args()

A = load_planner_attrs()
out = defaultdict(Counter)
for r in json.loads(args.runs.read_text()):
    bags, fam = defaultdict(list), Counter()
    for item, bag in r["placed"]:
        bags[bag].append(A[re.sub(r"#\d+$", "", item)])
    for its in bags.values():
        for i in range(len(its)):
            for j in range(i + 1, len(its)):
                fam.update(psa.pair_violations_for_items(its[i], its[j], ordered=True))
    o = out[r["beta"]]
    o["trials"] += 1
    o["violations"] += sum(fam.values())
    o["trials_affected"] += bool(fam)
    o.update(fam)
for b, o in out.items():
    print(b, dict(o))
if args.json:
    args.json.write_text(json.dumps({b: dict(o) for b, o in out.items()}, indent=1) + "\n")

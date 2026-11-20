#!/usr/bin/env python3
"""Summarise runs of external/risk_aware_mcts as in the dissertation's Tables 5.2 and 5.3:
mean, p95, p99 and max of the episode cost with 90% bootstrap intervals, the
risky placements (sum of the three violation counts) and the episodes or
scenes with at least one, and bags for the benchmark runs.

  .venv/bin/python3.12 tools/planner/summarize_risk_aware.py runs/eval/risk_aware_v18
"""
import json
import sys
from pathlib import Path

import numpy as np

TYPES = ["fragile_weight_on_top", "spill_risk_weight_on_top", "spill_vulnerable_with_spill_risk"]


def boot(x, stat, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    s = [stat(rng.choice(x, len(x))) for _ in range(n)]
    return stat(x), np.percentile(s, 5), np.percentile(s, 95)


def main():
    root = Path(sys.argv[1])
    rows = []
    for f in sorted(root.glob("*/*/exp_data.json")):
        d = json.loads(json.load(open(f)))  # his files hold a JSON string
        x = np.asarray(d["f_vals"], float)
        risky = sum(np.asarray(d[t]) for t in TYPES)
        label = d.get("algo") or f.parent.name.split("_")[1]
        if d.get("erm_beta") is not None or "beta_" in f.parent.name:
            label += f" beta={d.get('erm_beta') or f.parent.name.split('beta_')[1].split('_')[0]}"
        r = {"run": f.parent.parent.name + "/" + label, "n": len(x),
             "mean": boot(x, np.mean), "p95": boot(x, lambda v: np.percentile(v, 95)),
             "p99": boot(x, lambda v: np.percentile(v, 99)), "max": float(x.max()),
             "risky": int(risky.sum()), "affected": int((risky > 0).sum())}
        if "n_bags_used" in d:
            r["bags"] = int(np.sum(d["n_bags_used"]))
        rows.append(r)
    for r in rows:
        ci = lambda k: f"{r[k][0]:.2f} [{r[k][1]:.2f}, {r[k][2]:.2f}]"
        print(f"{r['run']:40s} n={r['n']:3d}  mean {ci('mean')}  p95 {ci('p95')}  p99 {ci('p99')}  "
              f"max {r['max']:.0f}  risky {r['risky']} [{r['affected']}/{r['n']}]"
              + (f"  bags {r['bags']}" if "bags" in r else ""))
    out = root / "summary.json"
    out.write_text(json.dumps(rows, indent=1, default=float))


if __name__ == "__main__":
    main()

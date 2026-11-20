#!/usr/bin/env python3
"""Write a copy of the planner environment whose perception-noise table comes
from a given perception_uncertainty.py file.

binpacking_env_v3.py holds the noise model as a hardcoded FEATURE_DISTRIBUTION
built from the v15b detector. This rebuilds that table from another file (the
deployed v18) and writes the environment under a new name; the original is not
touched. Construction follows the original exactly:

  miss outcome        fraction = 1 - detection_rate, perceived_as None
  each named outcome  fraction = detection_rate * share, features of the
                      product it was named as, from the env's own ITEMS_DATA
  unknown_object      dropped (the env's choices() renormalises)
  observations        per_object[name]["observations"]

  python tools/eval/build_planner_noise_env.py \
      --noise runs/eval/planner_noise/perception_uncertainty_our_perception_v18.json \
      --out tools/planner/binpacking_env_v3_v18.py

--check FILE rebuilds from FILE and compares against the original table.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "external/risk_aware_mcts/envs/binpacking_env_v3.py"
START = "FEATURE_DISTRIBUTION = {"
END = "\n}\n\nNOISY_ITEMS"


def load_env():
    sys.path.insert(0, str(ROOT / "external/risk_aware_mcts"))
    spec = importlib.util.spec_from_file_location("bp_v3", SRC)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def features(item):
    return {"weight": item.weight, "volume": item.volume,
            "crush_score": item.crush_score, "category": item.category,
            "temperature": item.temperature, "spill_risk": item.spill_risk,
            "spill_vulnerable": item.spill_vulnerable,
            "orientation_sensitive": item.orientation_sensitive}


def build(per_object, items, strict=True):
    by_name = {it.name: it for it in items}
    fd = {}
    for it in items:
        if it.name not in per_object and not strict:
            continue
        po = per_object[it.name]  # every catalog product must be in the noise file
        det = po["detection_rate"]
        outs = [{"fraction": round(1 - det, 7), "perceived_as": None, "features": None}]
        for label, share in po["classified_as"].items():
            if label == "unknown_object":
                continue
            outs.append({"fraction": round(det * share, 7), "perceived_as": label,
                         "features": features(by_name[label])})
        fd[it.name] = {"id": it.id, "observations": po["observations"], "outcomes": outs}
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", required=True)
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true",
                    help="compare the rebuilt table with the original instead of writing")
    a = ap.parse_args()
    env = load_env()
    fd = build(json.load(open(a.noise))["per_object"], env.ITEMS_DATA, strict=not a.check)

    if a.check:
        bad = [k for k, v in fd.items()
               if [(o["perceived_as"], round(o["fraction"], 6), o["features"]) for o in v["outcomes"]]
               != [(o["perceived_as"], round(o["fraction"], 6), o["features"]) for o in env.FEATURE_DISTRIBUTION[k]["outcomes"]]
               or v["observations"] != env.FEATURE_DISTRIBUTION[k]["observations"]]
        print(f"{len(fd) - len(bad)}/{len(fd)} products identical; differ: {bad}")
        return

    src = SRC.read_text()
    i, j = src.index(START), src.index(END)
    body = json.dumps(fd, indent=4).replace("true", "True").replace("false", "False").replace("null", "None")
    header = f"# Noise model rebuilt from {Path(a.noise).name} by tools/eval/build_planner_noise_env.py\n"
    Path(a.out).write_text(src[:i] + header + "FEATURE_DISTRIBUTION = " + body + src[j + 2:])
    print(f"[OK] {a.out}")


if __name__ == "__main__":
    main()

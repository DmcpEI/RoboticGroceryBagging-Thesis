#!/usr/bin/env python3
"""What our side feeds the planner env, and what the env expects of it.

The env is the collaborator's file and is dropped in unmodified when a new
version arrives. This checks the seam rather than the contents: every field
perception puts in the planner record must exist on the env's Item, and the
category strings our adapter emits must be the ones the env's cost tests
against. A silent mismatch here does not raise -- it just stops charging for
something -- so it needs a test rather than a run.

    python3 test_planner_env_contract.py
"""
import inspect
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from envs.binpacking_env_v2 import Item, CATEGORY_MAP, TEMPERATURE_MAP  # noqa: E402
from robot_pc_package.perceive_local import to_planner_format  # noqa: E402
from simulate_fixed_items_interface import RobotPlanner  # noqa: E402

fails = []


def check(ok, msg):
    print(("  PASS " if ok else "  FAIL ") + msg)
    if not ok:
        fails.append(msg)


def demo():
    # 1. every field perception exports is a constructor argument of Item
    sample = [{"name": "bleach bottle", "est_weight_g": 900, "est_volume_cc": 900,
               "crush_score": 5, "category": "Cleaning", "temperature": "Ambient",
               "spill_risk": True, "spill_vulnerable": False,
               "orientation_sensitive": None}]
    exported = set(to_planner_format(sample)["items_data"]["bleach bottle#1"])
    accepted = set(inspect.signature(Item.__init__).parameters) - {"self"}
    missing = exported - accepted
    check(not missing, "every exported field is an Item argument (extra: {})".format(
        sorted(missing) or "none"))

    # 2. the category strings the catalog uses are ones the env knows
    lookup = json.loads((HERE / "robot_pc_package/objects_lookup.json").read_text())
    cats = {v["category"] for k, v in lookup.items()
            if not k.startswith("_") and v.get("category")}
    temps = {v["temperature"] for k, v in lookup.items()
             if not k.startswith("_") and v.get("temperature")}
    check(cats <= set(CATEGORY_MAP), "catalog categories are known to the env ({})".format(
        sorted(cats - set(CATEGORY_MAP)) or "all known"))
    check(temps <= set(TEMPERATURE_MAP), "catalog temperatures are known to the env ({})".format(
        sorted(temps - set(TEMPERATURE_MAP)) or "all known"))

    # 3. the two products that can trigger the chemical spill cost still can
    chem = {k for k, v in lookup.items()
            if not k.startswith("_") and v.get("category") == "Cleaning"
            and v.get("spill_risk")}
    check(len(chem) >= 1, "at least one product is both Cleaning and spill_risk: {}".format(
        sorted(chem)))
    vuln = {k for k, v in lookup.items()
            if not k.startswith("_") and v.get("spill_vulnerable")}
    check(len(vuln) >= 1, "{} products are spill_vulnerable".format(len(vuln)))

    # The live export and the delivery bundle must agree on the planner
    # contract. They drifted once: the live path carried orientation_sensitive,
    # which nothing populates, so the planner read False on every item.
    live = (Path(__file__).parent / "robot_pc_package/perceive_local.py").read_text()
    live = live.split("PLANNER_FIELDS = (")[1].split(")")[0]
    check("orientation_sensitive" not in live,
          "the live export does not carry orientation_sensitive")
    for f in ("est_weight_g", "est_volume_cc", "crush_score", "category",
              "temperature", "spill_risk", "spill_vulnerable"):
        check(f in live, "the live export carries {}".format(f))

    # A pick the arm reports as lost must not be placed in a bag. The planner
    # holds its choice and applies it at the START of the next call, so there
    # has to be a way to take it back before then.
    pl = RobotPlanner(num_bags=2, num_items_to_pack=4, n_iter_per_timestep=8)
    cands = [dict(sample[0], name="item_{}".format(k)) for k in range(3)]
    pl.plan_action(cands)
    check(pl._pending is not None, "a choice is held, not placed immediately")
    check(pl.cancel_pending() is True, "a lost pick can be cancelled")
    check(pl._pending is None and pl.mcts is None,
          "the placement and the stale search tree are both dropped")
    check(pl.cancel_pending() is False, "cancelling twice is a no-op")
    pl.plan_action(cands)
    check(pl.t == 0, "and nothing was ever added to a bag")

    print("\n" + ("ALL CHECKS PASSED" if not fails else "{} FAILURE(S)".format(len(fails))))
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(demo())

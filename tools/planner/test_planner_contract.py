#!/usr/bin/env python3
"""Contract checks for the extended bagging solver.

Covers the three things the 2026-09-19 refactor could silently break: that the
solver reads the perception schema directly, that it still accepts the derived
enums the collaborator's solver and the audit code emit, and that the bag is
the size the collaborator's environment uses.

  .venv/bin/python3.12 tools/planner/test_planner_contract.py
"""
import sys; sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
# CP-SAT hangs under default multi-threading in this sandbox (CLAUDE.md, 2026-07-15)
from ortools.sat.python import cp_model as _cm
_orig = _cm.CpSolver
class _Solver(_orig):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.parameters.num_search_workers = 1
        self.parameters.max_time_in_seconds = 20
_cm.CpSolver = _Solver
from extended_packbot_cp import _is_chemical, _is_raw_meat, _needs_cold, solve_bagging_extended
# perception-native records
assert _is_chemical({"group":"cleaning"}) and not _is_chemical({"group":"hygiene"})
assert _is_raw_meat({"group":"seafood"}) and not _is_raw_meat({"group":"pantry"})
assert _needs_cold({"cold_chain":True,"group":"dairy"})
assert _needs_cold({"cold_chain":False,"group":"frozen"}), "frozen is non-ambient even if the flag is unset"
assert not _needs_cold({"cold_chain":False,"group":"pantry"})
# legacy records still work (collaborator's solver output, audit code)
assert _is_chemical({"category":"Cleaning"}) and not _is_chemical({"category":"Household"})
assert _is_raw_meat({"category":"Raw Meat"})
assert _needs_cold({"temperature":"Refrigerated"}) and not _needs_cold({"temperature":"Ambient"})
# a chemical and a food may not share a bag; two foods may
def item(**kw):
    d={"est_weight_g":100,"est_volume_cc":100,"crush_score":5,"group":"pantry",
       "cold_chain":False,"spill_risk":False,"spill_vulnerable":False}
    d.update(kw); return d
r=solve_bagging_extended({"a":item(group="cleaning"),"b":item()},["a","b"])
assert r["total_bags_used"]==2, r
r=solve_bagging_extended({"a":item(),"b":item()},["a","b"])
assert r["total_bags_used"]==1, r
# a chemical may share a bag with a non-food item (a sponge), in the solver and the audit
r=solve_bagging_extended({"a":item(group="cleaning",edible=False),"b":item(group="household",edible=False)},["a","b"])
assert r["total_bags_used"]==1, r
from planner_safety_audit import pair_violations_for_items
from soft_packbot_cp import _pair_violations_for_items as soft_pairs
chem={"category":"Cleaning","edible":False}; sponge={"category":"Household","edible":False}
food={"category":"Pantry","edible":True}
for f in (lambda a,b: pair_violations_for_items(a,b,ordered=False),
          lambda a,b: soft_pairs(a,b,ordered=False)):
    assert "chemical_with_food" in f(chem,food) and not f(chem,sponge) and not f(chem,chem)
# capacity now binds at 5000cc, not 20000
r=solve_bagging_extended({"a":item(est_volume_cc=3000),"b":item(est_volume_cc=3000)},["a","b"])
assert r["total_bags_used"]==2, ("6000cc must not fit one 5000cc bag", r)
print("planner selfcheck ok")

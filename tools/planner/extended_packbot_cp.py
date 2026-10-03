"""Extended CP-SAT packing solver.

Adds one new constraint on top of or_approach.solve_bagging():
  Constraint A — Spill isolation: items with spill_risk=True cannot share a bag
      with items that are spill_vulnerable=True (Bakery, Produce, Snacks).

Items dict must include all fields required by the baseline solver PLUS:
  spill_risk          (bool)  — spill risk if tipped or leaking
  spill_vulnerable    (bool)  — category in {Bakery, Produce, Snacks}

Use enable_spill to isolate the constraint.

REMOVED 2026-07-23: Constraint B (upright handling / orientation_sensitive).
Jacopo's actual planner never consumed it; it was a Diogo-side exploratory
extension that, across every audit run (static 49-scene, 21 removal
sequences, multiple threshold recalibrations), never fired on real data —
either 0/49 or 0/21 depending on the run, and one run found zero
orientation_sensitive=True examples anywhere in the catalog/GT. See CLAUDE.md
Settled Decision 14.
"""

import os

from ortools.sat.python import cp_model

# Reproduces the pre-2026-09-19 planner exactly: the 7000 g / 20000 cc bag from
# or_approach.py, and the derived `category`/`temperature` enums as the source
# of the safety predicates. Every planner figure published before that date was
# produced under this setting, so the thesis can show a before and after rather
# than silently restating numbers. Follows the VMT_CRUSH_V1 precedent.
PLANNER_V1 = os.environ.get("VMT_PLANNER_V1") == "1"

# CP-SAT deadlocks under its default multi-threading in some sandboxes, which is
# why run_end_to_end_cp.py monkeypatched the solver rather than fixing it here.
# The patch never reached the other callers, so run_soft_cp_sweep.py hung
# indefinitely on two scenes. Set it once, in the solver, for everyone.
CP_WORKERS = int(os.environ.get("VMT_CP_WORKERS", "1"))
CP_MAX_SECONDS = float(os.environ.get("VMT_CP_MAX_SECONDS", "60"))

# The solver needs three predicates over an item, and until 2026-09-19 it read
# them out of two derived enums -- `category` (12 values) and `temperature`
# (3 values) -- that between them carried exactly three bits. Both are now read
# from the perception schema directly, so the planner consumes the same
# attributes perception produces and there is no translation layer to keep in
# step. `category`/`temperature` are still accepted, because the collaborator's
# baseline solver emits them and the audit and reporting code reads them.

CHEMICAL_GROUPS = {"cleaning", "pharmacy", "batteries"}
RAW_MEAT_GROUPS = {"raw_meat", "seafood"}


def _group(p) -> str:
    return str(p.get("group", "") or "").lower().strip()


def _is_chemical(p) -> bool:
    if "group" in p and not PLANNER_V1:
        return _group(p) in CHEMICAL_GROUPS
    return "Cleaning" in str(p.get("category", ""))


def _is_raw_meat(p) -> bool:
    if "group" in p and not PLANNER_V1:
        return _group(p) in RAW_MEAT_GROUPS
    return "Raw Meat" in str(p.get("category", ""))


def _is_food(p) -> bool:
    """`edible`, read from the item or its `_source`; unknown counts as food."""
    v = p.get("edible")
    if v is None:
        v = (p.get("_source") or {}).get("edible")
    return True if v is None else bool(v)


def _needs_cold(p) -> bool:
    """True when the item may not share a bag with ambient goods.

    The constraint only ever distinguished ambient from non-ambient, so the
    three-valued `temperature` enum was carrying one bit. `cold_chain` is that
    bit, and it is a perception field rather than a derived one.
    """
    if "cold_chain" in p and not PLANNER_V1:
        return bool(p["cold_chain"]) or _group(p) == "frozen"
    return "Ambient" not in str(p.get("temperature", "Ambient"))


def solve_bagging_extended(
    items_data: dict,
    arrival_order: list,
    enable_spill: bool = True,
    crush_threshold_heavy: int = 2,
    max_time_seconds: float | None = None,
) -> dict:
    """Extended bagging solver — superset of or_approach.solve_bagging().

    crush_threshold_heavy default (2) matches the original general-supermarket
    dataset's crush_score scale. The robot-lab planner catalog uses a
    different scale ({3,5,6,7,9}, never <=2) -- callers on that dataset must
    pass a calibrated value (see tools/eval/run_extended_cp_robot_lab.py) or
    Constraint B / the baseline crush constraint can never fire.
    """

    # 2026-09-19: matched to the collaborator's bagging environment
    # (envs/binpacking_env_v{2,3}.py, Bag.MAX_WEIGHT / Bag.MAX_VOLUME), which is
    # what the risk-aware planner this work is compared against actually packs
    # into. The previous 7000 g / 20000 cc came from or_approach.py, a different
    # artefact of the same project, and a bag four times larger by volume makes
    # any bag-count comparison between the two solvers meaningless.
    MAX_BAG_WEIGHT_G = 7000 if PLANNER_V1 else 5000
    MAX_BAG_VOLUME_CC = 20000 if PLANNER_V1 else 5000
    CRUSH_THRESHOLD_FRAGILE = 8
    CRUSH_THRESHOLD_HEAVY = crush_threshold_heavy

    model = cp_model.CpModel()

    item_list = list(items_data.keys())
    num_items = len(item_list)
    num_bags = num_items

    item_to_idx = {name: i for i, name in enumerate(item_list)}

    all_items = range(num_items)
    all_bags = range(num_bags)

    x = {}
    for i in all_items:
        for j in all_bags:
            x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

    y = [model.NewBoolVar(f'y_{j}') for j in all_bags]

    # Constraint 1: each item in exactly one bag
    for i in all_items:
        model.AddExactlyOne(x[i, j] for j in all_bags)

    # Constraint 2: bag usage link
    for i in all_items:
        for j in all_bags:
            model.Add(x[i, j] <= y[j])

    # Constraint 3: capacity
    for j in all_bags:
        bag_weight = sum(items_data[item_list[i]]['est_weight_g'] * x[i, j] for i in all_items)
        model.Add(bag_weight <= MAX_BAG_WEIGHT_G * y[j])
        bag_volume = sum(items_data[item_list[i]]['est_volume_cc'] * x[i, j] for i in all_items)
        model.Add(bag_volume <= MAX_BAG_VOLUME_CC * y[j])

    # Constraint 4: baseline separation (cleaning / raw meat / temperature)
    for j in all_bags:
        for i1 in all_items:
            for i2 in range(i1 + 1, num_items):
                p1 = items_data[item_list[i1]]
                p2 = items_data[item_list[i2]]

                if PLANNER_V1:
                    is_chemical_food_mix = _is_chemical(p1) != _is_chemical(p2)
                else:  # a chemical may not share a bag with food
                    is_chemical_food_mix = ((_is_chemical(p1) and _is_food(p2))
                                            or (_is_chemical(p2) and _is_food(p1)))
                is_raw_meat_mix = _is_raw_meat(p1) != _is_raw_meat(p2)
                is_temp_mix = _needs_cold(p1) != _needs_cold(p2)

                if is_chemical_food_mix or is_raw_meat_mix or is_temp_mix:
                    model.AddBoolOr([x[i1, j].Not(), x[i2, j].Not()])

    # Constraint 5: crushability (baseline)
    for bag_idx in all_bags:
        for i in range(len(arrival_order)):
            for k in range(i + 1, len(arrival_order)):
                bottom_name = arrival_order[i]
                top_name = arrival_order[k]
                bp = items_data[bottom_name]
                tp = items_data[top_name]

                if (bp['crush_score'] >= CRUSH_THRESHOLD_FRAGILE and
                        tp['crush_score'] <= CRUSH_THRESHOLD_HEAVY):
                    bi = item_to_idx[bottom_name]
                    ti = item_to_idx[top_name]
                    model.AddBoolOr([x[bi, bag_idx].Not(), x[ti, bag_idx].Not()])

    # Constraint A: spill isolation
    if enable_spill:
        for j in all_bags:
            for i1 in all_items:
                for i2 in range(i1 + 1, num_items):
                    p1 = items_data[item_list[i1]]
                    p2 = items_data[item_list[i2]]

                    spill_pair = (
                        (p1.get('spill_risk', False) and p2.get('spill_vulnerable', False)) or
                        (p2.get('spill_risk', False) and p1.get('spill_vulnerable', False))
                    )
                    if spill_pair:
                        model.AddBoolOr([x[i1, j].Not(), x[i2, j].Not()])

    model.Minimize(sum(y))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = CP_WORKERS
    solver.parameters.max_time_in_seconds = (
        float(max_time_seconds) if max_time_seconds else CP_MAX_SECONDS)
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution = []
        bag_count = 0
        for j in all_bags:
            if solver.Value(y[j]) == 1:
                bag_count += 1
                bag_items = []
                total_weight = 0
                total_volume = 0
                for i in all_items:
                    if solver.Value(x[i, j]) == 1:
                        item_name = item_list[i]
                        bag_items.append(item_name)
                        total_weight += items_data[item_name]['est_weight_g']
                        total_volume += items_data[item_name]['est_volume_cc']

                bag_items.sort(key=lambda name: arrival_order.index(name))

                first_props = items_data[bag_items[0]]
                bag_type = f"{first_props.get('temperature', 'Ambient')}/{first_props.get('category', 'Pantry')}"
                if "Cleaning" in bag_type:
                    bag_type = "Chemicals"
                if "Raw Meat" in bag_type:
                    bag_type = "Raw Meat"

                solution.append({
                    "bag_number": bag_count,
                    "bag_type": bag_type,
                    "items": bag_items,
                    "total_weight_g": total_weight,
                    "total_volume_cc": total_volume,
                    "weight_fullness_%": round((total_weight / MAX_BAG_WEIGHT_G) * 100),
                    "volume_fullness_%": round((total_volume / MAX_BAG_VOLUME_CC) * 100),
                })

        return {
            "status": solver.StatusName(status),
            "total_bags_used": bag_count,
            "solution": solution,
        }
    else:
        return {"status": solver.StatusName(status), "solution": "No solution found."}

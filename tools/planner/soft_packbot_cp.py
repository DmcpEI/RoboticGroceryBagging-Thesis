"""Soft-constraint CP-SAT packing solver.

This solver is an experimental analysis companion to the hard PACKBOT CP-SAT
planner. Assignment and capacity constraints remain hard, while safety rules are
converted into weighted violation variables in the objective.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from ortools.sat.python import cp_model

_CP_WORKERS = int(__import__('os').environ.get('VMT_CP_WORKERS', '1'))


# The bag of the collaborator's bagging environment, as in extended_packbot_cp.
MAX_BAG_WEIGHT_G = 5000
MAX_BAG_VOLUME_CC = 5000
CRUSH_THRESHOLD_FRAGILE = 8
CRUSH_THRESHOLD_HEAVY = 2

NON_AMBIENT_TEMPS = {"Frozen", "Refrigerated"}

SAFETY_FAMILIES = (
    "raw_meat_with_non_raw",
    "chemical_with_food",
    "ambient_with_nonambient",
    "crush",
    "spill_risk_with_vulnerable",
    "orientation_with_heavy",
)


def _contains(value: Any, token: str) -> bool:
    return token in str(value or "")


def _is_raw_meat(props: Mapping[str, Any]) -> bool:
    return _contains(props.get("category"), "Raw Meat")


def _is_cleaning(props: Mapping[str, Any]) -> bool:
    return _contains(props.get("category"), "Cleaning")


def _is_food(props: Mapping[str, Any]) -> bool:
    """`edible`, read from the item or its `_source`; an item of unknown
    edibility counts as food, so a chemical is never bagged with it."""
    v = props.get("edible")
    if v is None:
        v = (props.get("_source") or {}).get("edible")
    return True if v is None else bool(v)


def _is_ambient(props: Mapping[str, Any]) -> bool:
    return _contains(props.get("temperature"), "Ambient")


def _is_nonambient(props: Mapping[str, Any]) -> bool:
    return not _is_ambient(props)


def _is_heavy(props: Mapping[str, Any]) -> bool:
    return int(props.get("crush_score", 5)) <= CRUSH_THRESHOLD_HEAVY


def _is_fragile(props: Mapping[str, Any]) -> bool:
    return int(props.get("crush_score", 5)) >= CRUSH_THRESHOLD_FRAGILE


def _upright_required(props: Mapping[str, Any]) -> bool:
    return bool(props.get("upright_required", props.get("orientation_sensitive", False)))


def _family_penalty(
    family: str,
    safety_penalty: int,
    penalty_weights: Mapping[str, int] | None,
) -> int:
    if penalty_weights and family in penalty_weights:
        return int(penalty_weights[family])
    return int(safety_penalty)


def _families_enabled(enabled_families: Iterable[str] | None) -> set[str]:
    if enabled_families is None:
        return set(SAFETY_FAMILIES)
    unknown = set(enabled_families) - set(SAFETY_FAMILIES)
    if unknown:
        raise ValueError(f"unknown safety families: {sorted(unknown)}")
    return set(enabled_families)


def _nonnegative_int_field(item_id: str, props: Mapping[str, Any], field: str) -> int:
    try:
        value = int(props[field])
    except KeyError as exc:
        raise ValueError(f"item {item_id!r} is missing required planner field {field!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"item {item_id!r} has non-integer planner field {field!r}: {props.get(field)!r}"
        ) from exc
    if value < 0:
        raise ValueError(f"item {item_id!r} has negative planner field {field!r}: {value}")
    return value


def _pair_violations_for_items(
    bottom_props: Mapping[str, Any],
    top_props: Mapping[str, Any],
    *,
    ordered: bool,
) -> List[str]:
    """Return safety families violated if the two items share a bag.

    `ordered=True` means bottom_props arrives before top_props, so arrival-order
    constraints such as crush and orientation are meaningful.
    """
    families: List[str] = []

    is_raw_meat_mix = (
        (_is_raw_meat(bottom_props) and not _is_raw_meat(top_props))
        or (_is_raw_meat(top_props) and not _is_raw_meat(bottom_props))
    )
    if is_raw_meat_mix:
        families.append("raw_meat_with_non_raw")

    # A chemical may not share a bag with food (`edible`).
    is_chemical_food_mix = (
        (_is_cleaning(bottom_props) and _is_food(top_props))
        or (_is_cleaning(top_props) and _is_food(bottom_props))
    )
    if is_chemical_food_mix:
        families.append("chemical_with_food")

    is_temp_mix = (
        (_is_ambient(bottom_props) and _is_nonambient(top_props))
        or (_is_ambient(top_props) and _is_nonambient(bottom_props))
    )
    if is_temp_mix:
        families.append("ambient_with_nonambient")

    spill_pair = (
        (bool(bottom_props.get("spill_risk", False)) and bool(top_props.get("spill_vulnerable", False)))
        or (bool(top_props.get("spill_risk", False)) and bool(bottom_props.get("spill_vulnerable", False)))
    )
    if spill_pair:
        families.append("spill_risk_with_vulnerable")

    if ordered:
        if _is_fragile(bottom_props) and _is_heavy(top_props):
            families.append("crush")
        if _upright_required(bottom_props) and _is_heavy(top_props):
            families.append("orientation_with_heavy")

    return families


def bag_violation_flags(
    bag_item_dicts: Sequence[Mapping[str, Any]],
    bag_item_ids: Sequence[str] | None = None,
    arrival_order: Sequence[str] | None = None,
) -> Dict[str, bool]:
    categories = [str(item.get("category", "")) for item in bag_item_dicts]
    temps = [str(item.get("temperature", "")) for item in bag_item_dicts]

    has_raw_meat = any(cat == "Raw Meat" for cat in categories)
    has_non_raw = any(cat != "Raw Meat" for cat in categories)
    has_cleaning = any(cat == "Cleaning" for cat in categories)
    has_food = any(_is_food(item) for item in bag_item_dicts)
    has_frozen = any(temp == "Frozen" for temp in temps)
    has_ambient = any(temp == "Ambient" for temp in temps)
    has_nonambient = any(temp in NON_AMBIENT_TEMPS for temp in temps)
    has_spill_risk = any(bool(item.get("spill_risk", False)) for item in bag_item_dicts)
    has_spill_vulnerable = any(bool(item.get("spill_vulnerable", False)) for item in bag_item_dicts)

    has_crush = False
    has_orientation_with_heavy = False
    if bag_item_ids is not None and arrival_order is not None:
        by_id = dict(zip(bag_item_ids, bag_item_dicts))
        positions = {item_id: idx for idx, item_id in enumerate(arrival_order)}
        ordered_ids = sorted(bag_item_ids, key=lambda item_id: positions.get(item_id, 10**9))
        for pos, bottom_id in enumerate(ordered_ids):
            for top_id in ordered_ids[pos + 1 :]:
                bottom = by_id[bottom_id]
                top = by_id[top_id]
                if _is_fragile(bottom) and _is_heavy(top):
                    has_crush = True
                if _upright_required(bottom) and _is_heavy(top):
                    has_orientation_with_heavy = True
    else:
        has_crush = any(_is_fragile(item) for item in bag_item_dicts) and any(
            _is_heavy(item) for item in bag_item_dicts
        )
        has_orientation_with_heavy = any(
            _upright_required(item) for item in bag_item_dicts
        ) and any(_is_heavy(item) for item in bag_item_dicts)

    return {
        "raw_meat_with_non_raw": bool(has_raw_meat and has_non_raw),
        "chemical_with_food": bool(has_cleaning and has_food),
        "frozen_with_ambient": bool(has_frozen and has_ambient),
        "ambient_with_nonambient": bool(has_ambient and has_nonambient),
        "crush": has_crush,
        "spill_risk_with_vulnerable": bool(has_spill_risk and has_spill_vulnerable),
        "orientation_with_heavy": has_orientation_with_heavy,
    }


def _empty_violation_totals() -> Dict[str, int]:
    return {
        "raw_meat_with_non_raw": 0,
        "chemical_with_food": 0,
        "frozen_with_ambient": 0,
        "ambient_with_nonambient": 0,
        "crush": 0,
        "spill_risk_with_vulnerable": 0,
        "orientation_with_heavy": 0,
    }


def _solution_records(
    *,
    solver: cp_model.CpSolver,
    x: Mapping[Tuple[int, int], cp_model.IntVar],
    y: Sequence[cp_model.IntVar],
    items_data: Mapping[str, Mapping[str, Any]],
    item_list: Sequence[str],
    arrival_order: Sequence[str],
    all_items: range,
    all_bags: range,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    solution = []
    bag_level_totals = _empty_violation_totals()
    bag_count = 0

    for j in all_bags:
        if solver.Value(y[j]) != 1:
            continue
        bag_count += 1
        bag_item_ids: List[str] = []
        total_weight = 0
        total_volume = 0
        for i in all_items:
            if solver.Value(x[i, j]) == 1:
                item_id = item_list[i]
                bag_item_ids.append(item_id)
                total_weight += int(items_data[item_id]["est_weight_g"])
                total_volume += int(items_data[item_id]["est_volume_cc"])

        bag_item_ids.sort(key=lambda item_id: arrival_order.index(item_id))
        bag_items = [items_data[item_id] for item_id in bag_item_ids]
        flags = bag_violation_flags(bag_items, bag_item_ids, arrival_order)
        for key, value in flags.items():
            bag_level_totals[key] += int(value)

        first_props = items_data[bag_item_ids[0]]
        bag_type = f"{first_props['temperature']}/{first_props['category']}"
        if "Cleaning" in bag_type:
            bag_type = "Chemicals"
        if "Raw Meat" in bag_type:
            bag_type = "Raw Meat"

        solution.append(
            {
                "bag_number": bag_count,
                "bag_type": bag_type,
                "items": bag_item_ids,
                "item_display_names": [
                    str(items_data[item_id].get("display_name", item_id)) for item_id in bag_item_ids
                ],
                "item_categories": [
                    str(items_data[item_id].get("category", "")) for item_id in bag_item_ids
                ],
                "item_temperatures": [
                    str(items_data[item_id].get("temperature", "")) for item_id in bag_item_ids
                ],
                "violation_flags": flags,
                "total_weight_g": total_weight,
                "total_volume_cc": total_volume,
                "weight_fullness_%": round((total_weight / MAX_BAG_WEIGHT_G) * 100),
                "volume_fullness_%": round((total_volume / MAX_BAG_VOLUME_CC) * 100),
            }
        )

    return solution, bag_level_totals


def solve_bagging_soft(
    items_data: Dict[str, Dict[str, Any]],
    arrival_order: List[str],
    *,
    safety_penalty: int = 1000,
    bag_cost: int = 1000,
    penalty_weights: Mapping[str, int] | None = None,
    enabled_families: Iterable[str] | None = None,
    max_time_seconds: float | None = None,
) -> Dict[str, Any]:
    """Solve bagging with soft safety constraints.

    Args:
        items_data: PACKBOT planner input dict. The extended fields
            `spill_risk`, `spill_vulnerable`, and optional `upright_required`
            are supported. Legacy `orientation_sensitive` remains accepted.
        arrival_order: item ids in bottom-to-top order if packed together.
        safety_penalty: default integer penalty for each pairwise safety
            violation.
        bag_cost: integer objective cost for each used bag.
        penalty_weights: optional per-family penalty overrides.
        enabled_families: optional subset of safety families to soften/penalize.
        max_time_seconds: optional CP-SAT time limit.
    """
    enabled = _families_enabled(enabled_families)

    items_data = {item_id: dict(props) for item_id, props in items_data.items()}
    for item_id, props in items_data.items():
        props["est_weight_g"] = _nonnegative_int_field(item_id, props, "est_weight_g")
        props["est_volume_cc"] = _nonnegative_int_field(item_id, props, "est_volume_cc")

    model = cp_model.CpModel()
    item_list = list(items_data.keys())
    missing_order = [item_id for item_id in item_list if item_id not in arrival_order]
    if missing_order:
        arrival_order = list(arrival_order) + missing_order
    num_items = len(item_list)
    num_bags = num_items
    item_to_idx = {name: i for i, name in enumerate(item_list)}

    all_items = range(num_items)
    all_bags = range(num_bags)

    x: Dict[Tuple[int, int], cp_model.IntVar] = {}
    for i in all_items:
        for j in all_bags:
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")
    y = [model.NewBoolVar(f"y_{j}") for j in all_bags]

    for i in all_items:
        model.AddExactlyOne(x[i, j] for j in all_bags)
        for j in all_bags:
            model.Add(x[i, j] <= y[j])

    for j in all_bags:
        bag_weight = sum(int(items_data[item_list[i]]["est_weight_g"]) * x[i, j] for i in all_items)
        model.Add(bag_weight <= MAX_BAG_WEIGHT_G * y[j])
        bag_volume = sum(int(items_data[item_list[i]]["est_volume_cc"]) * x[i, j] for i in all_items)
        model.Add(bag_volume <= MAX_BAG_VOLUME_CC * y[j])

    violation_vars: Dict[str, List[cp_model.IntVar]] = defaultdict(list)
    objective_terms = [int(bag_cost) * y[j] for j in all_bags]

    def add_pair_violation(i1: int, i2: int, bag_idx: int, family: str) -> None:
        if family not in enabled:
            return
        var = model.NewBoolVar(f"v_{family}_{i1}_{i2}_{bag_idx}")
        model.Add(var >= x[i1, bag_idx] + x[i2, bag_idx] - 1)
        model.Add(var <= x[i1, bag_idx])
        model.Add(var <= x[i2, bag_idx])
        violation_vars[family].append(var)
        penalty = _family_penalty(family, safety_penalty, penalty_weights)
        objective_terms.append(int(penalty) * var)

    # Unordered pair constraints: separation, temperature, spill.
    for j in all_bags:
        for i1 in all_items:
            for i2 in range(i1 + 1, num_items):
                p1 = items_data[item_list[i1]]
                p2 = items_data[item_list[i2]]
                for family in _pair_violations_for_items(p1, p2, ordered=False):
                    add_pair_violation(i1, i2, j, family)

    # Ordered pair constraints: crush and orientation depend on arrival order.
    for bag_idx in all_bags:
        for pos, bottom_name in enumerate(arrival_order):
            if bottom_name not in item_to_idx:
                continue
            for top_name in arrival_order[pos + 1 :]:
                if top_name not in item_to_idx:
                    continue
                bottom_idx = item_to_idx[bottom_name]
                top_idx = item_to_idx[top_name]
                bottom_props = items_data[bottom_name]
                top_props = items_data[top_name]
                for family in _pair_violations_for_items(bottom_props, top_props, ordered=True):
                    if family in {"crush", "orientation_with_heavy"}:
                        add_pair_violation(bottom_idx, top_idx, bag_idx, family)

    model.Minimize(sum(objective_terms))

    solver = cp_model.CpSolver()
    # Same deadlock as the hard solver: CP-SAT's default multi-threading hangs
    # in some sandboxes. VMT_CP_WORKERS overrides it.
    solver.parameters.num_search_workers = _CP_WORKERS
    if max_time_seconds is not None:
        solver.parameters.max_time_in_seconds = float(max_time_seconds)
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {
            "status": solver.StatusName(status),
            "solution": "No solution found.",
            "total_bags_used": None,
            "objective_value": None,
            "pairwise_violation_totals": {family: 0 for family in SAFETY_FAMILIES},
            "bag_violation_totals": _empty_violation_totals(),
        }

    solution, bag_level_totals = _solution_records(
        solver=solver,
        x=x,
        y=y,
        items_data=items_data,
        item_list=item_list,
        arrival_order=arrival_order,
        all_items=all_items,
        all_bags=all_bags,
    )
    pairwise_totals = {
        family: sum(int(solver.Value(var)) for var in violation_vars.get(family, []))
        for family in SAFETY_FAMILIES
    }
    total_pairwise = sum(pairwise_totals.values())

    return {
        "status": solver.StatusName(status),
        "total_bags_used": len(solution),
        "objective_value": solver.ObjectiveValue(),
        "best_objective_bound": solver.BestObjectiveBound(),
        "bag_cost": int(bag_cost),
        "safety_penalty": int(safety_penalty),
        "penalty_weights": dict(penalty_weights or {}),
        "enabled_families": sorted(enabled),
        "pairwise_violation_totals": pairwise_totals,
        "total_pairwise_violations": total_pairwise,
        "bag_violation_totals": bag_level_totals,
        "total_bag_level_violations": sum(bag_level_totals.values()),
        "solution": solution,
    }

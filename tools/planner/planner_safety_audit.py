"""Planner safety predicates shared by CP-SAT and LLM baseline audits.

The orientation family was removed on 2026-08-31. Settled Decision 14 deleted
`upright_required`/`orientation_sensitive` from the schema, the catalogs and the
solver, so nothing has set it True since; the audit went on reporting a
violation family that could no longer fire, and every run that quoted it quoted
a structural zero. tools/planner/soft_packbot_cp.py keeps its own copy for the
old general-supermarket dataset, which was explicitly out of scope for
Decisions 13 and 14 and whose ground truth still carries the field.

`_is_fragile` and `_is_heavy` read crush_score, not the `fragile` flag: the
names are historical. crush_score is load tolerance (Decision 15), so "fragile"
here means "little may be stacked on it".
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence


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


def family_penalty(
    family: str,
    safety_penalty: int,
    penalty_weights: Mapping[str, int] | None,
) -> int:
    if penalty_weights and family in penalty_weights:
        return int(penalty_weights[family])
    return int(safety_penalty)


def families_enabled(enabled_families: Iterable[str] | None) -> set[str]:
    if enabled_families is None:
        return set(SAFETY_FAMILIES)
    unknown = set(enabled_families) - set(SAFETY_FAMILIES)
    if unknown:
        raise ValueError(f"unknown safety families: {sorted(unknown)}")
    return set(enabled_families)


def pair_violations_for_items(
    bottom_props: Mapping[str, Any],
    top_props: Mapping[str, Any],
    *,
    ordered: bool,
) -> List[str]:
    """Return safety families violated if the two items share a bag.

    `ordered=True` means bottom_props arrives before top_props, so the
    arrival-order constraint (crush) is meaningful.
    """
    families: List[str] = []

    is_raw_meat_mix = (
        (_is_raw_meat(bottom_props) and not _is_raw_meat(top_props))
        or (_is_raw_meat(top_props) and not _is_raw_meat(bottom_props))
    )
    if is_raw_meat_mix:
        families.append("raw_meat_with_non_raw")

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
    else:
        has_crush = any(_is_fragile(item) for item in bag_item_dicts) and any(
            _is_heavy(item) for item in bag_item_dicts
        )

    return {
        "raw_meat_with_non_raw": bool(has_raw_meat and has_non_raw),
        "chemical_with_food": bool(has_cleaning and has_food),
        "frozen_with_ambient": bool(has_frozen and has_ambient),
        "ambient_with_nonambient": bool(has_ambient and has_nonambient),
        "crush": has_crush,
        "spill_risk_with_vulnerable": bool(has_spill_risk and has_spill_vulnerable),
    }


def empty_violation_totals() -> Dict[str, int]:
    return {
        "raw_meat_with_non_raw": 0,
        "chemical_with_food": 0,
        "frozen_with_ambient": 0,
        "ambient_with_nonambient": 0,
        "crush": 0,
        "spill_risk_with_vulnerable": 0,
    }

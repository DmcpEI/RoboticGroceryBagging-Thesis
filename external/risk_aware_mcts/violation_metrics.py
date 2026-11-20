"""
Planner-agnostic bin-packing safety-violation metric, shared by
simulate_mcts_multiple_betas.py, simulate_rule_based_planner.py, and
simulate_gemini_planner.py.

Extracted from simulate_benchmark_scenes.py's count_violations/VIOLATION_TYPES/
_scenes_affected_stats (left in place there, unmodified, to avoid touching
that script's existing behavior/output schema).

count_violations computes a static STRUCTURAL risk audit of the final TRUE
bag contents after an episode ends -- NOT derived from env.step()'s cost,
NOT tied to which planner produced the placements, and NOT the same as
actual crush/spill outcomes (item.is_crushed/item.is_spilled), which the
env's cost mechanics already charge for separately.

Because it only reads bag.items (placement order) and each item's
crush_score/spill_risk/spill_vulnerable, count_violations works unmodified on
bags that never went through simulated physics at all -- e.g. the real-robot
planner interfaces (gemini_planner_interface.py, attribute_aware_gemini_
planner_interface.py, simulate_fixed_items_interface.py's RobotPlanner),
whose Bag.place_observed_item accumulation never rolls dice. format_bag_
allocation/bag_allocation_to_dict/save_session_log below are for exactly that
use: printing/saving a live real-robot session's current bag allocation.
"""

from __future__ import annotations

import json
import time

FRAGILE_THRESHOLD = 8  # crush_score >= this -> fragile; matches the env's own crush-risk trigger

VIOLATION_TYPES = ("fragile_weight_on_top", "spill_risk_weight_on_top", "spill_vulnerable_with_spill_risk")


def count_violations(bags, phantom_name=None) -> dict:
    """bags: iterable of Bag objects (the final TRUE bags after an episode).

    phantom_name: optional item name to treat as "not really there" (used by
    simulate_benchmark_scenes.py for hallucinated-detection placeholders --
    irrelevant, and a no-op, for callers that never construct such items)."""
    fragile_weight_on_top = 0
    spill_risk_weight_on_top = 0
    spill_vulnerable_with_spill_risk = 0
    for bag in bags:
        items = bag.items  # placement order, bottom to top
        for i, item in enumerate(items):
            if item.name == phantom_name:
                continue
            later_real = any(it.name != phantom_name for it in items[i + 1:])
            if item.crush_score >= FRAGILE_THRESHOLD and later_real:
                fragile_weight_on_top += 1
            if item.spill_risk and later_real:
                spill_risk_weight_on_top += 1
        real_items = [it for it in items if it.name != phantom_name]
        if any(it.spill_risk for it in real_items):
            spill_vulnerable_with_spill_risk += sum(1 for it in real_items if it.spill_vulnerable)
    return {
        "fragile_weight_on_top": fragile_weight_on_top,
        "spill_risk_weight_on_top": spill_risk_weight_on_top,
        "spill_vulnerable_with_spill_risk": spill_vulnerable_with_spill_risk,
    }


def violations_affected_stats(violations_list: list) -> dict:
    """Per-type and combined count of episodes with >=1 violation of that
    kind, out of len(violations_list) episodes -- "combined"/"any" means an
    episode with >=1 violation of ANY of the three types, not summed across
    types (an episode with 3 fragile violations and 0 of anything else still
    counts once toward "any")."""
    n = len(violations_list)
    by_type = {k: sum(1 for v in violations_list if v[k] > 0) for k in VIOLATION_TYPES}
    affected_any = sum(1 for v in violations_list if any(v[k] > 0 for k in VIOLATION_TYPES))
    return {
        "n_episodes": n,
        "episodes_affected_any": affected_any,
        "fraction_episodes_affected_any": (affected_any / n) if n else 0.0,
        "episodes_affected_by_type": by_type,
    }


def _item_flags(item) -> list:
    flags = []
    if item.crush_score >= FRAGILE_THRESHOLD:
        flags.append("fragile")
    if item.spill_risk:
        flags.append("spill_risk")
    if item.spill_vulnerable:
        flags.append("vulnerable")
    return flags


def num_bags_used(bags) -> int:
    return sum(1 for bag in bags if bag.total_items > 0)


def format_bag_allocation(bags) -> str:
    """Human-readable bag allocation: one block per non-empty bag, one line per
    item with its fragile/spill_risk/vulnerable flags. Mirrors
    simulate_gemini_planner.py's print_allocation(), but for bags that never
    went through simulated physics (e.g. the real-robot interfaces'
    Bag.place_observed_item accumulation, which never rolls dice) -- so
    instead of CRUSHED/SPILLED outcome tags (always False there), this shows
    each item's accrued risk_level/spill_risk_level when nonzero, which is the
    quantity that's actually available in that setting."""
    lines = []
    any_used = False
    for i, bag in enumerate(bags):
        if bag.total_items == 0:
            continue
        any_used = True
        lines.append(f"  Bag {i}  [vol={bag.volume_used}/{bag.MAX_VOLUME}  "
                      f"wt={bag.weight_used}/{bag.MAX_WEIGHT}  items={bag.total_items}]")
        for item in bag.items:
            flag_str = f"  [{', '.join(_item_flags(item))}]" if _item_flags(item) else ""
            risk_bits = []
            if getattr(item, "risk_level", 0):
                risk_bits.append(f"crush_risk={item.risk_level:.2f}")
            if getattr(item, "spill_risk_level", 0):
                risk_bits.append(f"spill_risk={item.spill_risk_level:.2f}")
            risk_str = f"  ({', '.join(risk_bits)})" if risk_bits else ""
            lines.append(f"    - {item.name:<30} (id={item.id:2d}){flag_str}{risk_str}")
    if not any_used:
        lines.append("  (all bags empty)")
    return "\n".join(lines)


def bag_allocation_to_dict(bags) -> list:
    """JSON-serializable form of the same data format_bag_allocation() prints:
    a list of bags, each a list of item dicts in placement order (bottom to
    top)."""
    out = []
    for bag in bags:
        out.append({
            "volume_used": bag.volume_used, "weight_used": bag.weight_used,
            "items": [
                {
                    "name": item.name, "weight": item.weight, "volume": item.volume,
                    "flags": _item_flags(item),
                    "risk_level": item.risk_level,
                    "spill_risk_level": item.spill_risk_level,
                }
                for item in bag.items
            ],
        })
    return out


def save_session_log(path, bags, extra=None) -> dict:
    """Overwrite `path` with the CURRENT final bag allocation + violation
    counts + bags-used count. Meant to be called every loop iteration of a
    live real-robot session (not just once at the end), so an interrupted run
    never loses its most recent state -- same resilience philosophy as
    simulate_gemini_planner.py's `_save()`, just triggered per-placement
    instead of per-episode. Returns the dict that was written, for callers
    that also want to print it."""
    data = {
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bags": bag_allocation_to_dict(bags),
        "num_bags_used": num_bags_used(bags),
        "violations": count_violations(bags),
    }
    if extra:
        data.update(extra)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data

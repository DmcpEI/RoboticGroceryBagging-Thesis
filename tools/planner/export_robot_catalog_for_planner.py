#!/usr/bin/env python3
"""Export the robot-lab item catalog as planner-facing PACKBOT input.

The catalog is a perception/GT-style list and may contain quantity > 1.
Jacopo's planner receives individual item instances in:

  {
    "items_data": {"item_000": {...planner fields...}},
    "arrival_order": ["item_000", ...]
  }

This exporter expands quantities and writes only planner-facing item fields by
default. It deliberately removes perception bookkeeping fields such as
canonical_name and free_name from the planner payload.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from tools.common import norm_qty
from tools.planner.perception_to_packbot_adapter import _build_extended_packbot_item


PLANNER_FIELDS = [
    "display_name",
    "est_weight_g",
    "weight_source",
    "est_volume_cc",
    "volume_source",
    "crush_score",
    "category",
    "temperature",
    "spill_risk",
    "spill_vulnerable",
    "edible",
]


def _new_item_id(counter: int) -> str:
    return f"item_{counter:03d}"


def _trace_name(item: Dict[str, Any]) -> str:
    canonical = str(item.get("canonical_name", "") or "").strip()
    free = str(item.get("free_name", "") or "").strip()
    legacy = str(item.get("name", "") or "").strip()
    if canonical and canonical != "unknown_item":
        return canonical
    return free or legacy or "unknown item"


def export_catalog(
    catalog: Dict[str, Any],
    *,
    include_source: bool,
    include_upright_required: bool,
) -> Dict[str, Any]:
    items_data: Dict[str, Dict[str, Any]] = {}
    item_id_map: Dict[str, Dict[str, Any]] = {}
    arrival_order: list[str] = []
    counter = 0

    for entry_index, item in enumerate(catalog.get("items", [])):
        if not isinstance(item, dict):
            continue
        quantity = norm_qty(item.get("quantity", 1))
        for occurrence_index in range(quantity):
            item_id = _new_item_id(counter)
            counter += 1
            planner_item = _build_extended_packbot_item(item)
            if not include_source:
                planner_item.pop("_source", None)
            planner_item.pop("orientation_sensitive", None)
            if not include_upright_required:
                planner_item.pop("upright_required", None)
            items_data[item_id] = planner_item
            item_id_map[item_id] = {
                "catalog_entry_index": entry_index,
                "occurrence_index": occurrence_index,
                "source_name": _trace_name(item),
            }
            arrival_order.append(item_id)

    return {
        "_meta": {
            "source": "robot_lab_item_catalog",
            "source_catalog": catalog.get("_meta", {}),
            "planner_contract": "PACKBOT extended CP input",
            "note": (
                "Catalog quantities are expanded into individual item IDs. "
                "arrival_order follows catalog order and is not a real robot packing sequence."
            ),
            "planner_item_fields": PLANNER_FIELDS + (["upright_required"] if include_upright_required else []),
            "item_instances": len(items_data),
            "source_entries": len(catalog.get("items", [])),
            "include_source": include_source,
            "include_upright_required": include_upright_required,
        },
        "items_data": items_data,
        "arrival_order": arrival_order,
        "item_id_map": item_id_map,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog-json",
        type=Path,
        default=Path("data/robot_lab/robot_item_catalog_perception.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("data/robot_lab/robot_item_catalog_planner.json"),
    )
    parser.add_argument(
        "--include-source",
        action="store_true",
        help="Keep adapter _source fields for debugging. Off by default for Jacopo-facing payloads.",
    )
    parser.add_argument(
        "--include-upright-required",
        action="store_true",
        help="Include optional upright_required field. Off by default for the compact v1 handoff.",
    )
    args = parser.parse_args()

    catalog = json.loads(args.catalog_json.read_text(encoding="utf-8"))
    out = export_catalog(
        catalog,
        include_source=bool(args.include_source),
        include_upright_required=bool(args.include_upright_required),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "item_instances": out["_meta"]["item_instances"],
                "source_entries": out["_meta"]["source_entries"],
                "planner_item_fields": out["_meta"]["planner_item_fields"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Load planner-facing item records from GT or predicted scene directories."""

from __future__ import annotations

import sys
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Deque, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from tools.common import norm_qty
from tools.planner.perception_to_packbot_adapter import (
    SIGNATURE_WIDTHS,
    _build_extended_packbot_item,
    _gt_instance_signature,
    _gt_packable_arrival_keys,
    _instance_signature,
    _iter_scene_frame_files,
    _load_json,
    _new_item_id,
    set_signature_width,
)


def _extract_gt_arrival_events(gt_scene_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen_true_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
    counter = 0
    for frame_path in _iter_scene_frame_files(gt_scene_dir):
        frame_data = _load_json(frame_path)
        current_true_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
        exemplar_by_sig: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for item in frame_data.get("items", []):
            if not isinstance(item, dict):
                continue
            if not bool(item.get("packable_now", False)):
                continue
            sig = _gt_instance_signature(item)
            current_true_counts[sig] += norm_qty(item.get("quantity", 1))
            exemplar_by_sig[sig] = item

        for sig, qty in current_true_counts.items():
            prev = seen_true_counts[sig]
            if qty <= prev:
                continue
            exemplar = exemplar_by_sig[sig]
            for occurrence_index in range(prev, qty):
                entity_id = _new_item_id(counter)
                counter += 1
                events.append(
                    {
                        "entity_id": entity_id,
                        "occurrence_index": occurrence_index,
                        "source_signature": sig,
                        "item": deepcopy(exemplar),
                    }
                )
            seen_true_counts[sig] = qty
    return events


def _items_from_gt_scene(gt_scene_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    events = _extract_gt_arrival_events(gt_scene_dir)
    items_data: Dict[str, Dict[str, Any]] = {}
    arrival_order: List[str] = []
    for event in events:
        item_id = str(event["entity_id"])
        items_data[item_id] = _build_extended_packbot_item(event["item"])
        arrival_order.append(item_id)
    return (
        items_data,
        arrival_order,
        {
            "mode": "gt",
            "scene_dir": str(gt_scene_dir),
            "item_instances": len(items_data),
            "arrival_strategy": "gt_packable_now_progression",
        },
    )


def _build_predicted_extended_instances_from_scene(
    pred_scene_dir: Path,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[Any, ...], Deque[str]], List[str], Dict[str, Any]]:
    items_data: Dict[str, Dict[str, Any]] = {}
    ids_by_sig: Dict[Tuple[Any, ...], Deque[str]] = defaultdict(deque)
    first_seen_order: List[str] = []
    seen_count_by_sig: Dict[Tuple[Any, ...], int] = defaultdict(int)
    counter = 0
    frame_files = _iter_scene_frame_files(pred_scene_dir)

    for frame_path in frame_files:
        frame_data = _load_json(frame_path)
        for item in frame_data.get("items", []):
            if not isinstance(item, dict):
                continue
            sig = _instance_signature(item)
            qty = norm_qty(item.get("quantity", 1))
            current_seen = seen_count_by_sig[sig]
            if qty <= current_seen:
                continue
            for _ in range(qty - current_seen):
                item_id = _new_item_id(counter)
                counter += 1
                items_data[item_id] = _build_extended_packbot_item(item)
                ids_by_sig[sig].append(item_id)
                first_seen_order.append(item_id)
            seen_count_by_sig[sig] = qty

    return (
        items_data,
        ids_by_sig,
        first_seen_order,
        {
            "scene_dir": str(pred_scene_dir),
            "frame_count": len(frame_files),
            "item_instances": len(items_data),
            "signature_count": len(ids_by_sig),
        },
    )


def _items_from_predicted_scene(
    pred_scene_dir: Path,
    gt_scene_dir: Path | None,
    *,
    oracle_order: bool,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    items_data, ids_by_sig, first_seen_order, pred_meta = _build_predicted_extended_instances_from_scene(
        pred_scene_dir
    )
    if not oracle_order:
        meta = dict(pred_meta)
        meta["mode"] = "predicted_order"
        meta["arrival_strategy"] = "first_seen_predicted_signature"
        return items_data, list(first_seen_order), meta

    if gt_scene_dir is None:
        raise ValueError("oracle_order mode requires a GT scene directory")

    gt_arrival_keys, gt_meta = _gt_packable_arrival_keys(gt_scene_dir)
    ids_by_packing_key: Dict[Tuple[str, str, str], Deque[str]] = defaultdict(deque)
    for sig, ids in ids_by_sig.items():
        key = (sig[0], sig[1], sig[2])
        for item_id in ids:
            ids_by_packing_key[key].append(item_id)

    arrival_order: List[str] = []
    used_ids = set()
    gt_events_with_match = 0
    gt_events_without_match = 0
    for key in gt_arrival_keys:
        pool = ids_by_packing_key.get(key)
        if pool:
            item_id = pool.popleft()
            arrival_order.append(item_id)
            used_ids.add(item_id)
            gt_events_with_match += 1
        else:
            gt_events_without_match += 1

    for item_id in first_seen_order:
        if item_id not in used_ids:
            arrival_order.append(item_id)
            used_ids.add(item_id)

    meta = {
        "mode": "oracle_order",
        **pred_meta,
        **gt_meta,
        "arrival_strategy": "gt_packable_now_by_packing_key_then_predicted_leftovers",
        "gt_events_with_match": gt_events_with_match,
        "gt_events_without_match": gt_events_without_match,
    }
    return items_data, arrival_order, meta


def load_scene_inputs(
    *,
    mode: str,
    scene_id: str,
    gt_root: Path,
    pred_root: Path | None,
    signature_width: str = "wide",
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    if signature_width not in SIGNATURE_WIDTHS:
        raise ValueError(f"unsupported signature width: {signature_width}")
    set_signature_width(signature_width)
    if mode == "gt":
        items, order, meta = _items_from_gt_scene(gt_root / scene_id)
        meta["signature_width"] = signature_width
        return items, order, meta
    if mode == "predicted_order":
        if pred_root is None:
            raise ValueError("predicted_order mode requires --pred-root")
        items, order, meta = _items_from_predicted_scene(pred_root / scene_id, None, oracle_order=False)
        meta["signature_width"] = signature_width
        return items, order, meta
    if mode == "oracle_order":
        if pred_root is None:
            raise ValueError("oracle_order mode requires --pred-root")
        items, order, meta = _items_from_predicted_scene(pred_root / scene_id, gt_root / scene_id, oracle_order=True)
        meta["signature_width"] = signature_width
        return items, order, meta
    raise ValueError(f"unsupported mode: {mode}")

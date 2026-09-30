#!/usr/bin/env python3
"""Sweep soft CP-SAT safety penalties across grocery bagging scenes."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from tools.common import norm_qty
from tools.planner.extended_packbot_cp import solve_bagging_extended
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
from tools.planner.soft_packbot_cp import SAFETY_FAMILIES, solve_bagging_soft


DEFAULT_LAMBDAS = [0, 10, 25, 50, 100, 250, 500, 750, 999, 1000, 1001, 2000, 5000, 10000]


def _parse_lambdas(text: str | None) -> List[int]:
    if not text:
        return DEFAULT_LAMBDAS
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return sorted(set(values))


def _scene_dirs(root: Path, scene_ids: Sequence[str] | None) -> List[Path]:
    if scene_ids:
        return [root / scene_id for scene_id in scene_ids]
    # Found by CONTENT, not by a name prefix. The prefix test silently returned
    # nothing on datasets_seq_GT_b50, whose directories are named b50_*, and the
    # sweep reported an empty aggregate rather than an error. The same bug was
    # fixed in the Gemini full-pipeline runner on 2026-09-05.
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / "frames").is_dir() and any((d / "frames").glob("frame_*.json"))
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
            # ABSENT IS NOT FALSE. `packable_now` belongs to the sequential
            # protocol of the old supermarket dataset, where an item could be
            # visible but not yet reachable. Static benchmarks such as b50 do
            # not carry the field at all, and defaulting it to False skipped
            # every item and produced an empty sweep that still exited zero.
            # Same guard as the unknown-graspable output: a missing flag means
            # unknown, and the safe reading here is that the item is packable.
            if "packable_now" in item and not bool(item["packable_now"]):
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


def _load_scene_inputs(
    *,
    mode: str,
    scene_id: str,
    gt_root: Path,
    pred_root: Path | None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    if mode == "gt":
        return _items_from_gt_scene(gt_root / scene_id)
    if mode == "predicted_order":
        if pred_root is None:
            raise ValueError("predicted_order mode requires --pred-root")
        return _items_from_predicted_scene(pred_root / scene_id, None, oracle_order=False)
    if mode == "oracle_order":
        if pred_root is None:
            raise ValueError("oracle_order mode requires --pred-root")
        return _items_from_predicted_scene(pred_root / scene_id, gt_root / scene_id, oracle_order=True)
    raise ValueError(f"unsupported mode: {mode}")


def _total_pairwise(result: Dict[str, Any]) -> int:
    return int(result.get("total_pairwise_violations") or 0)


def _hard_violation_totals() -> Dict[str, int]:
    return {family: 0 for family in SAFETY_FAMILIES}


def _run_scene_sweep(
    *,
    scene_id: str,
    items_data: Dict[str, Dict[str, Any]],
    arrival_order: List[str],
    lambdas: Sequence[int],
    bag_cost: int,
    max_time_seconds: float | None,
) -> Dict[str, Any]:
    hard = solve_bagging_extended(items_data, arrival_order, enable_spill=True)
    hard_bags = int(hard.get("total_bags_used") or 0)
    rows = []
    results_by_lambda = {}

    for lam in lambdas:
        soft = solve_bagging_soft(
            items_data,
            arrival_order,
            safety_penalty=int(lam),
            bag_cost=bag_cost,
            max_time_seconds=max_time_seconds,
        )
        pairwise = dict(soft.get("pairwise_violation_totals") or _hard_violation_totals())
        bag_level = dict(soft.get("bag_violation_totals") or {})
        total_pairwise = _total_pairwise(soft)
        bags = int(soft.get("total_bags_used") or 0)
        row = {
            "scene_id": scene_id,
            "lambda": int(lam),
            "bag_cost": int(bag_cost),
            "status": soft.get("status"),
            "hard_status": hard.get("status"),
            "n_items": len(items_data),
            "hard_bags": hard_bags,
            "soft_bags": bags,
            "bag_savings_vs_hard": hard_bags - bags,
            "total_pairwise_violations": total_pairwise,
            "total_bag_level_violations": int(soft.get("total_bag_level_violations") or 0),
            "converged_to_hard": bool(total_pairwise == 0 and bags == hard_bags),
            "objective_value": soft.get("objective_value"),
        }
        for family in SAFETY_FAMILIES:
            row[f"pairwise_{family}"] = int(pairwise.get(family, 0))
        for key, value in bag_level.items():
            row[f"baglevel_{key}"] = int(value)
        rows.append(row)
        results_by_lambda[str(lam)] = soft

    zero_violation_lambda = None
    hard_equivalent_lambda = None
    for row in rows:
        if zero_violation_lambda is None and row["total_pairwise_violations"] == 0:
            zero_violation_lambda = row["lambda"]
        if hard_equivalent_lambda is None and row["converged_to_hard"]:
            hard_equivalent_lambda = row["lambda"]

    return {
        "scene_id": scene_id,
        "hard": hard,
        "hard_bags": hard_bags,
        "zero_violation_lambda": zero_violation_lambda,
        "hard_equivalent_lambda": hard_equivalent_lambda,
        "rows": rows,
        "soft_results_by_lambda": results_by_lambda,
    }


def _aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_lambda: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_lambda[int(row["lambda"])].append(row)

    out = []
    for lam, lam_rows in sorted(by_lambda.items()):
        record = {
            "lambda": lam,
            "scene_count": len(lam_rows),
            "optimal_count": sum(int(row["status"] == "OPTIMAL") for row in lam_rows),
            "total_hard_bags": sum(int(row["hard_bags"]) for row in lam_rows),
            "total_soft_bags": sum(int(row["soft_bags"]) for row in lam_rows),
            "total_bag_savings_vs_hard": sum(int(row["bag_savings_vs_hard"]) for row in lam_rows),
            "total_pairwise_violations": sum(
                int(row["total_pairwise_violations"]) for row in lam_rows
            ),
            "total_bag_level_violations": sum(
                int(row["total_bag_level_violations"]) for row in lam_rows
            ),
            "converged_scenes": sum(int(row["converged_to_hard"]) for row in lam_rows),
        }
        for family in SAFETY_FAMILIES:
            record[f"pairwise_{family}"] = sum(
                int(row.get(f"pairwise_{family}", 0)) for row in lam_rows
            )
        out.append(record)
    return out


def _threshold_rows(scene_results: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "scene_id": result["scene_id"],
            "hard_bags": result["hard_bags"],
            "zero_violation_lambda": result["zero_violation_lambda"],
            "hard_equivalent_lambda": result["hard_equivalent_lambda"],
        }
        for result in scene_results
    ]


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    preferred = [
        "scene_id",
        "lambda",
        "bag_cost",
        "status",
        "hard_status",
        "n_items",
        "hard_bags",
        "soft_bags",
        "bag_savings_vs_hard",
        "total_pairwise_violations",
        "total_bag_level_violations",
        "converged_to_hard",
        "objective_value",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path: Path, aggregate: Sequence[Dict[str, Any]], thresholds: Sequence[Dict[str, Any]]) -> None:
    lines = [
        "# Soft CP-SAT Sweep Summary",
        "",
        "## Aggregate Tradeoff",
        "",
        "| lambda | soft bags | hard bags | bag savings | pairwise violations | converged scenes |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        lines.append(
            "| {lambda} | {total_soft_bags} | {total_hard_bags} | "
            "{total_bag_savings_vs_hard} | {total_pairwise_violations} | "
            "{converged_scenes}/{scene_count} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Per-Scene Thresholds",
            "",
            "| scene | hard bags | first zero-violation lambda | first hard-equivalent lambda |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in thresholds:
        lines.append(
            f"| {row['scene_id']} | {row['hard_bags']} | "
            f"{row['zero_violation_lambda']} | {row['hard_equivalent_lambda']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_sweep(
    *,
    mode: str,
    gt_root: Path,
    pred_root: Path | None,
    scene_ids: Sequence[str] | None,
    lambdas: Sequence[int],
    bag_cost: int,
    max_time_seconds: float | None,
    signature_width: str = "wide",
) -> Dict[str, Any]:
    set_signature_width(signature_width)
    source_root = gt_root if mode == "gt" else pred_root
    if source_root is None:
        raise ValueError(f"{mode} mode requires --pred-root")
    scenes = _scene_dirs(source_root, scene_ids)

    scene_results = []
    flat_rows = []
    for scene_dir in scenes:
        scene_id = scene_dir.name
        if mode != "gt" and not scene_dir.exists():
            continue
        if not (gt_root / scene_id).exists() and mode in {"gt", "oracle_order"}:
            continue
        items_data, arrival_order, metadata = _load_scene_inputs(
            mode=mode,
            scene_id=scene_id,
            gt_root=gt_root,
            pred_root=pred_root,
        )
        if not items_data:
            continue
        result = _run_scene_sweep(
            scene_id=scene_id,
            items_data=items_data,
            arrival_order=arrival_order,
            lambdas=lambdas,
            bag_cost=bag_cost,
            max_time_seconds=max_time_seconds,
        )
        result["metadata"] = metadata
        scene_results.append(result)
        flat_rows.extend(result["rows"])

    aggregate = _aggregate_rows(flat_rows)
    thresholds = _threshold_rows(scene_results)
    return {
        "config": {
            "mode": mode,
            "gt_root": str(gt_root),
            "pred_root": str(pred_root) if pred_root else None,
            "lambdas": list(lambdas),
            "bag_cost": bag_cost,
            "max_time_seconds": max_time_seconds,
            "signature_width": signature_width,
        },
        "aggregate": aggregate,
        "thresholds": thresholds,
        "rows": flat_rows,
        "scenes": scene_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gt", "predicted_order", "oracle_order"], default="gt")
    parser.add_argument("--gt-root", type=Path, default=ROOT_DIR / "datasets_seq_GT")
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--scene", action="append", dest="scene_ids", help="Optional scene id; repeatable.")
    parser.add_argument("--lambdas", default=None, help="Comma-separated integer penalties.")
    parser.add_argument("--bag-cost", type=int, default=1000)
    parser.add_argument("--max-time-seconds", type=float, default=None)
    parser.add_argument(
        "--signature-width",
        choices=sorted(SIGNATURE_WIDTHS.keys()),
        default="wide",
        help="Sequential dedup signature width for GT/predicted planner inputs.",
    )
    parser.add_argument("--output-json", type=Path, default=ROOT_DIR / "runs/eval/soft_cp_sweep.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "runs/eval/soft_cp_sweep.csv")
    parser.add_argument(
        "--output-summary-csv",
        type=Path,
        default=ROOT_DIR / "runs/eval/soft_cp_sweep_summary.csv",
    )
    parser.add_argument(
        "--output-thresholds-csv",
        type=Path,
        default=ROOT_DIR / "runs/eval/soft_cp_sweep_thresholds.csv",
    )
    parser.add_argument("--output-md", type=Path, default=ROOT_DIR / "runs/eval/soft_cp_sweep_summary.md")
    args = parser.parse_args()

    summary = run_sweep(
        mode=args.mode,
        gt_root=args.gt_root,
        pred_root=args.pred_root,
        scene_ids=args.scene_ids,
        lambdas=_parse_lambdas(args.lambdas),
        bag_cost=args.bag_cost,
        max_time_seconds=args.max_time_seconds,
        signature_width=args.signature_width,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_csv, summary["rows"])
    _write_csv(args.output_summary_csv, summary["aggregate"])
    _write_csv(args.output_thresholds_csv, summary["thresholds"])
    _write_summary_md(args.output_md, summary["aggregate"], summary["thresholds"])

    print(f"[OK] JSON: {args.output_json}")
    print(f"[OK] CSV: {args.output_csv}")
    print(f"[OK] aggregate: {args.output_summary_csv}")
    print(f"[OK] thresholds: {args.output_thresholds_csv}")
    print(f"[OK] markdown: {args.output_md}")
    print(json.dumps({"aggregate": summary["aggregate"], "thresholds": summary["thresholds"]}, indent=2))


if __name__ == "__main__":
    main()

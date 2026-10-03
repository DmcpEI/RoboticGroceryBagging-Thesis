#!/usr/bin/env python3
"""Run a text-only LLM grocery-bagging planner baseline.

The LLM receives the same planner-facing item records used by CP-SAT and must
return JSON bag assignments. The script repairs malformed assignments only
enough to make them auditable: duplicate item ids are dropped after first use,
and missing item ids are added as singleton bags.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from tools.planner.planner_safety_audit import (
    MAX_BAG_VOLUME_CC,
    MAX_BAG_WEIGHT_G,
    SAFETY_FAMILIES,
    bag_violation_flags,
    pair_violations_for_items,
)
from tools.planner.perception_to_packbot_adapter import SIGNATURE_WIDTHS
from tools.planner.planner_scene_inputs import load_scene_inputs


def _scene_ids_for_mode(mode: str, gt_root: Path, pred_root: Path | None, scene_ids: Sequence[str] | None) -> List[str]:
    if scene_ids:
        return list(scene_ids)
    root = gt_root if mode == "gt" else pred_root
    if root is None:
        raise ValueError(f"{mode} mode requires --pred-root")
    return [
        p.name
        for p in sorted(root.iterdir())
        if p.is_dir() and (p.name.startswith("scene_") or p.name.startswith("syn_scene_"))
    ]


def _prompt_for_scene(scene_id: str, items_data: Mapping[str, Mapping[str, Any]], arrival_order: Sequence[str]) -> str:
    records = []
    for item_id in arrival_order:
        item = items_data[item_id]
        records.append(
            {
                "id": item_id,
                "name": item.get("display_name", item_id),
                "category": item.get("category"),
                "edible": bool(item.get("edible", True)),
                "temperature": item.get("temperature"),
                "est_weight_g": item.get("est_weight_g"),
                "est_volume_cc": item.get("est_volume_cc"),
                "crush_score": item.get("crush_score"),
                "spill_risk": bool(item.get("spill_risk", False)),
                "spill_vulnerable": bool(item.get("spill_vulnerable", False)),
            }
        )

    return f"""You are a grocery-bagging planner.

Pack the listed items into grocery bags. Use as few bags as possible, but avoid unsafe combinations.

Hard capacity limits:
- maximum bag weight: {MAX_BAG_WEIGHT_G} g
- maximum bag volume: {MAX_BAG_VOLUME_CC} cc

Safety preferences:
- keep raw meat separate from all non-raw items
- keep cleaning products out of any bag holding food (edible items)
- keep ambient items separate from refrigerated/frozen items
- avoid putting items with a low crush_score above items with a high one; crush_score is load tolerance, so 10 means nothing may rest on the item and 1 means it bears anything. Arrival order is bottom-to-top if two items share a bag
- keep spill-risk items away from spill-vulnerable bakery/produce/snack items

Return only valid JSON in this exact shape:
{{
  "bags": [
    {{"bag_id": 1, "items": ["item_000", "item_001"], "rationale": "short reason"}}
  ]
}}

Rules:
- Use each item id exactly once.
- Do not invent item ids.
- The items list inside each bag must contain only ids from the input.
- Keep rationale short.

Scene id: {scene_id}
Arrival order, bottom to top if shared:
{json.dumps(list(arrival_order), indent=2)}

Items:
{json.dumps(records, indent=2)}
"""


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[idx:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in LLM response")


def _normalise_bags(parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
    bags = parsed.get("bags", [])
    if not isinstance(bags, list):
        raise ValueError("JSON response does not contain a list field named 'bags'")
    out = []
    for idx, bag in enumerate(bags, start=1):
        if not isinstance(bag, dict):
            continue
        raw_items = bag.get("items", [])
        if not isinstance(raw_items, list):
            raw_items = []
        out.append(
            {
                "bag_id": bag.get("bag_id", idx),
                "items": [str(item_id) for item_id in raw_items],
                "rationale": str(bag.get("rationale", ""))[:500],
            }
        )
    return out


def _repair_assignment(
    bags: List[Dict[str, Any]],
    valid_item_ids: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    valid = set(valid_item_ids)
    seen = set()
    repaired = []
    duplicates = []
    unknown = []

    for bag in bags:
        clean_items = []
        for item_id in bag["items"]:
            if item_id not in valid:
                unknown.append(item_id)
                continue
            if item_id in seen:
                duplicates.append(item_id)
                continue
            seen.add(item_id)
            clean_items.append(item_id)
        if clean_items:
            repaired.append({**bag, "items": clean_items})

    missing = [item_id for item_id in valid_item_ids if item_id not in seen]
    next_id = len(repaired) + 1
    for item_id in missing:
        repaired.append(
            {
                "bag_id": next_id,
                "items": [item_id],
                "rationale": "repair: missing item added as singleton",
            }
        )
        next_id += 1

    for idx, bag in enumerate(repaired, start=1):
        bag["bag_id"] = idx

    return repaired, {
        "duplicates_removed": duplicates,
        "unknown_item_ids_removed": unknown,
        "missing_item_ids_added": missing,
        "repaired": bool(duplicates or unknown or missing),
    }


def _capacity_flags(bag_items: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    total_weight = sum(int(item.get("est_weight_g", 0)) for item in bag_items)
    total_volume = sum(int(item.get("est_volume_cc", 0)) for item in bag_items)
    return {
        "over_weight": total_weight > MAX_BAG_WEIGHT_G,
        "over_volume": total_volume > MAX_BAG_VOLUME_CC,
    }


def _pairwise_violation_totals_for_bag(
    bag_item_ids: Sequence[str],
    items_data: Mapping[str, Mapping[str, Any]],
    arrival_order: Sequence[str],
) -> Dict[str, int]:
    totals = {family: 0 for family in SAFETY_FAMILIES}
    order_pos = {item_id: idx for idx, item_id in enumerate(arrival_order)}
    ordered_ids = sorted(bag_item_ids, key=lambda item_id: order_pos.get(item_id, 10**9))

    for idx, first_id in enumerate(ordered_ids):
        for second_id in ordered_ids[idx + 1 :]:
            first = items_data[first_id]
            second = items_data[second_id]
            for family in pair_violations_for_items(first, second, ordered=False):
                totals[family] += 1
            for family in pair_violations_for_items(first, second, ordered=True):
                if family == "crush":
                    totals[family] += 1
    return totals


def _audit_bags(
    bags: Sequence[Mapping[str, Any]],
    items_data: Mapping[str, Mapping[str, Any]],
    arrival_order: Sequence[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    totals: Dict[str, int] = {
        "over_weight": 0,
        "over_volume": 0,
        "raw_meat_with_non_raw": 0,
        "chemical_with_food": 0,
        "frozen_with_ambient": 0,
        "ambient_with_nonambient": 0,
        "crush": 0,
        "spill_risk_with_vulnerable": 0,
        **{f"pairwise_{family}": 0 for family in SAFETY_FAMILIES},
    }
    audited = []
    for bag in bags:
        item_ids = [item_id for item_id in bag.get("items", []) if item_id in items_data]
        bag_items = [items_data[item_id] for item_id in item_ids]
        flags = bag_violation_flags(bag_items, item_ids, arrival_order)
        cap = _capacity_flags(bag_items)
        pairwise = _pairwise_violation_totals_for_bag(item_ids, items_data, arrival_order)
        for key, value in {**flags, **cap}.items():
            totals[key] = totals.get(key, 0) + int(value)
        for key, value in pairwise.items():
            totals[f"pairwise_{key}"] += int(value)

        audited.append(
            {
                **bag,
                "item_display_names": [items_data[item_id].get("display_name", item_id) for item_id in item_ids],
                "total_weight_g": sum(int(item.get("est_weight_g", 0)) for item in bag_items),
                "total_volume_cc": sum(int(item.get("est_volume_cc", 0)) for item in bag_items),
                "violation_flags": flags,
                "capacity_flags": cap,
                "pairwise_violation_totals": pairwise,
            }
        )
    return audited, totals


def _client(provider: str, model: str, temperature: float):
    if provider == "gemini":
        from vlm_client import GeminiClient

        return GeminiClient(model=model, temperature=temperature)
    if provider == "ollama":
        from vlm_client import OllamaClient

        return OllamaClient(model=model, options={"temperature": temperature})
    raise ValueError(f"unsupported provider: {provider}")


def _call_llm(provider: str, model: str, prompt: str, temperature: float,
              max_retries: int = 6, sleep: float = 3.5) -> str:
    client = _client(provider, model, temperature)
    text = ""
    for attempt in range(max_retries):
        text = client.generate(prompt)
        if text:
            break
        wait = sleep * (2 ** attempt)
        print(f"  empty response (attempt {attempt + 1}/{max_retries}), retry in {wait:.1f}s", file=sys.stderr)
        time.sleep(wait)
    time.sleep(sleep)
    return text


def _run_scene(
    *,
    scene_id: str,
    mode: str,
    gt_root: Path,
    pred_root: Path | None,
    provider: str,
    model: str,
    temperature: float,
    raw_response_dir: Path,
    prompt_dir: Path,
    dry_run: bool,
    signature_width: str,
) -> Dict[str, Any]:
    items_data, arrival_order, metadata = load_scene_inputs(
        mode=mode,
        scene_id=scene_id,
        gt_root=gt_root,
        pred_root=pred_root,
        signature_width=signature_width,
    )
    prompt = _prompt_for_scene(scene_id, items_data, arrival_order)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    (prompt_dir / f"{scene_id}.txt").write_text(prompt, encoding="utf-8")

    if dry_run:
        raw_text = '{"bags":[]}'
    else:
        raw_text = _call_llm(provider, model, prompt, temperature)

    raw_response_dir.mkdir(parents=True, exist_ok=True)
    (raw_response_dir / f"{scene_id}.txt").write_text(raw_text, encoding="utf-8")

    parse_error = None
    try:
        parsed = _extract_json_object(raw_text)
        bags = _normalise_bags(parsed)
    except Exception as exc:
        parse_error = repr(exc)
        bags = []

    repaired, repair_info = _repair_assignment(bags, arrival_order)
    audited_bags, totals = _audit_bags(repaired, items_data, arrival_order)

    total_pairwise = sum(totals.get(f"pairwise_{family}", 0) for family in SAFETY_FAMILIES)
    total_bag_level = sum(
        totals.get(key, 0)
        for key in (
            "raw_meat_with_non_raw",
            "chemical_with_food",
            "ambient_with_nonambient",
            "crush",
            "spill_risk_with_vulnerable",
        )
    )
    total_capacity = totals.get("over_weight", 0) + totals.get("over_volume", 0)

    return {
        "scene_id": scene_id,
        "provider": provider,
        "model": model,
        "mode": mode,
        "metadata": metadata,
        "status": "DRY_RUN" if dry_run else ("PARSE_ERROR" if parse_error else "OK"),
        "parse_error": parse_error,
        "repair_info": repair_info,
        "n_items": len(items_data),
        "total_bags_used": len(repaired),
        "total_pairwise_violations": total_pairwise,
        "total_bag_level_violations": total_bag_level,
        "total_capacity_violations": total_capacity,
        "violation_totals": totals,
        "bags": audited_bags,
    }


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in records:
        row = {
            "scene_id": rec["scene_id"],
            "status": rec["status"],
            "n_items": rec["n_items"],
            "total_bags_used": rec["total_bags_used"],
            "total_pairwise_violations": rec["total_pairwise_violations"],
            "total_bag_level_violations": rec["total_bag_level_violations"],
            "total_capacity_violations": rec["total_capacity_violations"],
        }
        for key, value in rec.get("violation_totals", {}).items():
            row[key] = value
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    preferred = [
        "scene_id",
        "status",
        "n_items",
        "total_bags_used",
        "total_pairwise_violations",
        "total_bag_level_violations",
        "total_capacity_violations",
    ]
    ordered = [key for key in preferred if key in fieldnames] + [
        key for key in fieldnames if key not in preferred
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _overall(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "scene_count": len(records),
        "ok_count": sum(int(rec["status"] == "OK") for rec in records),
        "parse_error_count": sum(int(rec["status"] == "PARSE_ERROR") for rec in records),
        "total_items": sum(int(rec["n_items"]) for rec in records),
        "total_bags_used": sum(int(rec["total_bags_used"]) for rec in records),
        "total_pairwise_violations": sum(int(rec["total_pairwise_violations"]) for rec in records),
        "total_bag_level_violations": sum(int(rec["total_bag_level_violations"]) for rec in records),
        "total_capacity_violations": sum(int(rec["total_capacity_violations"]) for rec in records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gt", "predicted_order", "oracle_order"], default="gt")
    parser.add_argument("--gt-root", type=Path, default=ROOT_DIR / "datasets_seq_GT")
    parser.add_argument("--pred-root", type=Path, default=None)
    parser.add_argument("--scene", action="append", dest="scene_ids")
    parser.add_argument("--provider", choices=["gemini", "ollama"], default="gemini")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--signature-width",
        choices=sorted(SIGNATURE_WIDTHS.keys()),
        default="wide",
        help="Sequential dedup signature width for GT/predicted planner inputs.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write prompts and run parser/auditor without calling an LLM.")
    parser.add_argument("--output-json", type=Path, default=ROOT_DIR / "runs/eval/llm_planner_baseline.json")
    parser.add_argument("--output-csv", type=Path, default=ROOT_DIR / "runs/eval/llm_planner_baseline.csv")
    parser.add_argument("--prompt-dir", type=Path, default=ROOT_DIR / "runs/llm_planner_prompts")
    parser.add_argument("--raw-response-dir", type=Path, default=ROOT_DIR / "runs/llm_planner_raw")
    args = parser.parse_args()

    scene_ids = _scene_ids_for_mode(args.mode, args.gt_root, args.pred_root, args.scene_ids)
    records = []
    for scene_id in scene_ids:
        if args.mode in {"gt", "oracle_order"} and not (args.gt_root / scene_id).exists():
            continue
        if args.mode != "gt" and args.pred_root and not (args.pred_root / scene_id).exists():
            continue
        records.append(
            _run_scene(
                scene_id=scene_id,
                mode=args.mode,
                gt_root=args.gt_root,
                pred_root=args.pred_root,
                provider=args.provider,
                model=args.model,
                temperature=args.temperature,
                raw_response_dir=args.raw_response_dir,
                prompt_dir=args.prompt_dir,
                dry_run=args.dry_run,
                signature_width=args.signature_width,
            )
        )

    summary = {
        "config": {
            "mode": args.mode,
            "gt_root": str(args.gt_root),
            "pred_root": str(args.pred_root) if args.pred_root else None,
            "provider": args.provider,
            "model": args.model,
            "temperature": args.temperature,
            "dry_run": args.dry_run,
            "signature_width": args.signature_width,
        },
        "overall": _overall(records),
        "per_scene": records,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.output_csv, records)

    print(f"[OK] JSON: {args.output_json}")
    print(f"[OK] CSV: {args.output_csv}")
    print(json.dumps(summary["overall"], indent=2))


if __name__ == "__main__":
    main()

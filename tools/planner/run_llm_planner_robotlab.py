#!/usr/bin/env python3
"""Gemini text-LLM bagging planner for robot-lab static scenes.

run_llm_planner_baseline.py's GT loader (_items_from_gt_scene) assumes the
old temporal dataset's packable_now field, which robot-lab GT frames don't
have (single static snapshot per scene, everything present at once) -- it
would silently produce 0 items for every scene. This script builds items_data
directly from the flat scene.json / predicted frame_000.json item lists
(same approach as run_extended_cp_robot_lab.py's build_items_data), then
reuses run_llm_planner_baseline's prompt/parse/repair/audit machinery.

  python tools/planner/run_llm_planner_robotlab.py --mode gt \
      --gt datasets_seq_GT_robot_49 --out runs/eval/gemini_llm_planner_gt_49.json

  python tools/planner/run_llm_planner_robotlab.py --mode predicted \
      --gt datasets_seq_GT_robot_49 --pred runs/robot_lab_prod_49_gemini_er \
      --out runs/eval/gemini_llm_planner_pred_49.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/eval"))
sys.path.insert(0, str(ROOT / "tools/planner"))

from build_planner_output import match_attrs, estimate_attrs, load_planner_attrs  # noqa: E402
from run_gemini_robotics import load_key  # noqa: E402
from run_llm_planner_baseline import (  # noqa: E402
    _prompt_for_scene, _extract_json_object, _normalise_bags,
    _repair_assignment, _audit_bags, _call_llm, _overall, _write_csv,
)
from vlm_client import QuotaExhausted  # noqa: E402


def build_items_data(items: list[dict], attrs_table: dict) -> tuple[dict, list[str]]:
    items_data, order, counts = {}, [], {}
    for it in items:
        name = it.get("name") or "unknown"
        if name == "unknown_object":
            continue
        q = int(it.get("quantity", 1) or 1)
        pa = match_attrs(name, attrs_table) or estimate_attrs(it)
        for _ in range(max(1, q)):
            counts[name] = counts.get(name, 0) + 1
            key = f"{name}#{counts[name]}"
            items_data[key] = {"display_name": name, **pa}
            order.append(key)
    return items_data, order


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["gt", "predicted"], required=True)
    ap.add_argument("--gt", type=Path, default=ROOT / "datasets_seq_GT_robot_49")
    ap.add_argument("--pred", type=Path, default=None)
    ap.add_argument("--provider", default="gemini")
    ap.add_argument("--model", default="gemini-robotics-er-1.6-preview")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--sleep", type=float, default=3.5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt-dir", type=Path, default=ROOT / "runs/llm_planner_prompts_robotlab2")
    ap.add_argument("--raw-response-dir", type=Path, default=ROOT / "runs/llm_planner_raw_robotlab2")
    ap.add_argument("--resume", action="store_true", help="skip scenes already OK in --out")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY",
                     help="which .env var to read the key from (use a different account's key for a fresh quota bucket)")
    args = ap.parse_args()

    if args.provider == "gemini":
        os.environ["GEMINI_API_KEY"] = load_key(args.api_key_env)

    attrs_table = load_planner_attrs()
    # Found by content, not by a "scene_" prefix: the b50 benchmark names its
    # scenes b50_*, and a prefix glob returned zero scenes there without error --
    # the same silent-empty failure fixed in run_soft_cp_sweep on 2026-09-19.
    scenes = sorted(p.name for p in args.gt.iterdir()
                    if (p / "scene.json").exists() or (p / "frames" / "frame_000.json").exists())
    if not scenes:
        raise SystemExit(f"no scenes under {args.gt}")

    prior = {}
    if args.resume and args.out.exists():
        prior = {r["scene_id"]: r for r in json.loads(args.out.read_text())["per_scene"] if r["status"] == "OK"}

    records = []
    for scene_id in scenes:
        if scene_id in prior:
            records.append(prior[scene_id])
            continue
        if args.mode == "gt":
            fp = args.gt / scene_id / "scene.json"
            if not fp.exists():   # b50 layout
                fp = args.gt / scene_id / "frames" / "frame_000.json"
        else:
            fp = args.pred / scene_id / "frames" / "frame_000.json"
        if not fp.exists():
            continue
        items = json.loads(fp.read_text())["items"]
        items_data, arrival_order = build_items_data(items, attrs_table)

        prompt = _prompt_for_scene(scene_id, items_data, arrival_order)
        args.prompt_dir.mkdir(parents=True, exist_ok=True)
        (args.prompt_dir / f"{scene_id}.txt").write_text(prompt, encoding="utf-8")

        if items_data:
            try:
                raw_text = _call_llm(args.provider, args.model, prompt, args.temperature, sleep=args.sleep)
            except QuotaExhausted as e:
                print(f"[STOP] quota exhausted at {scene_id}: {e}")
                print(f"{len(records)} scenes done this run -> {args.out}. Rerun with --resume once quota resets.")
                summary = {"overall": _overall(records), "per_scene": records}
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
                _write_csv(args.out.with_suffix(".csv"), records)
                return 1
        else:
            raw_text = '{"bags":[]}'
        args.raw_response_dir.mkdir(parents=True, exist_ok=True)
        (args.raw_response_dir / f"{scene_id}.txt").write_text(raw_text, encoding="utf-8")

        parse_error = None
        try:
            parsed = _extract_json_object(raw_text)
            bags = _normalise_bags(parsed)
        except Exception as exc:  # noqa: BLE001
            parse_error = repr(exc)
            bags = []

        repaired, repair_info = _repair_assignment(bags, arrival_order)
        audited_bags, totals = _audit_bags(repaired, items_data, arrival_order)
        total_pairwise = sum(v for k, v in totals.items() if k.startswith("pairwise_"))
        total_bag_level = sum(totals.get(k, 0) for k in (
            "raw_meat_with_non_raw", "cleaning_with_non_cleaning", "ambient_with_nonambient",
            "crush", "spill_risk_with_vulnerable"))
        total_capacity = totals.get("over_weight", 0) + totals.get("over_volume", 0)

        rec = {
            "scene_id": scene_id, "provider": args.provider, "model": args.model, "mode": args.mode,
            "status": "PARSE_ERROR" if parse_error else "OK", "parse_error": parse_error,
            "repair_info": repair_info, "n_items": len(items_data), "total_bags_used": len(repaired),
            "total_pairwise_violations": total_pairwise, "total_bag_level_violations": total_bag_level,
            "total_capacity_violations": total_capacity, "violation_totals": totals, "bags": audited_bags,
        }
        records.append(rec)
        print(f"{scene_id}: status={rec['status']} n_items={rec['n_items']} bags={rec['total_bags_used']}")

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"overall": _overall(records), "per_scene": records}, indent=2), encoding="utf-8")

    summary = {"overall": _overall(records), "per_scene": records}
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.out.with_suffix(".csv"), records)
    print(json.dumps(summary["overall"], indent=1))
    print(f"[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

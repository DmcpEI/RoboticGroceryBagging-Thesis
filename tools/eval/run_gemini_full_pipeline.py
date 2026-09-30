#!/usr/bin/env python3
"""Full-pipeline Gemini baseline: Gemini perceives, Gemini packs. No rules, no CP-SAT.

This is the "what does a normal online VLM do out of the box" baseline the paper
promises. It is deliberately NOT the same experiment as run_llm_planner_baseline.py,
which hands the model our structured Layer-3 records AND every safety rule and
capacity limit -- that measures rule-following, not out-of-the-box behaviour.

Here the model gets only a list of item NAMES (its own perception output in
--mode pred) and the instruction "pack these into bags". No attributes, no
capacity limits, no safety constraints, no hints that safety matters at all.
The returned bags are then audited with the same planner_safety_audit
predicates used for CP-SAT, resolving each name to the robot planner catalog.

    python tools/eval/run_gemini_full_pipeline.py --mode pred \
        --perception-run runs/robot_lab_prod_49_gemini_er_v1_namelookup \
        --out runs/eval/gemini_full_pipeline_pred_49.json

--mode gt feeds GT item names instead, which separates perception error from
packing error (same packing prompt, perfect inventory).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/pipeline"))

from apply_robot_catalog_weight import _tokens  # noqa: E402
from tools.planner import planner_safety_audit  # noqa: E402
from tools.planner.planner_safety_audit import (  # noqa: E402
    MAX_BAG_VOLUME_CC,
    MAX_BAG_WEIGHT_G,
    SAFETY_FAMILIES,
    pair_violations_for_items,
)

# The module default (2) is calibrated to the old general-supermarket dataset;
# the robot-lab CP runs all use 3, the lowest crush tier the catalog contains.
# Audit at 3 so this baseline is comparable to those numbers.
CRUSH_THRESHOLD_HEAVY = 3
planner_safety_audit.CRUSH_THRESHOLD_HEAVY = CRUSH_THRESHOLD_HEAVY

# Layer-3 planner records, keyed by display_name (the catalog repeats names for
# quantity expansion, so dedupe -- all copies of a name carry identical props).
_CATALOG = json.loads((ROOT / "data/robot_lab/robot_item_catalog_planner.json").read_text())
PROPS_BY_NAME = {}
for _rec in _CATALOG["items_data"].values():
    PROPS_BY_NAME.setdefault(_rec["display_name"], _rec)
_NAME_INDEX = [(_tokens(n), n) for n in PROPS_BY_NAME]
# Longest first so "wine cup" wins over "cup" and "potato chip can" over "can".
_LONGEST_FIRST = sorted(PROPS_BY_NAME, key=len, reverse=True)

# No capacity limits, no safety rules, no attributes -- only names and a format.
PACK_PROMPT = """These items are at a supermarket checkout. Pack them into bags.

{items}

Reply with ONLY JSON, no prose and no markdown fences:
{{"bags": [[1, 4, 7], [2, 3]]}}
Each number is an item number above. Every item must appear in exactly one bag.
List the numbers within a bag in the order you would place them, first at the bottom."""


def resolve(name: str, min_overlap: float = 0.34):
    """Token-overlap match to the planner catalog (same matcher as the weight
    override and the Gemini-ER attribute reattach). Returns None for a name
    outside the catalog -- reported, never silently dropped."""
    # _tokens strips packaging words, so catalog entries whose whole name IS a
    # packaging word ("cup", "can") tokenize to the empty set and can never be
    # matched by overlap. Try a literal match first.
    low = re.sub(r"[^a-z ]+", " ", name.lower())
    low = " ".join(low.split())
    if low in PROPS_BY_NAME:
        return low
    for cname in _LONGEST_FIRST:
        if re.search(rf"\b{re.escape(cname)}\b", low):
            return cname

    toks = _tokens(name)
    best, best_score = None, 0.0
    for ctoks, cname in _NAME_INDEX:
        if not toks or not ctoks:
            continue
        inter = len(toks & ctoks)
        if inter == 0:
            continue
        score = inter / min(len(toks), len(ctoks))
        if score > best_score:
            best, best_score = cname, score
    return best if best_score >= min_overlap else None


def load_key(env_var: str = "GEMINI_API_KEY") -> str:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith(f"{env_var}="):
                os.environ[env_var] = line.split("=", 1)[1].strip().strip('"')
    key = os.environ.get(env_var)
    if not key:
        sys.exit(f"{env_var} not set (add to .env)")
    return key


def scene_items(scene_dir: Path) -> list[str]:
    """Item names for one scene, quantity-expanded."""
    f = scene_dir / "frames" / "frame_000.json"
    if not f.exists():
        # scenes 045-049 are the standalone GT entries and store items flat.
        f = scene_dir / "scene.json"
    if not f.exists():
        return []
    names = []
    for it in json.loads(f.read_text()).get("items", []):
        nm = str(it.get("name") or "").strip()
        if not nm:
            continue
        try:
            q = max(1, int(it.get("quantity") or 1))
        except Exception:  # noqa: BLE001
            q = 1
        names.extend([nm] * q)
    return names


def parse_bags(text: str, n_items: int):
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    data = None
    try:
        data = json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\{.*\}", t, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:  # noqa: BLE001
                data = None
    if not isinstance(data, dict):
        return None
    bags = data.get("bags")
    if not isinstance(bags, list):
        return None
    out, seen = [], set()
    for bag in bags:
        if not isinstance(bag, list):
            continue
        keep = []
        for v in bag:
            try:
                i = int(v) - 1
            except Exception:  # noqa: BLE001
                continue
            if 0 <= i < n_items and i not in seen:
                seen.add(i)
                keep.append(i)
        if keep:
            out.append(keep)
    # Repair, and count it: unassigned items become singleton bags rather than
    # vanishing, which would flatter both the bag count and the violation count.
    dropped = [i for i in range(n_items) if i not in seen]
    for i in dropped:
        out.append([i])
    return out, len(dropped)


def audit_bags(bags, names):
    """Pair-audit each bag with the CP-SAT safety predicates. Within-bag list
    order is treated as bottom-to-top, which is what the prompt asked for."""
    fam = {f: 0 for f in SAFETY_FAMILIES}
    n_pairs = 0
    over_weight = over_volume = 0
    unresolved = []
    for bag in bags:
        props = []
        for i in bag:
            cname = resolve(names[i])
            if cname is None:
                unresolved.append(names[i])
                continue
            props.append(PROPS_BY_NAME[cname])
        if sum(int(p.get("est_weight_g", 0)) for p in props) > MAX_BAG_WEIGHT_G:
            over_weight += 1
        if sum(int(p.get("est_volume_cc", 0)) for p in props) > MAX_BAG_VOLUME_CC:
            over_volume += 1
        for a in range(len(props)):
            for b in range(a + 1, len(props)):
                n_pairs += 1
                for f in pair_violations_for_items(props[a], props[b], ordered=True):
                    fam[f] += 1
    return fam, n_pairs, over_weight, over_volume, unresolved


def main() -> int:
    import google.generativeai as genai

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["pred", "gt"], default="pred")
    ap.add_argument("--perception-run", type=lambda v: Path(v).resolve(),
                    default=ROOT / "runs/robot_lab_prod_49_gemini_er_v1_namelookup")
    ap.add_argument("--gt-root", type=lambda v: Path(v).resolve(), default=ROOT / "datasets_seq_GT_robot_49")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-robotics-er-1.6-preview")
    ap.add_argument("--sleep", type=float, default=13.0,
                    help="throttle between scenes; free-tier ER allows 5 req/min")
    ap.add_argument("--max-retries", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY,GEMINI_API_KEY_ALT,"
                                             "GEMINI_API_KEY_ALT2,GEMINI_API_KEY_ALT3",
                    help="comma-separated .env vars; rotated when one hits its daily cap")
    args = ap.parse_args()

    # Free-tier keys have both a per-minute and a per-day cap. The per-minute one
    # is waited out; the per-day one only clears by moving to the next account.
    key_envs = [k.strip() for k in args.api_key_env.split(",") if k.strip()]
    keys = []
    for env_var in key_envs:
        try:
            keys.append(load_key(env_var))
        except SystemExit:
            pass
    if not keys:
        sys.exit(f"no usable key in {key_envs}")
    key_i = 0
    genai.configure(api_key=keys[key_i])
    model = genai.GenerativeModel(args.model)

    def rotate_key() -> bool:
        nonlocal key_i, model
        if key_i + 1 >= len(keys):
            return False
        key_i += 1
        genai.configure(api_key=keys[key_i])
        model = genai.GenerativeModel(args.model)
        print(f"  -> switching to key {key_i + 1}/{len(keys)} ({key_envs[key_i]})", flush=True)
        return True

    root = args.perception_run if args.mode == "pred" else args.gt_root
    # Scene directories are named scene_NNN on the 49-scene set and by their
    # arrangement on the b50 captures, so take every directory that holds a
    # scene rather than matching a prefix.
    scenes = sorted(p for p in root.iterdir() if p.is_dir()
                    and ((p / "frames/frame_000.json").exists() or (p / "scene.json").exists()))
    if args.limit:
        scenes = scenes[: args.limit]

    # Raw model replies are cached per scene so an interrupted run (rate limits,
    # network) resumes without re-spending quota.
    cache_path = args.out.with_suffix(".raw.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    cache_hits = set(cache)  # no throttle needed for scenes served from cache

    per_scene = []
    totals = {f: 0 for f in SAFETY_FAMILIES}
    n_bags = n_items = n_pairs = n_dropped = 0
    n_over_w = n_over_v = 0
    unresolved_all: list[str] = []
    failed: list[str] = []
    t0 = time.time()

    for sc in scenes:
        names = scene_items(sc)
        if not names:
            per_scene.append({"scene": sc.name, "n_items": 0, "n_bags": 0, "note": "empty perception"})
            continue
        listing = "\n".join(f"{i + 1}. {nm}" for i, nm in enumerate(names))
        prompt = PACK_PROMPT.format(items=listing)

        parsed = None
        rate_limited = 0
        if sc.name in cache:
            parsed = parse_bags(cache[sc.name], len(names))
        for attempt in range(args.max_retries):
            if parsed is not None:
                break
            try:
                r = model.generate_content(prompt)
                parsed = parse_bags(r.text, len(names))
                if parsed is not None:
                    cache[sc.name] = r.text
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(json.dumps(cache, indent=1))
                    break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "quota" in msg.lower():
                    rate_limited += 1
                    # A per-minute limit clears by waiting; a per-day one never
                    # does, and presents as the same 429 repeating. Wait twice,
                    # then treat it as the daily cap and move to the next key.
                    if rate_limited > 2 and rotate_key():
                        rate_limited = 0
                        continue
                    m = re.search(r"retry in ([\d.]+)s", msg)
                    wait = float(m.group(1)) + 2 if m else 30.0
                    print(f"  rate limited at {sc.name}, waiting {wait:.0f}s", flush=True)
                    time.sleep(wait)
                    continue
                time.sleep(2 * (attempt + 1))
        if parsed is None:
            failed.append(sc.name)
            continue

        bags, dropped = parsed
        fam, pairs, ow, ov, unres = audit_bags(bags, names)
        for f, v in fam.items():
            totals[f] += v
        n_bags += len(bags)
        n_items += len(names)
        n_pairs += pairs
        n_dropped += dropped
        n_over_w += ow
        n_over_v += ov
        unresolved_all.extend(unres)
        per_scene.append({
            "scene": sc.name,
            "n_items": len(names),
            "n_bags": len(bags),
            "bags": [[names[i] for i in bag] for bag in bags],
            "violations": {k: v for k, v in fam.items() if v},
            "unassigned_repaired": dropped,
            "over_weight_bags": ow,
            "over_volume_bags": ov,
        })
        print(f"{sc.name}: {len(names)} items -> {len(bags)} bags, "
              f"{sum(fam.values())} violations", flush=True)
        if sc.name not in cache_hits:
            time.sleep(args.sleep)

    from collections import Counter
    summary = {
        "mode": args.mode,
        "model": args.model,
        "source": str(root.relative_to(ROOT)),
        "protocol": "Gemini perceives, Gemini packs. Prompt gives item names only -- "
                    "no attributes, no capacity limits, no safety rules. Bags audited "
                    "post hoc with planner_safety_audit against the robot planner catalog.",
        "n_scenes": len([s for s in per_scene if s.get("n_items")]),
        "n_scenes_failed": len(failed),
        "scenes_failed": failed,
        "total_items": n_items,
        "total_bags": n_bags,
        "total_pairs_audited": n_pairs,
        "violations": totals,
        "total_violations": sum(totals.values()),
        "items_unassigned_repaired": n_dropped,
        "bags_over_weight_limit": n_over_w,
        "bags_over_volume_limit": n_over_v,
        "names_unresolved_to_catalog": len(unresolved_all),
        "unresolved_examples": Counter(unresolved_all).most_common(15),
        "per_scene": per_scene,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"\n{args.mode}: {n_items} items, {n_bags} bags, {sum(totals.values())} violations "
          f"({n_pairs} pairs audited), {n_over_w} over-weight / {n_over_v} over-volume bags, "
          f"{len(unresolved_all)} names off-catalog -> {args.out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

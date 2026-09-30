#!/usr/bin/env python3
"""Supply weight from the fixed robot-lab catalog after identification.

weight_class is not visually grounded for replica objects: both Qwen3-VL-8B and
MiniCPM-V predict ~90% `light` while the GT is balanced (286 light / 160 medium
/ 22 heavy). For the robot lab the object set is fixed, so weight is a property
of identity, not appearance. This maps each predicted item name to the closest
catalog entry (token overlap + a small synonym map) and supplies:

  - weight_class  (always, when matched)
  - measured_weight_g + weight_source  (when the catalog has a measured gram value)

Unmatched items keep the VLM-predicted weight_class and get weight_source=`vlm`.

When the catalog carries a measured gram value, weight_class is DERIVED from it
via documented thresholds (so class and grams never disagree). The adapter then
passes measured grams straight through to est_weight_g, replacing the
200/800/2500 proxy.

This is a robot-lab Layer-2 deterministic prior. Do NOT use it for general
supermarket scenes (no closed catalog there).

Usage:
  # mirror a run into <pred>_catwt with weight supplied, then eval
  python tools/pipeline/apply_robot_catalog_weight.py \
      --pred runs/robot_lab_open_35_base --out runs/robot_lab_open_35_catwt_base
  # write an audit CSV of every override
  python tools/pipeline/apply_robot_catalog_weight.py \
      --pred runs/robot_lab_open_35_base --out /tmp/x --audit-csv runs/eval/catwt_audit.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

CATALOG = Path("data/robot_lab/robot_item_catalog_perception.json")

# Documented weight_class thresholds (grams). Single rule, reproducible:
#   weight < 250 g          -> light
#   250 g <= weight < 1000  -> medium
#   weight >= 1000 g        -> heavy
WEIGHT_CLASS_THRESHOLDS = ((250, "light"), (1000, "medium"))  # else "heavy"


def class_from_grams(grams: float) -> str:
    for limit, label in WEIGHT_CLASS_THRESHOLDS:
        if grams < limit:
            return label
    return "heavy"


# Generic packaging/colour words that do not help disambiguate identity.
STOPWORDS = {
    "box", "can", "bottle", "jar", "carton", "cup", "bag", "tube", "tub", "tray",
    "pouch", "wrap", "loose", "other", "the", "a", "of", "and",
    "red", "green", "yellow", "blue", "white", "orange-coloured",
}

# Model-vocab -> catalog-token synonyms (applied to predicted name tokens).
SYNONYMS = {
    "drill": "screwdriver", "cordless": "screwdriver",
    "cheez": "cracker", "cheezit": "cracker", "cheezits": "cracker", "crackers": "cracker",
    "jello": "gelatin", "jell": "gelatin", "jelly": "gelatin",
    "domino": "sugar",
    "soda": "energy", "redbull": "energy", "bull": "energy", "monster": "energy",
    "coke": "cola", "sprite": "cola", "pop": "cola", "soft": "cola",  # soft drink
    "pringles": "potato", "chips": "potato", "chip": "potato", "crisps": "potato",
    "mug": "cup",  # `plate` and `dish` removed: plate is its own catalog item now
    "colgate": "toothpaste", "toothpastes": "toothpaste",
    "windex": "glass", "scrub": "glass",  # soft scrub / windex -> cleaning bottle
    "windelx": "glass", "windshield": "glass", "washer": "glass",  # misspellings / windshield-washer blue bottle
    "salmon": "tuna", "sardine": "tuna",  # generic food can (same weight class)
    "eggs": "egg", "cherry": "cherries",
    # YCB 002 "master chef can" IS the lab's coffee can replica
    "master": "coffee", "chef": "coffee",
    # only one yellow squeeze-bottle replica in the lab (YCB 006 mustard)
    "mayo": "mustard", "mayonnaise": "mustard", "ketchup": "mustard",
}

# Fallback weight by packaging when name match fails. box/loose/other left out
# (ambiguous: a box can be light gelatin or medium cracker), keep VLM weight.
PACKAGING_FALLBACK_WEIGHT = {
    "bottle": "medium",
    "can": "medium",
    "jar": "medium",
    "aerosol": "medium",
    "cup": "light",
    "tube": "light",
    "carton": "light",
}

TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(name: str, keep_stopwords: bool = False):
    toks = TOKEN_RE.findall(str(name or "").lower())
    out = set()
    for t in toks:
        t = SYNONYMS.get(t, t)
        if keep_stopwords or t not in STOPWORDS:
            out.add(t)
    return out


def load_catalog_index():
    """Return list of (tokens, weight_class, measured_weight_g, catalog_name).

    Catalog items whose name consists ONLY of stopwords (e.g. "cup") would get
    an empty token set and become unreachable by name match; for those, keep
    the raw tokens so predictions like "plastic cup" can still reach them.
    The raw tokens are flagged so match_weight only uses them as a second tier
    (a bare "can" prediction must NOT match "tuna can").
    """
    cat = json.loads(CATALOG.read_text())["items"]
    idx = []
    for it in cat:
        toks = _tokens(it["name"])
        bare_container = not toks
        if bare_container:
            toks = _tokens(it["name"], keep_stopwords=True)
        idx.append((toks, it["weight_class"], it.get("measured_weight_g"), it["name"], bare_container))
    return idx


def match_weight(name: str, packaging: str, idx, min_overlap: float = 0.34):
    """Return (weight_class, measured_g, source, catalog_name).

    source is 'catalog_name', 'packaging_fallback', or None (no match).
    When the matched catalog entry has measured grams, weight_class is derived
    from those grams so the two never disagree.
    """
    toks = _tokens(name)
    best, best_score = None, 0.0
    for ctoks, weight, grams, cname, bare in idx:
        if bare or not ctoks or not toks:
            continue
        inter = len(toks & ctoks)
        if inter == 0:
            continue
        # overlap normalized by the smaller token set (substring-friendly)
        score = inter / min(len(toks), len(ctoks))
        if score > best_score:
            best, best_score = (weight, grams, cname), score
    if best is None:
        # Second tier: bare-container catalog items (e.g. "cup") matched on RAW
        # tokens, so "plastic cup"/"red cup" resolve. A lone "can" prediction
        # cannot land here ("cup" is the only bare item; multi-can items are
        # tier 1 only), so this cannot steal "tuna can"-style matches.
        raw = _tokens(name, keep_stopwords=True)
        for ctoks, weight, grams, cname, bare in idx:
            if not bare or not raw:
                continue
            inter = len(raw & ctoks)
            if inter == 0:
                continue
            score = inter / min(len(raw), len(ctoks))
            if score > best_score:
                best, best_score = (weight, grams, cname), score
    if best and best_score >= min_overlap:
        weight, grams, cname = best
        if isinstance(grams, (int, float)) and grams > 0:
            weight = class_from_grams(grams)
        return weight, grams, "catalog_name", cname
    fb = PACKAGING_FALLBACK_WEIGHT.get(str(packaging or "").lower())
    if fb:
        return fb, None, "packaging_fallback", None
    return None, None, None, None


def apply_to_items(items, idx, debug: bool = False, audit_rows=None):
    """Supply weight in place. Returns counts dict.

    Counts (renamed for clarity per 2026-06-07 review):
      matched_by_name        : items whose name matched a catalog entry
      matched_by_packaging   : items resolved only via the packaging fallback
      changed_weight_class   : items whose weight_class value actually changed
      set_weight_g           : items given a measured_weight_g from the catalog
    """
    counts = {
        "matched_by_name": 0,
        "matched_by_packaging": 0,
        "changed_weight_class": 0,
        "set_weight_g": 0,
    }
    for it in items:
        old_wc = it.get("weight_class")
        w, grams, src, cname = match_weight(it.get("name", ""), it.get("packaging", ""), idx)
        if not src:
            it.setdefault("weight_source", "vlm")
            continue
        if src == "catalog_name":
            counts["matched_by_name"] += 1
        elif src == "packaging_fallback":
            counts["matched_by_packaging"] += 1
        if debug:
            it["_weight_class_vlm"] = old_wc
        it["weight_class"] = w
        it["weight_source"] = src
        if w != old_wc:
            counts["changed_weight_class"] += 1
        if isinstance(grams, (int, float)) and grams > 0:
            it["measured_weight_g"] = grams
            counts["set_weight_g"] += 1
        if audit_rows is not None:
            audit_rows.append({
                "pred_name": it.get("name", ""),
                "matched_catalog_name": cname or "",
                "weight_source": src,
                "old_weight_class": old_wc,
                "new_weight_class": w,
                "measured_weight_g": grams if grams is not None else "",
            })
    return counts


def write_audit_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["pred_name", "matched_catalog_name", "weight_source",
              "old_weight_class", "new_weight_class", "measured_weight_g"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", type=Path, required=True, help="source run base dir")
    ap.add_argument("--out", type=Path, required=True, help="destination run dir (mirrored)")
    ap.add_argument("--debug", action="store_true", help="write _weight_class_vlm debug field")
    ap.add_argument("--audit-csv", type=Path, default=None, help="write a per-item override audit CSV")
    args = ap.parse_args()

    idx = load_catalog_index()
    frames = sorted(args.pred.glob("scene_*/frames/frame_000.json"))
    agg = {"matched_by_name": 0, "matched_by_packaging": 0, "changed_weight_class": 0, "set_weight_g": 0}
    audit_rows = [] if args.audit_csv else None
    total_items = 0
    for fp in frames:
        data = json.loads(fp.read_text())
        items = data.get("items", [])
        total_items += len(items)
        c = apply_to_items(items, idx, debug=args.debug, audit_rows=audit_rows)
        for k in agg:
            agg[k] += c[k]
        rel = fp.relative_to(args.pred)
        dst = args.out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n")
    print(f"mirrored {len(frames)} frames -> {args.out}")
    print(f"items={total_items} matched_by_name={agg['matched_by_name']} "
          f"matched_by_packaging={agg['matched_by_packaging']} "
          f"changed_weight_class={agg['changed_weight_class']} set_weight_g={agg['set_weight_g']}")
    if args.audit_csv:
        write_audit_csv(audit_rows, args.audit_csv)
        print(f"audit CSV -> {args.audit_csv} ({len(audit_rows)} rows)")


if __name__ == "__main__":
    main()

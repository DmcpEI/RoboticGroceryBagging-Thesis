#!/usr/bin/env python3
"""Adapt perception outputs into PACKBOT CP planner inputs.

This module bridges the sequential perception outputs in this repo to the input
contract expected by `external/colleague_packing_system/code/or_approach.py`.

The CP planner expects:
- `items_data`: dict[item_id -> properties]
- `arrival_order`: list[item_id]

where each item provides:
- `est_weight_g`
- `est_volume_cc`
- `crush_score`
- `category`
- `temperature`

Important implementation notes validated against the repo:
- The current best prediction outputs are per-frame JSON files, not a single
  scene JSON with a top-level `frames` array.
- Current best predictions do not include `packable_now`, so scene-level arrival
  order must be approximated from first appearance, or borrowed from GT in an
  oracle-order mode.
- PACKBOT CP uses high crush_score = "keep it on top" and low = heavy/sturdy.
  The scale is LOAD TOLERANCE (crushability), not breakability: the
  collaborator's own item_properties.json scores `Loaf of Bread` 10 and
  `Can of Soup` 1. See compute_crush_score_v2 and Settled Decision 7.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
root_str = str(ROOT_DIR)
if root_str not in sys.path:
    sys.path.insert(0, root_str)

from tools.common import norm_key_part, norm_qty

WEIGHT_ESTIMATION = {
    "light": 200,
    "medium": 800,
    "heavy": 2500,
}

VOLUME_ESTIMATION = {
    ("bottle", "light"): 250,
    ("bottle", "medium"): 750,
    ("bottle", "heavy"): 2000,
    ("carton", "light"): 300,
    ("carton", "medium"): 1100,
    ("carton", "heavy"): 2500,
    ("can", "light"): 200,
    ("can", "medium"): 400,
    ("can", "heavy"): 800,
    ("bag", "light"): 300,
    ("bag", "medium"): 800,
    ("bag", "heavy"): 1500,
    ("box", "light"): 400,
    ("box", "medium"): 1000,
    ("box", "heavy"): 2000,
    ("jar", "light"): 200,
    ("jar", "medium"): 500,
    ("jar", "heavy"): 1000,
    ("tray", "light"): 300,
    ("tray", "medium"): 600,
    ("tray", "heavy"): 1200,
    ("tub", "light"): 200,
    ("tub", "medium"): 500,
    ("tub", "heavy"): 1000,
    ("cup", "light"): 150,
    ("cup", "medium"): 300,
    ("cup", "heavy"): 500,
    ("wrap", "light"): 200,
    ("wrap", "medium"): 500,
    ("wrap", "heavy"): 800,
    ("pouch", "light"): 200,
    ("pouch", "medium"): 500,
    ("pouch", "heavy"): 800,
    ("tube", "light"): 150,
    ("tube", "medium"): 300,
    ("tube", "heavy"): 600,
    ("blister", "light"): 150,
    ("blister", "medium"): 250,
    ("blister", "heavy"): 500,
    ("loose", "light"): 350,
    ("loose", "medium"): 1200,
    ("loose", "heavy"): 4000,
    ("aerosol", "light"): 250,
    ("aerosol", "medium"): 500,
    ("aerosol", "heavy"): 900,
}

DEFAULT_VOLUME = {"light": 300, "medium": 700, "heavy": 1500}

# Name-driven volume overrides help align a few large footprint cases with
# PACKBOT's item_properties without requiring a richer geometry model.
NAME_VOLUME_OVERRIDES = {
    "paper towel": 8000,
    "paper towels": 8000,
    "toilet paper": 7000,
    "tissue": 5000,
    "bread": 3000,
    "baguette": 2500,
    "lettuce": 2500,
}


def estimate_weight(weight_class: str) -> int:
    """Return estimated weight in grams from coarse weight class."""
    return WEIGHT_ESTIMATION.get(norm_key_part(weight_class, "medium"), 800)


def estimate_volume(packaging: str, weight_class: str, display_name: str = "") -> int:
    """Return estimated volume in cc from packaging type and weight class.

    Name-driven overrides apply first (e.g. bread, toilet paper) to handle
    large-footprint items that deviate from the packaging/weight lookup.
    """
    text = str(display_name or "").lower()
    for token, volume in NAME_VOLUME_OVERRIDES.items():
        if token in text:
            return volume
    key = (norm_key_part(packaging, "other"), norm_key_part(weight_class, "medium"))
    if key in VOLUME_ESTIMATION:
        return VOLUME_ESTIMATION[key]
    return DEFAULT_VOLUME.get(key[1], 700)


# --- crush_score v2: crushability, not breakability (DEFAULT since 2026-07-28)
#
# Set VMT_CRUSH_V1=1 to restore the old mapping for historical reproduction.
#
# The v1 derivation below opens with `if fragile: return 9`, which imports
# BREAKABILITY into a field whose reference data encodes CRUSHABILITY under
# static load. The collaborator's own item_properties.json settles the
# semantics: `Loaf of Bread` and `Bag of Potato Chips` score 10 (neither is
# breakable, both crush), `Can of Soup` and `Bleach` score 1 (the crushers).
# No glass item appears in their reference set at all.
#
# Two consequences of v1, both live on the robot catalog:
#   * its (rigidity, weight_class) table maxes at 7 while the solver's fragile
#     threshold is 8, so the crushability channel can NEVER grant protection --
#     100% of it comes from `fragile`;
#   * 4 rigid crockery SKUs (wine cup, cup, bowl, plate) get max protection
#     despite being load-bearing, while genuinely crushable boxes/bags get none.
#
# Operational definition (Diogo + Jacopo, 2026-07-28): crush_score answers
# "how much load may be placed ON TOP of this item?" -- 10 = nothing or only
# very light items, 1 = anything may go on top. Three distinct physical causes
# raise it, and the schema discriminates all three:
#
#   permanent deformation  fresh produce bruises, bread/crisps squash  -> group
#   breakage under load    open drinkware and stemware                 -> packaging
#   (elastic deformation)  sponges, rubber balls: recover, NO damage   -> group
#
# The elastic case is why `rigidity` alone cannot carry this: a banana and a
# kitchen sponge are both loose+soft, but squashing the sponge costs nothing.
# `group` is what separates them -- consistent with the E3 finding that
# group-keyed rules transfer where packaging-keyed ones do not.
CRUSHABLE_FOOD_GROUPS = {"produce", "bakery"}
# Edible contents -> a `cup` is a sealed food pot, not open drinkware.
FOOD_GROUPS = {"produce", "bakery", "snack", "pantry", "dairy", "drink",
               "frozen", "raw_meat", "seafood"}
ELASTIC_NONFOOD_GROUPS = {"household", "other", "pet", "hygiene"}
LOAD_BEARING_PACKAGING = {"bottle", "jar", "can", "aerosol"}
SEMI_RIGID_PACKAGING = {"box", "carton", "tray", "tub", "wrap", "tube"}


def compute_crush_score_v2(packaging: str, rigidity: str, display_name: str = "",
                           group: str = "", deformation: str = "") -> int:
    """Load tolerance on the collaborator's 1-10 scale (high = keep it on top).

    Ordered rules, first match wins. Divergence to raise with Jacopo: the
    collaborator's reference set scores `Apples` 5 and `Lettuce` 8, i.e. it
    treats firm produce as load-bearing. Bruising makes produce unsaleable, so
    all produce here sits at or above the protection threshold, graded by
    rigidity (soft 10 > semi 8) to keep their lettuce/apple ordering.
    """
    pkg = norm_key_part(packaging, "other")
    rig = norm_key_part(rigidity, "semi")
    grp = norm_key_part(group, "other")
    name = str(display_name or "").lower()
    dfm = str(deformation or "").strip().lower()

    # `deformation` answers the one question this function cannot otherwise
    # ask: does a load DAMAGE the item, or does it squash and recover? Only
    # `elastic` overrides. `plastic` is the default path anyway, and `brittle`
    # deliberately does NOT override: brittleness alone does not determine load
    # tolerance -- a glass bottle and a plate are both brittle and both stack
    # fine, while a wine glass fails because of its GEOMETRY, which `packaging`
    # already encodes (`cup`). Measured: forcing brittle -> 8 sends `plate` and
    # `bowl` from 2 back to 8, re-creating the v1 over-protection bug.
    if dfm == "elastic":
        return 5

    if "egg" in name:                       # crushes; matches reference value 10
        return 9
    if grp in CRUSHABLE_FOOD_GROUPS:        # bruises/squashes -> must ride on top
        return {"soft": 10, "semi": 8}.get(rig, 7)
    # Open drinkware/stemware takes no load. Must be group-conditional: `cup`
    # means reusable glass/ceramic drinkware in the robot catalog, but a SEALED
    # noodle pot / pudding cup / instant-drink cup in RPC (10 SKUs), which bears
    # load fine. Keying on packaging alone over-protected all 10 -- the
    # packaging-keyed non-transfer problem (E3) recurring inside this function.
    if pkg == "cup" and grp not in FOOD_GROUPS:
        return 8
    if grp == "snack" and pkg in {"bag", "pouch", "wrap"}:
        return 9                            # crisps: reference value 10
    if pkg in {"bag", "pouch"}:
        return 8
    if pkg == "loose" and rig == "soft" and grp in ELASTIC_NONFOOD_GROUPS:
        return 5        # bare sponge/ball/toy: deforms elastically, no damage done
    if pkg in LOAD_BEARING_PACKAGING:
        return 2
    if pkg in SEMI_RIGID_PACKAGING:
        if rig == "rigid":
            return 3                        # stiff when full, e.g. a milk carton
        return 6 if grp == "snack" else 5   # cracker box crushes; frozen pizza does not
    if rig == "rigid":
        return 2
    return 4


def compute_crush_score(fragile: bool, rigidity: str, weight_class: str,
                        packaging: str = "", display_name: str = "",
                        group: str = "", deformation: str = "") -> int:
    """Map perception attributes to PACKBOT CP crush_score semantics.

    Important: PACKBOT CP treats HIGH scores as fragile and LOW scores as
    heavy/sturdy. The solver thresholds are:
    - fragile: score >= 8
    - heavy/crush-causing: score <= 2

    v2 (load tolerance) is the DEFAULT since 2026-07-28. `VMT_CRUSH_V1=1`
    restores the pre-2026-07-28 breakability mapping, which is needed only to
    reproduce historical numbers -- see Settled Decision 7 in CLAUDE.md.
    """
    if os.environ.get("VMT_CRUSH_V1") != "1":
        return compute_crush_score_v2(packaging, rigidity, display_name, group, deformation)
    return compute_crush_score_v1(fragile, rigidity, weight_class)


def compute_crush_score_v1(fragile: bool, rigidity: str, weight_class: str) -> int:
    """Pre-2026-07-28 breakability mapping. Kept callable so comparisons can
    reference it directly rather than via the env-gated dispatcher, which
    silently returns v2 now that v2 is the default."""
    if fragile:
        return 9

    score_table = {
        ("soft", "light"): 7,
        ("soft", "medium"): 6,
        ("soft", "heavy"): 4,
        ("semi", "light"): 6,
        ("semi", "medium"): 5,
        ("semi", "heavy"): 3,
        ("rigid", "light"): 5,
        ("rigid", "medium"): 3,
        ("rigid", "heavy"): 1,
    }
    return score_table.get((norm_key_part(rigidity, "semi"), norm_key_part(weight_class, "medium")), 5)


def map_category(group: str, edible: bool, hazard_class: Sequence[str] = (), packaging: str = "", display_name: str = "") -> str:
    """Map perception group/edibility to PACKBOT CP category string.

    Priority: raw_meat/seafood → Cleaning (non-edible) → group-based food mapping.
    Household items are Cleaning only when packaging/name suggests chemical products.

    hazard_class parameter is retained for backward-compat only (deprecated 2026-05-12,
    decision 10). Field is empty in all current GT and predicted output; the
    Raw Meat branch now relies solely on `group in {raw_meat, seafood}`.
    """
    group_norm = norm_key_part(group, "other")
    packaging_norm = norm_key_part(packaging, "other")
    name_norm = str(display_name or "").lower()

    if group_norm in {"raw_meat", "seafood"}:
        return "Raw Meat"

    if not edible:
        # Tools/hardware get their own neutral category (not lumped with chemicals).
        # CP safety only isolates "Cleaning" and "Raw Meat"; "Hardware" is neutral.
        if group_norm == "hardware":
            return "Hardware"
        # A `household` item in a bottle was called a chemical until 2026-09-19.
        # That is a packaging-keyed derivation, the family the 200-SKU transfer
        # study showed does not survive a change of catalog, and it sat on a
        # safety path. A chemical is identified by what it IS -- group ==
        # cleaning -- which is the group-keyed form that does transfer.
        _ = (packaging_norm, name_norm)  # retained for signature compatibility
        # `hygiene` was mapped to Cleaning until 2026-09-19 and should not have
        # been: toothpaste does not contaminate food. It accounted for 51 of the
        # 76 cleaning-with-food violation pairs in the Gemini audit, inflating
        # that headline threefold. Pharmacy and batteries stay, being genuinely
        # hazardous to ingest.
        if _PLANNER_V1 and group_norm == "household" and (
            packaging_norm in {"bottle", "aerosol"}
            or any(tok in name_norm for tok in {"cleaner", "detergent", "bleach", "spray"})
        ):
            return "Cleaning"
        if group_norm in ({"cleaning", "hygiene", "pharmacy", "batteries"} if _PLANNER_V1
                          else {"cleaning", "pharmacy", "batteries"}):
            return "Cleaning"
        return "Household"

    mapping = {
        "dairy": "Dairy",
        "frozen": "Frozen",
        "bakery": "Bakery",
        "pantry": "Pantry",
        "snack": "Snacks",
        "produce": "Produce",
        "drink": "Pantry",
        "raw_meat": "Raw Meat",
        "seafood": "Raw Meat",
        "household": "Household",
        "cleaning": "Cleaning",
        "hygiene": "Cleaning" if _PLANNER_V1 else "Household",
    }
    return mapping.get(group_norm, "Pantry")


def map_temperature(cold_chain: bool, group: str) -> str:
    """Map cold_chain + group to PACKBOT temperature class (Ambient/Refrigerated/Frozen).

    group=frozen always maps to Frozen regardless of cold_chain.
    cold_chain=True maps to Refrigerated for all other groups.
    This is a known abstraction lossy point — boolean cold_chain cannot distinguish
    Refrigerated vs Frozen without the group field.
    """
    group_norm = norm_key_part(group, "other")
    if group_norm == "frozen":
        return "Frozen"
    if bool(cold_chain):
        return "Refrigerated"
    return "Ambient"


def _display_name(item: Dict[str, Any]) -> str:
    for key in ("free_name", "name", "canonical_name"):
        value = str(item.get(key, "") or "").strip()
        if value and value != "unknown_item":
            return value
    return "unknown item"


def _build_packbot_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Build a PACKBOT CP planner item dict from a perception item dict.

    Returns the Layer 3 planner interface fields: est_weight_g, est_volume_cc,
    crush_score, category, temperature. Also includes _source for traceability.
    """
    group = norm_key_part(item.get("group"), "other")
    packaging = norm_key_part(item.get("packaging"), "other")
    weight_class = norm_key_part(item.get("weight_class"), "medium")
    rigidity = norm_key_part(item.get("rigidity"), "semi")
    # Optional; "" (absent) means compute_crush_score_v2 falls back to the
    # group/packaging heuristic, so catalogs without it behave exactly as before.
    deformation = str(item.get("deformation") or "")
    fragile = bool(item.get("fragile", False))
    # dict.get's default only applies when the key is ABSENT, not when present-and-None
    # (e.g. an unknown_object entry explicitly sets edible: null) -- so this can't use
    # the same bool(item.get(k, default)) pattern as fragile/cold_chain above without
    # silently defaulting an unidentified item to non-edible.
    edible_raw = item.get("edible")
    edible = True if edible_raw is None else bool(edible_raw)
    cold_chain = bool(item.get("cold_chain", False))
    display_name = _display_name(item)

    # Prefer a measured/catalog gram value when present (robot-lab catwt path):
    # weight is supplied by identity, not the 200/800/2500 weight_class proxy.
    measured_g = item.get("measured_weight_g")
    if isinstance(measured_g, (int, float)) and measured_g > 0:
        est_weight_g = int(round(measured_g))
        weight_source = str(item.get("weight_source") or "catalog_measured")
    else:
        est_weight_g = estimate_weight(weight_class)
        weight_source = "weight_class_proxy"

    # Same for volume: a true/measured volume (YCB spec or scale) overrides the
    # packaging/weight proxy. Lets Jacopo's simulator use the true value.
    true_vol = item.get("true_volume_cc")
    if isinstance(true_vol, (int, float)) and true_vol > 0:
        est_volume_cc = int(round(true_vol))
        volume_source = str(item.get("volume_source") or "catalog_measured")
    else:
        est_volume_cc = estimate_volume(packaging, weight_class, display_name=display_name)
        volume_source = "packaging_proxy"

    return {
        "display_name": display_name,
        "est_weight_g": est_weight_g,
        "weight_source": weight_source,
        "est_volume_cc": est_volume_cc,
        "volume_source": volume_source,
        "crush_score": compute_crush_score(fragile, rigidity, weight_class,
                                           packaging=packaging, display_name=display_name,
                                           group=group, deformation=deformation),
        "category": map_category(group, edible, packaging=packaging, display_name=display_name),
        "temperature": map_temperature(cold_chain, group),
        "_source": {
            "group": group,
            "packaging": packaging,
            "weight_class": weight_class,
            "weight_source": weight_source,
            "volume_source": volume_source,
            "rigidity": rigidity,
            "fragile": fragile,
            "cold_chain": cold_chain,
            "edible": edible,
        },
    }


_PLANNER_V1 = __import__("os").environ.get("VMT_PLANNER_V1") == "1"

SPILL_VULNERABLE_CATEGORIES = {"Bakery", "Produce", "Snacks"}
# Derived from `group` rather than from `category` since 2026-09-19. The two
# agree by construction, but deriving a field from a derived field hides that
# the real criterion is what the object IS -- unwrapped or porous food that a
# leak ruins -- and group is where that lives.
SPILL_VULNERABLE_GROUPS = {"bakery", "produce", "snack"}


def _build_extended_packbot_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Like _build_packbot_item but adds the spill planner signal.

    New handoff files can provide `spill_risk` directly. Legacy perception
    outputs that still use `leak_risk`/`is_liquid` remain supported (2026-07-23
    collapse, see CLAUDE.md Settled Decision 13). `upright_required`/
    `orientation_sensitive` REMOVED 2026-07-23 -- Jacopo's planner never
    consumed it and it never fired in any audit (Settled Decision 14).
    """
    base = _build_packbot_item(item)

    spill_risk = bool(item.get("spill_risk", item.get("leak_risk") or item.get("is_liquid", False)))

    base["spill_risk"] = spill_risk
    base["edible"] = base["_source"]["edible"]
    base["spill_vulnerable"] = (
        base["category"] in SPILL_VULNERABLE_CATEGORIES if _PLANNER_V1
        else norm_key_part(item.get("group", ""), "other") in SPILL_VULNERABLE_GROUPS)
    base["_source"]["spill_risk"] = spill_risk
    return base


def _frame_index_from_path(path: Path) -> int:
    stem = path.stem
    if stem.startswith("frame_"):
        try:
            return int(stem.split("_", 1)[1])
        except Exception:
            return 0
    return 0


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_scene_frame_files(scene_dir: Path) -> List[Path]:
    frame_dir = scene_dir / "frames" if scene_dir.name != "frames" else scene_dir
    files = sorted(frame_dir.glob("frame_*.json"), key=_frame_index_from_path)
    return [p for p in files if p.is_file()]


def _packing_key(item: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        norm_key_part(item.get("group"), "other"),
        norm_key_part(item.get("packaging"), "other"),
        norm_key_part(item.get("weight_class"), "medium"),
    )


SIGNATURE_WIDTHS = {
    "narrow": ("group", "packaging", "weight_class"),
    "medium": ("group", "packaging", "weight_class", "rigidity"),
    # 8 fields since 2026-07-23 (was 9: leak_risk/is_liquid collapsed into
    # spill_risk -- they only ever diverged via an annotation bug, never a
    # meaningful distinction; see CLAUDE.md Settled Decision 13).
    "wide":   ("group", "packaging", "weight_class", "rigidity",
               "fragile", "spill_risk", "cold_chain", "edible"),
}
# Module-level switch. Defaults to "wide" (legacy 8-field behaviour).
# Override via tools/planner/run_packbot_cp.py --signature-width CLI flag
# or by setting this module attribute before calling adapt_scene_dir.
ACTIVE_SIGNATURE_WIDTH = "wide"


def set_signature_width(width: str) -> None:
    if width not in SIGNATURE_WIDTHS:
        raise ValueError(f"unknown signature width '{width}', expected one of {sorted(SIGNATURE_WIDTHS)}")
    global ACTIVE_SIGNATURE_WIDTH
    ACTIVE_SIGNATURE_WIDTH = width


def _signature_from_fields(item: Dict[str, Any], fields: Tuple[str, ...]) -> Tuple[Any, ...]:
    out = []
    for f in fields:
        if f == "group":
            out.append(norm_key_part(item.get("group"), "other"))
        elif f == "packaging":
            out.append(norm_key_part(item.get("packaging"), "other"))
        elif f == "weight_class":
            out.append(norm_key_part(item.get("weight_class"), "medium"))
        elif f == "rigidity":
            out.append(norm_key_part(item.get("rigidity"), "semi"))
        elif f == "edible":
            out.append(bool(item.get("edible", True)))
        elif f == "spill_risk":
            # collapses the old leak_risk/is_liquid pair (2026-07-23); fall
            # back to OR'ing the legacy fields for items that predate it.
            out.append(bool(item.get("spill_risk", item.get("leak_risk") or item.get("is_liquid"))))
        else:
            out.append(bool(item.get(f, False)))
    return tuple(out)


def _instance_signature(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """8-field (default) or narrower signature for sequential dedup.

    hazard_class element removed 2026-05-13 (decision 10). Field is empty in
    all current GT and produces empty tuple — removal is mathematically
    equivalent on current data. leak_risk/is_liquid collapsed into spill_risk
    2026-07-23 (was 9 fields), see CLAUDE.md Settled Decision 13.
    """
    return _signature_from_fields(item, SIGNATURE_WIDTHS[ACTIVE_SIGNATURE_WIDTH])


def _gt_instance_signature(item: Dict[str, Any]) -> Tuple[Any, ...]:
    """Stable GT signature for packable instance reconstruction.

    Adds `name` as a prefix to the active-width signature so GT items with
    distinct canonical names but identical Layer 1/2 attributes are
    distinguishable (e.g. two different snack bags with same group+packaging).
    hazard_class removed 2026-05-13.
    """
    base = _signature_from_fields(item, SIGNATURE_WIDTHS[ACTIVE_SIGNATURE_WIDTH])
    return (norm_key_part(item.get("name"), "unknown_item"),) + base


def _new_item_id(counter: int) -> str:
    return f"item_{counter:03d}"


def adapt_single_frame_data(frame_data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    items_data: Dict[str, Dict[str, Any]] = {}
    arrival_order: List[str] = []
    counter = 0

    for item in frame_data.get("items", []):
        if not isinstance(item, dict):
            continue
        qty = norm_qty(item.get("quantity", 1))
        for _ in range(qty):
            item_id = _new_item_id(counter)
            counter += 1
            items_data[item_id] = _build_packbot_item(item)
            arrival_order.append(item_id)

    meta = {
        "mode": "single_frame",
        "item_instances": len(items_data),
        "source_items": len(frame_data.get("items", [])),
    }
    return items_data, arrival_order, meta


def adapt_single_frame_path(frame_path: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    return adapt_single_frame_data(_load_json(frame_path))


def _build_predicted_instances_from_scene(scene_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[Any, ...], Deque[str]], List[str], Dict[str, Any]]:
    items_data: Dict[str, Dict[str, Any]] = {}
    ids_by_sig: Dict[Tuple[Any, ...], Deque[str]] = defaultdict(deque)
    first_seen_order: List[str] = []
    seen_count_by_sig: Dict[Tuple[Any, ...], int] = defaultdict(int)
    counter = 0
    frame_files = _iter_scene_frame_files(scene_dir)

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
                items_data[item_id] = _build_packbot_item(item)
                ids_by_sig[sig].append(item_id)
                first_seen_order.append(item_id)
            seen_count_by_sig[sig] = qty

    meta = {
        "scene_dir": str(scene_dir),
        "frame_count": len(frame_files),
        "item_instances": len(items_data),
        "signature_count": len(ids_by_sig),
    }
    return items_data, ids_by_sig, first_seen_order, meta


def adapt_scene_dir(scene_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    items_data, _, first_seen_order, meta = _build_predicted_instances_from_scene(scene_dir)
    meta = dict(meta)
    meta["mode"] = "scene_dir"
    meta["arrival_strategy"] = "first_seen_predicted_signature"
    return items_data, list(first_seen_order), meta


def adapt_gt_scene_dir(gt_scene_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    """Build planner input directly from GT packable progression."""
    items_data: Dict[str, Dict[str, Any]] = {}
    arrival_order: List[str] = []
    seen_true_counts: Dict[Tuple[Any, ...], int] = defaultdict(int)
    counter = 0
    frame_files = _iter_scene_frame_files(gt_scene_dir)

    for frame_path in frame_files:
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
            for _ in range(qty - prev):
                item_id = _new_item_id(counter)
                counter += 1
                items_data[item_id] = _build_packbot_item(exemplar)
                arrival_order.append(item_id)
            seen_true_counts[sig] = qty

    meta = {
        "mode": "gt_scene_dir",
        "scene_dir": str(gt_scene_dir),
        "frame_count": len(frame_files),
        "item_instances": len(items_data),
        "arrival_strategy": "gt_packable_now_progression",
    }
    return items_data, arrival_order, meta


def _gt_packable_arrival_keys(gt_scene_dir: Path) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
    seen_true_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
    arrival_keys: List[Tuple[str, str, str]] = []
    frame_files = _iter_scene_frame_files(gt_scene_dir)

    for frame_path in frame_files:
        frame_data = _load_json(frame_path)
        current_true_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        for item in frame_data.get("items", []):
            if not isinstance(item, dict):
                continue
            if bool(item.get("packable_now", False)):
                current_true_counts[_packing_key(item)] += norm_qty(item.get("quantity", 1))

        for key, qty in current_true_counts.items():
            prev = seen_true_counts[key]
            if qty > prev:
                arrival_keys.extend([key] * (qty - prev))
                seen_true_counts[key] = qty

    return arrival_keys, {"gt_frame_count": len(frame_files), "gt_arrival_events": len(arrival_keys)}


def adapt_scene_dir_with_oracle_order(pred_scene_dir: Path, gt_scene_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    items_data, ids_by_sig, first_seen_order, pred_meta = _build_predicted_instances_from_scene(pred_scene_dir)
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


def adapt_input(
    *,
    mode: str,
    input_path: Path,
    gt_scene_dir: Optional[Path] = None,
) -> Tuple[Dict[str, Dict[str, Any]], List[str], Dict[str, Any]]:
    mode = str(mode)
    if mode == "single_frame":
        return adapt_single_frame_path(input_path)
    if mode == "scene_dir":
        return adapt_scene_dir(input_path)
    if mode == "gt_scene_dir":
        return adapt_gt_scene_dir(input_path)
    if mode == "oracle_order":
        if gt_scene_dir is None:
            raise ValueError("oracle_order mode requires --gt-scene-dir")
        return adapt_scene_dir_with_oracle_order(input_path, gt_scene_dir)
    raise ValueError(f"Unsupported mode: {mode}")


def _build_output_json(items_data: Dict[str, Dict[str, Any]], arrival_order: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "items_data": items_data,
        "arrival_order": arrival_order,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt perception outputs into PACKBOT CP input format.")
    parser.add_argument(
        "--mode",
        choices=["single_frame", "scene_dir", "gt_scene_dir", "oracle_order"],
        required=True,
        help="Input interpretation mode.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Frame JSON path for single_frame, or scene directory path for scene_dir/oracle_order.",
    )
    parser.add_argument(
        "--gt-scene-dir",
        help="Required for oracle_order mode: GT scene directory containing frames/*.json.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Where to write the combined PACKBOT input JSON.",
    )
    parser.add_argument(
        "--signature-width",
        choices=sorted(SIGNATURE_WIDTHS.keys()),
        default="wide",
        help="Item dedup signature width. 'narrow' = (group, packaging, weight_class) — "
             "less drift-prone; 'medium' adds rigidity; 'wide' (default) is the legacy "
             "9-field signature including Layer 2 booleans. See "
             "docs/PIPELINE_REVIEW_2026-05-13.md §P3.3.",
    )
    args = parser.parse_args()

    set_signature_width(args.signature_width)

    input_path = Path(args.input)
    gt_scene_dir = Path(args.gt_scene_dir) if args.gt_scene_dir else None
    items_data, arrival_order, metadata = adapt_input(mode=args.mode, input_path=input_path, gt_scene_dir=gt_scene_dir)
    metadata = dict(metadata)
    metadata["signature_width"] = args.signature_width

    out = _build_output_json(items_data, arrival_order, metadata)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "output_json": str(out_path),
        "mode": args.mode,
        "item_instances": len(items_data),
        "arrival_len": len(arrival_order),
        "metadata": metadata,
    }, indent=2))


if __name__ == "__main__":
    main()

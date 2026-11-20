# Inventory Schema Reference

This document describes every field in the output produced by `perception_static.py`
and `perception_sequential.py`. Fields are organised by the pipeline layer that
produces them.

---

## Output Structure

```json
{
  "items": {
    "<display_name>": { ...item fields... },
    "<display_name>": { ...item fields... }
  },
  "arrival_order": ["<display_name>", "<display_name>", ...]
}
```

- **`items`** — dict keyed by item display name. Items with `quantity > 1` appear
  as separate keys with a numeric suffix, e.g. `"Whole Milk (1)"`, `"Whole Milk (2)"`.
- **`arrival_order`** — list of display names in scan/arrival order. Items placed earlier
  on the conveyor come first; they will be packed at the bag bottom. The CP-SAT solver
  uses this to enforce crush and orientation constraints.

---

## Layer 1 — Direct VLM Targets

These fields are predicted by the VLM (Qwen3-VL) directly from the image.
They are inputs to the planner_rules derivation and are **not** present in the
final planner output dict, but they drive every field that is.

| Field | Type | Description |
|-------|------|-------------|
| `group` | string | Item food/product category. One of: `dairy`, `frozen`, `bakery`, `produce`, `snack`, `pantry`, `drink`, `raw_meat`, `seafood`, `cleaning`, `hygiene`, `household`. Drives `category` and `temperature` in Layer 3. |
| `packaging` | string | Container type. One of: `bottle`, `can`, `carton`, `jar`, `bag`, `box`, `tray`, `tub`, `cup`, `wrap`, `pouch`, `aerosol`, `loose`, `other`. Drives most Layer 2 derivations. |
| `quantity` | integer | Number of identical items visible in the frame. Minimum 1. Causes multiple entries in `arrival_order`. |
| `weight_class` | string | Coarse weight estimate. One of: `light` (~200 g), `medium` (~800 g), `heavy` (~2500 g). Used in `est_weight_g` and `crush_score`. |
| `cold_chain` | boolean | True if the item requires refrigeration or is frozen. Used in `temperature`. |

---

## Layer 2 — Deterministic Derivations (planner_rules_v1)

These fields are computed from Layer 1 by `_apply_planner_rules()` using
packaging-driven lookup tables. The VLM does not predict them directly —
rule-based derivation is more reliable than VLM prediction for these attributes
(packaging type is a strong proxy for physical behaviour).

| Field | Type | Description | Derivation |
|-------|------|-------------|------------|
| `fragile` | boolean | True if the item breaks or is easily damaged under direct pressure. | `jar` → always fragile. `carton` with "egg" in name → fragile. `packaging=bottle` → not fragile (rigid). All other rules lock this deterministically from packaging. |
| `leak_risk` | boolean | True if the item can spill or leak if tipped or punctured. | `bottle`, `carton` (non-egg), `tray` → True. `jar`, `can` → False. Drives `spill_risk` in Layer 3. |
| `is_liquid` | boolean | True if the item contains a pourable liquid. More specific than `leak_risk` — a leaky meat tray has `leak_risk=True` but `is_liquid=False`. | `bottle`, `carton` (non-egg) → True. All others → False. Also drives `spill_risk`. |
| `orientation_sensitive` | boolean | True if the item must stay upright to avoid leaking or spillage. | `bottle`, `carton`, `tray` → True. `jar`, `can`, `cup` → False. Used in Constraint B of the extended planner. |
| `edible` | boolean | True if the item is food or drink. | Derived from `group`: food groups → True, non-food groups (cleaning, hygiene, etc.) → False. Used in `category` mapping. |
| `rigidity` | string | Structural stiffness of the packaging. One of: `rigid`, `semi`, `soft`. | Derived from `packaging`: `bottle/can/jar/box/aerosol` → rigid; `carton/tray/tub/cup` → semi; `bag/wrap/pouch` → soft. Used in `crush_score`. |

---

## Layer 3 — Planner Interface (adapter-computed)

These are the fields that actually appear in the output dict and are consumed
by the CP-SAT packing solver. All are computed deterministically from Layers 1+2.

| Field | Type | Description | How it's computed |
|-------|------|-------------|-------------------|
| `est_weight_g` | integer | Estimated item weight in grams. | Lookup by `weight_class`: light=200g, medium=800g, heavy=2500g. |
| `est_volume_cc` | integer | Estimated item volume in cubic centimetres. | Lookup by `(packaging, weight_class)`. Name-based overrides for large-footprint items (bread=3000cc, paper towels=8000cc). |
| `crush_score` | integer | Fragility score for CP-SAT crush constraint. Range 1–10. **High = fragile (threshold ≥8). Low = heavy/sturdy (threshold ≤2).** | `fragile=True` → 9. Otherwise lookup by `(rigidity, weight_class)`: rigid/heavy → 1, soft/light → 7. |
| `category` | string | CP-SAT safety category. One of: `Dairy`, `Bakery`, `Produce`, `Snacks`, `Pantry`, `Frozen`, `Raw Meat`, `Cleaning`, `Household`. | Priority mapping: raw_meat/seafood → `Raw Meat`; non-edible → `Cleaning` or `Household`; food groups → category mapping. |
| `temperature` | string | Temperature zone. One of: `Ambient`, `Refrigerated`, `Frozen`. | `group=frozen` → `Frozen`. `cold_chain=True` → `Refrigerated`. Otherwise → `Ambient`. |
| `spill_risk` | boolean | True if the item is a spill hazard (`leak_risk OR is_liquid`). | Inputs to Constraint A in the extended planner: spill_risk items cannot share a bag with spill_vulnerable items. |
| `spill_vulnerable` | boolean | True if the item is damaged by moisture (`category` in `{Bakery, Produce, Snacks}`). | Paired with `spill_risk` for Constraint A. |
| `orientation_sensitive` | boolean | True if the item must stay upright (passed through from Layer 2). | Used in Constraint B: orientation_sensitive items cannot be below a heavy item in arrival order. |

---

## arrival_order

`arrival_order` is a sequence-level variable, not an item attribute. It is not
stored inside any item dict. It encodes the scan order (conveyor belt order) of
items as they pass the checkout. The CP-SAT solver uses it for two purposes:

1. **Crush prevention (baseline):** A fragile item arriving early (at the bottom)
   cannot share a bag with a heavy item arriving later (on top).
2. **Orientation protection (extended):** An orientation-sensitive item arriving
   early cannot share a bag with a heavy item arriving later.

In the static pipeline, `arrival_order` reflects the left-to-right / top-to-bottom
visual order within the single frame. In the sequential pipeline, it reflects the
temporal order across frames (first-seen frame determines position).

---

## Notes on Crush Score Convention

`crush_score` uses **inverted semantics** relative to plain English:

- `crush_score = 9` → item is **fragile** (eggs, bread, chips). Must not have
  heavy items placed on top.
- `crush_score = 1` → item is **heavy and sturdy** (canned goods, jugs). Safe
  to place under lighter items. Causes crush violations if placed on top of
  fragile items in the same bag.

This convention matches `rewardEllicitation/or_approach.py` and
`or_approach_extended.py` (thresholds: fragile ≥ 8, heavy ≤ 2).

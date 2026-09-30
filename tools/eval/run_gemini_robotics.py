#!/usr/bin/env python3
"""Gemini Robotics-ER baseline: off-the-shelf embodied VLM -> canonical run format.

Runs Gemini Robotics-ER (default 1.6-preview) as the paper's baseline perception
system. For each scene image it asks for a JSON list of graspable tabletop objects
with 2D boxes, then maps each detected name to the fixed-SKU attribute catalog (the
SAME Layer-3 lookup the detector uses), and writes the canonical evaluate_sequences
run format so it is scored identically to our own systems (score_name_f1.py +
evaluate_sequences.py).

API only -- runs anywhere with the key (no GPU). Key from .env (GEMINI_API_KEY).

  python tools/eval/run_gemini_robotics.py \
      --images datasets_perception/robot_lab_prod_44 \
      --out runs/robot_lab_prod_44_gemini_er
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
SKU = json.loads((ROOT / "data/robot_lab/robot_sku_attrs.json").read_text())
SKU = {k: v for k, v in SKU.items() if not str(k).startswith("_")}

# The attribute catalog holds 53 rows; the DEPLOYED detector knows 32. The extra
# 21 are loose single fruits and lab props, kept because the E1/E2/E3 transfer
# study was measured against that snapshot. Nothing can detect them now, so they
# are out of scope for scoring -- and leaving them in the matcher was not
# harmless. Two measured errors on the 150-scene benchmark:
#
#   "Jell-O strawberry gelatin box" -> strawberry      (x3)
#   "red apple"                     -> green apple     (x2)
#
# The first is the scoring rule biting itself: overlap is normalised by the
# SHORTER name, so a one-word catalog row scores a perfect 1.0 on a single token
# while the right answer, "gelatin dessert box", scores 0.5. The second is worse
# -- `_tokens` treats colour words as stopwords, so "red apple" and "green apple"
# both reduce to {apple} and become indistinguishable. Neither row exists in the
# deployed vocabulary, so restricting to it removes both without touching the
# scoring rule or the catalog file.
#
# VMT_FULL_CATALOG=1 restores all 53, for reproducing the E1/E2/E3 numbers.
def _deployed_vocab():
    for rel in ("robot/robot_pc_package/classes.json",):
        path = ROOT / rel
        if not path.exists():
            continue
        names = json.loads(path.read_text())
        names = names if isinstance(names, list) else names.get("names", names)
        return {str(n) for n in names}
    return None


if os.environ.get("VMT_FULL_CATALOG") != "1":
    _VOCAB = _deployed_vocab()
    if _VOCAB:
        _dropped = sorted(set(SKU) - _VOCAB)
        SKU = {k: v for k, v in SKU.items() if k in _VOCAB}
        if os.environ.get("VMT_VERBOSE_CATALOG") == "1":
            print(f"[catalog] {len(SKU)} deployed rows; {len(_dropped)} out of "
                  f"vocabulary and not matchable: {_dropped}")
ATTR_KEYS = list(next(iter(SKU.values())).keys())  # group, packaging, weight_class, cold_chain, ...

sys.path.insert(0, str(ROOT / "tools/pipeline"))
from apply_robot_catalog_weight import _tokens  # noqa: E402  (shared tokenizer + synonym map)

PROMPT = """This is a top-down photo of a robot work table. Detect EVERY distinct
graspable object resting on the table (grocery items and any other tabletop
object the robot may pick up). Ignore the table itself, robot arms/grippers,
floor, background equipment, and tape/markings.

Read any visible brand or product text and use it to identify the specific
product (e.g. "Cheez-It" -> cracker box, "SPAM" -> spam can, "Colgate" ->
toothpaste box). Do not guess a generic shape when the text is readable.

Return ONLY a JSON list, no markdown fences, no prose. Each element:
{"label": <specific name>, "quantity": <int, count of identical adjacent
  instances of this exact object>,
  "box_2d": [ymin,xmin,ymax,xmax] normalized 0-1000}
"""


# Same ask our own vision-language model gets in the direct path: identity plus
# the four characteristics it is prompted for, each against a closed vocabulary.
# Used to measure what the baseline predicts itself, rather than what the
# catalog supplies once its name has been looked up.
PROMPT_CHARACTERISTICS = PROMPT.rstrip().replace(
    '  "box_2d": [ymin,xmin,ymax,xmax] normalized 0-1000}',
    '  "box_2d": [ymin,xmin,ymax,xmax] normalized 0-1000,\n'
    '  "group": <one of: baby, bakery, batteries, cleaning, dairy, drink, frozen, hardware, household, hygiene, other, pantry, pet, pharmacy, produce, raw_meat, seafood, snack>,\n'
    '  "packaging": <one of: aerosol, bag, blister, bottle, box, can, carton, cup, jar, loose, other, pouch, tray, tub, tube, wrap>,\n'
    '  "weight_class": <one of: light, medium, heavy>,\n'
    '  "rigidity": <one of: soft, semi, rigid>,\n'
    '  "cold_chain": <true if the product needs refrigeration or freezing, else false>}'
) + """

rigidity is how much load the item carries without deforming: a bag or wrapped
pack is soft, a thin card box or plastic bottle is semi, a can, jar, glass or
thick rigid box is rigid.

Use exactly the vocabulary given for group, packaging, weight_class and
rigidity. Do not invent a value and do not leave a field out. Do not output
spill_risk, edible or crush_score: those are derived from the fields above.
"""

# Every characteristic the planner reads, asked of the model directly. The five
# above plus spill_risk, so a model-predicts-everything row can be compared with
# a catalog-lookup row on the SAME field set and no rule layer inside either.
#
# spill_risk gets its operational definition in the prompt, not just its name.
# Settled Decision 18 exists because this attribute was annotated twice under
# two different readings -- "is it sealed" and "does it contain liquid" -- and
# the disagreement presented as a rule failure. Asking for it by name alone
# would reproduce that at prediction time.
PROMPT_FULL_RECORD = PROMPT_CHARACTERISTICS.replace(
    '  "cold_chain": <true if the product needs refrigeration or freezing, else false>}',
    '  "cold_chain": <true if the product needs refrigeration or freezing, else false>,\n'
    '  "spill_risk": <true or false, see below>}'
).replace(
    "Do not output\nspill_risk, edible or crush_score: those are derived from the fields above.",
    "Do not output\nedible or crush_score: those are derived from the fields above.\n"
    "\n"
    "spill_risk: if the bag is tipped over, or the item is laid on its side, can\n"
    "the CONTENTS get out? TRUE needs BOTH: the contents are liquid or pourable,\n"
    "AND the container has a closure that opens and closes -- a screw cap, a\n"
    "spray head, a snap lid -- whether or not it has been opened yet. Everything\n"
    "else is false. A metal can is false even when it is full of liquid, because\n"
    "it must be destroyed to open: a can of soup is false. A sealed carton is\n"
    "false. A box, a bag or a wrapped pack is false, and a resealable lid on dry\n"
    "goods such as a crisp tube is still false. Loose produce, crockery and a\n"
    "sponge have no contents at all: false."
)


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


def _match_tokens(name):
    """Tokens worth matching on. Single characters are dropped: "Jell-O" splits
    into {jell, o}, and a stray one-letter token is overlap that means nothing."""
    return {t for t in _tokens(name) if len(t) > 1}


def _tiebreak_tokens(name):
    """Every word, stopwords included, for separating candidates that TIE.

    `_tokens` drops colour words, so "bag of red apples" and "bag of green
    apples" -- both real products in the deployed vocabulary -- reduce to the
    same {apples} and the winner is decided by dictionary order. Measured: a
    predicted "bag of red apples" resolved to the GREEN one.

    Colours cannot simply be un-stopworded: "mug" is a synonym for "cup" and
    "cup" is itself a stopword, so "red mug" would reduce to {red} and match
    "bag of red apples" outright. Keeping them for the tie-break only means the
    ordinary score is unchanged and this can never create a match, only choose
    between equals.
    """
    return _tokens(name, keep_stopwords=True)


_SKU_INDEX = [(_match_tokens(name), _tiebreak_tokens(name), name, attrs)
              for name, attrs in SKU.items()]


def full_catalog_match(name: str, min_overlap: float = 0.34):
    """Which catalog product this name selects, or None. Same rule as
    full_catalog_attrs, exposed so a scorer can ask whether a prediction
    resolves to the RIGHT product rather than merely to some product."""
    toks = _match_tokens(name)
    if not toks:
        key = name if name in SKU else str(name or "").strip().lower()
        return key if key in SKU else None
    full = _tiebreak_tokens(name)
    best, best_score, best_tie = None, 0.0, -1
    for ctoks, cfull, cname, _ in _SKU_INDEX:
        if not ctoks:
            continue
        inter = len(toks & ctoks)
        if inter == 0:
            continue
        score = inter / min(len(toks), len(ctoks))
        tie = len(full & cfull)
        # A remaining tie is broken by NAME, so the answer does not depend on the
        # order of a JSON file. Six real model outputs -- "Jell-O Chocolate",
        # "scrub sponge" and four like them -- tie on both the overlap and the
        # full-name test, and two implementations of this rule disagreed purely
        # because they iterated the catalog in different orders.
        better = (score, tie, best is None or cname < best)
        if score > best_score or (score == best_score and (
                tie > best_tie or (tie == best_tie and cname < best))):
            best, best_score, best_tie = cname, score, tie
    return best if best_score >= min_overlap else None


def full_catalog_attrs(name: str, min_overlap: float = 0.34) -> dict:
    """Closed-set lookup: every non-identity field (group, packaging,
    weight_class, cold_chain, fragile, edible, spill_risk, rigidity,
    orientation_sensitive, leak_risk, is_liquid) comes from the 32-item
    catalog indexed by identity, never predicted from the image (matches
    Section IV-C: identity is perceived, attributes are looked up). Gemini's
    names are free-form, so matching uses the same token-overlap + synonym
    matcher apply_robot_catalog_weight.py uses for weight alone, generalized
    to the full attribute set. An object whose name matches nothing in the
    catalog is a real lab prop outside scope, not a bug -- it correctly gets
    all-None attrs."""
    toks = _match_tokens(name)
    if not toks:
        # Every word was a container word, so token overlap has nothing to work
        # with. That is the right answer for a vague "can" or "bottle", but the
        # catalog also holds a product whose whole name is one of those words:
        # the detector reports "cup", correctly, and the module returned an
        # empty row for it, so a correctly identified cup reached the planner
        # with no characteristics at all. An exact name is an exact match, and
        # it cannot let a vague name back in: "can" is not a product.
        exact = SKU.get(name if name in SKU else str(name or "").strip().lower())
        return dict(exact) if exact else {k: None for k in ATTR_KEYS}
    full = _tiebreak_tokens(name)
    best, best_name, best_score, best_tie = None, "\uffff", 0.0, -1
    for ctoks, cfull, _cname, attrs in _SKU_INDEX:
        if not toks or not ctoks:
            continue
        inter = len(toks & ctoks)
        if inter == 0:
            continue
        score = inter / min(len(toks), len(ctoks))
        tie = len(full & cfull)
        if score > best_score or (score == best_score and (
                tie > best_tie or (tie == best_tie and _cname < best_name))):
            best, best_name, best_score, best_tie = attrs, _cname, score, tie
    if best is not None and best_score >= min_overlap:
        return dict(best)
    return {k: None for k in ATTR_KEYS}


def parse_objects(text: str):
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        data = json.loads(t)
    except Exception:  # noqa: BLE001
        m = re.search(r"\[.*\]", t, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:  # noqa: BLE001
            return []
    return data if isinstance(data, list) else []


def main() -> int:
    import google.generativeai as genai
    import PIL.Image

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True, help="root with scene_*/frames/frame_000.png")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="gemini-robotics-er-1.6-preview")
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--sleep", type=float, default=1.0, help="throttle between scenes (s)")
    ap.add_argument("--resume", action="store_true",
                     help="skip scenes whose output already has >=1 detected item")
    ap.add_argument("--emit-full-record", action="store_true",
                    help="ask for every characteristic the planner reads, "
                         "spill_risk included, so a model-predicts-everything "
                         "row is comparable with a catalog-lookup row")
    ap.add_argument("--emit-characteristics", action="store_true",
                     help="ask the model for group/packaging/weight_class/cold_chain "
                          "and keep its answers instead of the catalog lookup")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY",
                     help="which .env var to read the key from (use a different account's key to get a fresh quota bucket)")
    args = ap.parse_args()

    key = load_key(args.api_key_env)
    genai.configure(api_key=key)
    model = genai.GenerativeModel(args.model)

    scenes = sorted(p for p in args.images.iterdir() if p.is_dir() and (p / "frames/frame_000.png").exists())
    n = 0; total_items = 0; t0 = time.time()
    for sc in scenes:
        if args.resume:
            existing = args.out / sc.name / "frames" / "frame_000.json"
            # File existing means a real model response was already written (the
            # quota-exhaustion path above returns before writing anything), so an
            # empty item list here is a genuine "nothing detected", not a masked
            # failure -- safe to skip rather than re-asking every relaunch.
            if existing.exists():
                continue
        png = sc / "frames/frame_000.png"
        im = PIL.Image.open(png)
        objs = None
        for attempt in range(args.max_retries):
            try:
                r = model.generate_content(
                    [PROMPT_FULL_RECORD if args.emit_full_record
                     else PROMPT_CHARACTERISTICS if args.emit_characteristics
                     else PROMPT, im])
                objs = parse_objects(r.text)
                break
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                if "429" in msg or "ResourceExhausted" in type(e).__name__ or "quota" in msg.lower():
                    # Daily quota is gone for this key -- retrying just burns more of
                    # it for nothing, and so would moving on to the next scene.
                    print(f"  {sc.name}: quota exhausted on {args.api_key_env} ({msg[:120]})")
                    print(f"[STOP] {n} scenes written this run, {total_items} items -> {args.out}. "
                          f"Switch --api-key-env or wait for reset, then rerun with --resume.")
                    return 1
                wait = 2 ** attempt
                print(f"  {sc.name} attempt {attempt+1} failed ({msg[:80]}), retry in {wait}s")
                time.sleep(wait)
        if objs is None:
            print(f"  {sc.name}: no response after {args.max_retries} attempts, skipping")
            continue
        items = []
        for o in objs:
            name = str(o.get("label", "")).strip()
            if not name:
                continue
            box = o.get("box_2d") or [0, 0, 0, 0]
            # normalized [ymin,xmin,ymax,xmax] 0-1000 -> pixel [x1,y1,x2,y2]
            try:
                ymin, xmin, ymax, xmax = box
                px = [round(xmin / 1000 * im.width, 1), round(ymin / 1000 * im.height, 1),
                      round(xmax / 1000 * im.width, 1), round(ymax / 1000 * im.height, 1)]
            except Exception:  # noqa: BLE001
                px = None
            qty = o.get("quantity", 1)
            try:
                qty = max(1, int(qty))
            except (TypeError, ValueError):
                qty = 1
            row = {"name": name, "name_confidence": None, "bbox_2d": px, "quantity": qty}
            # Whatever the model was ASKED for is kept as the model answered it.
            # Falling through to the catalog here would silently turn a
            # model-predicts-everything run into a lookup run: the prompt asks,
            # the answer is parsed, and then it is overwritten by the row the
            # name resolves to. That is the whole distinction the row exists to
            # measure, so it has to follow the prompt flags, not just one.
            if args.emit_characteristics or args.emit_full_record:
                for f in ("group", "packaging", "weight_class", "rigidity"):
                    v = o.get(f)
                    row[f] = str(v).strip().lower() if v is not None else None
                row["cold_chain"] = bool(o.get("cold_chain")) if o.get("cold_chain") is not None else None
                if args.emit_full_record:
                    sr = o.get("spill_risk")
                    row["spill_risk"] = bool(sr) if sr is not None else None
            else:
                row.update(full_catalog_attrs(name))
            items.append(row)
        od = args.out / sc.name / "frames"; od.mkdir(parents=True, exist_ok=True)
        (od / "frame_000.json").write_text(json.dumps({"items": items, "metadata": {"model": args.model}}, indent=1, ensure_ascii=False) + "\n")
        n += 1; total_items += len(items)
        print(f"  {sc.name}: {len(items)} objects")
        time.sleep(args.sleep)
    print(f"gemini-er ({args.model}): {n} scenes, {total_items} objects -> {args.out}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

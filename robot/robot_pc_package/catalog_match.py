#!/usr/bin/env python3
"""Free-text product name -> one row of objects_lookup.json, or nothing.

The closed-set detector emits catalog names, so `build_detections` can look a
row up by exact key. A vision-language model emits whatever it likes -- "Domino
sugar box", "Cheez-It cracker box", "a metal can" -- and something has to decide
which product, if any, that denotes.

This is a PORT of the matcher used to score the offline experiments
(`tools/eval/run_gemini_robotics.full_catalog_match`), kept here so the robot
package stays self-contained on a machine that has no copy of the repository.
`test_catalog_match.py` pins it against that original on real model output; if
the two ever disagree the test says so, because a second implementation of a
production rule is worth nothing unless something checks it is the same rule.

Four properties are load-bearing and each exists because of a measured failure:

  * Container words are stopwords. A prediction of "can" must select none of the
    four cans rather than an arbitrary one.
  * ...but a product whose whole NAME is a container word must still resolve. A
    correctly detected "cup" tokenised to nothing, matched no row, and reached
    the planner with every characteristic unset. An empty token set falls back
    to exact name equality, which no vague name can exploit.
  * Colour words are stopwords too, so "bag of red apples" and "bag of green
    apples" -- both real products -- reduce to the same tokens. Ties are broken
    on the full name INCLUDING stopwords. Colours cannot simply be un-stopworded:
    "mug" is a synonym of "cup" and "cup" is itself a stopword, so "red mug"
    would reduce to {red} and match the red apples outright.
  * Single characters are dropped: "Jell-O" splits into {jell, o}, and a stray
    one-letter token is overlap that means nothing.
"""
from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "box", "can", "bottle", "jar", "carton", "cup", "bag", "tube", "tub", "tray",
    "pouch", "wrap", "loose", "other", "the", "a", "of", "and",
    "red", "green", "yellow", "blue", "white", "orange-coloured",
}

SYNONYMS = {
    "drill": "screwdriver", "cordless": "screwdriver",
    "cheez": "cracker", "cheezit": "cracker", "cheezits": "cracker",
    "crackers": "cracker",
    "jello": "gelatin", "jell": "gelatin", "jelly": "gelatin",
    "domino": "sugar",
    "soda": "energy", "redbull": "energy", "bull": "energy", "monster": "energy",
    "coke": "cola", "sprite": "cola", "pop": "cola", "soft": "cola",
    "pringles": "potato", "chips": "potato", "chip": "potato", "crisps": "potato",
    "mug": "cup",
    "colgate": "toothpaste", "toothpastes": "toothpaste",
    "windex": "glass", "scrub": "glass", "windelx": "glass",
    "windshield": "glass", "washer": "glass",
    "salmon": "tuna", "sardine": "tuna",
    "eggs": "egg", "cherry": "cherries",
    "master": "coffee", "chef": "coffee",
    "mayo": "mustard", "mayonnaise": "mustard", "ketchup": "mustard",
}

MIN_OVERLAP = 0.34


def _tokens(name, keep_stopwords=False):
    out = set()
    for t in TOKEN_RE.findall(str(name or "").lower()):
        t = SYNONYMS.get(t, t)
        if keep_stopwords or t not in STOPWORDS:
            out.add(t)
    return out


def match_tokens(name):
    """Tokens worth matching on: stopwords and single characters removed."""
    return {t for t in _tokens(name) if len(t) > 1}


def tiebreak_tokens(name):
    """Every word, stopwords included, for separating candidates that TIE."""
    return _tokens(name, keep_stopwords=True)


def build_index(catalog, vocab=None):
    """Precompute the token sets. `catalog` is objects_lookup.json, whose
    private `_`-prefixed keys are defaults rather than products.

    `vocab` restricts matching to the products the deployed perception can
    actually name, and passing it is not optional bookkeeping. The lookup table
    carries rows kept for historical comparison -- loose single fruits among
    them -- which nothing can detect any more. Left in, they win matches they
    should not: overlap is normalised by the SHORTER name, so a one-word row
    scores a perfect 1.0 on a single shared token, and "Jell-O strawberry
    gelatin box" selects `strawberry` over `gelatin dessert box`.
    """
    keys = [k for k in catalog if not str(k).startswith("_")]
    if vocab is not None:
        allowed = set(vocab)
        keys = [k for k in keys if k in allowed]
    return [(match_tokens(k), tiebreak_tokens(k), k) for k in keys]


def match(name, catalog, index=None, min_overlap=MIN_OVERLAP, vocab=None):
    """The catalog key this name denotes, or None if it denotes no single one."""
    allowed = None if vocab is None else set(vocab)

    def usable(k):
        return (k in catalog and not str(k).startswith("_")
                and (allowed is None or k in allowed))

    toks = match_tokens(name)
    if not toks:
        if usable(name):
            return name
        key = str(name or "").strip().lower()
        return key if usable(key) else None

    idx = index if index is not None else build_index(catalog, vocab)
    full = tiebreak_tokens(name)
    best, best_score, best_tie = None, 0.0, -1
    for ctoks, cfull, cname in idx:
        if not ctoks:
            continue
        inter = len(toks & ctoks)
        if inter == 0:
            continue
        score = inter / float(min(len(toks), len(ctoks)))
        tie = len(full & cfull)
        # A remaining tie is broken by NAME, so the answer does not depend on the
        # order of a JSON file. Six real model outputs -- "Jell-O Chocolate",
        # "scrub sponge" and four like them -- tie on both the overlap and the
        # full-name test, and two implementations of this rule disagreed purely
        # because they iterated the catalog in different orders.
        if score > best_score or (score == best_score and (
                tie > best_tie or (tie == best_tie and cname < best))):
            best, best_score, best_tie = cname, score, tie
    return best if best_score >= min_overlap else None

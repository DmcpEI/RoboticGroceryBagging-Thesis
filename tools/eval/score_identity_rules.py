#!/usr/bin/env python3
"""Identification F1 under the three matching rules the paper reports.

  lenient     a prediction counts if it contains, is contained by, or shares a
              first word with the ground-truth name. The repo's historical rule.
  exact       the two normalized names are equal.
  resolvable  the pipeline's own catalog matcher resolves the predicted name to
              the ground-truth PRODUCT. This is what decides what the planner
              receives: a name resolving to nothing arrives with every
              characteristic unset, and a name resolving to the wrong product
              arrives with the wrong ones.

Lenient is reported because it is the rule this benchmark has always used, but
it should not be read as an accuracy. It counts a prediction correct when it
shares a first word with the truth, and the Robot catalog holds ten products
whose names begin with "bag of", so under that rule a bag of pears predicted
for a bag of limes is correct, as is a "red can" for a "red apple". Every
system gains from this, and a system that names by shape gains most.

The resolvable rule calls the SAME matcher the perception module uses to attach
characteristics, rather than a second string rule written for scoring, so the
column cannot drift from the behaviour it claims to describe.

A closed-set detector scores identically under all three, because it can only
emit catalog names. Nothing else has that property, which is the argument for a
closed output vocabulary.

  python tools/eval/score_identity_rules.py --gt datasets_seq_GT_b50 \
      --run "Closed-set detector=runs/b50_detector_v15b" \
      --run "VLM 32B + LoRA=runs/b50_vlm32b_rawattrs"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/eval"))
from run_gemini_robotics import full_catalog_match  # noqa: E402  the pipeline's matcher


def norm(s) -> str:
    return str(s or "").strip().lower()


def lenient(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a in b or b in a or a.split()[0] == b.split()[0])


def expand(items):
    out = []
    for it in items or []:
        n = it.get("name")
        # unknown_object is kept: it is an object reported to the planner with no
        # product, i.e. an unresolvable prediction, and whole-record scoring
        # charges it the same way.
        if n:
            out += [n] * max(1, int(it.get("quantity", 1) or 1))
    return out


def greedy(gt, pred, ok):
    used = [False] * len(pred)
    tp = 0
    for g in gt:
        for i, p in enumerate(pred):
            if not used[i] and ok(g, p):
                used[i] = True
                tp += 1
                break
    return tp, len(pred) - tp, len(gt) - tp


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    ap.add_argument("--catalog", type=Path,
                    default=ROOT / "data/robot_lab/robot_sku_attrs.json")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    catalog = [n.lower() for n in json.loads(args.catalog.read_text())]
    cache: dict[str, object] = {}

    def resolved(pred: str):
        """The catalog product the perception module would attach, or None."""
        k = norm(pred)
        if k not in cache:
            cache[k] = full_catalog_match(pred)
        return cache[k]

    rules = {"lenient": lambda g, p: lenient(g, p),
             "exact": lambda g, p: norm(g) == norm(p),
             "resolvable": lambda g, p: resolved(p) is not None
                                        and norm(resolved(p)) == norm(resolved(g) or g)}

    scenes = sorted(p.parent.parent.name
                    for p in args.gt.glob("*/frames/frame_000.json"))
    print(f"{len(scenes)} scenes, catalog of {len(catalog)} products\n")
    hdr = f"{'system':32s}" + "".join(f"{r:>12s}" for r in rules) + f"{'unresolvable':>14s}"
    print(hdr)
    print("-" * len(hdr))

    report = {"gt": str(args.gt), "scenes": len(scenes), "systems": {}}
    for spec in args.run:
        label, _, path = spec.partition("=")
        root = Path(path)
        tot = {r: [0, 0, 0] for r in rules}
        n_pred = n_unres = n_missing = 0
        for sc in scenes:
            fp = root / sc / "frames/frame_000.json"
            if not fp.exists():
                n_missing += 1
                continue
            gt = expand(json.loads((args.gt / sc / "frames/frame_000.json").read_text())["items"])
            pr = expand(json.loads(fp.read_text()).get("items", []))
            n_pred += len(pr)
            n_unres += sum(1 for p in pr if resolved(p) is None)
            for r, ok in rules.items():
                for i, v in enumerate(greedy(gt, pr, ok)):
                    tot[r][i] += v

        cells = ""
        row = {}
        for r in rules:
            p, rc, f = prf(*tot[r])
            cells += f"{f:12.3f}"
            row[r] = {"precision": round(p, 4), "recall": round(rc, 4), "f1": round(f, 4),
                      "tp": tot[r][0], "fp": tot[r][1], "fn": tot[r][2]}
        frac = n_unres / n_pred if n_pred else 0.0
        row["unresolvable_rate"] = round(frac, 4)
        row["predicted_objects"] = n_pred
        row["scenes_missing"] = n_missing
        report["systems"][label] = row
        note = f"   ({n_missing} scenes missing)" if n_missing else ""
        print(f"{label:32s}{cells}{frac:13.1%}{note}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n")
        print(f"\n[OK] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

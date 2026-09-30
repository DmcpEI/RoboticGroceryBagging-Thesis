#!/usr/bin/env python3
"""Compare detectors on b50 across TRAINING SEEDS, not single runs.

A single run's per-class F1 on this benchmark is not a reportable quantity.
Measured 2026-09-17 on three seeds of one model, same data, same
hyperparameters: the median per-class spread is 0.052 and the worst class
(`bag of yellow lemons`, 11 GT instances) spans 0.633. Pooled F1 is far more
stable -- 0.002 across two seeds of the shipped detector -- so a pooled
comparison of single runs is usually fine and a per-class one is not.

This scores every (model, seed) pair, groups them by model, and reports the
mean and the spread within each group. A class counts as separated only when
the two groups' ranges do not overlap at all, which is the weakest claim the
data supports with a handful of seeds. Anything else is reported as
overlapping rather than as a delta, because a delta invites a causal story
that the spread does not license.

  python tools/eval/compare_detectors_seeds.py \
      --model v15b runs/.../syn_v15b/weights/best.pt \
              runs/.../syn_v15b_seed1/weights/best.pt \
      --classes-for v15b data/robot_lab/synthetic_det_v15b/classes.json \
      --model v16b runs/.../syn_v16b/weights/best.pt \
              runs/.../syn_v16b_seed1/weights/best.pt \
              runs/.../syn_v16b_seed2/weights/best.pt \
      --classes-for v16b data/robot_lab/synthetic_det_v16b/classes.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def f1(counts) -> float:
    tp, fp, fn = counts
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def load_gt(gt_root: Path):
    gt = {}
    for d in sorted(gt_root.glob("*")):
        f = d / "frames/frame_000.json"
        if not f.exists():
            continue
        c = Counter()
        for i in json.loads(f.read_text())["items"]:
            c[i["name"]] += i.get("quantity", 1)
        gt[d.name] = c
    return gt


def score(weights: Path, classes: Path, gt, img_root: Path, conf: float):
    from ultralytics import YOLO
    names = json.loads(classes.read_text())
    model = YOLO(str(weights))
    tp, fp, fn = Counter(), Counter(), Counter()
    for scene, want in gt.items():
        img = img_root / scene / "color.png"
        if not img.exists():
            continue
        r = model(str(img), conf=conf, verbose=False)[0]
        got = Counter(names[int(c)] for c in (r.boxes.cls.tolist() if r.boxes is not None else []))
        for k in set(got) | set(want):
            hit = min(got[k], want[k])
            tp[k] += hit
            fp[k] += got[k] - hit
            fn[k] += want[k] - hit
    return {k: (tp[k], fp[k], fn[k]) for k in set(tp) | set(fp) | set(fn)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", nargs="+", action="append", metavar=("LABEL", "WEIGHTS"),
                    required=True, help="label followed by one weights path per seed")
    ap.add_argument("--classes-for", nargs=2, action="append", metavar=("LABEL", "JSON"),
                    required=True)
    ap.add_argument("--gt", type=Path, default=ROOT / "datasets_seq_GT_b50")
    ap.add_argument("--images", type=Path, default=ROOT / "data/robot_lab/rgbd")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    classes = {lab: Path(p) for lab, p in args.classes_for}
    gt = load_gt(args.gt)
    runs = defaultdict(list)
    for entry in args.model:
        lab, weights = entry[0], entry[1:]
        for w in weights:
            runs[lab].append(score(Path(w), classes[lab], gt, args.images, args.conf))

    print(f"{len(gt)} scenes, conf={args.conf}\n=== POOLED ===")
    pooled = {}
    for lab, seeds in runs.items():
        vals = []
        for per in seeds:
            t = sum(v[0] for v in per.values())
            p = sum(v[1] for v in per.values())
            n = sum(v[2] for v in per.values())
            vals.append(f1((t, p, n)))
        pooled[lab] = vals
        print(f"  {lab:10s} n={len(vals)}  mean {sum(vals)/len(vals):.4f}  "
              f"spread {max(vals)-min(vals):.4f}  [{', '.join(f'{v:.4f}' for v in vals)}]")
    labs = list(runs)
    if len(labs) == 2:
        a, b = (sum(pooled[l]) / len(pooled[l]) for l in labs)
        worst = max(max(pooled[l]) - min(pooled[l]) for l in labs)
        verdict = "INSIDE the seed spread -- not a measurable difference" if abs(b - a) < worst \
            else "larger than the seed spread"
        print(f"  difference {b - a:+.4f}  vs worst spread {worst:.4f}  -> {verdict}")

    keys = sorted(set().union(*[set(p) for seeds in runs.values() for p in seeds]))
    print("\n=== PER CLASS (mean over seeds; spread = max-min) ===")
    hdr = f"{'class':28s}{'GT':>5s}"
    for lab in labs:
        hdr += f"{lab:>9s}{'spr':>7s}"
    print(hdr + "  verdict")
    rows = []
    for k in keys:
        stats, gtn = {}, 0
        for lab in labs:
            vals = [f1(p.get(k, (0, 0, 0))) for p in runs[lab]]
            stats[lab] = vals
            c = runs[lab][0].get(k, (0, 0, 0))
            gtn = max(gtn, c[0] + c[2])
        if len(labs) == 2:
            A, B = stats[labs[0]], stats[labs[1]]
            sep = min(B) > max(A) or max(B) < min(A)
            delta = sum(B) / len(B) - sum(A) / len(A)
        else:
            sep, delta = False, 0.0
        rows.append((delta, k, gtn, stats, sep))
    rows.sort()
    for delta, k, gtn, stats, sep in rows:
        line = f"{k:28s}{gtn:5d}"
        for lab in labs:
            v = stats[lab]
            line += f"{sum(v)/len(v):9.3f}{max(v)-min(v):7.3f}"
        print(f"{line}  {'SEPARATED' if sep else 'ranges overlap'}")

    for lab in labs:
        sp = [max(v) - min(v) for _, _, _, s, _ in rows for lab2, v in s.items() if lab2 == lab]
        sp.sort()
        if len(runs[lab]) > 1:
            print(f"\n{lab} per-class spread over {len(runs[lab])} seeds: "
                  f"median {sp[len(sp)//2]:.3f}  mean {sum(sp)/len(sp):.3f}  max {max(sp):.3f}")

    if args.json:
        args.json.write_text(json.dumps(
            {lab: [{k: list(v) for k, v in p.items()} for p in seeds]
             for lab, seeds in runs.items()}, indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

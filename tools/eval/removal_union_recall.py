#!/usr/bin/env python3
"""Single-frame vs sequential-union recall on the removal sequences.

The question this answers is not "is recall better with more frames" -- it is
always weakly better, because the union can only add. It is WHICH classes the
extra frames recover, and that separates two failure modes an aggregate score
cannot tell apart:

  occlusion  -- the object is there and hidden. Some frame in the sequence sees
                it, so the union recovers it. The sparse control (nothing
                touching) finds it too.
  confusion  -- the object is visible and named as something else. Every frame
                names it the same way, so the union recovers nothing, and the
                sparse control fails on it as well.

A class that fails in the sparse control is not an occlusion problem and no
amount of re-perception fixes it; it needs photographs. Reporting the union gain
without that split invites the opposite conclusion.

GT is free: `--removal` capture records `gt_objects` per step, derived by
removal differencing. Detection is the deployed detector at its deployed
confidence, run on the raw frame -- the same instrument as
compare_detectors_seeds.py, so numbers here sit beside the b50 ones.

  python tools/eval/removal_union_recall.py --json runs/eval/removal_union.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sequences(rgbd: Path):
    """{sequence_id: [(step, meta, image), ...]} ordered by step."""
    out = defaultdict(list)
    for d in sorted(rgbd.glob("rem[23]*_step*")):
        meta, img = d / "meta.json", d / "color.png"
        if not (meta.exists() and img.exists()):
            continue
        m = json.loads(meta.read_text())
        out[m.get("scene") or d.name.rsplit("_step", 1)[0]].append(
            (int(m.get("shot", 0)), Counter(m["gt_objects"]), img))
    for k in out:
        out[k].sort(key=lambda t: t[0])
    return dict(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rgbd", type=Path, default=ROOT / "data/robot_lab/rgbd")
    # NOT robot_pc_package/weights/best.pt at the repo root -- that copy is syn_v7_bags
    # (2026-08-31) and scores b50 at F1 0.800. The deployed detector is the one inside
    # the planner package, syn_v15b, which reproduces the published 0.886. Two files with
    # the same name and different models is how a whole analysis gets run on the wrong one.
    ap.add_argument("--weights", type=Path,
                    default=ROOT / "robot/robot_pc_package/weights/best.pt")
    ap.add_argument("--classes", type=Path, default=ROOT / "robot/robot_pc_package/classes.json")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--control", default="rem3_sparse",
                    help="prefix of the no-occlusion control sequences")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO
    names = json.loads(args.classes.read_text())
    model = YOLO(str(args.weights))
    # Say which model this is. "it ran without error" does not mean it ran on the
    # intended weights, and the file name cannot tell them apart.
    try:
        import torch
        ck = torch.load(args.weights, map_location="cpu", weights_only=False)
        print(f"model: {(ck.get('train_args') or {}).get('name', '?')}  "
              f"trained {str(ck.get('date'))[:10]}  {len(ck['model'].names)} classes")
    except Exception as exc:
        print(f"model: could not read training metadata ({type(exc).__name__})")
    seqs = sequences(args.rgbd)
    if not seqs:
        raise SystemExit(f"no rem2_/rem3_ sequences with meta.json + color.png under {args.rgbd}")

    # per sequence: first-frame / every-frame / union hits, and the same per class
    rows, cls = {}, defaultdict(lambda: Counter())
    for sid, steps in sorted(seqs.items()):
        first = steps[0][1]
        ever, allf_hit, allf_gt = Counter(), 0, 0
        first_hit, fp = Counter(), Counter()
        for i, (_, gt, img) in enumerate(steps):
            r = model(str(img), conf=args.conf, verbose=False)[0]
            got = Counter(names[int(c)] for c in
                          (r.boxes.cls.tolist() if r.boxes is not None else []))
            for c in set(gt) | set(got):
                hit = min(got[c], gt[c])
                ever[c] = max(ever[c], hit)       # best any single frame managed
                allf_hit += hit
                allf_gt += gt[c]
                fp[c] += got[c] - hit
                if i == 0:
                    first_hit[c] = hit
        rows[sid] = dict(steps=len(steps), n0=sum(first.values()),
                         first=sum(first_hit.values()), union=sum(ever.values()),
                         frame_hit=allf_hit, frame_gt=allf_gt)
        for c, n in first.items():
            cls[c]["n"] += n
            cls[c]["first"] += first_hit[c]
            cls[c]["union"] += ever[c]
            cls[c]["fp"] += fp[c]
            if sid.startswith(args.control):
                cls[c]["ctrl_n"] += n
                cls[c]["ctrl_hit"] += ever[c]

    def rate(a, b):
        return a / b if b else float("nan")

    dense = {k: v for k, v in rows.items() if not k.startswith(args.control)}
    ctrl = {k: v for k, v in rows.items() if k.startswith(args.control)}

    print(f"{len(rows)} sequences ({len(dense)} dense, {len(ctrl)} control), "
          f"{sum(v['steps'] for v in rows.values())} frames, conf={args.conf}\n")
    hdr = f"{'':22}{'seqs':>6}{'objs':>7}{'first frame':>13}{'union':>9}{'gain':>8}{'per frame':>11}"
    print(hdr)
    for label, grp in (("dense", dense), ("control (sparse)", ctrl), ("all", rows)):
        n0 = sum(v["n0"] for v in grp.values())
        f = rate(sum(v["first"] for v in grp.values()), n0)
        u = rate(sum(v["union"] for v in grp.values()), n0)
        pf = rate(sum(v["frame_hit"] for v in grp.values()),
                  sum(v["frame_gt"] for v in grp.values()))
        print(f"{label:22}{len(grp):6d}{n0:7d}{f:13.3f}{u:9.3f}{u - f:+8.3f}{pf:11.3f}")

    print(f"\n{'sequence':22}{'steps':>6}{'objs':>6}{'first':>8}{'union':>8}{'gain':>8}")
    for sid, v in sorted(rows.items(), key=lambda kv: -(rate(kv[1]['union'], kv[1]['n0'])
                                                        - rate(kv[1]['first'], kv[1]['n0']))):
        f, u = rate(v["first"], v["n0"]), rate(v["union"], v["n0"])
        print(f"{sid:22}{v['steps']:6d}{v['n0']:6d}{f:8.3f}{u:8.3f}{u - f:+8.3f}")

    # The safety row. A chemical that is never perceived never enters the inventory,
    # so the planner reports zero violations on a bag it had no way to know was unsafe.
    # On the retired rig this was the one family the sequence did NOT help: first
    # exposure and ever-found were both 0.667.
    HAZ = ("bleach bottle", "glass cleaner spray bottle", "mustard bottle")
    hn = sum(cls[c]["n"] for c in HAZ)
    hf = sum(cls[c]["first"] for c in HAZ)
    hu = sum(cls[c]["union"] for c in HAZ)
    print(f"\nhazard classes ({', '.join(HAZ)}): {hn} instances, "
          f"first frame {rate(hf, hn):.3f}, union {rate(hu, hn):.3f}")

    print(f"\n{'class':30}{'n':>4}{'first':>8}{'union':>8}{'gain':>8}{'sparse':>9}{'fp':>5}  verdict")
    out_cls = {}
    for c, v in sorted(cls.items(), key=lambda kv: rate(kv[1]["union"], kv[1]["n"])):
        f, u = rate(v["first"], v["n"]), rate(v["union"], v["n"])
        cn, ch = v["ctrl_n"], v["ctrl_hit"]
        s = rate(ch, cn)
        if v["n"] < 5:
            verdict = f"n={v['n']} -- too few to read"
        elif not cn:
            verdict = "no control"
        elif s < 0.5:
            # It was not hidden and was still missed, so no number of frames helps.
            # WHY it was missed is a separate question the control cannot answer, and
            # the false positives do: a class that is misnamed appears where it is not,
            # a class that is never proposed appears nowhere at all. The remedies
            # differ -- one needs the confusable pair separated, the other needs the
            # object to survive the cutout builder.
            verdict = ("NOT OCCLUSION, misnamed -- confusable with another class"
                       if v["fp"] else
                       "NOT OCCLUSION, never proposed -- no false positives either")
        elif u - f >= 0.10:
            verdict = "OCCLUSION -- the sequence recovers it"
        elif u < 0.9:
            verdict = "mixed"
        else:
            verdict = "fine"
        st = "  n/a" if not cn else f"{s:9.3f}"
        print(f"{c:30}{v['n']:4d}{f:8.3f}{u:8.3f}{u - f:+8.3f}{st}{v['fp']:5d}  {verdict}")
        out_cls[c] = dict(n=v["n"], first=f, union=u, fp=v["fp"],
                          control_n=cn, control=s, verdict=verdict)

    if args.json:
        args.json.write_text(json.dumps(
            dict(conf=args.conf, weights=str(args.weights),
                 sequences=rows, classes=out_cls), indent=1) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

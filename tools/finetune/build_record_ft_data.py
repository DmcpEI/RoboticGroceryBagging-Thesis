#!/usr/bin/env python3
"""Fine-tuning data for the vision-language model, from the detector's own sources.

Two kinds of example, both with targets read from the catalog:
  - single-product photographs: the captures the v18 cutout library was cut from,
    a few per product, each a one-item target;
  - composed scenes: v18 synthetic training scenes, whose generated labels give
    the products present (objects below the visibility threshold carry no label,
    so they are absent from the target as they are from the detector's).

  python tools/finetune/build_record_ft_data.py --out data/fine_tuning_record
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/pipeline"))
from record_prompt import FIELDS, PROMPT  # noqa: E402

LAB = ROOT / "data/robot_lab"


def record(name: str, attrs: dict, quantity: int) -> dict:
    a = attrs[name]
    r = {"name": name, "group": a["group"], "packaging": a["packaging"], "quantity": quantity,
         "weight_class": a["weight_class"], "rigidity": a["rigidity"],
         "cold_chain": bool(a["cold_chain"]), "spill_risk": bool(a["spill_risk"])}
    assert tuple(r) == FIELDS
    return r


def row(rid: str, split: str, source: str, image: Path, items: list, meta: dict) -> dict:
    target = {"items": items}
    return {"id": rid, "split": split, "source": source,
            "image_path": str(image.relative_to(ROOT)), "prompt": PROMPT, "target": target,
            "target_text": json.dumps(target, separators=(",", ":")), "metadata": meta}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data/fine_tuning_record")
    ap.add_argument("--singles-per-product", type=int, default=6)
    ap.add_argument("--scenes", type=int, default=80)
    ap.add_argument("--val-scenes", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    attrs = json.loads((LAB / "robot_sku_attrs.json").read_text())

    # single-product captures used for the v18 cutouts, grouped by product
    srcs: dict[str, list[str]] = OrderedDict()
    for e in json.loads((LAB / "cutouts_v18/cutout_manifest.json").read_text()):
        if (LAB / "rgbd" / e["src"] / "color.png").exists():
            srcs.setdefault(e["label"], [])
            if e["src"] not in srcs[e["label"]]:
                srcs[e["label"]].append(e["src"])
    # the plate is cut by colour, so its manifest entries name no capture folder;
    # take its single-product captures directly
    for label in {e["label"] for e in json.loads((LAB / "cutouts_v18/cutout_manifest.json").read_text())}:
        if label not in srcs:
            srcs[label] = [d.name for d in sorted((LAB / "rgbd").glob(f"single_{label.replace(' ', '_')}_*"))
                           if (d / "color.png").exists()]
    train, val = [], []
    for label, s in srcs.items():
        s = sorted(s)
        step = max(1, len(s) // (args.singles_per_product + 1))
        picks = s[::step][: args.singles_per_product + 1]
        val_pick, train_picks = picks[-1], picks[:-1]
        for split, lst in (("train", train_picks), ("val", [val_pick])):
            for src in lst:
                (train if split == "train" else val).append(row(
                    f"single:{src}", split, "robot_lab_singles", LAB / "rgbd" / src / "color.png",
                    [record(label, attrs, 1)], {"product": label}))

    # composed scenes
    syn = LAB / "synthetic_det_v18"
    classes = json.loads((syn / "classes.json").read_text())
    for split, n, bucket in (("train", args.scenes, train), ("val", args.val_scenes, val)):
        imgs = sorted((syn / "images/train").glob("*.jpg")) if split == "train" \
            else sorted((syn / "images/val").glob("*.jpg"))
        for img in rng.sample(imgs, n):
            lab = syn / "labels" / img.parent.name / f"{img.stem}.txt"
            counts = Counter(classes[int(l.split()[0])] for l in lab.read_text().split("\n") if l.strip())
            items = [record(name, attrs, q) for name, q in sorted(counts.items())]
            bucket.append(row(f"composed:{img.stem}", split, "synthetic_v18", img, items,
                              {"objects": sum(counts.values())}))

    rng.shuffle(train)
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in (("train", train), ("val", val), ("test", [dict(r, split="test") for r in val])):
        with (args.out / f"{split}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    print(f"products {len(srcs)}  train {len(train)}  val {len(val)}  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

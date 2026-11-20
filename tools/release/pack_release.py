#!/usr/bin/env python3
"""Build the data release of the dissertation from the working copies.

Writes a self-contained folder (default ../robotic-grocery-bagging-data) with
clean names, then checks its counts against the dissertation:

  catalog/products.json          the 32 products: characteristics, grams, cm3
  camera/intrinsics.json         the ceiling RealSense, colour and aligned depth
  camera/empty_table/NNN/        photographs of the empty table (training background)
  benchmark/scenes.csv           the 150 static scenes: arrangement, density, objects
  benchmark/<scene_id>/          color.png, depth.png, objects.json
  removal_sequences/<seq_id>/    step_NN/{color,depth}.png, sequence.json
  removal_sequences/located_objects.json   the 137 regions found by depth differencing
  products/<product>/<capture>/  single-product photographs, color, depth, meta.json
  predictions/<method>/<scene_id>.json     each method's output on the benchmark
  robot_trials/trials.json       the 60 robot trials, with the operator's notes
  models/                        detector weights and classes; LoRA adapter if given
  id_map.csv                     old capture ids -> release ids

Depth is stored as 16-bit PNG in millimetres (the camera's own unit, lossless);
0 means no reading. Scene ids drop the internal "b50_" prefix and name the
density as the dissertation does: easy -> sparse, med -> medium, hard -> dense.

  .venv/bin/python3.12 tools/release/pack_release.py \
      [--out ../robotic-grocery-bagging-data] [--lora PATH_TO_ADAPTER_DIR]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.planner.perception_to_packbot_adapter import compute_crush_score_v2  # noqa: E402

RGBD = ROOT / "data/robot_lab/rgbd"
GT_B50 = ROOT / "datasets_seq_GT_b50"
DENSITY = {"easy": "sparse", "med": "medium", "hard": "dense"}
FRUIT = {"banana": "bananas", "cherries": "cherries", "green_apple": "green apples",
         "lime": "limes", "mango": "mangoes", "orange": "oranges", "pear": "pears",
         "red_apple": "red apples", "strawberry": "strawberries",
         "yellow_lemon": "yellow lemons"}
# Run directories behind Table 4.2 (whole record) and 4.3 (identity). "catalog":
# characteristics read from the catalog through the predicted names; "own": as
# the model stated them.
PREDICTIONS = {
    "detector": "b50_detector_v18",
    "hybrid_catalog": "b50_hybrid_record015_lookup",
    "hybrid_own": "b50_hybrid_record015_own",
    "gemini_catalog": "b50_gemini_er",
    "gemini_own": "b50_gemini_er_fullrecord",
    "finetuned_qwen_catalog": "b50_vlm32b_record_lookup",
    "finetuned_qwen_own": "b50_vlm32b_record_own",
    "base_qwen_catalog": "b50_vlm32b_base_lookup",
    "base_qwen_own": "b50_vlm32b_base_own",
}
CATALOG_FIELDS = ["group", "packaging", "weight_class", "rigidity", "cold_chain",
                  "spill_risk", "edible"]

id_map: list[tuple[str, str, str]] = []   # kind, old, new


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def copy_frame(src: Path, dst: Path) -> None:
    """color.png as captured; depth.npy (float metres) -> 16-bit PNG millimetres."""
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src / "color.png", dst / "color.png")
    d = np.load(src / "depth.npy")
    mm = np.round(np.nan_to_num(d, nan=0.0) * 1000.0)
    assert mm.min() >= 0 and mm.max() <= 65535, src
    assert np.abs(mm / 1000.0 - np.nan_to_num(d)).max() < 1e-6, f"depth not in whole mm: {src}"
    Image.fromarray(mm.astype(np.uint16)).save(dst / "depth.png")


def scene_id(old: str) -> str:
    m = re.fullmatch(r"b50_(easy|med|hard)_(.+)_(\d{3})", old)
    assert m, old
    return f"{DENSITY[m.group(1)]}_{m.group(2)}_{m.group(3)}"


def pack_catalog(out: Path) -> dict:
    # The perception catalog also holds single fruits and retired lab props; the
    # 32 products are the detector's classes.
    names = set(json.loads((ROOT / "robot/robot_pc_package/classes.json").read_text()))
    products = json.loads((ROOT / "data/robot_lab/robot_item_catalog_perception.json").read_text())["items"]
    rows = []
    for p in sorted((p for p in products if p["name"] in names), key=lambda p: p["name"]):
        row = {"name": p["name"], **{k: p[k] for k in CATALOG_FIELDS}}
        row["crush_score"] = compute_crush_score_v2(p["packaging"], p["rigidity"], p["name"],
                                                    p["group"], p.get("deformation") or "")
        row["weight_g"] = p["measured_weight_g"]
        row["volume_cm3"] = p["true_volume_cc"]
        row["weight_source"] = p.get("weight_source")
        row["volume_source"] = p.get("volume_source")
        rows.append(row)
    assert len(rows) == 32, len(rows)
    write_json(out / "catalog/products.json", rows)
    return {r["name"]: r for r in rows}


def pack_camera(out: Path) -> None:
    shutil.copyfile(RGBD / "b50_easy_bag_bananas_000/intrinsics.json", out / "camera/intrinsics.json")
    for i, src in enumerate(sorted(RGBD.glob("empty_*"))):
        copy_frame(src, out / f"camera/empty_table/{i:03d}")
        id_map.append(("empty_table", src.name, f"camera/empty_table/{i:03d}"))


def pack_benchmark(out: Path, catalog: dict) -> tuple[int, int]:
    rows, total = [], 0
    for src in sorted(RGBD.glob("b50_*")):
        new = scene_id(src.name)
        meta = json.loads((src / "meta.json").read_text())
        gt = json.loads((GT_B50 / src.name / "frames/frame_000.json").read_text())["items"]
        objects = [{"name": it["name"], "quantity": int(it.get("quantity") or 1)} for it in gt]
        n = sum(o["quantity"] for o in objects)
        assert Counter(meta["gt_objects"]) == Counter({o["name"]: o["quantity"] for o in objects}), src
        assert all(o["name"] in catalog for o in objects), src
        copy_frame(src, out / "benchmark" / new)
        m = re.fullmatch(r"(sparse|medium|dense)_(.+)_(\d{3})", new)
        write_json(out / "benchmark" / new / "objects.json",
                   {"scene_id": new, "density": m.group(1), "arrangement": m.group(2),
                    "shot": int(m.group(3)), "objects": objects})
        rows.append([new, m.group(2), m.group(1), int(m.group(3)), n])
        id_map.append(("benchmark", src.name, new))
        total += n
    with (out / "benchmark/scenes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scene_id", "arrangement", "density", "shot", "n_objects"])
        w.writerows(rows)
    return len(rows), total


def pack_removal(out: Path) -> tuple[int, int, int, int]:
    seqs: dict[str, list[Path]] = {}
    for src in sorted(RGBD.glob("rem[23]_*_step*")):
        old_seq, step = re.fullmatch(r"(rem[23]_.+)_step(\d+)", src.name).groups()
        seqs.setdefault(old_seq, []).append(src)
    n_frames = n_objects = 0
    seq_names = {}
    for old_seq, frames in seqs.items():
        new_seq = old_seq.split("_", 1)[1]
        seq_names[old_seq] = new_seq
        steps = []
        for src in frames:
            k = int(src.name.rsplit("step", 1)[1])
            meta = json.loads((src / "meta.json").read_text())
            copy_frame(src, out / "removal_sequences" / new_seq / f"step_{k:02d}")
            steps.append({"step": k, "removed_before_this_step": meta.get("removed") or None,
                          "objects_on_table": meta.get("gt_objects", [])})
            id_map.append(("removal_frame", src.name, f"{new_seq}/step_{k:02d}"))
        steps.sort(key=lambda s: s["step"])
        n_frames += len(steps)
        n_objects += len(steps[0]["objects_on_table"])
        write_json(out / "removal_sequences" / new_seq / "sequence.json",
                   {"sequence_id": new_seq,
                    "occlusion": "none" if new_seq.startswith("sparse_") else "touching_or_stacked",
                    "steps": steps})
    boxes = [b for b in json.loads((ROOT / "runs/eval/removal_diff_boxes_v2.json").read_text())
             if b.get("ok")]
    located = [{"sequence_id": seq_names[b["scene"]], "step_before": b["step_prev"],
                "step_after": b["step_curr"], "name": b["name"],
                "bbox_xyxy_px": b["bbox_px"], "height_cm": b["height_cm"]} for b in boxes]
    write_json(out / "removal_sequences/located_objects.json", located)
    return len(seqs), n_frames, n_objects, len(located)


def pack_products(out: Path, catalog: dict) -> tuple[int, int]:
    n = 0
    for src in sorted(RGBD.glob("single_*")) + sorted(RGBD.glob("bag_*")):
        meta = json.loads((src / "meta.json").read_text())
        m = re.fullmatch(r"single_(.+?)(2?)(?:_(\d{3}))?", src.name)
        if m:
            base, copy2, shot = m.groups()
            name = base.replace("_", " ")
            # first RGB-D session (2026-06-17): two un-numbered captures per
            # product, the second suffixed "2"
            cap = shot if shot else f"early_{2 if copy2 else 1}"
            info = {"product": name, "shot": int(shot) if shot else None}
        else:
            fruit, k, shot = re.fullmatch(r"bag_(.+)_n(\d)_(\d{3})", src.name).groups()
            name = f"bag of {FRUIT[fruit]}"
            cap = f"n{k}_{shot}"
            info = {"product": name, "fruit_count": int(k), "shot": int(shot)}
        assert name in catalog, (src, name)
        assert meta.get("mode") == "produce" or [slug(g) for g in meta["gt_objects"]] == [slug(name)], src
        dst = out / "products" / slug(name) / cap
        copy_frame(src, dst)
        write_json(dst / "meta.json", info)
        id_map.append(("product_photo", src.name, f"{slug(name)}/{cap}"))
        n += 1
    return n, len(list((out / "products").iterdir()))


def pack_predictions(out: Path) -> None:
    for method, run in PREDICTIONS.items():
        files = sorted((ROOT / "runs" / run).glob("*/frames/frame_000.json"))
        assert len(files) == 150, (run, len(files))
        for f in files:
            items = json.loads(f.read_text()).get("items", [])
            write_json(out / "predictions" / method / f"{scene_id(f.parent.parent.name)}.json",
                       {"items": items})


def pack_trials(out: Path) -> dict:
    trials = json.loads((ROOT / "runs/eval/robot_real_runs_old5000.json").read_text())
    for t in trials:
        # file: name of the raw terminal capture (not released); v_any, v_hard:
        # violation counts under an analysis the dissertation does not report.
        for k in ("file", "v_any", "v_hard", "runs_in_file"):
            t.pop(k, None)
        t["stopped_early"] = t.pop("aborted")
    events = [e for t in trials for e in t["events"]]
    stats = {"trials": len(trials), "picks": len(events),
             "outcomes": dict(Counter(e[2] for e in events))}
    write_json(out / "robot_trials/trials.json", trials)
    return stats


def pack_models(out: Path, lora: Path | None) -> None:
    pc = ROOT / "robot/robot_pc_package"
    (out / "models").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pc / "weights/best.pt", out / "models/detector_yolo11s_robot32.pt")
    shutil.copyfile(pc / "classes.json", out / "models/detector_classes.json")
    if lora:
        dst = out / "models/qwen3vl32b_lora_record"
        dst.mkdir(exist_ok=True)
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copyfile(lora / name, dst / name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the data release.")
    ap.add_argument("--out", type=Path, default=ROOT.parent / "robotic-grocery-bagging-data")
    ap.add_argument("--lora", type=Path, default=None, help="adapter directory of lora_32b_record")
    args = ap.parse_args()
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "camera").mkdir(parents=True)

    catalog = pack_catalog(out)
    pack_camera(out)
    n_scenes, n_objects = pack_benchmark(out, catalog)
    n_seq, n_frames, n_seq_objects, n_located = pack_removal(out)
    n_photos, n_products = pack_products(out, catalog)
    pack_predictions(out)
    trials = pack_trials(out)
    pack_models(out, args.lora)
    shutil.copyfile(Path(__file__).with_name("DATA_README.md"), out / "README.md")
    (out / "LICENSE").write_text(
        "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
        "https://creativecommons.org/licenses/by/4.0/legalcode\n\n"
        "Copyright (c) 2026 Diogo Miguel Castro Paixão\n")
    with (out / "id_map.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("kind", "old_id", "release_id"), *id_map])

    # Counts the dissertation states (Chapters 4 and 5).
    checks = {
        "benchmark scenes": (n_scenes, 150), "benchmark objects": (n_objects, 1345),
        "removal sequences": (n_seq, 20), "removal frames": (n_frames, 215),
        "removal objects": (n_seq_objects, 195), "located objects": (n_located, 137),
        "products photographed": (n_products, 32), "robot trials": (trials["trials"], 60),
        "robot picks": (trials["picks"], 444),
    }
    ok = True
    for k, (got, want) in checks.items():
        flag = "ok" if got == want else "MISMATCH"
        ok &= got == want
        print(f"{k:24s} {got:6d}  expected {want:6d}  {flag}")
    print(f"product photographs       {n_photos}")
    print(f"pick outcomes            {trials['outcomes']}")
    print(f"written to {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

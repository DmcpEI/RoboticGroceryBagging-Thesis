#!/usr/bin/env python3
"""Local object perception -> planner-attribute JSON. Runs entirely on this
machine, no network, no external repo.

Reads a camera frame (a PNG file, refreshed by ros1_frame_writer.py or any
other means), runs a closed-set object detector on it, looks up each
detected object's packing attributes in objects_lookup.json, and writes:

  - <out-dir>/<shot>_detections.json  every detected object (for humans:
    name, confidence, pixel box, attributes)
  - <out-dir>/<shot>_planner.json     ONLY the fields a packing planner
    needs: {"items_data": {...}, "arrival_order": [...]}

Usage:
  python3 perceive_local.py --image /tmp/live_frame.png --loop
  python3 perceive_local.py --image /path/to/photo.png          # single shot
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = HERE / "weights" / "best.pt"
DEFAULT_CLASSES = HERE / "classes.json"
DEFAULT_LOOKUP = HERE / "objects_lookup.json"

# Detector runs inside this pixel box only (left, top, right, bottom), so the
# background around the table (arms, floor, cases) can't produce phantom
# detections. Update if the camera or table position changes.
DEFAULT_CROP = (290, 40, 970, 675)


def load_model(weights: Path, class_names: list = None):
    """Load the detector, and if class names are supplied, refuse to run on a
    mismatch.

    A YOLO checkpoint outputs a class INDEX; the names live in classes.json.
    Swap the weights and forget the names and every detection is silently
    renamed -- no error, just wrong products, wrong attributes, wrong bags.
    The v6 -> v7 upgrade shifts index 0 from "banana" to "bag of bananas" and
    changes the class count, so this is not hypothetical. The checkpoint
    carries its own names, so the two can simply be compared.
    """
    from ultralytics import YOLO
    model = YOLO(str(weights))
    if class_names is not None:
        embedded = [model.names[i] for i in range(len(model.names))]
        if embedded != list(class_names):
            extra = [c for c in embedded if c not in class_names]
            missing = [c for c in class_names if c not in embedded]
            raise SystemExit(
                "Detector weights and classes.json disagree.\n"
                "  weights ({}) carry {} classes, classes.json has {}.\n"
                "  in weights only : {}\n"
                "  in classes.json only: {}\n"
                "Every detection would be misnamed. Copy best.pt, classes.json and "
                "objects_lookup.json together -- they are one set.".format(
                    weights, len(embedded), len(class_names),
                    extra[:6] or "-", missing[:6] or "-"))
    return model


def detect(model, image_path: Path, class_names: list, imgsz: int, min_scan_conf: float,
           crop, crop_before_detect=False):
    """Every raw detection above min_scan_conf: list of (name, confidence, [x1,y1,x2,y2]).

    The crop keeps objects that are not on the table out of the inventory. It
    used to be applied to the IMAGE, before the network: a 455x287 crop handed
    to a 640-wide input makes every object arrive about 1.4 times larger than
    the detector was trained for, and the small products pay for it. Measured
    over the 133 benchmark scenes, cropping first scores 0.870 against 0.902
    for detecting on the whole frame and dropping the boxes that land off the
    table -- tuna can 0.619 against 0.738, gelatin dessert box 0.500 against
    0.611, coffee can 0.843 against 0.932. So the crop is now a filter on the
    boxes, not on the pixels. Pass crop_before_detect=True for the old
    behaviour.
    """
    src = str(image_path)
    ox = oy = 0
    if crop and crop_before_detect:
        from PIL import Image
        im = Image.open(image_path).convert("RGB")
        w, h = im.size
        left, top, right, bottom = crop
        left, top = max(0, left), max(0, top)
        right, bottom = min(w, right), min(h, bottom)
        im = im.crop((left, top, right, bottom))
        src, ox, oy = im, left, top
    result = model(src, imgsz=imgsz, verbose=False)[0]
    out = []
    if result.boxes is not None:
        for cls, conf, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()):
            c = float(conf)
            if c < min_scan_conf:
                continue
            box = [round(xyxy[0] + ox, 1), round(xyxy[1] + oy, 1),
                   round(xyxy[2] + ox, 1), round(xyxy[3] + oy, 1)]
            if crop and not crop_before_detect:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if not (crop[0] <= cx <= crop[2] and crop[1] <= cy <= crop[3]):
                    continue          # on the floor, or on the robot, not on the table
            out.append((class_names[int(cls)], c, box))
    return out


def build_detections(raw_dets, lookup: dict, min_conf: float, unknown_conf: float):
    """Confident detections get named + their looked-up attributes attached.
    Detections too unsure to name (but not pure noise) still get flagged as a
    graspable, unidentified object rather than silently dropped."""
    unknown_attrs = lookup["_unknown_object_default"]
    items = []
    for name, conf, box in raw_dets:
        if conf >= min_conf:
            attrs = lookup.get(name, unknown_attrs)
            row = {"name": name, "name_confidence": round(conf, 4), "bbox_2d": box}
            row.update(attrs)
            items.append(row)
        elif conf >= unknown_conf:
            row = {"name": "unknown_object", "name_confidence": round(conf, 4), "bbox_2d": box,
                   "graspable": True, "needs_rescan": True, "best_guess": name}
            row.update(unknown_attrs)
            items.append(row)
    return items


def planner_keys(items: list) -> dict:
    """{"name#N": item} for the items the planner is allowed to choose from.

    THE one place the "name#N" numbering is defined. A caller that needs the
    box or the depth of the chosen item must look it up here too: numbering
    over a different subset silently pairs a planner choice with another
    object's box, and the first excluded item is a BLOCKED one, so the arm is
    sent to the object mark_blocked exists to keep it away from.
    """
    keyed, counts = {}, {}
    for it in items:
        name = it.get("name")
        if not name or name == "unknown_object":
            continue
        if it.get("graspable") is False:
            continue
        counts[name] = counts.get(name, 0) + 1
        keyed[f"{name}#{counts[name]}"] = it
    return keyed


def to_planner_format(items: list) -> dict:
    """{"items_data": {...}, "arrival_order": [...]} -- the only two fields a
    packing planner needs. unknown_object is excluded (not a safely identified
    item yet); quantity is always 1 per detected box here (each box is its own
    physical object), so no explosion needed.

    An item carrying graspable=False is also excluded: something is resting on
    top of it (see mark_blocked). It is still a real object and still in the
    detections, it just cannot be picked yet -- offering it would let the
    planner choose an item whose removal drags whatever sits on it. The next
    perception cycle offers it once the top object is gone."""
    # orientation_sensitive was removed from the schema and nothing populates
    # it, so exporting it sent None on every item and _to_item turned that into
    # False -- a field the planner reads and never sees set. Dropped, matching
    # the bundle export, which already carries these seven.
    PLANNER_FIELDS = ("est_weight_g", "est_volume_cc", "crush_score", "category",
                      "temperature", "spill_risk", "spill_vulnerable")
    offered = planner_keys(items)
    return {"items_data": {k: {f: it.get(f) for f in PLANNER_FIELDS}
                           for k, it in offered.items()},
            "arrival_order": list(offered)}


def run_one_shot(model, class_names, lookup, image_path: Path, args, out_dir: Path, shot: int):
    if not image_path.exists():
        print(f"[perceive] no frame at {image_path}")
        return
    t0 = time.time()
    raw = detect(model, image_path, class_names, args.imgsz, args.unknown_conf, args.crop)
    items = build_detections(raw, lookup, args.min_conf, args.unknown_conf)
    planner = to_planner_format(items)
    dt = round(time.time() - t0, 3)

    out_dir.mkdir(parents=True, exist_ok=True)
    det_fp = out_dir / f"shot{shot:04d}_detections.json"
    plan_fp = out_dir / f"shot{shot:04d}_planner.json"
    det_fp.write_text(json.dumps({"items": items, "seconds": dt}, indent=1) + "\n")
    plan_fp.write_text(json.dumps(planner, indent=1) + "\n")

    named = [i for i in items if i["name"] != "unknown_object"]
    print(f"[perceive] shot {shot}: {len(named)} named + {len(items) - len(named)} unknown "
          f"in {dt}s -> {plan_fp.name}")
    for i in items[:15]:
        print(f"    {i['name']:22s} conf={i['name_confidence']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="frame PNG, re-read every shot")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--classes", type=Path, default=DEFAULT_CLASSES)
    ap.add_argument("--lookup", type=Path, default=DEFAULT_LOOKUP)
    ap.add_argument("--out-dir", type=Path, default=HERE / "output")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--min-conf", type=float, default=0.4, help="confidence to name an object")
    ap.add_argument("--unknown-conf", type=float, default=0.15,
                    help="below --min-conf but above this: flagged as unknown_object, not dropped")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                    default=list(DEFAULT_CROP), help="table region in pixels; --crop 0 0 0 0 for full frame")
    ap.add_argument("--loop", action="store_true", help="press ENTER for each new shot (Ctrl-C to stop)")
    args = ap.parse_args()
    if args.crop == [0, 0, 0, 0]:
        args.crop = None

    class_names = json.loads(args.classes.read_text())
    lookup = json.loads(args.lookup.read_text())
    model = load_model(args.weights, class_names)
    print(f"[perceive] {args.weights.name} ready ({len(class_names)} object classes).")

    shot = 0
    while True:
        if args.loop:
            try:
                input(f"[perceive] ENTER for shot {shot + 1} (Ctrl-C to stop)... ")
            except (EOFError, KeyboardInterrupt):
                print("\n[perceive] stopped.")
                return 0
        shot += 1
        run_one_shot(model, class_names, lookup, args.image, args, args.out_dir, shot)
        if not args.loop:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

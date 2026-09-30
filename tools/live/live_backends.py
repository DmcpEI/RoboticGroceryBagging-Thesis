#!/usr/bin/env python3
"""Perception backends for live deployment: YOLO, VLM, and YOLO+VLM hybrid.

Three interchangeable ways to turn one frame into a planner inventory, so the
same file-drop server/client (perception_server.py / remote_perceive.py) can run
any of them by launch-time --backend choice:

  yolo      closed-set detector only (PRIMARY; real name F1 ~0.79, real-time).
            Runs on a modest GPU -> also usable on the robot PC directly.
  vlm       whole-frame / crop VLM (open-set; handled by perceive_once elsewhere).
  yolo_vlm  YOLO boxes -> VLM re-classifies each crop (hybrid; measured WORSE than
            yolo alone on real crops, kept as a switchable comparison mode).

Items carry name + name_confidence + bbox_2d + the catalog planner attributes
(same schema as detector_inventory.py), with sub-threshold detections surfaced as
graspable unknown_object (needs_rescan) instead of dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools/eval"))


def load_yolo(weights: str):
    from ultralytics import YOLO
    return YOLO(str(weights))


def _attrs_for(name: str, planner):
    from build_planner_output import match_attrs, estimate_attrs, PLANNER_FIELDS
    pa = match_attrs(name, planner) or estimate_attrs({"name": name})
    return {f: pa.get(f) for f in PLANNER_FIELDS}


def _planner():
    from build_planner_output import load_planner_attrs
    return load_planner_attrs()


def to_planner_format(items: list) -> dict:
    """Convert a yolo/yolo_vlm/vlm inventory's items list into the exact
    {items_data, arrival_order} contract external/colleague_packing_system's
    solve_bagging()/solve_bagging_extended() (and the tools/planner/
    run_packbot_cp.py wrapper's --adapter-json) expect: items_data is a dict
    keyed by unique item id -> planner fields, arrival_order is that dict's
    keys in arrival order.

    unknown_object rows are excluded (not a real named item to pack yet --
    matches the convention in tools/eval/run_end_to_end_cp.py); quantity>1
    explodes into separate numbered instances.

    Single-frame arrival_order is arbitrary (list order) -- there is no real
    temporal signal in one shot; this matches every static-scene run this
    session.
    """
    items_data, arrival_order, counts = {}, [], {}
    for it in items:
        name = it.get("name")
        if not name or name == "unknown_object":
            continue
        qty = int(it.get("quantity", 1) or 1)
        for _ in range(max(1, qty)):
            counts[name] = counts.get(name, 0) + 1
            key = f"{name}#{counts[name]}"
            items_data[key] = {
                "est_weight_g": it.get("est_weight_g"),
                "est_volume_cc": it.get("est_volume_cc"),
                "crush_score": it.get("crush_score"),
                "category": it.get("category"),
                "temperature": it.get("temperature"),
                "spill_risk": bool(it.get("spill_risk", False)),
                "spill_vulnerable": bool(it.get("spill_vulnerable", False)),
            }
            arrival_order.append(key)
    return {"items_data": items_data, "arrival_order": arrival_order}


def yolo_detect(model, img_path: Path, labels, imgsz=640, min_conf=0.15, crop=None):
    """Raw YOLO detections above min_conf: list of (name, conf, [x1,y1,x2,y2]).

    crop=[L,T,R,B] restricts detection to the table ROI (fixed camera): the frame
    is cropped before inference so the lab background (robot arms, floor, cases)
    can't produce phantom detections. Boxes are offset back to full-frame pixels.
    """
    src = str(img_path)
    ox = oy = 0
    if crop:
        from PIL import Image
        im = Image.open(img_path).convert("RGB")
        w, h = im.size
        L, T, R, B = (int(v) for v in crop)
        L, T = max(0, L), max(0, T)
        R, B = min(w, R), min(h, B)
        im = im.crop((L, T, R, B))
        src, ox, oy = im, L, T
    r = model(src, imgsz=imgsz, verbose=False)[0]
    out = []
    if r.boxes is not None:
        for cls, conf, xyxy in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            c = float(conf)
            if c >= min_conf:
                box = [round(xyxy[0] + ox, 1), round(xyxy[1] + oy, 1),
                       round(xyxy[2] + ox, 1), round(xyxy[3] + oy, 1)]
                out.append((labels[int(cls)], c, box))
    return out


def _assemble(dets, planner, min_conf, unknown_conf, vlm_namer=None):
    """dets -> inventory items. If vlm_namer given, it re-names each det's crop."""
    items = []
    for name, conf, box in dets:
        final = name
        if vlm_namer is not None:
            vn = vlm_namer(box)
            if vn:
                final = vn
        if conf >= min_conf:
            row = {"name": final, "name_confidence": round(conf, 4), "bbox_2d": box}
            row.update(_attrs_for(final, planner))
            items.append(row)
        elif conf >= unknown_conf:
            row = {"name": "unknown_object", "name_confidence": round(conf, 4), "bbox_2d": box,
                   "graspable": True, "needs_rescan": True, "vlm_name": final}
            row.update(_attrs_for("unknown_object", planner))
            items.append(row)
    return items


def yolo_inventory(model, img_path: Path, labels, planner, imgsz=640,
                   min_conf=0.4, unknown_conf=0.15, crop=None):
    dets = yolo_detect(model, img_path, labels, imgsz=imgsz, min_conf=unknown_conf, crop=crop)
    return {"items": _assemble(dets, planner, min_conf, unknown_conf), "backend": "yolo",
            "crop": list(crop) if crop else None}


def yolo_vlm_inventory(model, img_path: Path, labels, planner, vlm_namer,
                       imgsz=640, min_conf=0.4, unknown_conf=0.15, crop=None):
    """YOLO boxes, VLM names each crop. vlm_namer(box)->name (crop-single VLM)."""
    dets = yolo_detect(model, img_path, labels, imgsz=imgsz, min_conf=unknown_conf, crop=crop)
    return {"items": _assemble(dets, planner, min_conf, unknown_conf, vlm_namer=vlm_namer),
            "backend": "yolo_vlm", "crop": list(crop) if crop else None}


def _bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


_DEPTH_ENVELOPES = None
_ENVELOPE_MARGIN_FRAC = 0.25  # same margin rule as depth_geometry_hazard_filter.py


def _envelope_ok(name, height_cm, diam_cm):
    """True if (height, diam) is plausible for the named class per the measured
    per-class envelopes (data/robot_lab/depth_envelopes.json), with the validated
    relative margin (25% of envelope width, floor 1cm). Unknown classes pass."""
    global _DEPTH_ENVELOPES
    if _DEPTH_ENVELOPES is None:
        import json as _json
        fp = ROOT / "data/robot_lab/depth_envelopes.json"
        _DEPTH_ENVELOPES = _json.loads(fp.read_text()) if fp.exists() else {}
    env = _DEPTH_ENVELOPES.get(str(name or "").strip().lower())
    if not env:
        return True
    for val, (lo, hi) in ((height_cm, env["h"]), (diam_cm, env["d"])):
        m = max(1.0, (hi - lo) * _ENVELOPE_MARGIN_FRAC)
        if not (lo - m <= val <= hi + m):
            return False
    return True


def depth_blob_crosscheck(items, step_dir: Path, planner, claim_iou=0.2):
    """Fail-closed depth cross-check, two tiers (validated on the 5 bleach-miss
    frames, 2026-07-17; see runs/eval/depth_crosscheck_eval.json):

    Tier 1 -- omission: every physically raised table blob must be claimed by a
    detection at bbox IoU >= claim_iou. IoU ONLY, no centroid rules: centroid
    containment lets a large adjacent box (e.g. a cracker box next to the bleach
    bottle) claim a blob it doesn't actually cover, which re-hides exactly the
    silent omissions this check exists to catch. Unclaimed blobs are appended as
    graspable unknown_object rows (needs_rescan, provenance depth_blob, measured
    height/diam attached).

    Tier 2 -- mislabel: each claimed blob's best-IoU claimant is checked against
    its own class's measured size envelope (depth_envelopes.json, same
    25%-margin rule as depth_geometry_hazard_filter.py). A claimant whose class
    cannot physically be this tall/wide (e.g. "coffee can" max 14.2cm claiming a
    17cm blob -- the classic swallowed-bleach signature) keeps its name but gets
    geometry_mismatch=True + needs_rescan=True.

    Depth blobs key on elevation, not appearance, so this catches the
    textureless-item failure no RGB confidence threshold can reach. CPU-only
    (numpy/scipy) -> runs on the robot PC where the VLM cannot. Requires
    depth.npy + intrinsics.json next to the color frame; if missing, items pass
    through unchanged.

    Returns (items_list, appended_unknown_rows). Tier-2 flags mutate the
    existing item dicts in place.
    """
    import numpy as np
    from eval_depth_separation import blobs  # tools/eval is on sys.path

    depth_fp = Path(step_dir) / "depth.npy"
    intr_fp = Path(step_dir) / "intrinsics.json"
    if not depth_fp.exists() or not intr_fp.exists():
        return items, []
    import json as _json
    depth = np.load(depth_fp)
    intr = _json.loads(intr_fp.read_text())

    added = []
    for bl in blobs(depth, intr, split_touching=True):
        best_it, best_iou = None, 0.0
        for it in items:
            if not it.get("bbox_2d"):
                continue
            v = _bbox_iou(bl["bbox_px"], it["bbox_2d"])
            if v > best_iou:
                best_it, best_iou = it, v
        if best_iou >= claim_iou:
            if (best_it["name"] != "unknown_object"
                    and not _envelope_ok(best_it["name"], bl["height_cm"], bl["diam_cm"])):
                best_it["geometry_mismatch"] = True
                best_it["needs_rescan"] = True
                best_it["blob_height_cm"] = bl["height_cm"]
                best_it["blob_diam_cm"] = bl["diam_cm"]
            continue
        row = {"name": "unknown_object", "name_confidence": None,
               "bbox_2d": [round(float(v), 1) for v in bl["bbox_px"]],
               "graspable": True, "needs_rescan": True,
               "provenance": "depth_blob",
               "height_cm": bl["height_cm"], "diam_cm": bl["diam_cm"]}
        row.update(_attrs_for("unknown_object", planner))
        added.append(row)
    return items + added, added

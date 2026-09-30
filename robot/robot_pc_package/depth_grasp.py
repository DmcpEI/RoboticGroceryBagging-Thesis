#!/usr/bin/env python3
"""Object height above the table, from the ceiling camera's aligned depth.

WHY. The gripper used to descend to the table surface for every object and
close there. On a 20 cm bottle that grips the bottom 6 cm, so the centre of
mass sits 10 cm ABOVE the grip and the object pivots out of the jaws as it
lifts. Gripping near the top inverts that -- the mass hangs below the grip
like a pendulum -- and it is also the only way to pick anything stacked.

METHOD. Fit a plane to the table once per frame, then for each object:

    height = plane_depth_at(object centre)   <- where the table would be
           - percentile(depth in the box, low)  <- the object's top

Depth is distance from the camera, so the closer (smaller) value is the top.
Only the DIFFERENCE is used, which needs no camera-to-robot transform: the
ceiling camera looks down, so a difference along its optical axis is a height.
That approximation costs 1/cos(theta); for a camera ~1.2 m above a 0.8 m table
the worst ray is about 18 degrees off nadir, so a 20 cm object reads about 1 cm
tall -- under the grip tolerance, and it errs high (grips further up), which is
the safe direction.

A FITTED PLANE, not a local ring around each box. A ring reads whatever the
object is standing ON, so a box resting on another box measures its height
above the lower box rather than above the table -- wrong by the height of the
lower box, and wrong exactly in the stacking case that motivates using depth
at all. The plane also absorbs table tilt, which is real here: the four
corners of this table differ by 37 mm.

The fit is robust by trimming: objects are always NEARER than the table, so
they are one-sided outliers. Fitting to the far half and then re-fitting to
whatever lies close to that plane converges onto the table even when objects
cover much of the view.

Run `python3 depth_grasp.py` for a self-check on synthetic depth images.
"""
from __future__ import annotations

import math
import os

import numpy as np

# Anything outside this is not a grocery item standing on our table: reject and
# fall back rather than send the arm somewhere odd.
MIN_HEIGHT_M = 0.005
MAX_HEIGHT_M = 0.350
# A table fit whose inliers scatter more than this is not a table.
MAX_PLANE_RMS_M = 0.020


def _valid(a):
    """Depth images carry 0 where the sensor returned nothing."""
    return a[(a > 0) & np.isfinite(a)]


def fit_table_plane(depth_m, crop=None, iters=3, band_m=0.020):
    """Least-squares plane through the table, as depth = a*x + b*y + c.

    Objects sit ABOVE the table, so in depth they are all nearer -- one-sided
    outliers. Seeding on the far half and re-fitting to points near that plane
    walks the fit onto the table and off the objects.

    Returns (coeffs, info); coeffs is None when no plane could be trusted.
    """

    d = np.asarray(depth_m)
    h_img, w_img = d.shape[:2]
    if crop:
        l, t, r, b = [int(v) for v in crop]
        l, t = max(0, l), max(0, t)
        r, b = min(w_img, r), min(h_img, b)
    else:
        l, t, r, b = 0, 0, w_img, h_img
    sub = d[t:b, l:r]
    ys, xs = np.mgrid[t:b, l:r]
    m = (sub > 0) & np.isfinite(sub)
    if m.sum() < 200:
        return None, {"reason": "too few valid depth pixels on the table ({})".format(int(m.sum()))}

    X, Y, Z = xs[m].astype("float64"), ys[m].astype("float64"), sub[m].astype("float64")
    # Seed on the far half: the table cannot be nearer than the things on it.
    keep = Z >= np.percentile(Z, 50)
    coeffs = None
    for _ in range(iters):
        if keep.sum() < 100:
            return None, {"reason": "table fit lost its inliers"}
        A = np.column_stack([X[keep], Y[keep], np.ones(keep.sum())])
        coeffs, *_ = np.linalg.lstsq(A, Z[keep], rcond=None)
        resid = Z - (coeffs[0] * X + coeffs[1] * Y + coeffs[2])
        keep = np.abs(resid) < band_m

    rms = float(np.sqrt(np.mean(resid[keep] ** 2))) if keep.sum() else 1e9
    info = {"inliers": int(keep.sum()), "total": int(m.sum()), "rms": rms}
    if rms > MAX_PLANE_RMS_M:
        info["reason"] = "table fit too rough (rms {:.3f} m)".format(rms)
        return None, info
    if keep.sum() < 0.15 * m.sum():
        info["reason"] = "table plane explains only {:.0f}% of the view".format(
            100.0 * keep.sum() / m.sum())
        return None, info
    return tuple(float(c) for c in coeffs), info


def _clamp_bbox(depth_m, bbox):
    """(x1, y1, x2, y2) rounded and clipped to the depth frame."""
    h_img, w_img = depth_m.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    return max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)


def plane_depth_at(coeffs, x, y):
    return coeffs[0] * x + coeffs[1] * y + coeffs[2]


def object_height_m(depth_m, bbox, plane, obj_percentile=30, min_obj_px=40,
                    patch=0.40, other_boxes=None, max_mask=0.85):
    """Height of the object's top above the FITTED TABLE PLANE, in metres.

    depth_m : 2-D array of metres (see `to_metres` for raw RealSense frames)
    bbox    : (x1, y1, x2, y2) in the same pixel frame as depth_m
    plane   : coefficients from fit_table_plane

    Returns (height_m, info). height_m is None when the reading cannot be
    trusted, and info['reason'] says why -- callers fall back to the table.
    """

    if plane is None:
        return None, {"reason": "no table plane"}
    x1, y1, x2, y2 = _clamp_bbox(depth_m, bbox)
    info = {"bbox": (x1, y1, x2, y2)}
    if x2 - x1 < 3 or y2 - y1 < 3:
        info["reason"] = "box too small"
        return None, info

    # Central patch only, so the object's own edges do not enter the reading.
    # A TALL NEIGHBOUR intrudes at the box edge and, being closer, holds the
    # LOWEST depths -- a low percentile then returns ITS height and the arm
    # never descends. Defended three ways: tight patch, high percentile, and
    # masking other detections' boxes when known.
    f = (1.0 - patch) / 2.0
    mx, my = int((x2 - x1) * f), int((y2 - y1) * f)
    box = depth_m[y1:y2, x1:x2].copy()
    if other_boxes:
        whole = max(1.0, float(box.shape[0] * box.shape[1]))
        for ob in other_boxes:
            bx1, by1, bx2, by2 = [int(round(v)) for v in ob]
            cx1, cy1 = max(bx1 - x1, 0), max(by1 - y1, 0)
            cx2, cy2 = min(bx2 - x1, box.shape[1]), min(by2 - y1, box.shape[0])
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            if (cx2 - cx1) * (cy2 - cy1) > max_mask * whole:
                # It covers this box whole, so it is either the LARGER object
                # this one stands on or a duplicate of it. Masking it would
                # erase the object being measured; there is nothing above.
                continue
            box[cy1:cy2, cx1:cx2] = 0.0
    inner = box[my:box.shape[0] - my, mx:box.shape[1] - mx]
    iv = _valid(inner)
    if iv.size < min_obj_px:
        # Masking emptied the middle, which means something is STANDING there.
        # What is left of the box is the object's own visible rim, and the rim
        # IS its top surface -- the reading this function is asked for. Without
        # this the lower object of a stack returns None, gets merged away as a
        # duplicate, and the item on top is then gripped at table height.
        iv = _valid(box)
        info["from"] = "rim"
    if iv.size < min_obj_px:
        info["reason"] = "too few valid depth pixels on the object ({})".format(iv.size)
        return None, info

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    table = plane_depth_at(plane, cx, cy)
    top = float(np.percentile(iv, obj_percentile))   # closest -> highest
    height = table - top
    info.update({"table_depth": table, "object_depth": top, "height": height})
    if not (MIN_HEIGHT_M <= height <= MAX_HEIGHT_M):
        info["reason"] = "height {:.3f} m outside [{:.3f}, {:.3f}]".format(
            height, MIN_HEIGHT_M, MAX_HEIGHT_M)
        return None, info
    return height, info


def support_height_m(depth_m, bbox, plane, ring_px=14, min_ring_px=60):
    """Height above the table of whatever the object is STANDING ON.

    The object's own thickness is then (top height - support height), and that
    is what decides how deep the fingers may go. Without it a thin object on
    another thin object is gripped at the table and the jaws close around BOTH:
    a 2 cm gelatin box on a 3 cm tuna can tops out at 5 cm, the default 5 cm
    grip depth floors the lift at zero, and the fingers span 0-6 cm.

    Returns (support_height_m, info); None when the ring is not a clean read.
    """

    if plane is None:
        return None, {"reason": "no table plane"}
    h_img, w_img = depth_m.shape[:2]
    x1, y1, x2, y2 = _clamp_bbox(depth_m, bbox)

    rx1, ry1 = max(0, x1 - ring_px), max(0, y1 - ring_px)
    rx2, ry2 = min(w_img, x2 + ring_px), min(h_img, y2 + ring_px)
    ring = depth_m[ry1:ry2, rx1:rx2].copy()
    ring[y1 - ry1:y2 - ry1, x1 - rx1:x2 - rx1] = 0.0
    rv = _valid(ring)
    if rv.size < min_ring_px:
        return None, {"reason": "too few valid depth pixels around the object"}

    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # The ring's NEAR side is what the object stands on; its far side is table.
    # A low percentile takes the supporting surface where one exists.
    support_depth = float(np.percentile(rv, 30))
    h = plane_depth_at(plane, cx, cy) - support_depth
    if not (-0.02 <= h <= MAX_HEIGHT_M):
        return None, {"reason": "support height {:.3f} m implausible".format(h)}
    return max(0.0, h), {"support_height": h}


# How far a box must read ABOVE the tallest captured pose of its product
# before a second surface inside it is believed. A tuna can on a coffee can
# tops out 11 mm over the coffee can's own maximum, so this has to be small.
STACK_SLACK_M = 0.005


def step_support_m(depth_m, bbox, plane, top_height, max_own_height=None,
                   band_m=0.015, gap_m=0.015, min_frac=0.12, min_px=40,
                   pct=80, shelf_frac=0.5, slack_m=STACK_SLACK_M):
    """The SECOND surface INSIDE one box: what the top object is standing on.

    The detector often calls a stack one product -- a tuna can on a potato chip
    can comes back as a single box, named after one of them. `support_height_m`
    then reads the ring OUTSIDE that box, which is table, so the object is
    reported as thick as the whole stack, `grip_depth_m` takes its full 50 mm,
    and the fingertips close below the tuna's underside, around the can beneath.
    Neither the masking in `object_height_m` nor `infer_support_from_neighbours`
    helps: both need a second detection, and there is only one box.

    Depth already holds the answer. Inside that box the top face sits over a
    wide shelf. Drop everything at table level and everything near the top, and
    the high percentile of what is left is that shelf.

    A tall can photographed off the camera axis also shows pixels below its
    top -- its own side wall -- and must NOT be read as a shelf, or every tall
    object loses grip depth. A wall is a ramp: its pixels spread evenly from
    the table to the top. A shelf is flat. Requiring most of the lower pixels
    to sit within band_m of the answer separates the two.

    That test alone is not enough, and the robot showed why: a coffee can LYING
    DOWN reported a shelf at 0.067 m under its 0.089 m top, with 93% of the
    lower pixels inside the band. Its flank is curved, but a cylinder on its
    side only drops 45 mm from crown to table, so the whole curve fits inside a
    15 mm band and reads flat. Curvature beats any flatness test at this scale.

    `max_own_height` is what settles it: the tallest pose of this product ever
    measured. An object cannot be taller than itself, so a box reading ABOVE
    that is the only evidence of a stack that does not depend on shape. Passing
    it in is what separates the lying coffee can (0.089 m, its own range reaches
    0.156) from the real stack (0.252 m). Without it this function guesses.

    Returns (support_height_m, info); None when the box shows no second level.
    """

    if depth_m is None or plane is None or top_height is None:
        return None, {"reason": "no depth, plane or height"}
    if max_own_height is not None and top_height <= max_own_height + slack_m:
        return None, {"reason": "its own height explains the reading "
                                "({:.3f} m, it reaches {:.3f} m alone)".format(
                                    top_height, max_own_height)}
    d = np.asarray(depth_m)
    if d.ndim < 2:
        return None, {"reason": "depth is not an image"}
    x1, y1, x2, y2 = _clamp_bbox(d, bbox)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None, {"reason": "box too small"}

    sub = d[y1:y2, x1:x2]
    ys, xs = np.mgrid[y1:y2, x1:x2]
    hm = plane_depth_at(plane, xs, ys) - sub
    obj = (sub > 0) & np.isfinite(sub) & (hm > MIN_HEIGHT_M)
    n_obj = int(obj.sum())
    if n_obj < min_px:
        return None, {"reason": "too few object pixels"}

    lower = obj & (hm < top_height - gap_m)
    frac = int(lower.sum()) / float(n_obj)
    if frac < min_frac:
        return None, {"reason": "no second level ({:.2f} of the object)".format(frac)}

    hl = hm[lower]
    sup = float(np.percentile(hl, pct))
    if not (MIN_HEIGHT_M <= sup <= top_height - gap_m):
        return None, {"reason": "second level {:.3f} m implausible".format(sup)}
    shelf = float(np.mean(np.abs(hl - sup) <= band_m))
    if shelf < shelf_frac:
        return None, {"reason": "a slope, not a shelf ({:.2f} within the band)".format(shelf)}
    return sup, {"support_height": sup, "lower_frac": frac, "shelf_frac": shelf}


def grasp_pixel(depth_m, bbox, plane, height_m, principal=None,
                top_band_m=0.015, min_px=20):
    """Where to aim the gripper, corrected for viewing parallax.

    The camera looks down from about 1.6 m, so a raised object is NOT imaged
    directly above its footprint: its top face is projected outward, away from
    the point under the camera, and the further the object is from that point
    the larger the shift. Aiming at the box centre and mapping that pixel onto
    the table plane therefore targets a spot beside the object. It is invisible
    on a flat item near the middle of the table and reaches ~87 mm on a
    250 mm-tall can near the edge, which is most of a gripper width.

    A point at height h images at c + f*X/(H-h), while the same table position
    images at c + f*X/H. So the fix is to scale the offset from the principal
    point by (H-h)/H, pulling the aim back in towards the camera axis.

    That identity holds for points actually AT height h, so the top surface is
    located first (depth within top_band_m of the object's top) and its centroid
    is corrected. The box centre sits somewhere between the top face and the
    base, which is a different and pose-dependent height, so correcting it
    directly would over-shoot.

    Returns (px, py, info). Falls back to the box centre whenever the geometry
    is unavailable, which is the previous behaviour.
    """

    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    cx_box, cy_box = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # A frame can arrive with no depth at all, so check before touching it.
    if depth_m is None or plane is None or height_m is None:
        return cx_box, cy_box, {"corrected": False, "reason": "no depth, plane or height"}

    d = np.asarray(depth_m)
    if d.ndim < 2:
        return cx_box, cy_box, {"corrected": False, "reason": "depth is not an image"}
    h_img, w_img = d.shape[:2]
    if principal is None:
        principal = (w_img / 2.0, h_img / 2.0)
    px_c, py_c = float(principal[0]), float(principal[1])

    H = plane_depth_at(plane, px_c, py_c)
    if not (H and H > height_m > 0):
        return cx_box, cy_box, {"corrected": False, "reason": "implausible H/h"}

    xa, ya = max(0, x1), max(0, y1)
    xb, yb = min(w_img, x2), min(h_img, y2)
    if xb <= xa or yb <= ya:
        return cx_box, cy_box, {"corrected": False, "reason": "empty bbox"}

    sub = d[ya:yb, xa:xb]
    ys, xs = np.mgrid[ya:yb, xa:xb]
    valid = (sub > 0) & np.isfinite(sub)
    hm = np.where(valid, plane_depth_at(plane, xs, ys) - sub, -1.0)
    top = valid & (hm >= height_m - top_band_m)
    if int(top.sum()) < min_px:
        obs_x, obs_y, src = cx_box, cy_box, "bbox centre (no top surface)"
    else:
        obs_x, obs_y = float(xs[top].mean()), float(ys[top].mean())
        src = "top surface"

    scale = (H - height_m) / H
    gx = px_c + (obs_x - px_c) * scale
    gy = py_c + (obs_y - py_c) * scale
    shift = float(((gx - cx_box) ** 2 + (gy - cy_box) ** 2) ** 0.5)
    return gx, gy, {"corrected": True, "source": src, "H_m": round(H, 3),
                    "scale": round(scale, 4), "shift_px": round(shift, 1),
                    "top_px": int(top.sum())}


def grip_depth_m(top_height, support_height, default=0.05, minimum=0.010):
    """How far below an object's top the fingertips should sit.

    Capped at 80% of the object's own thickness, so the jaws stay inside the
    object rather than reaching past its underside into whatever it rests on.
    """
    if support_height is None:
        return default
    thickness = max(0.0, top_height - support_height)
    if thickness < MIN_THICKNESS_M:
        # Nothing in the catalog is this thin, so the support is wrong rather
        # than the object flat. Fall back to the default depth; for something
        # genuinely low the caller clamps the lift at the table anyway.
        return default
    return max(minimum, min(default, 0.8 * thickness))


def boxes_overlap_min(a, b):
    """Intersection over the SMALLER box's area.

    Plain IoU is the wrong measure for a box nested inside another: a label-
    sized box inside a whole-product box scores well under any usual NMS
    threshold and both survive, so one object is reported twice and the planner
    packs a phantom.
    """
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    sa = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    sb = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / min(sa, sb)


def suppress_nested(items, contain=0.75, area_ratio_no_depth=0.90, height_gap=0.020):
    """Drop a box nested inside a larger box OF THE SAME CLASS.

    Same class only. Cross-class nesting is exactly what a small object sitting
    ON a large one looks like from overhead, and suppressing that would delete
    the stacked items depth was added to pick.

    What separates a duplicate from a genuine stack is HEIGHT, not size. A
    first version also demanded the inner box be under 60% of the outer's area;
    on the real robot a whole-product box and its label-only twin came out
    105x117 and 78x109 -- fully nested, containment 1.00, but 69% of the area,
    so nothing was dropped. Two boxes on ONE object read the same height; two
    genuinely stacked objects do not. Height is the test, and the area ratio is
    kept only as a weak fallback for when no depth is available.

    items: dicts with 'bbox_2d', 'name', optionally 'height_m'.
    Returns (kept, dropped).
    """
    def area(b):
        return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

    order = sorted(range(len(items)), key=lambda i: -area(items[i]["bbox_2d"]))
    dead = set()
    for oi, i in enumerate(order):
        if i in dead:
            continue
        ai = items[i]
        for j in order[oi + 1:]:
            if j in dead:
                continue
            aj = items[j]
            if aj.get("name") != ai.get("name"):
                continue
            if boxes_overlap_min(ai["bbox_2d"], aj["bbox_2d"]) < contain:
                continue
            hi, hj = ai.get("height_m"), aj.get("height_m")
            if hi is not None and hj is not None:
                if abs(hi - hj) > height_gap:
                    continue                  # different levels: a real stack
            elif area(aj["bbox_2d"]) > area_ratio_no_depth * area(ai["bbox_2d"]):
                continue                      # no depth, and nearly the same size
            dead.add(j)
    return ([it for k, it in enumerate(items) if k not in dead],
            [it for k, it in enumerate(items) if k in dead])


def suppress_overlapping(items, iou=0.55, cross_class_height_gap=0.020):
    """Merge two boxes that overlap heavily, keeping the more confident one.

    Same class always. DIFFERENT classes only when their measured heights agree
    to within cross_class_height_gap, because that is one object the detector
    could not decide the name of: the robot reported a JELL-O box as both
    "gelatin dessert box" 0.75 and "chocolate pudding box" 0.42, two boxes on
    one item, and the planner packed a phantom. Neither suppressor fired, both
    being same-class by design.

    Height is what separates that from a genuine cross-class stack, and it is
    the same test the same-class rules already use. A tuna can resting on a
    spam can also overlaps heavily from overhead, but its top sits a can's
    thickness higher, so the heights disagree and both boxes survive. When
    either height is missing, nothing cross-class is dropped -- without depth
    there is no way to tell the two cases apart, and deleting a real stacked
    item is the worse error.

    A tall cylinder away from the camera's nadir shows both its top face and
    its side, and the detector can fire once on each: the robot reported one
    potato chip can at 0.47 and another at 0.94 on a single Pringles tube.
    Neither box contains the other, so the nested rule never sees it, and their
    heights genuinely differ -- top face versus body -- so the height guard
    would actively protect it as if it were a stack.

    The cost is a real stack of two IDENTICAL products, whose boxes would also
    overlap heavily. That is rare in this setup (stacks mix products), the
    lower object is mostly occluded anyway, and a phantom item reaching the
    planner is the failure actually being observed.
    """
    def area(b):
        return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

    def iou_of(a, b):
        ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        inter = ix * iy
        u = area(a) + area(b) - inter
        return inter / u if u > 0 else 0.0

    order = sorted(range(len(items)),
                   key=lambda i: -(items[i].get("name_confidence") or 0.0))
    dead = set()
    for oi, i in enumerate(order):
        if i in dead:
            continue
        for j in order[oi + 1:]:
            if j in dead:
                continue
            # The height guard applies to SAME-class pairs too. Two tuna cans
            # stacked used to be merged into one, because a matching name
            # skipped the check entirely and the inventory lost an object.
            same = items[j].get("name") == items[i].get("name")
            hi_, hj = items[i].get("height_m"), items[j].get("height_m")
            if hi_ is not None and hj is not None:
                if abs(hi_ - hj) > cross_class_height_gap:
                    continue   # a real stack, regardless of class
            elif not same:
                # No depth to judge by. Merging two DIFFERENT products on
                # overlap alone deletes one of them from the inventory, and
                # depth does drop out; two boxes of the same product are a
                # duplicate detection and still merge.
                continue
            if iou_of(items[i]["bbox_2d"], items[j]["bbox_2d"]) >= iou:
                dead.add(j)
    return ([it for k, it in enumerate(items) if k not in dead],
            [it for k, it in enumerate(items) if k in dead])


# Grippable width band, metres. MEASURE YOUR OWN: open and close the jaws fully
# and read the gap. Defaults are read off objects seen at each limit (coffee can
# at full open, tomato soup can near full close) -- right order, not calibrated.
GRIPPER_MAX_M = 0.135
GRIPPER_MIN_M = 0.030


def footprint_axes(depth_m, bbox, plane, height_m, band_m=0.015, min_px=40,
                   focal_px=615.0):
    """Which way to turn the wrist, and how wide the object is across the jaws.

    A parallel jaw closes along ONE axis, so for anything that is not round the
    wrist angle decides whether the object fits at all. A sugar box measures
    76-118 mm across its short axis and 131-210 mm along its long one: the first
    fits the gripper, the second does not. Gripping across the SHORT axis is
    therefore not a refinement, it is the difference between picking the box and
    not being able to.

    The object's top surface is taken from depth (as for the parallax fix) and
    its principal axes found. Returns (yaw_deg, minor_m, major_m, info) where
    yaw_deg is the direction the jaws should close along -- the MINOR axis --
    measured in image coordinates, and minor_m is the width they must span.

    Returns yaw None when depth cannot supply a footprint, which leaves the
    caller on its existing orientation order.
    """

    if depth_m is None or plane is None or height_m is None:
        return None, None, None, {"reason": "no depth, plane or height"}
    d = np.asarray(depth_m)
    if d.ndim < 2:
        return None, None, None, {"reason": "depth is not an image"}
    h_img, w_img = d.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    xa, ya, xb, yb = max(0, x1), max(0, y1), min(w_img, x2), min(h_img, y2)
    if xb <= xa or yb <= ya:
        return None, None, None, {"reason": "empty bbox"}

    sub = d[ya:yb, xa:xb]
    ys, xs = np.mgrid[ya:yb, xa:xb]
    valid = (sub > 0) & np.isfinite(sub)
    hm = np.where(valid, plane_depth_at(plane, xs, ys) - sub, -1.0)
    top = valid & (hm >= height_m - band_m)
    if int(top.sum()) < min_px:
        return None, None, None, {"reason": "no top surface"}

    z = float(np.median(sub[top]))
    # metres per pixel at the object's own distance; this RealSense is square
    # pixels at 640x480, fx = fy = 615, so one scale covers both axes
    mx = my = z / focal_px
    px = (xs[top] - xs[top].mean()) * mx
    py = (ys[top] - ys[top].mean()) * my
    P = np.stack([px, py])
    if P.shape[1] < min_px:
        return None, None, None, {"reason": "too few points"}
    _, evecs = np.linalg.eigh(np.cov(P))
    proj = evecs.T @ P
    ext = proj.max(1) - proj.min(1)
    order = np.argsort(ext)                     # [minor, major]
    minor_v = evecs[:, order[0]]
    minor_m, major_m = float(ext[order[0]]), float(ext[order[1]])
    yaw = math.degrees(math.atan2(float(minor_v[1]), float(minor_v[0]))) % 180.0
    return yaw, minor_m, major_m, {"top_px": int(top.sum()), "z_m": round(z, 3)}


def grasp_width_ok(minor_m, lo=GRIPPER_MIN_M, hi=GRIPPER_MAX_M):
    """Can the jaws span this width? Returns (ok, reason)."""
    if minor_m is None:
        return True, None                        # unmeasured: do not block
    if minor_m > hi:
        return False, "too wide for the gripper ({:.0f} mm > {:.0f})".format(
            minor_m * 1000, hi * 1000)
    if minor_m < lo:
        return False, "too narrow to grip ({:.0f} mm < {:.0f})".format(
            minor_m * 1000, lo * 1000)
    return True, None


# When does one item count as resting ON another, rather than standing beside
# it? Two conditions, and the HEIGHT one is what was wrong.
#
# On the robot an egg carton standing NEXT TO a mustard bottle blocked it: the
# carton is 16 mm taller, against a 15 mm gate, so it cleared it by 1 mm. A
# real stack is not marginal -- a spam can on an egg carton read 52 mm, and a
# can on the edge of a sugar box reads 35 mm -- so 25 mm separates the two
# cleanly while leaving genuine stacks well inside.
#
# The overlap is raised too, but only to 0.25: a can sitting on the EDGE of a
# box covers 29% of it, so 0.30 would start missing real partial stacks.
#
# Both are env-tunable so they can be dialled against the robot without an
# edit. A false block only defers an item one cycle; a missed one has the arm
# drag something off, so err high on height and low on cover.
BLOCK_COVER = float(os.environ.get("VMT_BLOCK_COVER", 0.25))
BLOCK_HEIGHT_GAP = float(os.environ.get("VMT_BLOCK_HEIGHT_GAP", 0.025))


# Thinnest object worth gripping. Below this a "thickness" is evidence that
# the support estimate is wrong, not that the object is a wafer: the flattest
# thing in the catalog is a plate at 17 mm.
MIN_THICKNESS_M = 0.010


def infer_support_from_neighbours(items, cover=BLOCK_COVER,
                                  min_gap=MIN_THICKNESS_M):
    """Raise an item's support to the top of whatever it is standing ON.

    `support_height_m` reads a ring around the item's box, which fails exactly
    when it matters most: a tuna can sitting on a coffee can is barely narrower
    than the can, so the ring straddles the can's rim and the table and the
    percentile falls to the table. Support then reads ~0, thickness reads the
    whole STACK, and `grip_depth_m` is free to take its full 50 mm -- which for
    a 30 mm tuna can puts the fingertips 20 mm BELOW its underside, closing the
    jaws on the coffee can instead. That is the failure seen on the robot.

    The detections already say what is underneath. If A's box substantially
    overlaps B's and A's top is above B's, then A is standing on B, so A's
    support is at least B's top. Only ever raises the support, never lowers it,
    so a correct depth read is left alone.

    B must be at least min_gap BELOW A. Without that, a second box on the SAME
    object qualifies: the robot detected one tuna can as both "tuna can" 0.67
    and "tomato soup can" 0.42, the two boxes read heights 5 microns apart, and
    the can was declared to be standing on itself -- thickness 0.000 m, grip
    down to its 10 mm floor, jaws closing on the top third of a 31 mm can.
    Suppression removes that duplicate, but only a real step is a support.
    """
    for it in items:
        b, h = it.get("bbox_2d"), it.get("height_m")
        if not b or h is None:
            continue
        a = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
        if a <= 0:
            continue
        best, on = it.get("support_m"), None
        for other in items:
            if other is it:
                continue
            b2, h2 = other.get("bbox_2d"), other.get("height_m")
            if not b2 or h2 is None or h2 > h - min_gap:
                continue                      # not a real step underneath
            ix = max(0.0, min(b[2], b2[2]) - max(b[0], b2[0]))
            iy = max(0.0, min(b[3], b2[3]) - max(b[1], b2[1]))
            if ix * iy / a < cover:
                continue
            if best is None or h2 > best:
                best, on = h2, other.get("name")
        if on is not None and best != it.get("support_m"):
            it["support_m"] = best
            it["support_from"] = on
    return items


def mark_blocked(items, cover=BLOCK_COVER, height_gap=BLOCK_HEIGHT_GAP):
    """Flag items that cannot be picked because something rests on top of them.

    Detection reports every object it can see, including one that is partly
    covered. The planner then treats a buried item as available and may choose
    it: on the robot a sugar box was detected with a spam can sitting on it, and
    picking the sugar would have dragged or dropped the can.

    An item is blocked when another item's box covers at least `cover` of its
    area AND that item's top is at least `height_gap` higher -- i.e. the other
    object is resting on it rather than standing beside it. Height is what makes
    this safe to decide: two neighbours whose boxes clip at the edges read
    similar heights and neither is blocked.

    Nothing is deleted. The item is real and becomes pickable as soon as what is
    on top of it is removed, which the next perception cycle sees. Only its
    availability to the planner changes.
    """
    def area(b):
        return max(0.0, (b[2] - b[0]) * (b[3] - b[1]))

    for it in items:
        it.pop("blocked_by", None)
        it["graspable"] = True

    for i, low in enumerate(items):
        b_low, h_low = low.get("bbox_2d"), low.get("height_m")
        if not b_low or h_low is None or area(b_low) <= 0:
            continue
        for j, high in enumerate(items):
            if i == j:
                continue
            b_hi, h_hi = high.get("bbox_2d"), high.get("height_m")
            if not b_hi or h_hi is None:
                continue
            if h_hi < h_low + height_gap:
                continue                      # beside it, or underneath it
            ix = max(0.0, min(b_low[2], b_hi[2]) - max(b_low[0], b_hi[0]))
            iy = max(0.0, min(b_low[3], b_hi[3]) - max(b_low[1], b_hi[1]))
            frac = ix * iy / area(b_low)
            if frac >= cover:
                low["graspable"] = False
                low["blocked_by"] = high.get("name")
                # The evidence, so a wrong call can be read off the log rather
                # than guessed at: a HEIGHT DIFFERENCE barely over height_gap
                # with a small overlap is two neighbours, one of them simply
                # taller, not one resting on the other.
                low["blocked_evidence"] = {
                    "overlap_frac": round(frac, 3),
                    "height_gap_m": round(h_hi - h_low, 4),
                    "top_support_m": high.get("support_m"),
                    "top_thickness_m": (None if high.get("support_m") is None
                                        else round(h_hi - high["support_m"], 4)),
                }
                break
    return items


def height_implausible(name, height_m, ranges, lo=0.6, hi=1.4):
    """True when a measured height cannot belong to the named product.

    A REJECTION test, not a correction: the ranges come from a handful of
    captured poses per product and several products overlap (a gelatin box and
    a tomato soup can share 5.6-8.4 cm), so this can say "that is not a soup
    can" but never "that is a gelatin box". Margins are wide because the pose
    sample is small.
    """
    r = (ranges or {}).get(name)
    if r is None or height_m is None:
        return False, None
    lo_m, hi_m = r[0] * lo, r[1] * hi
    if height_m < lo_m or height_m > hi_m:
        return True, "{} measures {:.3f} m; captured poses span {:.3f}-{:.3f} m".format(
            name, height_m, r[0], r[1])
    return False, None


def to_metres(depth_raw):
    """RealSense aligned depth arrives as uint16 millimetres; float frames are
    already metres. Decide by dtype, never by value, so a genuinely close
    object cannot be mistaken for the other encoding."""
    a = np.asarray(depth_raw)
    if a.dtype.kind in "ui":
        return a.astype("float32") / 1000.0
    return a.astype("float32")


# --------------------------------------------------------------------------
def _selftest():
    ok = True

    def check(c, msg):
        print("  {} {}".format("PASS" if c else "FAIL", msg))
        return c

    def scene(objs, table=1.200, tilt=0.0, noise=0.0, shape=(480, 640), seed=0):
        """Synthetic top-down depth. `tilt` is metres of table drop across the
        image width, so the 37 mm of real table tilt can be reproduced."""
        rng = np.random.RandomState(seed)
        ys, xs = np.mgrid[0:shape[0], 0:shape[1]]
        d = (table + tilt * (xs / float(shape[1]))).astype("float32")
        for (x1, y1, x2, y2, h) in objs:
            d[y1:y2, x1:x2] -= h
        if noise:
            d += rng.normal(0, noise, shape).astype("float32")
        return d

    def height(d, box, crop=None):
        pl, pi = fit_table_plane(d, crop)
        if pl is None:
            return None, pi
        return object_height_m(d, box, pl)

    print("depth_grasp self-check")
    BOX = (300, 200, 380, 280)

    h, _ = height(scene([BOX + (0.150,)]), BOX)
    ok &= check(h is not None and abs(h - 0.150) < 0.002,
                "recovers 0.150 m ({})".format("None" if h is None else "{:.4f}".format(h)))

    h, _ = height(scene([BOX + (0.150,)], noise=0.004, seed=1), BOX)
    ok &= check(h is not None and abs(h - 0.150) < 0.010,
                "within 1 cm under 4 mm depth noise ({:.4f})".format(h or -1))

    d = scene([BOX + (0.150,)])
    d[np.random.RandomState(2).rand(*d.shape) < 0.35] = 0.0
    h, _ = height(d, BOX)
    ok &= check(h is not None and abs(h - 0.150) < 0.005,
                "tolerates 35% missing depth ({:.4f})".format(h or -1))

    h, _ = height(scene([BOX + (0.020,)]), BOX)
    ok &= check(h is not None and abs(h - 0.020) < 0.003, "reads a 2 cm flat object")

    # THE CASE THE RING METHOD GOT WRONG: a box standing on another box must
    # report its height above the TABLE, not above the box under it.
    d = scene([(300, 200, 400, 300, 0.060)])
    d[220:280, 320:380] -= 0.070                      # +0.070 on top of 0.060
    h, _ = height(d, (320, 220, 380, 280))
    ok &= check(h is not None and abs(h - 0.130) < 0.005,
                "STACKED object reads 0.130 m above the TABLE ({:.4f}); a local "
                "ring would say 0.070".format(h or -1))

    # tilt: their real table drops 37 mm corner to corner
    d = scene([BOX + (0.150,)], tilt=0.037)
    h, _ = height(d, BOX)
    ok &= check(h is not None and abs(h - 0.150) < 0.004,
                "absorbs 37 mm of table tilt ({:.4f})".format(h or -1))
    flat_err = abs((1.200 - (1.200 + 0.037 * (340 / 640.0) - 0.150)) - 0.150)
    ok &= check(flat_err > 0.015,
                "  (a single flat table depth would be {:.0f} mm out there)".format(flat_err * 1000))

    # a neighbour does not corrupt a fitted plane the way it corrupts a ring
    d = scene([BOX + (0.150,), (382, 200, 460, 280, 0.220)])
    h, _ = height(d, BOX)
    ok &= check(h is not None and abs(h - 0.150) < 0.005,
                "unaffected by a taller neighbour touching the box ({:.4f})".format(h or -1))

    # crowded table: objects over most of the view
    d = scene([(x, y, x + 70, y + 70, 0.12) for x in range(60, 600, 90)
               for y in range(60, 420, 90)])
    pl, pi = fit_table_plane(d)
    ok &= check(pl is not None and abs(plane_depth_at(pl, 320, 240) - 1.200) < 0.005,
                "still finds the table when objects cover most of it")

    # failure directions all fall back rather than guess
    for name, d, box in (("flat table alone", scene([]), BOX),
                         ("implausibly tall", scene([BOX + (0.600,)]), BOX)):
        h, i = height(d, box)
        ok &= check(h is None, "{} -> no grasp height ({})".format(name, i.get("reason")))
    h, i = height(np.zeros((480, 640), dtype="float32"), BOX)
    ok &= check(h is None, "all-invalid depth refused ({})".format(i.get("reason")))

    mm = (scene([BOX + (0.150,)]) * 1000).astype("uint16")
    h, _ = height(to_metres(mm), BOX)
    ok &= check(h is not None and abs(h - 0.150) < 0.002, "uint16 millimetres converted")

    # ---- the thin-stack case: gelatin box on a tuna can ---------------
    # tuna can 3 cm on the table, gelatin box 2 cm on top of it.
    d = scene([(280, 180, 400, 300, 0.030)])          # the can
    d[210:270, 310:370] -= 0.020                      # the box on top
    BOXTOP = (310, 210, 370, 270)
    pl, _ = fit_table_plane(d)
    top, _ = object_height_m(d, BOXTOP, pl)
    sup, _ = support_height_m(d, BOXTOP, pl)
    ok &= check(top is not None and abs(top - 0.050) < 0.004,
                "stack top reads 0.050 m ({:.4f})".format(top or -1))
    ok &= check(sup is not None and abs(sup - 0.030) < 0.004,
                "the surface under it reads 0.030 m -- the can ({:.4f})".format(sup or -1))
    g = grip_depth_m(top, sup)
    ok &= check(abs(g - 0.016) < 0.004,
                "grip depth capped to 80% of the box's own 2 cm ({:.3f} m)".format(g))
    lift = max(0.0, top - g)
    ok &= check(lift > 0.030,
                "so the fingertips sit at {:.3f} m, ABOVE the can's 0.030 m top; "
                "with a fixed 5 cm grip depth the lift would be 0.000 and the "
                "jaws would close on both".format(lift))
    ok &= check(max(0.0, 0.050 - 0.05) == 0.0, "  (confirming the old behaviour was 0.000)")

    # a lone object on the table: no support, full grip depth
    d = scene([BOX + (0.150,)])
    pl, _ = fit_table_plane(d)
    sup, _ = support_height_m(d, BOX, pl)
    ok &= check(sup is not None and sup < 0.010, "a lone object rests on the table ({:.4f})".format(sup))
    ok &= check(abs(grip_depth_m(0.150, sup) - 0.05) < 1e-9, "and keeps the full 5 cm grip depth")

    # ---- nested duplicate boxes (the chocolate pudding case) ----------
    whole = {"name": "chocolate pudding box", "bbox_2d": [800, 470, 905, 600], "height_m": 0.04}
    label = {"name": "chocolate pudding box", "bbox_2d": [812, 480, 878, 592], "height_m": 0.04}
    kept, dropped = suppress_nested([whole, label])
    ok &= check(len(kept) == 1 and kept[0] is whole,
                "a label-sized box inside the whole-product box is dropped")
    inter = 78 * 109
    union = 105 * 117 + inter - inter
    union = 105 * 117 + 78 * 109 - inter
    ok &= check(inter / float(union) < 0.70,
                "  plain IoU is {:.2f}, under any usual NMS threshold -- which is "
                "why both survived".format(inter / float(union)))
    ok &= check((78 * 109) / float(105 * 117) > 0.60,
                "  and the inner box is {:.0f}% of the outer's area -- an area-ratio "
                "guard let it through on the robot".format(100 * (78 * 109) / float(105 * 117)))
    kept_nd, _ = suppress_nested([dict(whole, height_m=None), dict(label, height_m=None)])
    ok &= check(len(kept_nd) == 1, "still dropped when no depth is available")

    # two of the SAME product genuinely stacked must both survive
    lower = {"name": "sugar box", "bbox_2d": [300, 200, 420, 320], "height_m": 0.04}
    upper = {"name": "sugar box", "bbox_2d": [330, 230, 390, 290], "height_m": 0.09}
    kept, _ = suppress_nested([lower, upper])
    ok &= check(len(kept) == 2, "two stacked sugar boxes are BOTH kept (heights differ)")

    # different products nested: a small item on a big one, never suppressed
    kept, _ = suppress_nested([{"name": "cracker box", "bbox_2d": [300, 200, 420, 320]},
                               {"name": "tuna can", "bbox_2d": [330, 230, 390, 290]}])
    ok &= check(len(kept) == 2, "a different product nested inside is never suppressed")

    # ---- height plausibility gate ------------------------------------
    R = {"tomato soup can": [0.056, 0.097, 6], "gelatin dessert box": [0.030, 0.084, 6]}
    bad, why = height_implausible("tomato soup can", 0.030, R)
    ok &= check(bad, "a 3 cm object called a tomato soup can is flagged")
    ok &= check(not height_implausible("tomato soup can", 0.070, R)[0], "a 7 cm one is not")
    ok &= check(not height_implausible("gelatin dessert box", 0.030, R)[0],
                "and the same 3 cm object IS plausible as a gelatin box")
    ok &= check(not height_implausible("bag of oranges", 0.100, R)[0],
                "an unmeasured product is never flagged")

    # --- which way to turn the wrist -----------------------------------
    # A parallel jaw closes along one axis, so for a box the wrist angle decides
    # whether it fits between the fingers at all.
    TBL = 1.638
    wide = scene([(300, 200, 420, 260, 0.050)], table=TBL)   # 120 px x 60 px
    pl_w = fit_table_plane(wide)[0]
    yaw_w, minor_w, major_w, _ = footprint_axes(wide, [300, 200, 420, 260], pl_w, 0.050)
    ok &= check(yaw_w is not None and abs(((yaw_w - 90.0 + 90) % 180) - 90) < 10,
                "a box wider than it is deep closes along Y (yaw {:.0f})".format(yaw_w or -1))
    ok &= check(minor_w < major_w, "minor is the short side ({:.0f} vs {:.0f} mm)".format(
        minor_w * 1000, major_w * 1000))
    tall = scene([(300, 200, 360, 320, 0.050)], table=TBL)   # 60 px x 120 px
    pl_t2 = fit_table_plane(tall)[0]
    yaw_t, minor_t, _, _ = footprint_axes(tall, [300, 200, 360, 320], pl_t2, 0.050)
    ok &= check(abs(yaw_t % 180) < 10 or abs(yaw_t % 180 - 180) < 10,
                "turning the box 90 deg turns the grasp 90 deg (yaw {:.0f})".format(yaw_t))
    ok &= check(abs(minor_w - minor_t) < 0.010,
                "and the width to span is the same either way")
    # the width band decides what can be picked at all
    ok &= check(grasp_width_ok(0.090)[0], "a 90 mm span fits the jaws")
    ok &= check(not grasp_width_ok(0.200)[0], "a 200 mm span does not")
    ok &= check(not grasp_width_ok(0.010)[0], "nor does something too thin to hold")
    ok &= check(grasp_width_ok(None)[0], "an unmeasured object is never refused")
    ok &= check(footprint_axes(None, [0, 0, 10, 10], pl_w, 0.05)[0] is None,
                "no depth -> no yaw, caller keeps its existing order")

    # --- what is it standing ON? ---------------------------------------
    # The tuna-can-on-coffee-can failure: the ring around the tuna straddles
    # the can rim and the table, support reads ~0, and the grip takes its full
    # 50 mm -- 20 mm past the tuna's underside, into the coffee can.
    stack = [{"name": "coffee can", "height_m": 0.130,
              "bbox_2d": [440, 300, 515, 375], "support_m": 0.0},
             {"name": "tuna can", "height_m": 0.160,
              "bbox_2d": [448, 308, 508, 368], "support_m": 0.002}]
    ok &= check(0.160 - grip_depth_m(0.160, 0.002) < 0.130,
                "WITHOUT it the fingertips land below the coffee can's top")
    infer_support_from_neighbours(stack)
    tuna = stack[1]
    ok &= check(abs(tuna["support_m"] - 0.130) < 1e-9,
                "the tuna's support is raised to the coffee can's top")
    ok &= check(0.160 - grip_depth_m(0.160, tuna["support_m"]) > 0.130,
                "and the fingertips now sit inside the tuna can")
    ok &= check(stack[0].get("support_from") is None and stack[0]["support_m"] == 0.0,
                "the can underneath is left alone -- support is only ever raised")
    apart = [{"name": "tuna can", "height_m": 0.035,
              "bbox_2d": [100, 100, 160, 160], "support_m": 0.001},
             {"name": "coffee can", "height_m": 0.130,
              "bbox_2d": [400, 300, 475, 375], "support_m": 0.0}]
    infer_support_from_neighbours(apart)
    ok &= check(apart[0]["support_m"] == 0.001,
                "two objects side by side do not invent a support for each other")

    # --- something on top of it ----------------------------------------
    # A buried item must not be offered to the planner.
    stacked = [{"name": "sugar box", "height_m": 0.035, "bbox_2d": [480, 320, 595, 420]},
               {"name": "spam can", "height_m": 0.070, "bbox_2d": [485, 390, 595, 505]}]
    mark_blocked(stacked)
    ok &= check(stacked[0]["graspable"] is False
                and stacked[0]["blocked_by"] == "spam can",
                "a sugar box with a spam can on it is not offered to the planner")
    ok &= check(stacked[1]["graspable"] is True, "and the can on top still is")
    # two neighbours that merely clip at the edges are both pickable
    nbrs = [{"name": "sugar box", "height_m": 0.035, "bbox_2d": [300, 300, 400, 400]},
            {"name": "cracker box", "height_m": 0.120, "bbox_2d": [396, 300, 496, 400]}]
    mark_blocked(nbrs)
    ok &= check(all(i["graspable"] for i in nbrs),
                "two neighbours touching at the edge are both still pickable")
    # equal heights side by side: neither blocks the other
    same = [{"name": "tuna can", "height_m": 0.035, "bbox_2d": [300, 300, 400, 400]},
            {"name": "spam can", "height_m": 0.036, "bbox_2d": [340, 320, 440, 420]}]
    mark_blocked(same)
    ok &= check(all(i["graspable"] for i in same),
                "overlapping boxes at the same height are not a stack")
    # with no depth nothing is blocked, which is the old behaviour
    nod2 = [{"name": "sugar box", "bbox_2d": [480, 320, 595, 420]},
            {"name": "spam can", "bbox_2d": [485, 390, 595, 505]}]
    mark_blocked(nod2)
    ok &= check(all(i["graspable"] for i in nod2), "no depth -> nothing is blocked")

    # --- one object, two names -----------------------------------------
    # The detector can put two boxes with DIFFERENT names on one item when it
    # cannot decide between two similar products.
    twin = [{"name": "gelatin dessert box", "name_confidence": 0.75,
             "bbox_2d": [720, 255, 830, 400], "height_m": 0.030},
            {"name": "chocolate pudding box", "name_confidence": 0.42,
             "bbox_2d": [722, 258, 828, 398], "height_m": 0.031}]
    kept, gone = suppress_overlapping([dict(t) for t in twin])
    ok &= check(len(kept) == 1 and kept[0]["name"] == "gelatin dessert box",
                "one object with two names keeps only the confident name")
    # ...but a real cross-class stack must survive: a tuna can ON a spam can
    stack = [{"name": "spam can", "name_confidence": 0.89,
              "bbox_2d": [300, 300, 400, 400], "height_m": 0.035},
             {"name": "tuna can", "name_confidence": 0.83,
              "bbox_2d": [305, 305, 395, 395], "height_m": 0.070}]
    kept2, gone2 = suppress_overlapping([dict(t) for t in stack])
    ok &= check(len(kept2) == 2, "a tuna can stacked on a spam can keeps both boxes")
    # without depth, nothing cross-class is dropped
    nod = [{"name": "gelatin dessert box", "name_confidence": 0.75,
            "bbox_2d": [720, 255, 830, 400]},
           {"name": "chocolate pudding box", "name_confidence": 0.42,
            "bbox_2d": [722, 258, 828, 398]}]
    ok &= check(len(suppress_overlapping([dict(t) for t in nod])[0]) == 2,
                "with no height at all, a cross-class pair is left alone")
    # same class still merges with no depth, as before
    dup = [{"name": "potato chip can", "name_confidence": 0.94,
            "bbox_2d": [300, 300, 400, 400]},
           {"name": "potato chip can", "name_confidence": 0.47,
            "bbox_2d": [305, 305, 395, 395]}]
    ok &= check(len(suppress_overlapping([dict(t) for t in dup])[0]) == 1,
                "same-class merging is unchanged when there is no depth")

    # --- parallax correction on the grasp pixel ------------------------
    # A raised object is imaged outward from its footprint, so the aim must be
    # pulled back towards the camera axis by (H-h)/H.
    TABLE = 1.638                      # the real camera-to-table depth
    PRIN = (313.3, 242.8)              # the real principal point
    tall = (540, 300, 580, 340, 0.235)   # chip-can height, near the table edge
    flat = (540, 300, 580, 340, 0.005)   # same place, almost no height
    d_tall = scene([tall], table=TABLE)
    d_flat = scene([flat], table=TABLE)
    pl_t = fit_table_plane(d_tall)[0]
    pl_f = fit_table_plane(d_flat)[0]
    box = [540, 300, 580, 340]
    gx_t, gy_t, it = grasp_pixel(d_tall, box, pl_t, 0.235, PRIN)
    gx_f, gy_f, if_ = grasp_pixel(d_flat, box, pl_f, 0.005, PRIN)
    cx_box = 560.0
    exp = PRIN[0] + (cx_box - PRIN[0]) * (TABLE - 0.235) / TABLE
    ok &= check(it["corrected"] and abs(gx_t - exp) < 1.0,
                "tall object: aim matches (H-h)/H, {:.1f} px vs {:.1f}".format(gx_t, exp))
    ok &= check(gx_t < cx_box, "the correction pulls IN towards the camera axis")
    ok &= check(it["shift_px"] > 30, "and it is large: {:.0f} px".format(it["shift_px"]))
    ok &= check(if_["shift_px"] < 2,
                "a flat object at the same place barely moves ({:.1f} px)".format(if_["shift_px"]))
    # directly under the camera there is no parallax at any height
    ctr = (int(PRIN[0]) - 20, int(PRIN[1]) - 20, int(PRIN[0]) + 20, int(PRIN[1]) + 20)
    d_c = scene([ctr + (0.235,)], table=TABLE)
    gx_c, gy_c, ic = grasp_pixel(d_c, list(ctr), fit_table_plane(d_c)[0], 0.235, PRIN)
    ok &= check(ic["shift_px"] < 2.0,
                "an object under the camera axis is not moved ({:.1f} px)".format(ic["shift_px"]))
    # without depth the behaviour is exactly what it was before
    gx_n, gy_n, inf_n = grasp_pixel(d_tall, box, None, 0.235, PRIN)
    ok &= check(not inf_n["corrected"] and gx_n == cx_box,
                "no plane -> box centre, the old behaviour")
    gx_d, gy_d, inf_d = grasp_pixel(None, box, pl_t, 0.235, PRIN)
    ok &= check(not inf_d["corrected"] and gx_d == cx_box,
                "a frame with no depth at all returns the box centre, not a crash")
    gx_h, gy_h, inf_h = grasp_pixel(d_tall, box, pl_t, None, PRIN)
    ok &= check(not inf_h["corrected"] and gx_h == cx_box, "no height -> box centre")
    ok &= check(grasp_pixel(d_tall, box, pl_t, 99.0, PRIN)[2]["corrected"] is False,
                "an object taller than the camera is refused, not inverted")

    # THE ROBOT FAILURE: a tuna can on a coffee can. The coffee's box is filled
    # by the tuna, so an unmasked read gives BOTH the tuna's height, they merge
    # as one duplicated object, and the tuna is then gripped at table height.
    COFFEE = (300, 200, 380, 280)
    TUNA = (315, 215, 365, 265)
    d = scene([])                      # paint, not subtract: they overlap
    for (bx1, by1, bx2, by2), hh in ((COFFEE, 0.130), (TUNA, 0.167)):
        d[by1:by2, bx1:bx2] = 1.200 - hh
    pl, _ = fit_table_plane(d, None)
    h_un, _ = object_height_m(d, COFFEE, pl)
    ok &= check(h_un is not None and abs(h_un - 0.167) < 0.005,
                "unmasked, the coffee can wrongly reads the tuna on top of it "
                "({:.3f})".format(h_un or -1))
    h_m, i_m = object_height_m(d, COFFEE, pl, other_boxes=[TUNA])
    ok &= check(h_m is not None and abs(h_m - 0.130) < 0.005,
                "masked, it reads its own rim ({:.3f})".format(h_m or -1))
    ok &= check(i_m.get("from") == "rim", "and says the reading came from the rim")
    h_t, _ = object_height_m(d, TUNA, pl, other_boxes=[COFFEE])
    ok &= check(h_t is not None and abs(h_t - 0.167) < 0.005,
                "the tuna still reads its own top ({:.3f})".format(h_t or -1))
    a = {"name": "coffee can", "bbox_2d": list(COFFEE), "height_m": h_m,
         "name_confidence": 0.83}
    b = {"name": "tuna can", "bbox_2d": list(TUNA), "height_m": h_t,
         "name_confidence": 0.90}
    kept, drop = suppress_overlapping([a, b])
    ok &= check(len(kept) == 2, "both survive suppression, the stack is not merged")
    infer_support_from_neighbours(kept)
    thick = b["height_m"] - (b.get("support_m") or 0.0)
    ok &= check(b.get("support_from") == "coffee can" and abs(thick - 0.037) < 0.006,
                "the tuna stands on the coffee can, {:.3f} m thick".format(thick))
    ok &= check(grip_depth_m(b["height_m"], b["support_m"]) < 0.037,
                "so the fingers stay inside it instead of taking the can below")

    # Same stack, but the detector named the top can after the bottom one --
    # which is what the robot actually did. Nothing may key on the name.
    h_same, _ = object_height_m(d, COFFEE, pl, other_boxes=[TUNA])
    ok &= check(h_same is not None and abs(h_same - 0.130) < 0.005,
                "a stack of two SAME-named cans is still measured per can")
    c2 = {"name": "coffee can", "bbox_2d": list(COFFEE), "height_m": h_same,
          "name_confidence": 0.83}
    t2 = {"name": "coffee can", "bbox_2d": list(TUNA), "height_m": h_t,
          "name_confidence": 0.90}
    ok &= check(len(suppress_overlapping([c2, t2])[0]) == 2,
                "and both survive, though the names match")
    # A label-only box inside a product box: same surface, so masking it must
    # not change the product's height.
    LABEL = (320, 220, 360, 260)
    d2 = scene([])
    d2[200:280, 300:380] = 1.200 - 0.150
    h_l, _ = object_height_m(d2, (300, 200, 380, 280), pl, other_boxes=[LABEL])
    ok &= check(h_l is not None and abs(h_l - 0.150) < 0.005,
                "a nested label box does not disturb its own product ({:.3f})".format(h_l or -1))

    # ONE box over a whole stack, which is what the detector usually gives.
    # No second detection exists, so only depth inside that box can find the
    # top object's underside.
    st, si = step_support_m(d, COFFEE, pl, 0.167, max_own_height=0.156)
    ok &= check(st is not None and abs(st - 0.130) < 0.006,
                "one box over a stack: the shelf under the top can is found "
                "({})".format("None" if st is None else "{:.3f}".format(st)))
    ok &= check(st is not None and grip_depth_m(0.167, st) < 0.037,
                "so the fingers stay above the can below")

    # A tuna can on a potato chip can. The shelf IS there and is found when the
    # gate is open, but the chip can's own captured range spans 0.032-0.270 m
    # (standing and lying in one range), so a 0.240 m reading is inside what
    # the can reaches alone and the gate refuses it. That is the cost of the
    # gate, and it is the right trade: on the robot this stack picked correctly
    # through the ordinary full-depth grip, while the gate is what stops a lone
    # lying can being mistaken for a stack.
    d4 = scene([])
    d4[190:290, 290:390] = 1.200 - 0.213          # chip can lid
    d4[215:265, 315:365] = 1.200 - 0.240          # tuna on top of it
    st4, _ = step_support_m(d4, (290, 190, 390, 290), pl, 0.240)
    ok &= check(st4 is not None and abs(st4 - 0.213) < 0.006,
                "a tuna on a chip can: shelf at {}".format(
                    "None" if st4 is None else "{:.3f}".format(st4)))
    st4g, si4g = step_support_m(d4, (290, 190, 390, 290), pl, 0.240,
                                max_own_height=0.270)
    ok &= check(st4g is None,
                "but a chip can's own range covers 0.240 m, so it is left to "
                "the ordinary grip ({})".format(si4g.get("reason")))

    # A lone flat object has no second level and must be left alone.
    st2, si2 = step_support_m(scene([BOX + (0.150,)]), BOX, pl, 0.150)
    ok &= check(st2 is None, "a lone object reports no shelf ({})".format(si2.get("reason")))

    # A tall can off the camera axis shows its own SIDE WALL below the top.
    # That is a ramp, not a shelf, and reading it as support would rob every
    # tall object of grip depth.
    d3 = scene([])
    for k in range(40):               # wall sloping 0 -> 0.21 across 40 px
        d3[200:280, 300 + k] = 1.200 - 0.210 * (k + 1) / 40.0
    d3[200:280, 340:380] = 1.200 - 0.210
    st3, si3 = step_support_m(d3, BOX, pl, 0.210)
    ok &= check(st3 is None, "a side wall is not a shelf ({})".format(si3.get("reason")))

    # THE FALSE POSITIVE FROM THE ROBOT: a coffee can lying on its side, alone.
    # Its curved flank passed every shape test, so only its own known height
    # can rule the stack out. The numbers are the ones the robot printed.
    st5, si5 = step_support_m(d, COFFEE, pl, 0.089, max_own_height=0.156)
    ok &= check(st5 is None,
                "a lone lying can is not called a stack ({})".format(si5.get("reason")))

    # THE DUPLICATE-BOX FAILURE: one tuna can detected twice, the two boxes
    # reading heights 5 microns apart. The second box is not a support.
    dup = [{"name": "tuna can", "bbox_2d": [318, 163, 367, 219],
            "height_m": 0.0311050, "support_m": 0.0},
           {"name": "tomato soup can", "bbox_2d": [319, 164, 368, 220],
            "height_m": 0.0311000, "support_m": 0.0}]
    infer_support_from_neighbours(dup)
    ok &= check(dup[0].get("support_from") is None,
                "a duplicate box of the same object is not its support")
    ok &= check(grip_depth_m(0.031, 0.031) == 0.05,
                "and a zero thickness falls back to the full grip depth")
    ok &= check(abs(grip_depth_m(0.050, 0.030) - 0.016) < 1e-9,
                "while a real 0.020 m thickness still gives 0.8*t")

    print("SELF-CHECK {}".format("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

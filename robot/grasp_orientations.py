#!/usr/bin/env python
"""Candidate gripper orientations for Baxter top-down picking.

The original code demanded ONE rigid straight-down orientation
(x=0, y=1, z=0, w=0) on every IK call. That is one constraint too many:
Baxter loses a large slice of its reachable workspace at range, so far-side
table poses come back INVALID even though the arm could physically get there
with the wrist turned, or leaning a few degrees.

This module returns an ORDERED list of orientations to try. Order matters:

  1. pure vertical, four wrist yaws -- still a perfect top-down grasp, only
     the jaw axis rotates. Free for cylinders (cans, bottles); for boxes it
     just changes which face gets gripped.
  2. small tilts (15 deg), then larger (30 deg). These buy reach, but the
     approach is no longer perpendicular to the table, so they come last.

Self-check the quaternion math with:  python grasp_orientations.py
"""
from __future__ import division

import math

# Straight down = 180 deg about the base Y axis. Stored (w, x, y, z).
# In ROS Quaternion(x=0, y=1, z=0, w=0) -- the original hardcoded value.
STRAIGHT_DOWN = (0.0, 0.0, 1.0, 0.0)


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _axis_quat(axis, angle_rad):
    c, s = math.cos(angle_rad / 2.0), math.sin(angle_rad / 2.0)
    if axis == "x":
        return (c, s, 0.0, 0.0)
    if axis == "y":
        return (c, 0.0, s, 0.0)
    if axis == "z":
        return (c, 0.0, 0.0, s)
    raise ValueError("axis must be one of x, y, z")


def _pre(axis, deg):
    """Straight-down pose rotated by `deg` about a BASE-frame axis."""
    return _quat_mul(_axis_quat(axis, math.radians(deg)), STRAIGHT_DOWN)


def rotate(q, v):
    """Rotate 3-vector v by quaternion q=(w,x,y,z)."""
    qv = (0.0, v[0], v[1], v[2])
    conj = (q[0], -q[1], -q[2], -q[3])
    r = _quat_mul(_quat_mul(q, qv), conj)
    return (r[1], r[2], r[3])


def tilt_from_vertical_deg(q):
    """Angle between the gripper approach axis and straight down, in degrees."""
    ax = rotate(q, (0.0, 0.0, 1.0))          # gripper +z in base frame
    return math.degrees(math.acos(max(-1.0, min(1.0, -ax[2]))))


def candidates(allow_tilt=True, yaw_deg=None):
    """Ordered list of (label, (w, x, y, z)) orientations to try in IK.

    Index 0 is exactly the original hardcoded orientation, so with
    allow_tilt=False, yaw_deg=None and a single-candidate cut the behaviour is
    unchanged.

    `yaw_deg` is a REQUESTED wrist angle, measured from perception as the
    direction the jaws should close along -- the object's short axis. A parallel
    jaw only opens so far, so for anything not round this decides whether the
    object fits between the fingers: a sugar box spans 76-118 mm across its short
    axis and 131-210 mm along its long one, and only the first fits. When given,
    that angle is tried FIRST, and the fixed yaws follow as fallbacks for when it
    is unreachable.
    """
    out = []
    if yaw_deg is not None:
        out.append(("down_yaw{:.0f}_measured".format(yaw_deg % 180),
                    _pre("z", yaw_deg % 180)))
    out += [("down_yaw{:d}".format(d), _pre("z", d)) for d in (0, 90, 45, 135)]
    if allow_tilt:
        for mag in (15, 30):
            for axis in ("x", "y"):
                for sign in (1, -1):
                    out.append((
                        "tilt{:d}_{}{}".format(mag, axis, "+" if sign > 0 else "-"),
                        _pre(axis, sign * mag),
                    ))
    return out


def to_ros_xyzw(q):
    """(w,x,y,z) -> (x,y,z,w) for geometry_msgs Quaternion."""
    return (q[1], q[2], q[3], q[0])


def _selftest():
    ok = True

    def check(cond, msg):
        global_ok = cond
        print("  {} {}".format("PASS" if cond else "FAIL", msg))
        return global_ok

    print("grasp_orientations self-check")

    # 1. yaw 0 must reproduce the original hardcoded orientation exactly.
    _, q0 = candidates()[0]
    ok &= check(all(abs(a - b) < 1e-12 for a, b in zip(q0, STRAIGHT_DOWN)),
                "yaw0 == original straight-down {}".format(STRAIGHT_DOWN))
    ok &= check(to_ros_xyzw(q0) == (0.0, 1.0, 0.0, 0.0),
                "yaw0 in ROS xyzw order == (0,1,0,0)")

    # 2. every candidate is a unit quaternion.
    for label, q in candidates():
        n = math.sqrt(sum(c * c for c in q))
        ok &= check(abs(n - 1.0) < 1e-9, "{} unit norm ({:.12f})".format(label, n))

    # 3. every "down_*" candidate points the gripper straight down.
    for label, q in candidates():
        if not label.startswith("down_"):
            continue
        t = tilt_from_vertical_deg(q)
        ok &= check(t < 1e-9, "{} is vertical (tilt {:.9f} deg)".format(label, t))

    # 4. tilt candidates tilt by exactly their labelled magnitude.
    for label, q in candidates():
        if not label.startswith("tilt"):
            continue
        want = float(label[4:6])
        t = tilt_from_vertical_deg(q)
        ok &= check(abs(t - want) < 1e-6, "{} tilts {:.4f} deg (want {:.0f})".format(label, t, want))

    # 5. the four yaws must be four DISTINCT jaw axes (else they are useless).
    axes = [rotate(q, (1.0, 0.0, 0.0)) for lbl, q in candidates() if lbl.startswith("down_")]
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            d = abs(sum(a * b for a, b in zip(axes[i], axes[j])))
            ok &= check(d < 0.999, "yaw axes {} and {} differ (|dot|={:.4f})".format(i, j, d))

    # a measured yaw is tried first, and the fixed order still follows it
    base = candidates(allow_tilt=False)
    meas = candidates(allow_tilt=False, yaw_deg=37.0)
    ok &= check(len(meas) == len(base) + 1, "a measured yaw ADDS a candidate")
    ok &= check("measured" in meas[0][0], "and it is tried first ({})".format(meas[0][0]))
    ok &= check(meas[1:] == base, "the existing order is untouched behind it")
    ok &= check(candidates(allow_tilt=False, yaw_deg=None) == base,
                "no measured yaw -> byte-identical to before")
    # 180 deg of yaw is the whole range: a jaw axis repeats every half turn
    ok &= check(candidates(yaw_deg=200.0)[0][1] == candidates(yaw_deg=20.0)[0][1],
                "yaw wraps at 180 deg, since the jaw axis is symmetric")

    print("SELF-CHECK {}".format("PASSED" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())

#!/usr/bin/env python
from __future__ import division

import sys
import os
import json
import math
import time
import signal
import subprocess
import struct

import rospy
import actionlib
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from baxter_core_msgs.srv import SolvePositionIK, SolvePositionIKRequest
import baxter_interface
from control_msgs.msg import GripperCommandAction, GripperCommandGoal

# Gripper feedback. Optional: without it every holding() answer is "unknown"
# and the arm behaves exactly as it did before.
try:
    from baxter_core_msgs.msg import EndEffectorState
except Exception:
    EndEffectorState = None

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import grasp_orientations

# ---------------------------------------------------------------------------
# Motion tuning. All heights in metres, in Baxter base frame.
# ---------------------------------------------------------------------------
# Every lateral move happens at ONE height: tallest box rim + this clearance,
# so nothing sweeps through the boxes. Higher keeps the ELBOW clear but costs
# reach; map the trade-off with:  python reach_test.py --sweep
TRAVEL_CLEARANCE = float(os.environ.get("BAXTER_TRAVEL_CLEARANCE", 0.10))
# Release the item this far above the box rim.
DROP_MARGIN      = 0.03
# Two-stage descent: pause this far above the grasp before the final approach.
APPROACH_OFFSET  = 0.08
# Fingertip depth below an object's top. Grip near the top so the mass hangs
# BELOW the jaws instead of pivoting out on the lift. Capped at 80% of measured
# thickness, else a 2 cm box on a 3 cm can is gripped at the table, jaws round both.
GRIP_DEPTH       = 0.05
MIN_GRIP_DEPTH   = 0.010
# Extra depth for a cylinder lying on its side, past its widest point, and how
# far the fingertips must stay above whatever it rests on.
ROUND_GRIP_EXTRA = 0.028
TIP_CLEARANCE    = 0.015
# Thinnest object worth gripping. A thickness under this says the ceiling PC's
# support estimate is wrong, not that the object is a wafer -- the flattest
# thing in the catalog is a plate at 17 mm.
MIN_THICKNESS    = 0.010
# Fallbacks for a gripper that does not set its gripping flag. Measured on this
# arm: an EMPTY close reads position 4.0% / force 0.0, a held chip can reads
# 13.1% / 30.2. Position is the weaker of the two -- it depends on how wide the
# object is -- so force is tried first.
# Is the jaw POSITION allowed to decide a grip on its own, when force and the
# gripping flag both read nothing?
#
# Leave it True and a held item survives the gripper action timing out -- the
# case where force collapses to 0 with the item plainly in the jaws (the chip
# can did this every run). The cost is that an item removed in mid-air is only
# noticed once the jaws are re-commanded closed over the box.
#
# Set it False for force-only: anything not actively squeezing counts as gone.
# Cleaner while every product in the scene stalls the motor properly, and wrong
# the moment one of them times out instead.
GRIP_TRUST_POSITION = True
# Jaw opening with NOTHING in the jaws, in percent. Measured on this gripper
# at startup; this is only the fallback if that measurement cannot be taken.
GRIP_EMPTY_POS   = 4.0
# How far above the empty close the jaws must sit for something to be in them.
# Readings off this arm: empty 4.0, chip can held 6.5 / 6.6 / 7.1, spam 13.1,
# chip can pulled out of the jaws 5.3. 1.5 puts the line at 5.5, between the
# last two.
GRIP_HOLD_MARGIN = 1.5
GRIP_HOLD_FORCE  = 5.0
# Let the arm stop and the gripper publish a fresh state before reading it.
# Without this the check ran while the arm was still swinging over the box, on
# whatever message happened to have arrived last.
GRIP_SETTLE_S    = 1.0
# Baxter reports these as 0 / 1 / 2 (false / true / unknown).
_EE_TRUE, _EE_FALSE = 1, 0
# Refuse a measured object height above this; fall back to the table.
MAX_OBJECT_H     = 0.30
# A move is only a success if the endpoint actually got this close (metres).
ARRIVAL_TOL      = 0.030
# ...and the wrist must BE vertical, not just the tip in place. IK returning a
# vertical solution does not mean the controller achieved it.
ORIENT_TOL_DEG   = 8.0
# One IK waypoint per this much Cartesian distance.
WAYPOINT_STEP_M  = 0.08
MAX_WAYPOINTS    = 8
# Loose joint tolerance for pass-through waypoints keeps travel smooth;
# the final waypoint of every move uses the tight default.
LOOSE_JOINT_TOL  = 0.05
# Tilt buys reach, but a tilted grasp lands one finger high and knocks the
# object over. A reported FAILED is recoverable, a crushed item is not, so
# anything only a tilted pose can reach is failed instead. Travel may still
# tilt; only the final descent and close are forced vertical.
ALLOW_TILT_ON_GRASP = False
# Stream pass-through waypoints instead of stopping at each one (removes the
# stutter). False = blocking move at every waypoint.
STREAM_WAYPOINTS = True
# Seconds spent blending through each pass-through waypoint while streaming.
WAYPOINT_DWELL_S = 0.12
# Joint speed for the final, accuracy-critical move of each leg.
MOVE_SPEED = 0.6
# Warn on a large weighted joint-radian jump between waypoints: that is the arm
# reconfiguring, which throws the ELBOW across the table.
JOINT_JUMP_WARN = 1.5
# "Path blocked" = a straight line to a far box cuts across Baxter's own
# shoulder, not a travel-height problem. move_to_pose retries those as an ARC,
# triggered by IK actually failing, so nothing to tune per setup.
# Weights below: proximal joints swing the elbow, the wrist barely does.
JOINT_WEIGHTS = {"s0": 3.0, "s1": 3.0, "e0": 2.0, "e1": 2.0,
                 "w0": 0.6, "w1": 0.6, "w2": 0.3}

# Python 2 and Python 3 compatibility for input/raw_input
try:
    input = raw_input
except NameError:
    pass

class GripperClient(object):
    def __init__(self, gripper):
        ns = 'robot/end_effector/' + gripper + '_gripper/'
        self._client = actionlib.SimpleActionClient(
            ns + "gripper_action",
            GripperCommandAction,
        )
        self._goal = GripperCommandGoal()

        if not self._client.wait_for_server(rospy.Duration(10.0)):
            rospy.logerr("Gripper action server for {} not found. Exiting...".format(gripper))
            rospy.signal_shutdown("Action server not found")
            sys.exit(1)
        self.clear()

    def command(self, position, effort):
        self._goal.command.position = position
        self._goal.command.max_effort = effort
        self._client.send_goal(self._goal)
        self._client.wait_for_result(rospy.Duration(1.0))        

    def clear(self):
        self._goal = GripperCommandGoal()


def _solve3(M, rhs):
    """Gaussian elimination with partial pivoting on a 3x3 system."""
    a = [list(M[i]) + [rhs[i]] for i in range(3)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(a[r][c]))
        if abs(a[piv][c]) < 1e-15:
            return None
        a[c], a[piv] = a[piv], a[c]
        for r in range(3):
            if r == c:
                continue
            f = a[r][c] / a[c][c]
            for k in range(c, 4):
                a[r][k] -= f * a[c][k]
    return [a[i][3] / a[i][i] for i in range(3)]


def fit_affine(pts):
    """Least-squares fit of pixel -> Baxter (x, y, z) as a full affine map.

    pts: [(px, py, bx, by, bz), ...], at least 3 points.

    The old 2-point model had only two scale factors, which forces the
    assumption that the ceiling camera's axes are a PURE SWAP of Baxter's --
    no rotation, no shear, and identical scale on both axes. Measured on the
    real rig none of that holds: about 1.2 deg of camera rotation and 6.7%
    different scales, which alone put the tip 25 mm off at the far corner.
    That is most of a gripper jaw, and no amount of careful touching fixes it
    because the model cannot represent the error.

    Fitting z as a plane too means the pick height follows a table that is not
    perfectly level, instead of using one averaged height everywhere.

    Returns {"x": [c0,c1,c2], "y": [...], "z": [...]} where
    value = c0*px + c1*py + c2, or None if the points are degenerate
    (e.g. all collinear -- two points on the same row cannot pin down a map).
    """
    if len(pts) < 3:
        return None
    S = [[0.0] * 3 for _ in range(3)]
    for px, py, _, _, _ in pts:
        row = (px, py, 1.0)
        for i in range(3):
            for j in range(3):
                S[i][j] += row[i] * row[j]
    out = {}
    for idx, name in ((2, "x"), (3, "y"), (4, "z")):
        rhs = [0.0, 0.0, 0.0]
        for pt in pts:
            row = (pt[0], pt[1], 1.0)
            for i in range(3):
                rhs[i] += row[i] * pt[idx]
        c = _solve3(S, rhs)
        if c is None:
            return None
        out[name] = c
    return out


def affine_apply(aff, key, px, py):
    c = aff[key]
    return c[0] * px + c[1] * py + c[2]


# Which way the jaws close at down_yaw0 -- a hardware convention, observed not
# derived. 90 on this robot: at 0 everything was gripped along its LONG axis.
# Not a measurement error; the width gate reads the same minor axis and was
# passing sugar boxes, which it could not have done had the axes been swapped.
JAW_YAW_OFFSET_DEG = 90.0


def route_xy(x0, y0, x1, y1, mode, a):
    """Point a fraction `a` along the lateral route from (x0,y0) to (x1,y1).

    "line" is the straight one. "arc" interpolates radius and bearing instead,
    sweeping AROUND the base rather than cutting across it -- the motion the
    arm makes with the elbow out. Both the planner and the obstacle check call
    this, so what is checked for clearance is the path that actually gets flown.
    """
    if mode == "arc":
        r0, th0 = math.hypot(x0, y0), math.atan2(y0, x0)
        r1, th1 = math.hypot(x1, y1), math.atan2(y1, x1)
        dth = math.atan2(math.sin(th1 - th0), math.cos(th1 - th0))   # shortest way round
        r, th = r0 + a * (r1 - r0), th0 + a * dth
        return r * math.cos(th), r * math.sin(th)
    return x0 + a * (x1 - x0), y0 + a * (y1 - y0)


def image_yaw_to_baxter(aff, yaw_deg):
    """An angle measured in the camera image, expressed in Baxter's frame.

    Not a shift: the calibration found ~1.2 deg of rotation, 6.7% different
    scales per axis, and an axis SWAP. Directions are not preserved, so an
    image angle used directly as a Baxter yaw grips across the wrong axis.
    The affine's linear part maps displacements, hence directions too.
    Returns None with no affine fit, leaving the caller on its fixed order.
    """
    if not aff or yaw_deg is None:
        return None
    th = math.radians(yaw_deg)
    dx, dy = math.cos(th), math.sin(th)
    ax, ay = aff["x"][0], aff["x"][1]
    bx, by = aff["y"][0], aff["y"][1]
    rx = ax * dx + ay * dy
    ry = bx * dx + by * dy
    if abs(rx) < 1e-12 and abs(ry) < 1e-12:
        return None
    return (math.degrees(math.atan2(ry, rx)) + JAW_YAW_OFFSET_DEG) % 180.0


class BaxterExecutor(object):
    def __init__(self, limb_name="left", calib_file="baxter_calib.json", prompt=True):
        self.limb_name = limb_name
        self.calib_file = calib_file
        self.prompt = prompt
        
        # Start gripper action server
        self._start_gripper_action_server()
        
        # Initialize limb and gripper interfaces
        print("Initializing Baxter limb: {}...".format(self.limb_name))
        self.limb = baxter_interface.Limb(self.limb_name)
        self.gripper = GripperClient(self.limb_name)

        # Is anything actually between the jaws? Used to tell a failed grasp
        # from a dropped item, so the planner can be told the item was never
        # packed instead of assuming it was.
        self._ee_state = None
        if EndEffectorState is not None:
            rospy.Subscriber(
                "/robot/end_effector/{}_gripper/state".format(self.limb_name),
                EndEffectorState, self._ee_state_cb, queue_size=1)
        
        # Initialize IK service
        ik_ns = "ExternalTools/{}/PositionKinematicsNode/IKService".format(self.limb_name)
        self.iksvc = rospy.ServiceProxy(ik_ns, SolvePositionIK)
        self.ikreq = SolvePositionIKRequest()
        
        # Calibration state
        self.calib_data = None
        self.last_orientation = None
        self._warned_rim = False
        self.load_calibration()
        
        # Prompt for calibration if needed
        if not self.prompt:
            if not self.calib_data:
                raise ValueError("No calibration at {} and prompt=False".format(self.calib_file))
        elif not self.calib_data:
            self.calibrate()
        else:
            print("\nCalibration found. Options:")
            print("  [f] full recalibration (pixel mapping + boxes + home)")
            print("  [b] boxes only  <-- run this once to record real box RIM heights")
            print("  [p] refit from stored points (fix a mistyped pixel, no re-touching)")
            print("  [h] home pose only")
            print("  [r] carry route (transit + one pose above each box)")
            print("  [ENTER] keep everything as is")
            choice = input("Choice: ").strip().lower()
            if choice in ('f', 'y', 'yes'):
                self.calibrate()
            elif choice == 'b':
                self.calibrate_boxes()
            elif choice == 'p':
                self.refit_calibration()
            elif choice == 'h':
                self.record_home_pose()
            elif choice == 'r':
                self.record_transit_poses()

        boxes = (self.calib_data or {}).get("drop_boxes", {})
        if boxes and not all(self.has_taught_route(b) for b in boxes.values()):
            print("\n" + "=" * 62)
            print("  NO CARRY ROUTE TAUGHT FOR EVERY BOX")
            print("=" * 62)
            print("  Without it the arm plans the table->box move with IK alone.")
            print("  It picks whatever elbow configuration the solver returns,")
            print("  which is the roll close to the body that sweeps the table")
            print("  and knocks the boxes over. Teaching it takes two minutes and")
            print("  replaces that with the same fixed motion every time.")
            try:
                ans = input("\n  Teach the carry route now? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                ans = "n"
            if ans in ("", "y", "yes"):
                self.record_transit_poses()
            else:
                print("  Skipped. The arm will fly the Cartesian route.")

    def _start_gripper_action_server(self):
        try:
            print("Starting Baxter gripper action server...")
            subprocess.Popen(["rosrun", "baxter_interface", "gripper_action_server.py"])
            time.sleep(2.0)
        except Exception as e:
            rospy.logerr("Failed to start gripper action server: {}".format(e))

    def load_calibration(self):
        if os.path.exists(str(self.calib_file)):
            try:
                with open(str(self.calib_file), "r") as f:
                    self.calib_data = json.load(f)
                
                # Migrate older calibration format with single drop box to the new multi-box format
                if "drop_x_bax" in self.calib_data and "drop_boxes" not in self.calib_data:
                    self.calib_data["drop_boxes"] = {
                        "0": {
                            "x": self.calib_data["drop_x_bax"],
                            "y": self.calib_data["drop_y_bax"],
                            "z": self.calib_data["drop_z_bax"]
                        }
                    }
                print("Calibration loaded successfully from {}:".format(self.calib_file))
                for k, v in self.calib_data.items():
                    print("  {}: {}".format(k, v))
            except Exception as e:
                print("Failed to read calibration: {}".format(e))

    def calibrate(self):
        print("\n=== BAXTER N-POINT COORDINATE CALIBRATION ===")
        print("Touch the gripper tip to several points whose camera pixel you know")
        print("(the corners of the marked rectangle are ideal), and the pixel->Baxter")
        print("map is fitted to all of them at once.")
        print("")
        print("Use 4 or more. With 2 the fit has no rotation term at all, and on this")
        print("rig that alone puts the tip 25 mm off at the far corner. With 3 the fit")
        print("is exact and CANNOT report its own error. From 4 you get a real")
        print("leave-one-out estimate of how wrong it will be on a fresh point.")
        try:
            n = int(input("\nHow many calibration points? (2-8, 4 recommended): ").strip() or "4")
            n = max(2, min(8, n))
        except ValueError:
            n = 4

        pts = []
        for i in range(n):
            print("\n--- Calibration point {} of {} ---".format(i + 1, n))
            try:
                px = float(input("  Camera pixel X: ").strip())
                py = float(input("  Camera pixel Y: ").strip())
            except ValueError:
                print("  Bad input, skipping this point.")
                continue
            dup = [q for q in pts if abs(q[0] - px) < 0.5 and abs(q[1] - py) < 0.5]
            if dup:
                # Two different physical touches labelled with one pixel is a
                # contradiction, and the least-squares fit answers it by
                # splitting the difference -- silently mapping that pixel to
                # the midpoint between them. Caught in the lab only because
                # the arm visibly went to the middle of the table edge.
                print("  *** Pixel ({:.0f}, {:.0f}) was already used for a point at".format(px, py))
                print("  *** x={:+.4f} y={:+.4f}. Two touches cannot share one pixel.".format(dup[0][2], dup[0][3]))
                print("  *** Check the pixel you typed. Re-entering this point.")
                continue
            input("  Touch the gripper TIP to the physical spot at pixel ({:.0f}, {:.0f}), then ENTER...".format(px, py))
            pos = self.limb.endpoint_pose()['position']
            pts.append((px, py, pos.x, pos.y, pos.z))
            print("  Recorded: x={:+.4f} y={:+.4f} z={:+.4f}".format(pos.x, pos.y, pos.z))

        if len(pts) < 2:
            print("Not enough points; calibration aborted.")
            return

        print("\n--- Points collected ---")
        for i, (px, py, bx, by, bz) in enumerate(pts):
            print("  {}: pixel ({:6.1f},{:6.1f})  ->  x={:+.4f} y={:+.4f} z={:+.4f}".format(
                i, px, py, bx, by, bz))

        existing_home = self.calib_data.get("home_joint_angles", None) if self.calib_data else None
        mean_z = sum(p[4] for p in pts) / len(pts)
        self.calib_data = {
            "ref_x_img": pts[0][0], "ref_y_img": pts[0][1],
            "ref_x_bax": pts[0][2], "ref_y_bax": pts[0][3],
            "ref_z_bax": mean_z,
            "calib_points": [list(p) for p in pts],
        }

        aff = fit_affine(pts)
        if aff is None:
            if len(pts) >= 3:
                print("\n*** Could not fit a map to those points -- they are probably")
                print("*** collinear. Calibration points must NOT all sit on one row,")
                print("*** column, or line. Use opposite corners.")
            # legacy 2-scale fallback
            (p1x, p1y, b1x, b1y, _), (p2x, p2y, b2x, b2y, _) = pts[0], pts[-1]
            dx_img, dy_img = p2x - p1x, p2y - p1y
            self.calib_data["scale_x"] = ((b2x - b1x) / dy_img) if abs(dy_img) > 1e-5 else -0.0030
            self.calib_data["scale_y"] = ((b2y - b1y) / dx_img) if abs(dx_img) > 1e-5 else -0.0030
            print("\nUsing the legacy 2-scale model (no rotation term).")
        else:
            self.calib_data["affine"] = {k: list(v) for k, v in aff.items()}
            # keep legacy fields populated so older tooling still reads something sane
            self.calib_data["scale_x"] = aff["x"][1]
            self.calib_data["scale_y"] = aff["y"][0]
            if not self._report_fit(pts, aff):
                print("\nThis calibration is WORSE THAN USELESS -- it will send the arm")
                print("to the wrong place on every pick. The usual cause is a mistyped")
                print("pixel: check the points listed above for two entries with the")
                print("same pixel, or a pixel that does not match where you touched.")
                if input("Save it anyway? (y/N): ").strip().lower() not in ("y", "yes"):
                    print("Calibration NOT saved. Nothing changed.")
                    return

        self.calib_data["drop_boxes"] = self._prompt_drop_boxes()
        if existing_home:
            self.calib_data["home_joint_angles"] = existing_home

        with open(str(self.calib_file), "w") as f:
            json.dump(self.calib_data, f, indent=2)
        print("Calibration successfully saved to {}.\n".format(self.calib_file))

        rec_home = input("Do you want to record current position as the HOME pose joint angles? (Y/n): ").strip().lower()
        if rec_home not in ('n', 'no'):
            self.record_home_pose()

    def _report_fit(self, pts, aff):
        """Residuals, plus a leave-one-out estimate of error on a FRESH point.

        In-sample residuals flatter the fit: with 3 points it is exact and they
        are all zero, which says nothing about accuracy anywhere else. Refitting
        without each point in turn and predicting it is the honest number.
        """
        print("\n--- Fit quality ---")
        worst = 0.0
        for px, py, bx, by, _ in pts:
            e = math.sqrt((affine_apply(aff, "x", px, py) - bx) ** 2 +
                          (affine_apply(aff, "y", px, py) - by) ** 2)
            worst = max(worst, e)
            print("  pixel ({:6.1f},{:6.1f})  residual {:5.1f} mm".format(px, py, e * 1000))

        loo = []
        if len(pts) >= 4:
            for i in range(len(pts)):
                sub = pts[:i] + pts[i + 1:]
                a2 = fit_affine(sub)
                if a2 is None:
                    continue
                px, py, bx, by, _ = pts[i]
                loo.append(math.sqrt((affine_apply(a2, "x", px, py) - bx) ** 2 +
                                     (affine_apply(a2, "y", px, py) - by) ** 2))

        # Camera rotation relative to Baxter -- the term the 2-point model
        # cannot express, and the reason it was 25 mm out on this rig.
        rot = math.degrees(math.atan2(aff["x"][0], -aff["y"][0]))
        print("  implied camera rotation vs Baxter: {:+.2f} deg".format(rot))
        zs = [p[4] for p in pts]
        print("  table height spread across the touched points: {:.1f} mm".format(
            (max(zs) - min(zs)) * 1000))

        if loo:
            m = sum(loo) / len(loo)
            print("\n  LEAVE-ONE-OUT error on an unseen point: mean {:.1f} mm, worst {:.1f} mm".format(
                m * 1000, max(loo) * 1000))
            if m < 0.010:
                print("  EXCELLENT -- the mapping is not your centring problem.")
            elif m < 0.020:
                print("  ACCEPTABLE for grasping.")
            else:
                print("  TOO LARGE. Re-touch the points more carefully, or add more of them.")
        else:
            print("\n  Only {} points: the fit is exact by construction and CANNOT".format(len(pts)))
            print("  measure its own accuracy. Redo with 4+ to get a real error estimate.")
        return worst < 0.050

    def _prompt_drop_boxes(self):
        """Record each box's CENTRE and its RIM height.

        The rim height is the point of this: every lateral move is planned
        above the tallest rim, so the arm cannot sweep through the boxes.
        The old procedure recorded only wherever the arm happened to be
        hand-held "above" each box, which has no defined relation to the rim.
        """
        drop_boxes = {}
        try:
            num_boxes = int(input("\nHow many drop boxes/bags to calibrate? (1-4): ").strip() or "4")
            num_boxes = max(1, min(4, num_boxes))
        except ValueError:
            num_boxes = 4

        for i in range(num_boxes):
            print("\n--- Drop Box {} ---".format(i))
            print("Touch the gripper tip to the TOP RIM of box {} (the highest point".format(i))
            print("the arm must clear), roughly over the middle of its opening.")
            input("Press ENTER when the tip is ON THE RIM of box {}...".format(i))
            pose = self.limb.endpoint_pose()['position']
            drop_boxes[str(i)] = {"x": pose.x, "y": pose.y, "z": pose.z, "rim_z": pose.z}
            print("Recorded Box {}: x={:.4f}, y={:.4f}, rim_z={:.4f}".format(i, pose.x, pose.y, pose.z))

        rims = [b["rim_z"] for b in drop_boxes.values()]
        print("\nTallest rim {:.4f} m -> travel height will be {:.4f} m".format(
            max(rims), max(rims) + TRAVEL_CLEARANCE))
        return drop_boxes

    def refit_calibration(self):
        """Refit from the points already touched, after fixing a mistyped pixel.

        The touches are stored in the calibration file, so a wrong pixel LABEL
        costs nothing to fix -- there is no need to go back to the robot and
        touch four corners again.
        """
        pts = [tuple(q) for q in (self.calib_data or {}).get("calib_points", [])]
        if len(pts) < 3:
            print("No stored calibration points to refit (need 3+). Run full calibration.")
            return False

        while True:
            print("\n--- Stored calibration points ---")
            for i, (px, py, bx, by, bz) in enumerate(pts):
                print("  {}: pixel ({:6.1f},{:6.1f})  ->  x={:+.4f} y={:+.4f} z={:+.4f}".format(
                    i, px, py, bx, by, bz))
            seen = {}
            for i, q in enumerate(pts):
                key = (round(q[0]), round(q[1]))
                if key in seen:
                    print("  !! points {} and {} share pixel {} but were touched {:.0f} mm".format(
                        seen[key], i, key,
                        1000 * math.sqrt((pts[i][2] - pts[seen[key]][2]) ** 2 +
                                         (pts[i][3] - pts[seen[key]][3]) ** 2)))
                    print("  !! apart. One of those pixels is mistyped.")
                seen[key] = i

            aff = fit_affine(pts)
            if aff is not None:
                self._report_fit(pts, aff)

            ans = input("\nFix a point's pixel? (index, or ENTER to accept): ").strip()
            if ans == "":
                break
            try:
                idx = int(ans)
                npx = float(input("  correct pixel X: ").strip())
                npy = float(input("  correct pixel Y: ").strip())
            except (ValueError, IndexError):
                print("  bad input.")
                continue
            if not (0 <= idx < len(pts)):
                print("  no such point.")
                continue
            pts[idx] = (npx, npy) + pts[idx][2:]

        aff = fit_affine(pts)
        if aff is None:
            print("Still cannot fit those points. Run a full calibration.")
            return False
        self.calib_data["calib_points"] = [list(q) for q in pts]
        self.calib_data["affine"] = {k: list(v) for k, v in aff.items()}
        self.calib_data["scale_x"] = aff["x"][1]
        self.calib_data["scale_y"] = aff["y"][0]
        self.calib_data["ref_x_img"], self.calib_data["ref_y_img"] = pts[0][0], pts[0][1]
        self.calib_data["ref_x_bax"], self.calib_data["ref_y_bax"] = pts[0][2], pts[0][3]
        self.calib_data["ref_z_bax"] = sum(q[4] for q in pts) / len(pts)
        with open(str(self.calib_file), "w") as f:
            json.dump(self.calib_data, f, indent=2)
        print("\nRefit saved to {}.".format(self.calib_file))
        return True

    def calibrate_boxes(self):
        """Re-record the boxes only, keeping the pixel mapping and home pose."""
        if not self.calib_data:
            print("No calibration loaded; run full calibration first.")
            return False
        self.calib_data["drop_boxes"] = self._prompt_drop_boxes()
        self._warned_rim = False
        with open(str(self.calib_file), "w") as f:
            json.dump(self.calib_data, f, indent=2)
        print("Drop boxes saved to {}.\n".format(self.calib_file))
        return True

    def pixel_to_baxter(self, px_x, px_y):
        if not self.calib_data:
            raise ValueError("Calibration data is missing!")

        aff = self.calib_data.get("affine")
        if aff:
            return affine_apply(aff, "x", px_x, px_y), affine_apply(aff, "y", px_x, px_y)

        # Legacy 2-point model: pure axis swap, no rotation, no shear.
        dx_img = px_x - self.calib_data["ref_x_img"]
        dy_img = px_y - self.calib_data["ref_y_img"]
        x_bax = self.calib_data["ref_x_bax"] + dy_img * self.calib_data["scale_x"]
        y_bax = self.calib_data["ref_y_bax"] + dx_img * self.calib_data["scale_y"]
        return x_bax, y_bax

    def pixel_to_z(self, px_x, px_y):
        """Table height under a pixel. With an affine calibration this follows
        the fitted table plane; otherwise it is the single averaged height."""
        aff = self.calib_data.get("affine") if self.calib_data else None
        if aff:
            return affine_apply(aff, "z", px_x, px_y)
        return self.calib_data["ref_z_bax"]

    def _ee_state_cb(self, msg):
        self._ee_state = msg

    def grip_state_str(self):
        """What the gripper reports, for the log. The thresholds here are set
        from catalog geometry, not measured on this gripper -- printing the raw
        numbers on every pick is how they get calibrated against a real sponge
        and a real can instead of guessed."""
        st = self._ee_state
        if st is None:
            return "no feedback"
        return "position {:.1f}%  force {}  gripping={} missed={}".format(
            float(getattr(st, "position", -1.0)),
            getattr(st, "force", "?"),
            getattr(st, "gripping", "?"), getattr(st, "missed", "?"))

    def empty_grip_pos(self):
        return getattr(self, "_empty_grip_pos", None) or GRIP_EMPTY_POS

    def measure_empty_grip(self):
        """Close the empty jaws once and record where they stop.

        Every later "is something in there?" is this number plus a margin. It
        is measured rather than assumed because it is the only reading that
        survives the gripper action timing out, and because fingers can be
        re-mounted at a different spacing between sessions.
        """
        self.open_gripper()
        rospy.sleep(0.5)
        self.close_gripper()
        rospy.sleep(GRIP_SETTLE_S)
        pos = self.grip_position()
        self.open_gripper()
        if pos is None:
            print("[GRIP] No gripper feedback; assuming empty close at {:.1f}%.".format(
                GRIP_EMPTY_POS))
            return None
        self._empty_grip_pos = pos
        print("[GRIP] Empty jaws close at {:.1f}%. Anything above {:.1f}% is "
              "an item.".format(pos, pos + GRIP_HOLD_MARGIN))
        return pos

    def grip_position(self):
        """Jaw opening in percent, or None. The one reading that does not decay."""
        st = self._ee_state
        pos = None if st is None else getattr(st, "position", None)
        return None if pos is None else float(pos)

    def verify_grip(self, label, pick_pos=None, reclose=True):
        """Re-close the jaws, let them settle, then read. True/False/None.

        Reading the gripper some moves after the grasp does not work, and the
        chip can shows every reason why. Its close command reports "Gripper
        Command Not Achieved in Allotted Time": the motor gives up part way,
        stops driving, and force decays to 0. From then on the state says
        nothing about whether the can is there -- held it read 6.9% / force 0,
        pulled out by hand it read 5.3% / force 0. Position cannot separate
        1.6 percentage points, and force and `gripping` are both flat.

        So the jaws are commanded closed AGAIN before reading. The motor drives
        once more: against the item if it is there, to its stop if it is not,
        and force and `gripping` are live measurements rather than the remains
        of a command that timed out a move ago. Re-closing something already
        held costs nothing -- it is the same effort the grasp applied.

        `reclose` is off for the check after the lift and ON for the one over
        the box, and the difference is how much time has passed.

        Right after the lift the grasp's own goal may still be running. A new
        goal preempts it, the server says "Gripper Action Preempted", and the
        reading collapses to force 0 with the can still held -- a false loss.
        Nothing needs re-commanding there anyway: the jaws have not been asked
        to do anything since they closed on the item.

        Over the box the grasp goal is long finished, and re-closing is the
        only way to learn anything. The close command times out against an item
        it can never squeeze to zero, the motor stops driving, and the jaws
        then sit where they stopped no matter what happens to the item -- an
        item removed in mid-air left the reading at 6.1%, unchanged. Commanding
        the close again makes the jaws move: onto the item if it is there, down
        to the empty close if it is not.

        `pick_pos` is printed for context only. Comparing against it was tried
        and is WORSE than comparing against the empty close: a can that slips
        out reads 5.3% against a 6.9% grasp, 1.6 points, which no tolerance can
        separate from noise -- but 5.3% against an empty close of 4.0% is
        clearly empty.
        """
        if reclose:
            self.close_gripper()
        rospy.sleep(GRIP_SETTLE_S)
        print("      gripper {}: {}{}".format(
            label, self.grip_state_str(),
            "" if pick_pos is None else "   (was {:.1f}% at the grasp)".format(pick_pos)))
        return self.holding()

    def holding(self, use_position=True):
        """True / False / None -- is something in the jaws?

        None means the gripper never reported, so nothing is concluded and the
        caller carries on.

        `gripping` is the answer whenever it is a definite 0 or 1: Baxter sets
        it in firmware, where the encoder and the current sensor are. Measured
        on this arm, an empty close reports gripping=0 and a held can reports
        gripping=1, cleanly.

        `missed` is NOT a substitute. On this gripper it stays 0 even on a
        grasp that closed on nothing, so an earlier version that concluded
        "empty" only from missed=1 reported a missed chip can as SUCCESS.
        """
        st = self._ee_state
        if st is None:
            return None
        if getattr(st, "missed", None) == _EE_TRUE:
            return False

        # Force OUTRANKS the flag, in the holding direction only. An object
        # narrow enough that the jaws close almost all the way -- the chip can
        # is right at that limit -- can look to the firmware like the jaws
        # reached their commanded position, so gripping drops to 0 while the
        # can is plainly held. An empty close reads force 0.0 exactly, so any
        # real squeeze means something is in there whatever the flag says.
        force = getattr(st, "force", None)
        if force is not None and float(force) > GRIP_HOLD_FORCE:
            return True
        if getattr(st, "gripping", None) == _EE_TRUE:
            return True

        # Neither flag says yes -- which does NOT mean no. `close_gripper`
        # commands position 0, and with an item in the jaws that can never be
        # reached, so the action server eventually reports "Gripper Command Not
        # Achieved in Allotted Time", aborts, and force falls to 0. On this arm
        # the SAME chip can read 6.5% / force 29 before that timeout and 6.6% /
        # force 0 after it. Force and `gripping` are a race against the
        # timeout; the jaws are not.
        if not GRIP_TRUST_POSITION:
            return False                      # force-only: not squeezing, not held
        pos = getattr(st, "position", None)
        if pos is None or not use_position:
            return None
        return float(pos) > self.empty_grip_pos() + GRIP_HOLD_MARGIN

    def open_gripper(self):
        self.gripper.command(position=100.0, effort=50.0)

    def close_gripper(self):
        self.gripper.command(position=0.0, effort=50.0)

    def record_home_pose(self):
        """Record current joint angles as the home position and save to calibration file."""
        print("\n=== RECORD HOME JOINT ANGLES ===")
        print("Move Baxter's {} arm to the desired HOME position.".format(self.limb_name))
        input("Press ENTER when the arm is in position...")

        try:
            current_angles = self.limb.joint_angles()
        except Exception as e:
            print("Error reading joint angles: {}".format(e))
            return False

        if not current_angles:
            print("Error: Could not read joint angles from Baxter.")
            return False

        if self.calib_data is None:
            self.calib_data = {}

        self.calib_data["home_joint_angles"] = current_angles
        print("Recorded Home Joint Angles:")
        for joint, angle in sorted(current_angles.items()):
            print("  {}: {:.4f} rad".format(joint, angle))

        with open(str(self.calib_file), "w") as f:
            json.dump(self.calib_data, f, indent=2)
        print("Home position joint angles saved to {}.\n".format(self.calib_file))
        return True

    def _read_joints(self, what):
        """Current joint angles, or None with a printed reason."""
        try:
            j = self.limb.joint_angles()
        except Exception as e:
            print("  Could not read joint angles for {}: {}".format(what, e))
            return None
        if not j:
            print("  Baxter returned no joint angles for {}.".format(what))
            return None
        return j

    def _teach_pose(self, what, instruction):
        """Hand-guide the arm, press ENTER, keep the joint angles."""
        print("\n--- {} ---".format(what))
        print(instruction)
        input("Press ENTER when the arm is in position...")
        j = self._read_joints(what)
        if j:
            print("  Recorded: " + "  ".join(
                "{}={:+.3f}".format(k.split("_")[-1], v) for k, v in sorted(j.items())))
        return j

    def record_transit_poses(self):
        """Teach the carry route as JOINT poses instead of leaving it to IK.

        The long move from the table to a box is where the arm hit things. Run
        as Cartesian waypoints it is re-planned every time -- the IK service
        returns one of many 7-DOF solutions, so the elbow lands somewhere
        different run to run, and the same scene sweeps wide once and cuts
        across the table the next. Replaying a taught joint pose does the same
        thing every time, which is the property that was missing.

        Two kinds of pose: ONE transit pose (arm extended and high, clear of
        everything) and one per box (still extended, above that box). The carry
        becomes lift -> transit -> box pose -> descend.
        """
        if not self.calib_data:
            print("No calibration loaded; run full calibration first.")
            return False
        print("\n=== TEACH THE CARRY ROUTE ===")
        print("Hand-guide the arm (cuff button) to each pose. Keep the gripper")
        print("pointing DOWN and the arm well clear of the table and the boxes.")

        t = self._teach_pose(
            "TRANSIT pose",
            "Extend the arm up and out, clear of the table and of every box.\n"
            "This is the pose the arm passes through carrying an item.")
        if not t:
            return False
        self.calib_data["transit_joint_angles"] = t

        boxes = self.calib_data.get("drop_boxes", {})
        if not boxes:
            print("No drop boxes calibrated; do that first ('b').")
            return False
        for key in sorted(boxes, key=lambda k: int(k)):
            j = self._teach_pose(
                "ABOVE BOX {}".format(key),
                "Keeping the arm extended, swing it over box {} and stop high\n"
                "above the opening -- do NOT descend into it.".format(key))
            if not j:
                return False
            boxes[key]["transit_joint_angles"] = j

        with open(str(self.calib_file), "w") as f:
            json.dump(self.calib_data, f, indent=2)
        print("\nCarry route saved to {}.".format(self.calib_file))
        print("Watch it once empty with the motion test before running the pipeline.\n")
        return True

    def has_taught_route(self, box):
        """True when this box can be reached by replaying taught joint poses."""
        return bool((self.calib_data or {}).get("transit_joint_angles")
                    and box.get("transit_joint_angles"))

    def move_joints(self, angles, what, keep_wrist_roll=True, speed=0.4):
        """Replay a taught joint pose. No IK, no route choice, no elbow lottery.

        `keep_wrist_roll` holds w2 at its CURRENT value. w2 is the jaw axis: the
        grasp turned it to fit the object between the fingers, and twisting it
        while holding is how items are dropped. Every other joint -- including
        w0/w1, which is what brings a tilted grasp back to vertical -- comes
        from the taught pose.
        """
        angles = dict(angles)
        if keep_wrist_roll:
            now = self._read_joints("wrist roll") or {}
            for name in list(angles):
                if name.endswith("w2") and name in now:
                    angles[name] = now[name]
        try:
            self.limb.set_joint_position_speed(speed)
            self.limb.move_to_joint_positions(angles, timeout=15.0)
            return True
        except Exception as e:
            print("  Could not reach the taught {} pose: {}".format(what, e))
            return False

    def move_to_home(self):
        # Retract before any joint-space home move, so the arm does not
        # sweep laterally across the table on its way back.
        try:
            if self.calib_data:
                self.retract()
        except Exception as e:
            rospy.logwarn("Retract before home failed: {}".format(e))

        if self.calib_data and "home_joint_angles" in self.calib_data:
            print("[HOME] Moving arm to recorded home joint positions...")
            try:
                self.limb.set_joint_position_speed(0.3)
                self.limb.move_to_joint_positions(self.calib_data["home_joint_angles"], timeout=10.0)
                print("[HOME] Arm is at home position.")
                return True
            except Exception as e:
                print("[HOME] Warning: Could not move to saved home joint positions: {}".format(e))
                print("[HOME] Falling back to Cartesian home pose.")

        x = 0.3
        if self.limb_name == "left":
            y = 0.65
        elif self.limb_name == "right":
            y = -0.65
        z = 0.15
        print("[HOME] Moving arm to home/reference position (hover)...")
        ok = self.move_to_pose(x, y, z)
        if ok:
            print("[HOME] Arm is at home position.")
        else:
            print("[HOME] Warning: could not reach home position.")
        return ok

    def plan_path(self, x_target, y_target, z_target, allow_tilt=True, n_steps=None,
                  mode="line", prefer=None, max_skip=None):
        """Solve the WHOLE straight-line path before moving a millimetre, with
        ONE gripper orientation held for the entire path.

        Three properties this buys:

        1. Plan-or-park. The old code moved to a waypoint, solved the next,
           moved again. If waypoint 5 of 8 had no solution the arm was already
           halfway there, holding an item, and just stopped. Now a path either
           plans completely or the arm never leaves.
        2. ONE orientation per path. An earlier version fell back to a
           different orientation at any waypoint the preferred one failed --
           so the wrist could rotate or tilt mid-carry, which drops items.
           Now an orientation that cannot cover the whole path is rejected and
           the next one is tried from scratch.
        3. Chained seeds. Each waypoint is seeded from the previous waypoint's
           joint solution, so the joint trajectory is continuous instead of
           hopping between redundant configurations -- that hopping is what
           swings the elbow through the groceries.

        Returns (list_of_joint_dicts, orientation_label) or (None, None).
        """
        try:
            c = self.limb.endpoint_pose()['position']
            x0, y0, z0 = c.x, c.y, c.z
            j0 = self.limb.joint_angles()
        except Exception as e:
            rospy.logwarn("Cannot read arm state to plan: {}".format(e))
            return None, None

        if n_steps is None:
            dist = math.sqrt((x_target - x0) ** 2 + (y_target - y0) ** 2 + (z_target - z0) ** 2)
            n_steps = max(1, min(MAX_WAYPOINTS, int(math.ceil(dist / WAYPOINT_STEP_M))))

        def waypoint(i):
            if i == n_steps:
                return x_target, y_target, z_target
            a = float(i) / n_steps
            zi = z0 + a * (z_target - z0)
            wx, wy = route_xy(x0, y0, x_target, y_target, mode, a)
            return wx, wy, zi

        first_failure = None
        cands = grasp_orientations.candidates(
            allow_tilt=allow_tilt, yaw_deg=getattr(self, "grasp_yaw", None))
        if prefer is not None:
            # Keep the wrist a carried item was grasped with. Rotating or
            # tilting the wrist while holding something is how items slip.
            cands = ([c for c in cands if c[0] == prefer] +
                     [c for c in cands if c[0] != prefer])
        for label, q in cands:
            # Reject cheaply: if this orientation cannot reach the goal at all,
            # there is no point planning the path that leads to it.
            if self._ik_once(x_target, y_target, z_target, q, j0) is None:
                continue
            # Intermediate waypoints only SHAPE the path -- they are not
            # requirements. Dropping one that will not solve is far better
            # than abandoning a move whose goal is perfectly reachable, which
            # is what aborting on any single IK miss did on the robot. The
            # goal itself is mandatory, and too many misses means the path
            # really is blocked, not just awkward.
            path, seed, skipped = [], j0, 0
            for i in range(1, n_steps + 1):
                xi, yi, zi = waypoint(i)
                j = self._ik_once(xi, yi, zi, q, seed)
                if j is None:
                    if i == n_steps:
                        path = None
                        break
                    skipped += 1
                    if first_failure is None:
                        first_failure = (label, i, n_steps, xi, yi, zi)
                    continue
                path.append(j)
                seed = j
            budget = max(1, n_steps // 2) if max_skip is None else max_skip
            if path is None or skipped > budget:
                continue
            if skipped:
                rospy.logwarn("{} of {} pass-through waypoints had no IK solution and were "
                              "skipped; the path is coarser here.".format(skipped, n_steps))
            jump = max([self.joint_distance(a, b) for a, b in zip([j0] + path, path)] or [0.0])
            if jump > JOINT_JUMP_WARN:
                rospy.logwarn("Large arm reconfiguration on this path ({:.2f} weighted rad). "
                              "Watch the elbow.".format(jump))
            return path, label

        if first_failure is not None:
            lbl, i, n, xi, yi, zi = first_failure
            rospy.logwarn("Path blocked: no orientation covers the whole path. "
                          "Best attempt ({}) died at waypoint {}/{} x={:.3f} y={:.3f} z={:.3f}".format(
                              lbl, i, n, xi, yi, zi))
        else:
            rospy.logwarn("No orientation reaches goal x={:.4f} y={:.4f} z={:.4f}".format(
                x_target, y_target, z_target))
        return None, None

    def execute_path(self, path, x_target=None, y_target=None, z_target=None,
                     check_vertical=False):
        """Run a planned joint path. Pass-through waypoints are STREAMED at
        100 Hz so the arm blends through them in one continuous motion --
        the old code called the blocking move_to_joint_positions at every
        waypoint, which is the stop-start stutter. Only the final waypoint
        blocks, and only the final pose is checked for arrival."""
        if not path:
            return False
        if STREAM_WAYPOINTS and len(path) > 1:
            rate = rospy.Rate(100)
            for joints in path[:-1]:
                t0 = rospy.get_time()
                while rospy.get_time() - t0 < WAYPOINT_DWELL_S and not rospy.is_shutdown():
                    self.limb.set_joint_positions(joints)
                    rate.sleep()
        else:
            for joints in path[:-1]:
                self.limb.set_joint_position_speed(MOVE_SPEED)
                self.limb.move_to_joint_positions(joints, timeout=10.0,
                                                  threshold=LOOSE_JOINT_TOL)

        self.limb.set_joint_position_speed(MOVE_SPEED)
        self.limb.move_to_joint_positions(path[-1], timeout=10.0)

        if x_target is None:
            return True
        # move_to_joint_positions returns after its timeout whether or not it
        # converged, so a timed-out move used to be reported as a SUCCESS.
        err = self.endpoint_error(x_target, y_target, z_target)
        if err is not None and err > ARRIVAL_TOL:
            rospy.logwarn("ARRIVAL FAIL - asked x={:.3f} y={:.3f} z={:.3f}, off by {:.3f} m".format(
                x_target, y_target, z_target, err))
            return False
        if check_vertical:
            tilt = self.endpoint_tilt_deg()
            if tilt is not None and tilt > ORIENT_TOL_DEG:
                rospy.logwarn("ORIENTATION FAIL - gripper is {:.1f} deg off vertical (limit {:.0f}). "
                              "One finger is higher than the other; closing here knocks the "
                              "object over.".format(tilt, ORIENT_TOL_DEG))
                return False
        return True

    def move_to_pose(self, x_target, y_target, z_target, n_steps=None, allow_tilt=True,
                     prefer=None, check_vertical=False):
        """Plan, then execute in one flowing move.

        Straight Cartesian line first -- predictable, and what you want for the
        short vertical descents. If that has no solution, retry as an ARC that
        sweeps around the base. Long table->box legs cut across the shoulder
        when drawn straight, and a straight line is not something the move
        actually needs: the whole point of travel height is that everything
        underneath is already clear.
        """
        # Priority: a CLEAN straight line, then a CLEAN arc, and only then a
        # coarse path with unsolvable waypoints dropped. Skipping waypoints on
        # a line that crosses the shoulder would let the arm jump through that
        # region in joint space -- exactly the elbow swing the arc exists to
        # avoid -- so a clean arc must outrank a coarse line.
        for mode, max_skip in (("line", 0), ("arc", 0), ("line", None), ("arc", None)):
            path, label = self.plan_path(x_target, y_target, z_target,
                                         allow_tilt=allow_tilt, n_steps=n_steps,
                                         mode=mode, prefer=prefer, max_skip=max_skip)
            if path is not None:
                if mode == "arc":
                    print("      (straight line blocked; arcing around the base)")
                self.last_orientation = label
                return self.execute_path(path, x_target, y_target, z_target,
                                         check_vertical=check_vertical)
        return False

    def travel_z(self):
        """The ONE height every lateral move happens at: above the tallest box
        rim. The old code travelled to the boxes at `box_z + 0.15`, where
        box_z was wherever the arm happened to be hand-held during
        calibration -- no defined relation to the rims, which is why the arm
        swept through them."""
        z = self.calib_data["ref_z_bax"]
        missing = []
        for key, b in self.calib_data.get("drop_boxes", {}).items():
            if "rim_z" in b:
                z = max(z, b["rim_z"])
            else:
                missing.append(key)
                z = max(z, b["z"])
        if missing and not getattr(self, "_warned_rim", False):
            self._warned_rim = True
            print("[WARN] Boxes {} have no recorded rim height (old calibration).".format(sorted(missing)))
            print("[WARN] Falling back to their hover z, which is NOT the rim. Re-run box")
            print("[WARN] calibration ('b' at startup) to record real rim heights.")
        return z + TRAVEL_CLEARANCE

    def retract(self):
        """Straight up to travel height, no lateral motion. Do this before
        every lateral move so the arm never drags across the table."""
        try:
            p = self.limb.endpoint_pose()['position']
        except Exception as e:
            rospy.logwarn("Cannot read pose to retract: {}".format(e))
            return False
        zt = self.travel_z()
        if p.z >= zt - 0.01:
            return True
        return self.move_to_pose(p.x, p.y, zt)

    def _ik_seed(self, seed_joints=None):
        """Seed IK so the solution stays near a given configuration. When
        planning a whole path we CHAIN the seeds -- each waypoint is seeded
        with the previous waypoint's solution, not with the live arm pose --
        so the returned joint trajectory is continuous and the elbow does not
        jump between redundant configurations partway along a move."""
        try:
            if seed_joints is None:
                seed_joints = self.limb.joint_angles()
            seed = JointState()
            seed.name = list(seed_joints.keys())
            seed.position = list(seed_joints.values())
            self.ikreq.seed_angles = [seed]
            self.ikreq.seed_mode = SolvePositionIKRequest.SEED_USER
        except Exception as e:
            rospy.logwarn("Could not set IK seed: {}".format(e))
            self.ikreq.seed_mode = SolvePositionIKRequest.SEED_AUTO

    def _ik_once(self, x, y, z, q, seed_joints):
        """One IK service call at one orientation. -> joint dict or None.

        Baxter's SolvePositionIK is a NUMERICAL solver, so a seeded call can
        fail to converge on a pose that is perfectly reachable -- especially
        when the seed is a near-fully-extended configuration, where the
        Jacobian is close to singular. Seeing that on the robot: the arm sat
        at 0.947 m from the shoulder and every candidate died at waypoint 1,
        11 cm away, on poses much easier than the one it was already holding.
        So a SEED_USER miss is retried with the solver's own seed before the
        pose is declared unreachable.
        """
        qx, qy, qz, qw = grasp_orientations.to_ros_xyzw(q)
        for seed_mode in ("user", "auto"):
            hdr = Header(stamp=rospy.Time.now(), frame_id='base')
            self.ikreq.pose_stamp = [PoseStamped(
                header=hdr,
                pose=Pose(position=Point(x=x, y=y, z=z),
                          orientation=Quaternion(x=qx, y=qy, z=qz, w=qw)),
            )]
            if seed_mode == "user" and seed_joints:
                self._ik_seed(seed_joints)
            else:
                self.ikreq.seed_angles = []
                self.ikreq.seed_mode = SolvePositionIKRequest.SEED_AUTO
            try:
                rospy.wait_for_service(self.iksvc.resolved_name, 5.0)
                resp = self.iksvc(self.ikreq)
            except (rospy.ServiceException, rospy.ROSException) as e:
                rospy.logerr("IK service call failed: %s" % e)
                return None
            seeds = struct.unpack('<%dB' % len(resp.result_type), resp.result_type)
            if seeds[0] != resp.RESULT_INVALID:
                return dict(zip(resp.joints[0].name, resp.joints[0].position))
        return None

    @staticmethod
    def joint_distance(a, b):
        """Weighted joint-space distance. Proximal joints dominate because
        they are what move the elbow."""
        if not a or not b:
            return 0.0
        total = 0.0
        for name, va in a.items():
            if name not in b:
                continue
            try:
                delta = abs(float(va) - float(b[name]))
            except (TypeError, ValueError):
                continue                      # non-numeric entry, not a joint
            w = 1.0
            for suffix, weight in JOINT_WEIGHTS.items():
                if name.endswith(suffix):
                    w = weight
                    break
            total += w * delta
        return total

    def solve_ik(self, x, y, z, allow_tilt=True, seed_joints=None, prefer=None):
        """First orientation IN PREFERENCE ORDER that solves. NO MOTION.

        Preference order is fixed by grasp_orientations.candidates(): yaw0
        (the original hardcoded orientation) first, then the other vertical
        yaws, then tilts. Order is NOT negotiable and is deliberately not
        traded against joint distance.

        An earlier version scored solutions by how far they moved the arm and
        took the closest. That picked yaw135 over yaw0 at the grasp, which
        rotates a PARALLEL-JAW gripper 135 degrees -- it changes which axis
        the fingers squeeze, and items were dropped. Measured on the table
        map, the non-zero yaws recover only 1-2 cells of 35; every real reach
        gain comes from tilt. Not a trade worth making.

        Configuration continuity is handled by CHAINED SEEDS in plan_path
        instead, which is where it belongs: the IK service returns a solution
        near its seed, so seeding each waypoint from the previous waypoint's
        solution keeps the elbow continuous without touching orientation.

        Returns (label, joint_dict) or (None, None).
        """
        if seed_joints is None:
            try:
                seed_joints = self.limb.joint_angles()
            except Exception:
                seed_joints = None

        cands = grasp_orientations.candidates(
            allow_tilt=allow_tilt, yaw_deg=getattr(self, "grasp_yaw", None))
        if prefer is not None:
            cands = ([c for c in cands if c[0] == prefer] +
                     [c for c in cands if c[0] != prefer])
        for label, q in cands:
            j = self._ik_once(x, y, z, q, seed_joints)
            if j is not None:
                return label, j
        return None, None

    def endpoint_z(self):
        """Fingertip height the arm actually reached, or None if unreadable."""
        try:
            return self.limb.endpoint_pose()['position'].z
        except Exception:
            return None

    def endpoint_tilt_deg(self):
        """How far the gripper's approach axis ACTUALLY is from straight down,
        read back from the arm. Returns degrees, or None if unreadable."""
        try:
            o = self.limb.endpoint_pose()['orientation']
        except Exception:
            return None
        try:
            return grasp_orientations.tilt_from_vertical_deg((o.w, o.x, o.y, o.z))
        except Exception:
            return None

    def endpoint_error(self, x, y, z):
        """Straight-line distance from the current tip to (x, y, z), or None."""
        try:
            p = self.limb.endpoint_pose()['position']
        except Exception:
            return None
        return math.sqrt((p.x - x) ** 2 + (p.y - y) ** 2 + (p.z - z) ** 2)

    def _move_to_pose_direct(self, x, y, z, allow_tilt=True, tight=True, speed=MOVE_SPEED):
        """Single-waypoint move. Kept for callers that want no interpolation."""
        label, joints = self.solve_ik(x, y, z, allow_tilt=allow_tilt)
        if label is None:
            rospy.logwarn("INVALID POSE - no orientation solved for x={:.4f}, y={:.4f}, z={:.4f}".format(x, y, z))
            return False
        self.last_orientation = label
        return self.execute_path([joints], x if tight else None, y, z)

    def execute_pick_and_place(self, px_x, px_y, bag_index=0, grasp_dz=None,
                               grasp_thickness=None, grasp_yaw=None,
                               grasp_major=None, grasp_width=None):
        if not self.calib_data:
            print("Error: Baxter calibration is missing.")
            return "FAILED"

        # Close along the object's SHORT axis (sugar box: 76-118 mm vs 131-210).
        # With a fixed jaw stroke this decides whether it fits at all. Tried
        # first, fixed yaws as fallback, and kept for the carry so the wrist is
        # not rotated while holding the object.
        self.grasp_yaw = image_yaw_to_baxter(self.calib_data.get("affine"), grasp_yaw)
        if grasp_yaw is not None:
            print("  Object short axis: {:.0f} deg in image -> wrist yaw {} (Baxter frame)".format(
                grasp_yaw,
                "unavailable, using the fixed order" if self.grasp_yaw is None
                else "{:.0f} deg".format(self.grasp_yaw)))

        x_obj, y_obj = self.pixel_to_baxter(px_x, px_y)
        print("\n[BAXTER EXECUTION] Pick & Place:")
        print("  Pixel coordinates: ({:.1f}, {:.1f})".format(px_x, px_y))
        print("  Baxter target coordinates: x={:.4f}, y={:.4f}".format(x_obj, y_obj))
        print("  Target Bag/Box index: {}".format(bag_index))

        drop_boxes = self.calib_data.get("drop_boxes", {})
        bag_key = str(bag_index)
        if bag_key not in drop_boxes:
            print("Warning: Drop Box {} not calibrated. Falling back to Box 0.".format(bag_index))
            bag_key = "0"
        if bag_key not in drop_boxes:
            print("Error: No drop boxes calibrated.")
            return "FAILED"
        box = drop_boxes[bag_key]
        box_rim = box.get("rim_z", box["z"])

        z_travel = self.travel_z()
        z_table = self.pixel_to_z(px_x, px_y)

        # grasp_dz = object height above the table, from the ceiling depth.
        # Absent -> grasp at the table, which is right for flat objects anyway.
        # Footprint long axis: the LENGTH of a lying object, the DIAMETER of an
        # upright one. Read before the grip depth, which depends on which it is.
        major = width = None
        if grasp_major is not None:
            try:
                m = float(grasp_major)
                if 0.0 < m <= 2 * MAX_OBJECT_H:
                    major = m
            except (TypeError, ValueError):
                pass
        if grasp_width is not None:
            try:
                w = float(grasp_width)
                if 0.0 < w <= 2 * MAX_OBJECT_H:
                    width = w
            except (TypeError, ValueError):
                pass

        lift, thick = 0.0, None
        if grasp_dz is not None:
            try:
                h = float(grasp_dz)
            except (TypeError, ValueError):
                h = -1.0
            if 0.0 <= h <= MAX_OBJECT_H:
                grip, thick = GRIP_DEPTH, None
                try:
                    t = float(grasp_thickness)
                    if MIN_THICKNESS <= t <= MAX_OBJECT_H:
                        thick = t
                        grip = max(MIN_GRIP_DEPTH, min(GRIP_DEPTH, 0.8 * t))
                except (TypeError, ValueError):
                    pass
                # A cylinder ON ITS SIDE is longer than it is tall, and its
                # widest point is halfway up. Fingers that stop at the default
                # depth close on the curve above that, where the jaws push the
                # can away instead of holding it. Reach past the middle, and
                # stop only where the fingertips would touch what it rests on.
                # A cylinder on its side is ROUND across the jaws: as wide
                # as it is tall. A tuna can standing on its base is also
                # longer than it is tall, but its sides are straight and it
                # needs no extra depth -- `major > h` alone called it a lying
                # cylinder and gripped it 9 mm higher than it should.
                own = thick if thick is not None else h
                round_side = width is None or abs(width - h) <= 0.015
                if major is not None and major > h + 0.010 and own > 0.0 \
                        and round_side:
                    deepest = max(MIN_GRIP_DEPTH, own - TIP_CLEARANCE)
                    grip = min(max(grip, 0.5 * own + ROUND_GRIP_EXTRA), deepest)
                lift = max(0.0, h - grip)
                #is_on_table = (thick is not None and (h - thick) <= 0.015)
                if h<0.045 and (thick is None or (h - thick) <= 0.015):
                    lift = 0.0
            else:
                print("  Ignoring implausible object height {}; grasping at the table.".format(grasp_dz))
        z_pick = z_table + lift
        z_approach = z_pick + APPROACH_OFFSET

        # How far the item's LOWEST point sits below the fingertips once it is
        # in the jaws. Both the carry and release heights are built on this.
        #
        # `lift` is the object's VERTICAL height, which is the wrong number for
        # anything lying down: a chip can on its side measures 87 mm tall but is
        # 235 mm LONG, and a smooth cylinder gripped across its middle pivots in
        # the jaws until it hangs. Its low end then reaches half its length below
        # the fingertips -- 118 mm, not the 37 mm `lift` predicts. Carrying at
        # rim + 100 mm put that end 80 mm above the table, under the 91 mm tuna
        # can, and releasing at rim + 30 mm put it BELOW the tray rim, which is
        # the can hitting the table and then the tray.
        #
        # `major` is the footprint's long axis, measured from depth: the length
        # for a lying object, the diameter for an upright one. Taking the larger
        # of the two is right in both cases.
        hang = lift if major is None else max(lift, major / 2.0)
        z_carry = z_travel + hang
        z_release = box_rim + DROP_MARGIN + hang

        print("  Travel {:.3f}   pick {:.3f} (table {:.3f} + {:.3f})   "
              "carry {:.3f}   box {} release {:.3f}".format(
                  z_travel, z_pick, z_table, lift, z_carry, bag_key, z_release))
        if grasp_dz is not None and lift > 0:
            print("      object measures {:.3f} m tall{}; gripping {:.3f} m above the table".format(
                h, "" if thick is None else " ({:.3f} m thick)".format(thick), lift))
        if hang > lift + 1e-6:
            print("      it is {:.3f} m along its long axis, so it hangs {:.3f} m below the "
                  "fingertips once it pivots (not {:.3f})".format(grasp_major, hang, lift))

        print("Opening gripper...")
        self.open_gripper()
        rospy.sleep(0.5)

        # 1. Straight UP first. Never move sideways at table height.
        print("[1/8] Retracting to travel height...")
        if not self.retract():
            print("Aborted: could not retract to travel height.")
            return "FAILED"

        # 2. Cross to above the object, at travel height (clears every box).
        print("[2/8] Travelling above object...")
        if not self.move_to_pose(x_obj, y_obj, max(z_travel, z_approach)):
            print("Aborted: object not reachable at travel height.")
            return "FAILED"

        # 3. Two-stage descent onto the object.
        print("[3/8] Descending to object...")
        verify = not ALLOW_TILT_ON_GRASP
        if not (self.move_to_pose(x_obj, y_obj, z_approach, allow_tilt=ALLOW_TILT_ON_GRASP,
                                  check_vertical=verify)
                and self.move_to_pose(x_obj, y_obj, z_pick, allow_tilt=ALLOW_TILT_ON_GRASP,
                                      check_vertical=verify)):
            # Get OFF it first. The descent stopped short because something
            # stopped it, and until the arm goes back up the fingertips stay
            # leaning on whatever that was -- which on a slightly mis-aimed
            # grasp is the object itself, being pressed into the table.
            self.retract()

            # Reachable but did not arrive means the arm was blocked, not out
            # of range: a pose with a vertical IK solution is one the arm can
            # hold, so what stopped it was physical. Usually the fingertips
            # landing ON the object instead of beside it, from an aim that was
            # off by a centimetre or two.
            blocked = self.solve_ik(x_obj, y_obj, z_pick, allow_tilt=False)[0] is not None
            self.move_to_home()
            if blocked:
                print("Aborted: the descent was blocked. This pose is reachable, so "
                      "something physical stopped the arm -- most likely the gripper "
                      "came down ON the object. Backed off; it stays on the table.")
                return "BLOCKED"
            print("Aborted: no VERTICAL grasp at this pose.")
            if self.solve_ik(x_obj, y_obj, z_pick, allow_tilt=True)[0] is not None:
                print("        (a tilted grasp exists but is refused -- it hits the object")
                print("         at an angle and knocks it over. This spot is past the arm's")
                print("         usable envelope; move the table closer or use the other arm.)")
            return "FAILED"
        grasp_orientation = self.last_orientation
        tilt = self.endpoint_tilt_deg()
        print("      grasp orientation: {}   measured tilt from vertical: {}".format(
            grasp_orientation, "{:.1f} deg".format(tilt) if tilt is not None else "unknown"))

        print("[4/8] Closing gripper...")
        self.close_gripper()
        rospy.sleep(1.0)

        # No verdict here. Read the instant the jaws stop, the gripper is still
        # mid-command and the numbers move: a chip can read 7.1% here and 5.7%
        # moments later with nothing having happened to it. The item is checked
        # once the arm has lifted and the gripper has settled, below.

        # 5. Straight UP, vertically over the object just picked, so nothing is
        # passed over on the way.
        print("[5/8] Lifting to carry height...")
        if not self.move_to_pose(x_obj, y_obj, z_carry, prefer=grasp_orientation):
            print("Aborted: could not lift item to carry height.")
            return "FAILED"

        # Did anything come up with it? This is the FIRST grip reading: the arm
        # has stopped, the gripper has settled, and the grasp's own effort is
        # still applied. An item that is not here never left the table, so the
        # arm stops rather than flying an empty gripper to a bag and reporting
        # the item packed.
        held = self.verify_grip("after the lift", reclose=False)
        if held is False:
            print("      nothing in the jaws: the item was NOT picked up")
            self.open_gripper()
            self.move_to_home()
            return "LOST_PICKUP"
        if held is None:
            print("      (no gripper feedback; assuming the item is held)")
        # Where the jaws sit with the item in them, for the check over the box.
        pick_pos = self.grip_position()

        # 6. Cross to the box by REPLAYING TAUGHT JOINT POSES: extend and lift
        # clear of the table, swing over the box, and only then descend. The
        # route is fixed by what was taught, not re-derived by IK, so it is the
        # same motion every run -- which is what the Cartesian version could not
        # promise, since a 7-DOF arm has a whole family of solutions per pose
        # and the elbow landed somewhere different each time.
        print("[6/8] Travelling above box {}...".format(bag_key))
        if self.has_taught_route(box):
            if not (self.move_joints(self.calib_data["transit_joint_angles"], "transit")
                    and self.move_joints(box["transit_joint_angles"],
                                         "above box {}".format(bag_key))):
                print("Aborted: could not follow the taught carry route.")
                return "FAILED"
        else:
            print("      no taught carry route for box {} -- flying a straight "
                  "Cartesian line. Teach it with 'r' at startup.".format(bag_key))
            if not self.move_to_pose(box["x"], box["y"], z_carry, prefer=grasp_orientation):
                print("Aborted: box {} not reachable at travel height.".format(bag_key))
                return "FAILED"

        # Still holding it? Anything lost between the table and here fell on
        # the way, and where it landed is unknown -- back on the table if the
        # arm is still over it, on the floor otherwise. Either way it was not
        # packed, and the next photo is what settles where it is.
        if self.verify_grip("over the box", pick_pos) is False:
            print("      the item is gone from the jaws: LOST IN TRANSIT")
            self.open_gripper()
            self.move_to_home()
            return "LOST_TRANSIT"

        # 7. Descend to just above the rim, then release. The old code opened
        # the gripper at box_z + 0.15 -- an arbitrary height, so items were
        # dropped from wherever the arm was hand-held during calibration.
        # z_release already carries `hang`, which assumes the item has pivoted
        # as far as it can, so a tilted wrist needs no separate allowance.
        # `prefer` is only a hint here: the taught pose may hold the wrist at a
        # different yaw than the grasp did, and forcing the old one back would
        # twist the item in the jaws right above the box.
        print("[7/8] Descending to box rim and releasing...")
        if not self.move_to_pose(box["x"], box["y"], z_release,
                                 prefer=self.last_orientation or grasp_orientation):
            print("Warning: could not reach release height, releasing from carry height.")
        self.open_gripper()
        rospy.sleep(1.0)

        # 8. Straight home once the item is released. The gripper is empty and
        # already above the box, so there is nothing to route around.
        print("[8/8] Retracting and returning home...")
        self.retract()
        self.move_to_home()

        print("[BAXTER EXECUTION] Pick & Place complete.\n")
        return "SUCCESS"

    def test_bag_motions(self):
        """Visit every calibrated drop-box at hover height then return home.

        Use this before running the full pipeline to verify the arm can reach
        each bag without collisions.
        """
        if not self.calib_data:
            print("[TEST] No calibration data - cannot run motion test.")
            return False

        drop_boxes = self.calib_data.get("drop_boxes", {})
        if not drop_boxes:
            print("[TEST] No drop boxes found in calibration.")
            return False

        z_travel = self.travel_z()
        taught = sum(1 for b in drop_boxes.values() if self.has_taught_route(b))
        print("\n[TEST] Starting bag motion test ({} bags, {} with a taught carry "
              "route), travel height {:.3f} m...".format(len(drop_boxes), taught, z_travel))
        if taught < len(drop_boxes):
            print("[TEST] Boxes without a taught route fly a Cartesian line. "
                  "Teach them with 'r' at startup.")
        self.retract()
        all_ok = True
        for bag_key in sorted(drop_boxes.keys(), key=lambda k: int(k)):
            coords = drop_boxes[bag_key]
            bx = coords["x"]
            by = coords["y"]
            if self.has_taught_route(coords):
                print("[TEST] Bag {}: replaying the taught carry route...".format(bag_key))
                ok = (self.move_joints(self.calib_data["transit_joint_angles"], "transit")
                      and self.move_joints(coords["transit_joint_angles"],
                                           "above box {}".format(bag_key)))
            else:
                print("[TEST] Moving to Bag {} at travel height "
                      "(x={:.3f}, y={:.3f}, z={:.3f})...".format(bag_key, bx, by, z_travel))
                ok = self.move_to_pose(bx, by, z_travel)
            if ok:
                rim = coords.get("rim_z", coords["z"])
                print("[TEST]   descending to release height {:.3f} m...".format(rim + DROP_MARGIN))
                ok = self.move_to_pose(bx, by, rim + DROP_MARGIN)
                self.retract()
            if ok:
                print("[TEST] Bag {} reached OK.".format(bag_key))
            else:
                print("[TEST] WARNING: Could not reach Bag {}!".format(bag_key))
                all_ok = False
            rospy.sleep(0.5)

        print("[TEST] Returning to home...")
        self.move_to_home()
        if all_ok:
            print("[TEST] Motion test PASSED - all bags reachable.\n")
        else:
            print("[TEST] Motion test COMPLETED WITH WARNINGS - some bags unreachable.\n")
        return all_ok

def main():
    try:
        rospy.init_node("baxter_execution_listener", anonymous=True)
    except Exception as e:
        print("ROS node initialization error: {}".format(e))
        return

    limb_name = os.environ.get("BAXTER_LIMB", "left")
    print("==============================================")
    print("  BAXTER REMOTE EXECUTION LISTENER (OLD PC)   ")
    print("==============================================")
    print("Limb: {}".format(limb_name))

    executor = BaxterExecutor(limb_name=limb_name)
    # Non-latched: only current subscribers receive the status
    status_pub = rospy.Publisher("/baxter/pick_place_status", String, queue_size=1)

    def _shutdown_handler(signum, frame):
        """CTRL+C in this terminal: open gripper and return to home before exiting."""
        print("\n[CTRL+C] Shutdown requested - opening gripper and moving to home...")
        try:
            executor.open_gripper()
        except Exception as e:
            print("[CTRL+C] Could not open gripper: {}".format(e))
        try:
            executor.move_to_home()
        except Exception as e:
            print("[CTRL+C] Could not move to home: {}".format(e))
        print("[CTRL+C] Done. Shutting down.")
        rospy.signal_shutdown("User requested stop")

    signal.signal(signal.SIGINT, _shutdown_handler)

    # Move arm to home/reference and signal ready
    ok = executor.move_to_home()

    # --- Optional pre-flight motion test ---
    try:
        run_test = input("\nRun motion test to all bags before starting? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        run_test = "n"
    if run_test in ("y", "yes"):
        executor.test_bag_motions()
        try:
            proceed = input("Motion test complete. Start command listener? (Y/n): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            proceed = "y"
        if proceed in ("n", "no"):
            print("Exiting.")
            return

    executor.measure_empty_grip()

    print("\nListening for commands on topic '/baxter/pick_place_cmd'...")
    print("Press Ctrl-C to stop and return arm to home.\n")

    last_seq_id = None  # Track last processed sequence ID to avoid re-executing duplicates

    while not rospy.is_shutdown():
        try:
            # Wait for next command (2s timeout allows clean shutdown)
            msg = rospy.wait_for_message("/baxter/pick_place_cmd", String, timeout=2.0)
            data = json.loads(msg.data)
            seq_id = data.get("seq_id", None)

            # Skip if this is a duplicate of the last processed message
            if seq_id is not None and seq_id == last_seq_id:
                print("[SKIP] Already processed seq_id={}. Waiting for new command.".format(seq_id))
                rospy.sleep(0.5)
                continue

            cmd_type = data.get("cmd", "PICK_PLACE")
            print("\n[COMMAND RECEIVED] type={} seq_id={}".format(cmd_type, seq_id))

            if cmd_type == "STOP":
                # end_to_end_pipeline.py publishes this on CTRL+C. Without
                # this branch it fell through to "unknown command" and the
                # robot carried on.
                print("[STOP] Stop requested from the planner PC.")
                executor.open_gripper()
                executor.move_to_home()
                status_pub.publish("STOPPED:{}".format(seq_id))
                print("[STOP] Gripper open, arm home. Exiting.")
                return

            elif cmd_type == "MOVE_HOME":
                # Move arm to home/reference and signal ready
                ok = executor.move_to_home()
                response = "READY:{}" .format(seq_id)
                status_pub.publish(response)
                print("[COMMAND COMPLETED] Home move done. Published: {}".format(response))

            elif cmd_type == "PICK_PLACE":
                px_x = float(data["px_x"])
                px_y = float(data["px_y"])
                bag_index = int(data.get("bag_index", 0))
                grasp_dz = data.get("grasp_dz", None)
                grasp_thickness = data.get("grasp_thickness", None)
                grasp_yaw = data.get("grasp_yaw_deg", None)
                grasp_major = data.get("grasp_major_m", None)
                grasp_width = data.get("grasp_width_m", None)
                # Reported, not planned on: the carry route is taught, not
                # derived. Printing the tallest object is how you check the
                # taught transit pose is still high enough for the scene.
                obs = data.get("obstacles") or []
                tall = max([o.get("h") or 0.0 for o in obs] or [0.0])
                print("  px=({:.1f}, {:.1f})  bag={}  height={}  "
                      "{} other objects, tallest {:.3f} m".format(
                          px_x, px_y, bag_index,
                          "{:.3f} m".format(grasp_dz) if grasp_dz is not None else "not measured",
                          len(obs), tall))

                status_str = executor.execute_pick_and_place(px_x, px_y, bag_index=bag_index,
                                                          grasp_dz=grasp_dz,
                                                          grasp_thickness=grasp_thickness,
                                                          grasp_yaw=grasp_yaw,
                                                          grasp_major=grasp_major,
                                                          grasp_width=grasp_width)
                # execute_pick_and_place already moves back to home before
                # returning, and names its own outcome: SUCCESS, FAILED,
                # LOST_PICKUP or LOST_TRANSIT.
                response = "{}:{}".format(status_str, seq_id)
                status_pub.publish(response)
                print("[COMMAND COMPLETED] Result: {}".format(response))

            else:
                rospy.logwarn("Unknown command type: {}".format(cmd_type))
                status_pub.publish("ERROR:unknown_cmd:{}".format(seq_id))

            last_seq_id = seq_id

        except rospy.ROSInterruptException:
            break
        except rospy.ROSException:
            # Timeout - loop back and check shutdown
            continue
        except Exception as e:
            rospy.logerr("Error executing command: {}".format(e))
            status_pub.publish("ERROR:{}".format(e))

if __name__ == "__main__":
    from std_msgs.msg import String
    main()

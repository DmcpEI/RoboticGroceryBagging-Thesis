import os
import sys
import json
import time
import signal
import datetime
from pathlib import Path

import cv2
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# Setup paths to import from both packages
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "robot_pc_package"))

from robot_pc_package.perceive_local import (load_model, detect, build_detections,
                                            to_planner_format, planner_keys)
from robot_pc_package.depth_grasp import (fit_table_plane, object_height_m, support_height_m,
                                          grip_depth_m, suppress_nested, suppress_overlapping,
                                          height_implausible, to_metres, grasp_pixel,
                                          mark_blocked, footprint_axes, grasp_width_ok,
                                          infer_support_from_neighbours, step_support_m)
from simulate_fixed_items_interface import RobotPlanner

# Initialize global variables for ROS RealSense camera subscription & Baxter execution status
bridge = CvBridge()
color_image = None
received_color = False  # Flag to track when we receive color image

# Aligned depth: a detector box indexes the same pixels in both streams.
# Unaligned is a different viewpoint, ~2 cm off.
DEPTH_TOPIC = "/camera/aligned_depth_to_color/image_raw"
INFO_TOPIC = "/camera/color/camera_info"
# Pixel the camera looks straight down through, read from the camera:
# ~7 px off the image centre on this RealSense, and parallax measures from it.
PRINCIPAL = None
depth_image = None
received_depth = False

# Reject a detection whose height cannot belong to the product it was named as.
# OFF: the height ranges come from few poses, so it can reject a real one.
# Always prints what it would reject; VMT_HEIGHT_GATE=1 to enforce.
HEIGHT_GATE = os.environ.get("VMT_HEIGHT_GATE", "0") == "1"

# Table region (left, top, right, bottom) in the ceiling camera frame: detections
# outside it are not on the table. reach_test.py's DEFAULT_CROP mirrors this.
CROP = (110, 125, 565, 412)
SHOW_CROPPED_IMAGE = True

# Global variable to receive execution status back from Old PC
baxter_status = None

# Global flag set by CTRL+C handler
stop_requested = False

# Will be set to the cmd publisher once ROS is initialised
_cmd_pub = None

# Measured height range per product, for the plausibility gate above.
CLASS_HEIGHTS = {}

def _sigint_handler(signum, frame):
    """ Graceful CTRL+C: send a STOP command to the robot then exit. """
    global stop_requested, _cmd_pub
    if stop_requested:
        # Second CTRL+C – force quit immediately
        print("\nForce-quitting.")
        sys.exit(1)
    stop_requested = True
    print("\n[CTRL+C] Stop requested – sending STOP command to robot...")
    if _cmd_pub is not None:
        try:
            stop_payload = json.dumps({"cmd": "STOP"})
            _cmd_pub.publish(stop_payload)
            print(f"[CTRL+C] Published STOP command: {stop_payload}")
        except Exception as e:
            print(f"[CTRL+C] Failed to publish STOP command: {e}")

signal.signal(signal.SIGINT, _sigint_handler)

def color_image_callback(msg):
    """ Callback to capture one color image and store it globally. """
    global color_image, received_color
    try:
        # Convert ROS Image to OpenCV format
        color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        received_color = True  # Set flag to True after receiving one frame
    except Exception as e:
        rospy.logerr(f"Error processing color image: {e}")

def depth_image_callback(msg):
    """One aligned depth frame. Encoding passthrough: RealSense sends 16UC1
    millimetres, converted to metres at point of use."""
    global depth_image, received_depth
    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        received_depth = True
    except Exception as e:
        rospy.logerr(f"Error processing depth image: {e}")


def baxter_status_callback(msg):
    """ Callback to listen for action execution status from Baxter on Old PC. """
    global baxter_status
    baxter_status = msg.data
    print(f"\n[BAXTER STATUS FEEDBACK] Received execution status from Old PC: {baxter_status}")

def main():
    global received_color, color_image, received_depth, depth_image
    global baxter_status, stop_requested, _cmd_pub
    
    # ==========================================
    # 1. INITIALIZE PERCEPTION
    # ==========================================
    weights_path = HERE / "robot_pc_package" / "weights" / "best.pt"
    classes_path = HERE / "robot_pc_package" / "classes.json"
    lookup_path = HERE / "robot_pc_package" / "objects_lookup.json"
    
    if not weights_path.exists():
        print(f"Error: YOLO weights not found at {weights_path}")
        print("Please ensure the weights file is downloaded and placed correctly.")
        return

    class_names = json.loads(classes_path.read_text())
    lookup = json.loads(lookup_path.read_text())
    global CLASS_HEIGHTS
    heights_path = HERE / "robot_pc_package" / "class_heights.json"
    CLASS_HEIGHTS = (json.loads(heights_path.read_text()).get("heights", {})
                     if heights_path.exists() else {})
    print(f"Loaded measured height ranges for {len(CLASS_HEIGHTS)} products.")
    
    print("Loading perception model...")
    perception_model = load_model(weights_path, class_names)
    print("Perception model loaded successfully.\n")

    # ==========================================
    # 2. INITIALIZE PLANNER
    # ==========================================
    print("Initializing Robot Planner...")
    planner = RobotPlanner(
        num_bags=4,              # Adjust based on your setup
        num_items_to_pack=10,    # Expected total items per episode
        n_visible_items=5,       # Max items to consider at once
        erm_beta=15,            # Risk-neutral 1e-6 or 15 risk-averse
        # MCTS budget: DEFAULT_N_ITER_PER_TIMESTEP / $PLANNER_N_ITER
    )
    print("Planner initialized successfully.\n")

    # ==========================================
    # ROS INITIALIZATION & TOPIC crop
    # ==========================================
    print("Initializing ROS node on New PC...")
    try:
        rospy.init_node("end_to_end_planner_pipeline", anonymous=True)
        print("ROS node initialized successfully.\n")
    except Exception as e:
        print(f"Error initializing ROS: {e}")
        print("Please ensure roscore is running and ROS setup.bash is sourced.")
        return

    # Read the principal point only NOW: wait_for_message needs a node, and
    # this used to run before init_node, so it timed out on every start and
    # silently fell back to the image centre.
    global PRINCIPAL
    try:
        from sensor_msgs.msg import CameraInfo
        info = rospy.wait_for_message(INFO_TOPIC, CameraInfo, timeout=5.0)
        PRINCIPAL = (info.K[2], info.K[5])
        print(f"Camera principal point: ({PRINCIPAL[0]:.1f}, {PRINCIPAL[1]:.1f})")
    except Exception as e:
        print(f"No {INFO_TOPIC} ({e}); using the image centre for parallax correction.")

    # ROS Publishers and Subscribers for remote communication with Old PC (Baxter)
    # Not latched - messages are only delivered to currently connected subscribers
    cmd_pub = rospy.Publisher("/baxter/pick_place_cmd", String, queue_size=10)
    _cmd_pub = cmd_pub  # Expose to SIGINT handler so it can publish a STOP
    rospy.Subscriber("/baxter/pick_place_status", String, baxter_status_callback)

    # Give ROS connections time to negotiate
    print("Waiting for ROS connections to establish...")
    rospy.sleep(1.0)

    # ==========================================
    # 3. END-TO-END LOOP
    # ==========================================
    experiment = 'scene_1'
    cmd_seq = 0  # Monotonically increasing command sequence number
    print("Pipeline ready. Press CTRL+C at any time to stop the robot and exit.")
    
    while not rospy.is_shutdown() and not stop_requested:

        # Reset flags to capture fresh frame
        received_color = False
        color_image = None
        received_depth = False
        depth_image = None

        print("Subscribing to RealSense color and aligned depth to capture a frame...")
        sub_color = rospy.Subscriber("/camera/color/image_raw", Image, color_image_callback)
        sub_depth = rospy.Subscriber(DEPTH_TOPIC, Image, depth_image_callback)

        t_start = time.time()
        # Wait for frames to arrive. Depth is optional: without it the arm
        # grasps at table height, which is the old behaviour.
        while not received_color and not rospy.is_shutdown():
            time.sleep(0.01)
            if time.time() - t_start > 5.0:
                print("Timeout waiting for camera frame (5 seconds).")
                break
        while not received_depth and not rospy.is_shutdown() and time.time() - t_start < 5.0:
            time.sleep(0.01)

        # Clean up subscriptions to save CPU
        sub_color.unregister()
        sub_depth.unregister()
        if not received_depth:
            print(f"No depth on {DEPTH_TOPIC}; grasping at table height. "
                  f"(Is the RealSense started with align_depth:=true?)")

        if rospy.is_shutdown():
            break

        if color_image is None:
            print("Error: Failed to capture color image from '/camera/color/image_raw'.")
            continue

        # Create temporary directory for storing the captured/annotated images
        temp_dir = HERE / "output"
        temp_dir.mkdir(parents=True, exist_ok=True)

        min_sec = datetime.datetime.now().strftime("%d%H%M%S")
        image_path = temp_dir / f"{experiment}_{min_sec}_color.png"
        cv2.imwrite(str(image_path), color_image)

        # --- A. PERCEPTION PHASE ---
        print(f"\n--- Perceiving '{image_path.name}' ---")
        t0 = time.time()
        
        if CROP and SHOW_CROPPED_IMAGE:
            left, top, right, bottom = CROP
            h, w = color_image.shape[:2]
            l, t = max(0, left), max(0, top)
            r, b = min(w, right), min(h, bottom)
            cropped_img = color_image[t:b, l:r]
            cropped_path = temp_dir / f"{experiment}_{min_sec}_cropped.png"
            cv2.imwrite(str(cropped_path), cropped_img)

        # 0.25 over 0.40: -0.010 precision, +0.029 recall (26-scene bench, v7).
        # A miss is silent, a false positive is visible and only costs efficiency.
        # min_scan_conf <= unknown_conf, else nothing reaches the unknown branch.
        raw_dets = detect(perception_model, image_path, class_names, imgsz=640,
                          min_scan_conf=0.15, crop=CROP)
        items = build_detections(raw_dets, lookup, min_conf=0.25, unknown_conf=0.15)

        # --- depth: one table plane, then a height per object ------------
        # Every object, not just the chosen one: near-free once the plane is
        # fitted, and it tells a duplicate box from a genuinely stacked one.
        # Reset per frame, or a depth-less frame reuses the previous plane.
        plane = None
        depth_m = None
        if depth_image is not None:
            try:
                depth_m = to_metres(depth_image)
                plane, pinfo = fit_table_plane(depth_m, CROP)
                if plane is None:
                    print(f"Table plane not found ({pinfo.get('reason')}); grasping at table height.")
                else:
                    print(f"Table plane fitted: rms {pinfo.get('rms', 0):.4f} m over "
                          f"{pinfo.get('inliers', 0)} px")
                    for it in items:
                        b = it.get("bbox_2d")
                        if not b:
                            continue
                        # Mask every other box out of this one first. A can
                        # standing on a coffee can fills its box, so an unmasked
                        # read returns the CAN ON TOP's height for both, they
                        # look like one duplicated object, and the lower one is
                        # deleted. Same-name boxes are masked TOO: the tuna can
                        # on the robot was named "coffee can", matching the can
                        # under it, and skipping those put that stack straight
                        # back on the unmasked path. A duplicate box of one
                        # object is harmless here -- it lies on that object's
                        # own surface, so removing it changes no height -- and
                        # object_height_m ignores any box that covers this one
                        # whole, which is the case that would erase it.
                        other_b = [o["bbox_2d"] for o in items
                                   if o is not it and o.get("bbox_2d")]
                        h, hinfo = object_height_m(depth_m, b, plane,
                                                   other_boxes=other_b)
                        it["height_m"] = h
                        if h is None:
                            print(f"No height for {it.get('name')}: "
                                  f"{hinfo.get('reason')}")
                        elif hinfo.get("from") == "rim":
                            print(f"{it.get('name')} is measured from its rim "
                                  f"({h:.3f} m); something stands on its middle")
                        if h is not None:
                            sup, _ = support_height_m(depth_m, b, plane)
                            # A stack usually comes back as ONE box named after
                            # one of the two products, so the ring outside it is
                            # table and nothing above knows better. The step
                            # between the two levels is inside the box itself.
                            # The tallest pose of this product ever measured.
                            # Without it the shelf test calls a lying can's own
                            # curved flank a stack: the robot read a shelf at
                            # 0.067 m under a lone coffee can lying at 0.089 m.
                            rng = CLASS_HEIGHTS.get(it.get("name"))
                            st, sinfo = step_support_m(
                                depth_m, b, plane, h,
                                max_own_height=(rng[1] if rng else None))
                            if st is not None and (sup is None or st > sup):
                                sup = st
                                print(f"{it.get('name')}: a surface {st:.3f} m "
                                      f"inside its own box -- something is "
                                      f"stacked here; taking the top "
                                      f"{h - st:.3f} m")
                            elif st is None and sinfo.get("reason"):
                                print(f"{it.get('name')}: no shelf inside its "
                                      f"box ({sinfo['reason']})")
                            it["support_m"] = sup
                            # which way to turn the wrist, and whether the jaws
                            # can span the object at all
                            yaw, minor, major, _ = footprint_axes(depth_m, b, plane, h)
                            it["grasp_yaw_deg"] = yaw
                            it["grasp_width_m"] = minor
                            it["footprint_major_m"] = major
            except Exception as e:
                print(f"Depth processing failed ({e}); grasping at table height.")

        # --- one object, one box ----------------------------------------
        # Two duplicate shapes, two rules: a label-sized box INSIDE the product
        # box is nested; a cylinder found on both top face and body only overlaps.
        items, dropped = suppress_nested(items)
        for d in dropped:
            print(f"Dropped nested duplicate box: {d.get('name')} {d.get('bbox_2d')} "
                  f"(inside a larger box of the same product)")
        items, dropped = suppress_overlapping(items)
        for d in dropped:
            print(f"Dropped overlapping duplicate box: {d.get('name')} {d.get('bbox_2d')} "
                  f"conf {d.get('name_confidence')} (heavy overlap, same height)")

        # The ring around a small object sitting on a barely wider one reads the
        # table, not the surface it stands on. The other detections know better.
        # AFTER suppression: a second box on the same object is not a support,
        # and one tuna can detected twice had itself as its own support.
        infer_support_from_neighbours(items)
        for it in items:
            if it.get("support_from"):
                print(f"{it.get('name')} is standing on "
                      f"{it['support_from']}: support {it['support_m']:.3f} m, "
                      f"so it is {it['height_m'] - it['support_m']:.3f} m thick")

        # --- does the measured height agree with the name? ---------------
        for it in items:
            bad, why = height_implausible(it.get("name"), it.get("height_m"), CLASS_HEIGHTS)
            if not bad:
                continue
            if HEIGHT_GATE:
                print(f"HEIGHT GATE: {why} -- reporting as unknown_object")
                it["best_guess"] = it["name"]
                it["name"] = "unknown_object"
                it["needs_rescan"] = True
            else:
                print(f"height gate WOULD reject: {why}  (set VMT_HEIGHT_GATE=1 to enable)")

        # Flag buried objects before the planner chooses, or it picks one and
        # drags whatever is on top off with it.
        mark_blocked(items)
        # Depth working for this scene at all? If NO object was measured the
        # frame is unusable, and refusing every item would just stall the run.
        depth_works = any(it.get("height_m") is not None for it in items)
        for it in items:
            if it.get("graspable") is False:
                ev = it.get("blocked_evidence") or {}
                print(f"Not pickable yet: {it.get('name')} "
                      f"({it.get('blocked_by')} is resting on it; "
                      f"overlap {ev.get('overlap_frac')}, "
                      f"{1000 * (ev.get('height_gap_m') or 0):.0f} mm taller, "
                      f"its own thickness {ev.get('top_thickness_m')})")
                continue
            # No measured top, but depth is working for the rest of the scene:
            # this ONE object was not read. Grasping it at table height is what
            # slams the gripper into anything standing on something else, so
            # leave it for a later cycle instead of guessing.
            if depth_works and it.get("height_m") is None:
                it["graspable"] = False
                it["blocked_by"] = "unknown height"
                print(f"Not pickable: {it.get('name')} has no measured height "
                      f"and the arm will not descend blind")
                continue
            # Fixed jaw stroke: too wide across the SHORT axis is ungrippable at
            # any wrist angle. Say so here, not by failing in the arm.
            ok_w, why_w = grasp_width_ok(it.get("grasp_width_m"))
            if not ok_w:
                it["graspable"] = False
                it["blocked_by"] = "gripper"
                print(f"Not pickable: {it.get('name')} {why_w}")

        planner_format = to_planner_format(items)
        
        # Annotate image with detections & crop box
        annotated_img = color_image.copy()
        if CROP:
            cv2.rectangle(annotated_img, (CROP[0], CROP[1]), (CROP[2], CROP[3]), (255, 0, 0), 2)

        for it in items:
            bbox = it.get("bbox_2d")
            if bbox:
                x1, y1, x2, y2 = map(int, bbox)
                label = f"{it.get('name', 'obj')}: {it.get('name_confidence', 0):.2f}"
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, label, (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        annotated_path = temp_dir / f"{experiment}_{min_sec}_annotated.png"
        cv2.imwrite(str(annotated_path), annotated_img)
        print(f"Saved annotated image to: {annotated_path}")

        try:
            cv2.imshow("RealSense Detections", annotated_img)
            cv2.waitKey(1000)
        except Exception:
            pass

        # Same "name#N" numbering the planner is offered, so the key it picks
        # names the same object here. Numbering these separately skipped
        # mark_blocked's filter and paired the choice with a blocked item's box.
        keyed_items = planner_keys(items)
        
        # Convert planner_format to the flat list expected by RobotPlanner
        candidates = []
        for key in planner_format.get("arrival_order", []):
            item_data = planner_format["items_data"][key]
            item_data["name"] = key 
            candidates.append(item_data)
            
        print(f"Detected {len(candidates)} items on table: {[c.get('name') for c in candidates]}")
        print(f"Perception took {time.time() - t0:.2f} seconds")
        
        if not candidates:
            print("No items detected. Nothing to plan.")
            continue

        # --- B. PLANNING PHASE ---
        print("\n--- Planning ---")
        t0 = time.time()
        
        if len(candidates) > planner.env.n_visible_items:
            print(f"Warning: Truncating {len(candidates)} candidates to planner limit ({planner.env.n_visible_items})")
            candidates = candidates[:planner.env.n_visible_items]
            
        result = planner.plan_action(candidates)
        print(f"Planning took {time.time() - t0:.2f} seconds")

        if result["done"]:
            print("Planner signals: episode complete or no valid moves.")
            continue

        # --- C. REMOTE ROBOT EXECUTION PHASE ---
        print(f"\n>>> [DECISION]")
        chosen_item_key = result['chosen_item']['name']
        bag_index = result['bag_index']
        print(f">>> Action: Pick up '{chosen_item_key}' and place it into Bag {bag_index}")
        print(f">>> Items remaining for this episode: {result['items_remaining']}")
        
        chosen = keyed_items.get(chosen_item_key)
        bbox = None if chosen is None else chosen.get("bbox_2d")
        if bbox is not None:
            # Aim where the object MEETS THE TABLE. A raised object projects
            # outward, so its box centre lands beside it: ~87 mm for a tall can
            # near the frame edge.
            px_x, px_y, ginfo = grasp_pixel(
                depth_m, bbox, plane,
                None if chosen is None else chosen.get("height_m"), PRINCIPAL)
            if ginfo.get("corrected"):
                print(f"Chosen item pixel: ({px_x:.1f}, {px_y:.1f})  "
                      f"parallax {ginfo['shift_px']:.0f} px from box centre "
                      f"({ginfo['source']}, H {ginfo['H_m']:.2f} m)")
            else:
                print(f"Chosen item center pixel: ({px_x:.1f}, {px_y:.1f})  "
                      f"(uncorrected: {ginfo.get('reason')})")
        else:
            print(f"Warning: Bounding box for '{chosen_item_key}' not found. Sending pixel=-1 as sentinel.")
            px_x, px_y = -1.0, -1.0

        # --- height and thickness of the chosen object --------------------
        # Thickness (top minus its support) caps finger depth. Without it, a thin
        # object on another is gripped at the table and the jaws take both.
        grasp_dz = grasp_thickness = None
        if chosen is not None:
            grasp_dz = chosen.get("height_m")
            sup = chosen.get("support_m")
            if grasp_dz is not None and sup is not None:
                grasp_thickness = max(0.0, grasp_dz - sup)
        if grasp_dz is None:
            print("No usable height for the chosen object; grasping at table height.")
        else:
            g = grip_depth_m(grasp_dz, None if grasp_thickness is None else grasp_dz - grasp_thickness)
            print(f"Chosen object: height {grasp_dz:.3f} m, "
                  f"thickness {'unknown' if grasp_thickness is None else f'{grasp_thickness:.3f} m'}, "
                  f"grip {g:.3f} m below its top")

        # Send pick+place command with a new sequence ID
        cmd_seq += 1
        exec_seq = cmd_seq
        payload = json.dumps({
            "cmd": "PICK_PLACE",
            "grasp_yaw_deg": None if chosen is None else chosen.get("grasp_yaw_deg"),
            "seq_id": exec_seq,
            "px_x": px_x,
            "px_y": px_y,
            "bag_index": bag_index,
            "item_name": chosen_item_key,
            "grasp_dz": grasp_dz,
            "grasp_thickness": grasp_thickness,
            # Long axis of the chosen object: a tilted wrist swings it down by
            # half of this, which is what the release height has to allow for.
            "grasp_major_m": None if chosen is None else chosen.get("footprint_major_m"),
            # Short axis, the one the jaws close along. A cylinder ON ITS SIDE
            # is as tall as it is wide across the jaws; a flat disc standing on
            # its base is much wider than tall. Only the first needs a deeper
            # grip, and only this number tells them apart.
            "grasp_width_m": None if chosen is None else chosen.get("grasp_width_m"),
            # Everything ELSE still standing on the table, so the arm crosses
            # over it instead of through it. Heights are already measured for
            # every object; nothing extra is computed here.
            "obstacles": [
                {"name": it.get("name"),
                 "px": (it["bbox_2d"][0] + it["bbox_2d"][2]) / 2.0,
                 "py": (it["bbox_2d"][1] + it["bbox_2d"][3]) / 2.0,
                 "h": it.get("height_m")}
                for it in items
                if it is not chosen and it.get("bbox_2d") and it.get("height_m")
            ],
        })

        RETRY_INTERVAL = 5.0   # seconds between republishes
        EXEC_TIMEOUT   = 60.0  # overall timeout

        print(f">>> [SEQ {exec_seq}] Publishing PICK_PLACE command: {payload}")
        baxter_status = None
        cmd_pub.publish(payload)
        t_last_pub = time.time()

        # Wait for completion feedback matching our seq_id.
        # Re-publish every RETRY_INTERVAL seconds in case the first message was dropped.
        print(">>> Waiting for Baxter (Old PC) to complete pick and place... (CTRL+C to stop)")
        t_exec = time.time()
        while not rospy.is_shutdown() and not stop_requested:
            if baxter_status is not None:
                # Check that this response belongs to our current seq
                if ":{}".format(exec_seq) in baxter_status:
                    print(f">>> Execution finished with status: {baxter_status}\n")
                    outcome = baxter_status.split(":")[0]
                    baxter_status = None
                    # The planner holds its choice and places it at the start
                    # of the NEXT call. If the item never reached the bag, that
                    # placement must be cancelled, or the planner packs bags
                    # with items that are still on the table.
                    if outcome != "SUCCESS":
                        where = {"LOST_PICKUP": "never left the table",
                                 "LOST_TRANSIT": "was dropped on the way to the bag",
                                 "BLOCKED": "stopped the gripper on the way down, "
                                            "so the arm backed off it",
                                 }.get(outcome, "did not reach the bag")
                        if planner.cancel_pending():
                            print(f">>> '{chosen_item_key}' {where}; "
                                  f"unpacked and left for the next cycle.\n")
                    break
                # Stale response from a previous command - discard and keep waiting
                print(f">>> Discarding stale status (expected seq {exec_seq}): {baxter_status}")
                baxter_status = None
            time.sleep(0.2)
            elapsed = time.time() - t_exec
            if elapsed > EXEC_TIMEOUT:
                print("Timeout waiting for Baxter execution status (60s).")
                break
            # Re-publish if no response yet after RETRY_INTERVAL seconds
            if time.time() - t_last_pub >= RETRY_INTERVAL:
                retry_num = int((time.time() - t_exec) / RETRY_INTERVAL)
                print(f">>> [SEQ {exec_seq}] No response yet - republishing command (retry #{retry_num})...")
                cmd_pub.publish(payload)
                t_last_pub = time.time()
        
        if stop_requested:
            print(">>> Stop requested – exiting wait loop.")
            break


if __name__ == "__main__":
    main()

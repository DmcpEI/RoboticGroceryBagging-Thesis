#!/usr/bin/env python
"""
Baxter Robot & RealSense Camera Calibration Script

This script helps calibrate the spatial mapping between camera pixel coordinates (X, Y)
and Baxter's Cartesian coordinate frame (X, Y, Z in meters), as well as setting up
drop box / bin locations for packing.

Compatible with Python 2.7 (Ubuntu 16.04 / ROS Kinetic) and Python 3.
"""

import sys
import os
import json
import time

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from baxter_execution import BaxterExecutor, GripperClient

# Python 2 and Python 3 compatibility for input/raw_input
try:
    input = raw_input
except NameError:
    pass

# Global variables for image capture
bridge = CvBridge()
color_image = None
received_color = False

def camera_callback(msg):
    global color_image, received_color
    try:
        color_image = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        received_color = True
    except Exception as e:
        rospy.logerr("Camera callback error: {}".format(e))

def capture_single_frame(timeout=5.0):
    global color_image, received_color
    received_color = False
    color_image = None
    
    sub = rospy.Subscriber("/camera/color/image_raw", Image, camera_callback)
    t0 = time.time()
    while not received_color and not rospy.is_shutdown():
        time.sleep(0.02)
        if time.time() - t0 > timeout:
            break
    sub.unregister()
    return color_image

def main():
    print("==============================================")
    print("  BAXTER & REALSENSE CAMERA CALIBRATION SCRIPT  ")
    print("==============================================")
    
    # 1. Initialize ROS
    try:
        rospy.init_node("baxter_camera_calibration", anonymous=True)
        print("[ROS] Node initialized successfully.")
    except Exception as e:
        print("[ROS] Error initializing ROS node: {}".format(e))
        print("Please ensure roscore and Baxter SDK setup.bash are sourced.")
        return

    # 2. Capture a sample image from camera
    print("\nAttempting to capture image from '/camera/color/image_raw'...")
    frame = capture_single_frame()
    if frame is not None:
        h, w = frame.shape[:2]
        sample_path = os.path.join(HERE, "calibration_sample_frame.png")
        print("Captured camera frame successfully. Resolution: {}x{}".format(w, h))
        cv2.imwrite(sample_path, frame)
        print("Saved sample frame to: {}".format(sample_path))
    else:
        print("Warning: Could not capture frame from RealSense camera. (Proceeding with default dimensions 1280x720)")
        w, h = 1280, 720

    # 3. Initialize Baxter Executor
    limb_name = input("\nEnter Baxter arm to calibrate ('left' or 'right', default: 'left'): ").strip().lower() or "left"
    calib_file_path = os.path.join(HERE, "baxter_calib.json")
    
    print("\nInitializing Baxter limb interface for: {}...".format(limb_name))
    try:
        executor = BaxterExecutor(limb_name=limb_name, calib_file=calib_file_path)
    except Exception as e:
        print("Error initializing BaxterExecutor: {}".format(e))
        return

    # 4. Interactive Calibration Verification
    print("\n--- Calibration Verification & Test Phase ---")
    while True:
        choice = input("\nOptions: [t]est pixel move, [v]iew camera frame, [r]ecord home pose, [q]uit calibration test: ").strip().lower()
        if choice in ('q', 'quit', 'exit'):
            break
        elif choice == 'r':
            executor.record_home_pose()
        elif choice == 'v':
            img = capture_single_frame()
            if img is not None:
                # Overlay reference point and drop box predictions
                ref_px_x = executor.calib_data.get("ref_x_img", 640)
                ref_px_y = executor.calib_data.get("ref_y_img", 360)
                cv2.drawMarker(img, (int(ref_px_x), int(ref_px_y)), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                cv2.putText(img, "REF POINT", (int(ref_px_x) + 10, int(ref_px_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                out_path = os.path.join(HERE, "calibration_verified_frame.png")
                cv2.imwrite(out_path, img)
                print("Saved verified frame with reference overlay to: {}".format(out_path))
                try:
                    cv2.imshow("Calibration Reference", img)
                    cv2.waitKey(2000)
                except Exception:
                    pass
            else:
                print("Failed to capture frame from camera.")
        elif choice == 't':
            try:
                px_x = float(input("Enter target pixel X (0..{}): ".format(w)))
                px_y = float(input("Enter target pixel Y (0..{}): ".format(h)))
                
                bax_x, bax_y = executor.pixel_to_baxter(px_x, px_y)
                print("Mapped pixel ({}, {}) -> Baxter coordinates: X={:.4f}, Y={:.4f}".format(px_x, px_y, bax_x, bax_y))
                
                confirm = input("Move Baxter arm to hover over this target coordinate? (y/N): ").strip().lower()
                if confirm in ('y', 'yes'):
                    z_hover = executor.travel_z()
                    print("Moving arm to target hover pose (X={:.4f}, Y={:.4f}, Z={:.4f})...".format(bax_x, bax_y, z_hover))
                    success = executor.move_to_pose(bax_x, bax_y, z_hover)
                    if success:
                        print("Arm reached target pose successfully.")
                    else:
                        print("Pose unreachable or IK failed.")
            except ValueError:
                print("Invalid numerical input.")

    print("\nCalibration complete. Configuration saved at: {}".format(calib_file_path))

if __name__ == "__main__":
    main()

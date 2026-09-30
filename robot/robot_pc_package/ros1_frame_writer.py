#!/usr/bin/env python3
"""Continuously write the latest ROS1 camera frame to a PNG file.

Runs on the robot PC's SYSTEM python (the one with rospy — Ubuntu 20 / Noetic
python3.8). No cv_bridge, no torch, no repo venv needed: only rospy + numpy +
PIL (python3-pil), all present on a desktop Noetic install.

This is the bridge half of the two-process live setup:

  terminal 1 (system python, ROS sourced):
      source /opt/ros/noetic/setup.bash
      python3 tools/live/ros1_frame_writer.py \
          --topic /camera/color/image_raw --out /tmp/live_frame.png

  terminal 2 (model venv):
      .venv/bin/python tools/live/perceive_live.py \
          --hf-model <model_dir> --quantize 4bit --dtype float16 \
          --image /tmp/live_frame.png --loop

perceive_live re-reads the file on every ENTER, so each shot uses the newest
frame. Writes are atomic (tmp file + os.replace) so a half-written PNG is
never read.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def msg_to_array(msg):
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        arr = buf.reshape(h, w, 3)
        if enc == "bgr8":
            arr = arr[:, :, ::-1]
    elif enc in ("rgba8", "bgra8"):
        arr = buf.reshape(h, w, 4)[:, :, :3]
        if enc == "bgra8":
            arr = arr[:, :, ::-1]
    elif enc in ("mono8", "8uc1"):
        arr = np.repeat(buf.reshape(h, w, 1), 3, axis=2)
    else:
        raise ValueError(f"unsupported encoding '{enc}'")
    return arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--topic", default="/camera/color/image_raw")
    ap.add_argument("--out", default="/tmp/live_frame.png")
    ap.add_argument("--hz", type=float, default=1.0, help="max write rate (frames/s)")
    args = ap.parse_args()

    try:
        import rospy
        from sensor_msgs.msg import Image as RosImage
    except ImportError:
        print("rospy not importable — run with the SYSTEM python after "
              "'source /opt/ros/noetic/setup.bash'", file=sys.stderr)
        return 1
    from PIL import Image

    state = {"n": 0, "last": 0.0}

    def cb(msg):
        now = time.time()
        if now - state["last"] < 1.0 / args.hz:
            return
        state["last"] = now
        arr = msg_to_array(msg)
        tmp = args.out + ".tmp"
        Image.fromarray(arr, "RGB").save(tmp, "PNG")
        os.replace(tmp, args.out)
        state["n"] += 1
        if state["n"] == 1 or state["n"] % 30 == 0:
            print(f"[frame-writer] {state['n']} frames -> {args.out} "
                  f"({msg.width}x{msg.height} {msg.encoding})")

    rospy.init_node("live_frame_writer", anonymous=True, disable_signals=True)
    rospy.Subscriber(args.topic, RosImage, cb, queue_size=1, buff_size=2**24)
    print(f"[frame-writer] subscribed to {args.topic}, writing {args.out} at <={args.hz} Hz. Ctrl-C to stop.")
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("\n[frame-writer] stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

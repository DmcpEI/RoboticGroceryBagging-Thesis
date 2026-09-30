# Object perception -> planner attributes

Runs fully on this machine. No network, no cloud, no external repo needed.

Point it at a camera frame (an image file), it detects the objects on the
table and writes a JSON with exactly the attributes a packing planner needs.

## Files

- `perceive_local.py` — run this.
- `weights/best.pt` — the trained detector (small, ~19 MB, runs on CPU or a modest GPU).
- `classes.json` — the 32 object names the detector knows.
- `objects_lookup.json` — packing attributes (weight, volume, category, ...) per object.
- `ros1_frame_writer.py` — optional: writes the live camera feed to an image file, if using ROS.

## Which detector is in here

`*.pt` is gitignored, so the checkpoint is copied in by hand and the repo keeps
no record of which one. This section is that record.

**Current: `syn_v15b`** (yolo11s, 1500 synthetic composites, zero real training
images), built 2026-09-04 from the captures taken after the robot was moved to
its final position.

Scored on the 133 `b50_*` scenes captured on that same final rig
(`tools/eval/eval_b50_benchmark.py`, 1139 object instances, conf 0.40):

| detector | name-F1 | easy | med | hard |
|---|---|---|---|---|
| `syn_v15b` (this one) | **0.896** | 0.873 | 0.911 | 0.897 |
| `syn_v7_bags` (previous) | 0.806 | 0.786 | 0.827 | 0.801 |

Both seeds of v15b land within 0.002, well under the ~0.03 seed noise, so the
margin is real rather than a lucky run. Hazard recall — the reason the earlier
detector was not safe to ship — is now 1.000 for both `bleach bottle` and
`glass cleaner spray bottle`, up from 0.810.

Do not read that table as a general capability claim: v15b's training cutouts
and the b50 scenes come from the same capture session, so it is matched to this
rig and the previous detector is not. That is the deployment condition, which
is what this package is for, but it is not evidence about a different table.

Still weak: `gelatin dessert box` 0.524, `kitchen sponge` 0.278 and the thin
produce bags (`bag of limes` 0.222) — all classes whose cutouts still come from
the older session.

`classes.json` is unchanged from the previous detector, byte for byte and in the
same order, so swapping the checkpoint alone is correct. `perceive_local.py`
verifies this on load and refuses to run on a mismatch.

## Requirements

```bash
pip install ultralytics pillow
```

That's it for `perceive_local.py`. (`ros1_frame_writer.py` needs `rospy` +
`numpy` + `pillow` — already on any desktop ROS install; only needed if
you're feeding it from a ROS camera topic.)

## Running it

**If you already have a camera frame as an image file** (e.g. from ROS, or
any other source), just point at it:

```bash
python3 perceive_local.py --image /path/to/frame.png
```

**Live loop from the camera** — if the camera isn't already running as part
of your normal robot stack, you need ROS and the camera driver up first:

Terminal 1 — ROS core (skip if already running):
```bash
roscore
```

Terminal 2 — camera driver, publishing images (skip if the robot stack
already starts this; example below is for a RealSense camera, adjust for
your actual camera/launch file):
```bash
source /opt/ros/noetic/setup.bash
roslaunch realsense2_camera rs_camera.launch
```

Check it's actually publishing before going further:
```bash
rostopic hz /camera/color/image_raw
```
(Ctrl-C once you see frames coming in at a steady rate.)

Optional — visually confirm the camera sees the table correctly:
```bash
rosrun rviz rviz
```
Add an `Image` display, set its topic to `/camera/color/image_raw`.

Terminal 3 — write the camera feed to a file:
```bash
source /opt/ros/noetic/setup.bash
python3 ros1_frame_writer.py --topic /camera/color/image_raw --out /tmp/live_frame.png
```

Terminal 4 — perceive on demand:
```bash
python3 perceive_local.py --image /tmp/live_frame.png --loop
```
Press ENTER each time you want to perceive the current frame. Ctrl-C to stop.

## Output

Every shot writes two files to `output/` (next to the script):

- `shotNNNN_detections.json` — every detected object: name, confidence, pixel box, attributes. For checking by eye.
- **`shotNNNN_planner.json` — feed this to the planner.** Example:

```json
{
 "items_data": {
  "cracker box#1": {
   "est_weight_g": 411, "est_volume_cc": 1991, "crush_score": 5,
   "category": "Snacks", "temperature": "Ambient",
   "spill_risk": false, "spill_vulnerable": true, "orientation_sensitive": false
  }
 },
 "arrival_order": ["cracker box#1"]
}
```

Objects the detector isn't confident about come back as `"name":
"unknown_object"` in `shotNNNN_detections.json` (with a `best_guess` for
reference) — these are left OUT of `shotNNNN_planner.json` on purpose, since
they're not safely identified yet.

## If the camera moves or the table changes position

`--crop` (default `290 40 970 675`) restricts detection to the table area in
pixels, so background clutter can't produce false detections. If the camera
or table moves, update it (or pass `--crop 0 0 0 0` to disable and scan the
whole frame).

## Adjusting confidence

- `--min-conf` (default 0.4): how sure the detector must be to name an object.
- `--unknown-conf` (default 0.15): below `--min-conf` but above this, an
  object is flagged as `unknown_object` instead of being dropped entirely.

## Troubleshooting

- **`ros1_frame_writer.py` prints nothing / no file appears** — check
  `rostopic list` shows your camera topic, and `rostopic hz <topic>` shows
  frames arriving. If the topic name isn't `/camera/color/image_raw`, pass
  `--topic <your topic>` to `ros1_frame_writer.py`.
- **`rospy not importable`** — you forgot to `source
  /opt/ros/noetic/setup.bash` (or your distro's setup file) in that terminal.
- **Detections look shifted/missing objects at the edges** — the `--crop`
  box (see above) doesn't match the current table position; recheck it or
  disable it (`--crop 0 0 0 0`) while testing.

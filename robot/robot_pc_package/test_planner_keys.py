"""The planner's "name#N" key must name the same object the pipeline grasps.

The pipeline used to number the boxes itself, without mark_blocked's filter.
One blocked duplicate then shifted every later index by one and the arm was
sent to the buried object.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from robot_pc_package.perceive_local import planner_keys, to_planner_format

items = [
    {"name": "tuna can", "graspable": False, "bbox_2d": [1, 1, 2, 2]},      # buried
    {"name": "tuna can", "bbox_2d": [9, 9, 10, 10]},                        # pickable
    {"name": "unknown_object", "bbox_2d": [5, 5, 6, 6]},
    {"name": "sugar box", "bbox_2d": [3, 3, 4, 4]},
]
keyed = planner_keys(items)
offered = to_planner_format(items)["arrival_order"]

assert list(keyed) == offered, (list(keyed), offered)
assert keyed["tuna can#1"]["bbox_2d"] == [9, 9, 10, 10], "picked the blocked can"
assert "unknown_object" not in " ".join(keyed)
assert all(it.get("graspable") is not False for it in keyed.values())
print("ok")

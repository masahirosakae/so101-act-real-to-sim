#!/usr/bin/env python3
"""Run existing so101_sim.py, but render MP4 from the named car/goal camera.

Put this script beside the existing so101_sim.py.  Reuses the original ACT
worker and all original safety/conversion/CSV functionality unchanged.
Set SO101_VIDEO_CAMERA=car_goal_top for overhead view.
"""
from __future__ import annotations

import os
import so101_sim

CAMERA = os.environ.get("SO101_VIDEO_CAMERA", "car_goal_overview")
if CAMERA not in ("car_goal_overview", "car_goal_top"):
    raise ValueError("SO101_VIDEO_CAMERA must be car_goal_overview or car_goal_top")


def capture_from_fixed_camera(writer, renderer, data):
    if writer is None:
        return
    renderer.update_scene(data, camera=CAMERA)
    writer.append_data(renderer.render())


# Only the camera used for video changes. ACT inference and simulation
# control are provided by the original so101_sim.py, without modification.
so101_sim.capture = capture_from_fixed_camera

if __name__ == "__main__":
    print("Video camera:", CAMERA)
    so101_sim.main()

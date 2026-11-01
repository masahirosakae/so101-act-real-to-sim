#!/usr/bin/env python3
"""Add visual-only toy-car and red-goal props to a copy of official SO-101 MJCF.

The original scene.xml and its included SO-101 MJCF are never modified.
No third-party imports and no motor/serial interfaces.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def geom(parent: ET.Element, name: str, kind: str, size: str, pos: str, rgba: str, **extra: str) -> None:
    ET.SubElement(parent, "geom", {
        "name": name, "type": kind, "size": size, "pos": pos, "rgba": rgba,
        # Visual only; neither the toy car nor the goal participates in physics.
        "contype": "0", "conaffinity": "0", "group": "2", **extra,
    })


def build(scene: Path, destination: Path, car_x: float, car_y: float, goal_x: float, goal_y: float) -> None:
    if not scene.is_file():
        raise FileNotFoundError(f"Official SO-101 scene not found: {scene}")
    if destination == scene:
        raise ValueError("Will not overwrite the official scene.xml")
    if destination.parent != scene.parent:
        raise ValueError("New scene must be in the SAME directory as scene.xml to retain relative includes/assets")

    tree = ET.parse(scene)
    root = tree.getroot()
    if root.tag != "mujoco":
        raise ValueError("Not a MuJoCo XML model")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("No worldbody in scene.xml")
    if root.find("include") is None:
        raise ValueError("Expected official SO-101 include, refusing to replace scene")
    root.set("model", "scene_car_goal_visual_only")

    car = ET.SubElement(world, "body", {
        "name": "toy_car_VISUAL_ONLY", "pos": f"{car_x:.5f} {car_y:.5f} 0",
    })
    blue = "0.06 0.10 0.65 1"
    geom(car, "car_blue_body", "box", ".043 .025 .013", "0 0 .036", blue)
    geom(car, "car_blue_cabin", "box", ".025 .020 .011", "-.006 0 .056", "0.08 0.16 0.50 1")
    geom(car, "car_front_windshield", "box", ".004 .018 .008", ".019 0 .056", "0.70 0.84 0.91 1")
    for ax, sx in (("front", ".028"), ("rear", "-.028")):
        for side, sy in (("left", ".030"), ("right", "-.030")):
            geom(car, f"car_wheel_{ax}_{side}", "cylinder", ".014 .006",
                 f"{sx} {sy} .014", "0.07 0.07 0.07 1", quat=".7071068 .7071068 0 0")
    geom(car, "car_lightbar_red", "box", ".007 .008 .004", "-.010 -.010 .070", "0.95 0.04 0.05 1")
    geom(car, "car_lightbar_white", "box", ".007 .008 .004", "-.010 .010 .070", "0.94 0.94 0.94 1")

    goal = ET.SubElement(world, "body", {
        "name": "red_goal_VISUAL_ONLY", "pos": f"{goal_x:.5f} {goal_y:.5f} 0",
    })
    red = "0.96 0.07 0.05 1"
    geom(goal, "goal_left_edge", "box", ".006 .060 .0015", "-.077 0 .003", red)
    geom(goal, "goal_right_edge", "box", ".006 .060 .0015", ".077 0 .003", red)
    geom(goal, "goal_near_edge", "box", ".083 .006 .0015", "0 -.060 .003", red)
    geom(goal, "goal_far_edge", "box", ".083 .006 .0015", "0 .060 .003", red)
    ET.SubElement(goal, "site", {
        "name": "goal_center", "type": "sphere", "pos": "0 0 .003",
        "size": ".002", "rgba": "1 0 0 0", "group": "3",
    })

    # This camera is for reliable video framing, independent of the interactive Viewer.
    ET.SubElement(world, "camera", {
        "name": "car_goal_overview", "mode": "fixed",
        "pos": ".72 -.70 .78",
        "xyaxes": ".772 .663 0 -.419 .488 .785", "fovy": "53",
    })
    ET.SubElement(world, "camera", {
        "name": "car_goal_top", "mode": "fixed",
        "pos": ".18 -.06 1.10", "xyaxes": "1 0 0 0 1 0", "fovy": "48",
    })

    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    print("Created:", destination)
    print("Car (visual-only):", (car_x, car_y))
    print("Goal (visual-only):", (goal_x, goal_y))
    print("Cameras: car_goal_overview, car_goal_top")
    print("Original scene.xml: UNCHANGED")
    print("Physics: car and goal are STATIC, NO COLLISION, NO PUSHING")


def main():
    p = argparse.ArgumentParser(description="Generate visual-only car + goal scene for SO-101 MuJoCo")
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--car-x", type=float, default=0.20)
    p.add_argument("--car-y", type=float, default=-0.08)
    p.add_argument("--goal-x", type=float, default=0.37)
    p.add_argument("--goal-y", type=float, default=-0.08)
    a = p.parse_args()
    source = a.scene.resolve()
    dest = source.with_name("scene_car_goal_visual.xml")
    build(source, dest, a.car_x, a.car_y, a.goal_x, a.goal_y)

if __name__ == "__main__":
    main()

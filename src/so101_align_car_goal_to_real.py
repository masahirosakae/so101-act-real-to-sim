#!/usr/bin/env python3
"""Align SO-101 visual-only car/goal and top camera to a supplied EP2 top view.

This is approximate IMAGE COMPOSITION matching, not extrinsic calibration.
No MuJoCo import, physics, inference, serial port, or robot access.
Creates a NEW MJCF next to the input scene so relative mesh includes remain valid.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def pair(s: str) -> tuple[float, float]:
    try:
        values = tuple(float(x.strip()) for x in s.split(','))
    except ValueError as ex:
        raise argparse.ArgumentTypeError('Expected x,y numeric pair') from ex
    if len(values) != 2 or not all(math.isfinite(x) for x in values):
        raise argparse.ArgumentTypeError('Expected finite pair x,y, for example 288,101')
    return values


def vec(values) -> str:
    return ' '.join(f'{x:.8f}' for x in values)


def scale_geom(geom: ET.Element, sx: float, sy: float) -> None:
    if 'pos' in geom.attrib:
        xyz = [float(x) for x in geom.get('pos').split()]
        if len(xyz) != 3:
            raise ValueError(f'Unexpected geom.pos: {geom.get("name")}')
        xyz[0] *= sx
        xyz[1] *= sy
        geom.set('pos', vec(xyz))
    if 'size' in geom.attrib and geom.get('type') == 'box':
        size = [float(x) for x in geom.get('size').split()]
        if len(size) != 3:
            raise ValueError(f'Unexpected box.size: {geom.get("name")}')
        size[0] *= sx
        size[1] *= sy
        geom.set('size', vec(size))


def update_pose(body: ET.Element, xy: tuple[float, float], yaw_deg: float) -> None:
    body.set('pos', vec([xy[0], xy[1], 0.0]))
    angle = math.radians(yaw_deg) * 0.5
    body.set('quat', vec([math.cos(angle), 0, 0, math.sin(angle)]))


def run(args):
    source = args.scene.resolve()
    destination = args.output.resolve() if args.output else source.with_name('scene_car_goal_real_layout.xml')
    if not source.is_file():
        raise FileNotFoundError(f'Source MJCF is missing: {source}')
    if destination == source or destination.name in ('scene.xml', 'so101_new_calib.xml'):
        raise ValueError('Refusing to overwrite the source or robot model')
    if destination.parent != source.parent:
        raise ValueError('Output must be alongside source XML (relative assets/includes)')
    if any(x <= 0 for x in (args.goal_width, args.goal_height, args.fovy, args.car_depth)):
        raise ValueError('Physical dimensions, car depth and fovy must be positive')
    if args.fovy >= 90:
        raise ValueError('Camera fovy should be less than 90 degrees')
    W,H = args.frame_size
    if W < 100 or H < 100:
        raise ValueError('Unreasonable frame size')
    goal_w_px,goal_h_px = args.goal_box
    if goal_w_px < 5 or goal_h_px < 5:
        raise ValueError('Goal measured pixel size must be positive')
    # Two independent known goal dimensions supply a robust pixel/metre estimate.
    ppm_x = goal_w_px / args.goal_width
    ppm_y = goal_h_px / args.goal_height
    ppm = (ppm_x + ppm_y) / 2
    if abs(ppm_x - ppm_y) / ppm > 0.18:
        raise ValueError('Width/height scales differ by >18%; check goal dimensions or camera perspective')
    # Fixed overhead pinhole camera. Camera xyaxes = world +X to right, +Y upwards.
    zcam = H / (2 * ppm * math.tan(math.radians(args.fovy)/2))
    base_u,base_v = args.base_px
    cx = (W/2 - base_u) / ppm
    cy = (base_v - H/2) / ppm

    def xy_from_pixel(point, depth):
        # Depth is a best-guess z-height for the visible center; goal on table z=0.
        if depth >= zcam:
            raise ValueError('Object appears above camera height')
        scale = (zcam - depth) / zcam / ppm
        return (cx + (point[0]-W/2)*scale,
                cy + (H/2-point[1])*scale)

    car_xy = xy_from_pixel(args.car_px, args.car_depth)
    goal_xy = xy_from_pixel(args.goal_px, 0)

    tree = ET.parse(source)
    root = tree.getroot()
    if root.tag != 'mujoco':
        raise ValueError('Expected MuJoCo MJCF root')
    world = root.find('worldbody')
    if world is None:
        raise ValueError('Missing worldbody')
    car = world.find("body[@name='toy_car_VISUAL_ONLY']")
    goal = world.find("body[@name='red_goal_VISUAL_ONLY']")
    camera = world.find("camera[@name='car_goal_top']")
    if car is None or goal is None or camera is None:
        raise ValueError('Required car/goal/top camera not found. Start from scene_car_goal_visual.xml')
    if any(g.get('contype') != '0' or g.get('conaffinity') != '0' for b in (car,goal) for g in b.findall('geom')):
        raise ValueError('Expected VISUAL ONLY (non-colliding) car and goal')

    # Original car longitudinal axis is local +X, original bounding approx 0.086m x 0.072m.
    # New goal nominal rectangle approx 0.166m long x 0.132m wide.
    new_car_length = args.car_box[1] / ppm
    new_car_width = args.car_box[0] / ppm
    scale_car_long = new_car_length / 0.086
    scale_car_width = new_car_width / 0.072
    for g in car.findall('geom'):
        scale_geom(g, scale_car_long, scale_car_width)
    for g in goal.findall('geom'):
        scale_geom(g, args.goal_height / 0.166, args.goal_width / 0.132)

    update_pose(car, car_xy, args.car_yaw)
    update_pose(goal, goal_xy, args.goal_yaw)
    camera.set('pos', vec([cx,cy,zcam]))
    camera.set('xyaxes', '1 0 0 0 1 0')
    camera.set('fovy', f'{args.fovy:.5f}')
    root.set('model', 'scene_car_goal_real_layout_VISUAL_ONLY')
    # The original angled camera is left untouched for easy A/B comparison.
    ET.indent(tree, space='  ')
    tree.write(destination, encoding='utf-8', xml_declaration=True)

    print('Created:', destination)
    print('Source unchanged:',source)
    print(f'Reference image: {int(W)}x{int(H)} px, approximate scale {ppm:.1f} px/m')
    print(f'Robot base assumed at world(0,0), source pixel ({base_u:.0f},{base_v:.0f})')
    print(f'Car world XY: ({car_xy[0]:+.4f}, {car_xy[1]:+.4f}) m; yaw={args.car_yaw:.1f} degrees')
    print(f'Goal world XY: ({goal_xy[0]:+.4f}, {goal_xy[1]:+.4f}) m; yaw={args.goal_yaw:.1f} degrees')
    print(f'Car size approx {new_car_width*100:.1f} x {new_car_length*100:.1f} cm')
    print(f'Goal size approx {args.goal_width*100:.1f} x {args.goal_height*100:.1f} cm')
    print(f'Overhead camera pos: ({cx:.4f},{cy:.4f},{zcam:.4f}) m, fovy={args.fovy}')
    print('Use: --camera car_goal_top in so101_teacher_vs_act_replay.py')
    print('IMPORTANT: cosmetic image alignment only; car/goal remain STATIC NO COLLISION.')
    print('Joint calibration and physical task success remain UNVERIFIED.')


def main():
    p=argparse.ArgumentParser(description='Match 2D image arrangement in SO-101 MuJoCo top-view; VISUAL ONLY')
    p.add_argument('--scene',type=Path,required=True,help='Your scene_car_goal_visual.xml')
    p.add_argument('--output',type=Path,help='Output MJCF in the SAME folder; default scene_car_goal_real_layout.xml')
    p.add_argument('--frame-size',type=pair,default=(640.,480.),help='Reference frame width,height pixels')
    p.add_argument('--base-px',type=pair,default=(40.,285.),help='Robot base pixel center (40,285)')
    p.add_argument('--car-px',type=pair,default=(288.,101.),help='Car visual center in EP2 image')
    p.add_argument('--goal-px',type=pair,default=(297.,355.),help='Goal center in EP2 image')
    p.add_argument('--car-box',type=pair,default=(77.,128.),help='Car pixel width,height')
    p.add_argument('--goal-box',type=pair,default=(155.,190.),help='Goal pixel width,height')
    p.add_argument('--goal-width',type=float,default=0.145,help='ESTIMATED real goal width m; replace with measurement')
    p.add_argument('--goal-height',type=float,default=0.180,help='ESTIMATED real goal height m; replace with measurement')
    p.add_argument('--car-depth',type=float,default=0.04,help='Approximate visible car center z height m')
    p.add_argument('--car-yaw',type=float,default=90.0,help='Car longitudinal +X axis rotate toward image top (+90 deg)')
    p.add_argument('--goal-yaw',type=float,default=90.0,help='Goal longitudinal +X axis rotate toward image top (+90 deg)')
    p.add_argument('--fovy',type=float,default=48.0,help='Camera vertical field-of-view deg')
    a=p.parse_args()
    run(a)

if __name__=='__main__':
    main()

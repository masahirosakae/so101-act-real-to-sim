#!/usr/bin/env python3
"""Run MuJoCo without ACT while holding exactly the recorded initial target.

Designed to separate gravity/servo settling motion from ACT policy-induced motion.
No LeRobot imports, no hardware/COM ports, no dataset mutation.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']


def main():
    p = argparse.ArgumentParser(description='Simulation-only fixed-setpoint hold baseline')
    p.add_argument('--scene', required=True)
    p.add_argument('--pose-deg', required=True,
                   help='Five LeRobot initial joint degrees, comma-separated')
    p.add_argument('--frames', type=int, default=45)
    p.add_argument('--act-csv', help='ACT log to compare with hold baseline')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    try:
        initial_deg = np.asarray([float(x.strip()) for x in args.pose_deg.split(',')], dtype=float)
    except ValueError as e:
        p.error(f'pose-deg parse error: {e}')
    if len(initial_deg) != 5 or not np.isfinite(initial_deg).all():
        p.error('--pose-deg must have five finite values')
    if not 1 <= args.frames <= 5000:
        p.error('--frames must be 1..5000')
    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    angles = np.radians(initial_deg)
    ids = []
    aids = []
    for i, name in enumerate(JOINTS):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if jid < 0 or aid < 0:
            raise RuntimeError(f'Joint/actuator not found: {name}')
        ids.append(model.jnt_qposadr[jid])
        aids.append(aid)
        if i < 5:
            lo, hi = model.jnt_range[jid]
            if not lo <= angles[i] <= hi:
                raise ValueError(f'Initial joint outside range: {name}: {initial_deg[i]:.3f} deg')
            data.qpos[ids[i]] = angles[i]
            data.ctrl[aid] = angles[i]
        else:
            data.qpos[ids[i]] = 0.0
            data.ctrl[aid] = 0.0
    mujoco.mj_forward(model, data)
    steps = max(1, round(1 / (15 * model.opt.timestep)))
    logs = []
    for i in range(args.frames):
        for _ in range(steps):
            mujoco.mj_step(model, data)
        row = {'step': i, 'sim_time_s': data.time}
        row.update({f'actual_rad_{name}': float(data.qpos[idx]) for name, idx in zip(JOINTS, ids)})
        logs.append(row)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0]))
        writer.writeheader()
        writer.writerows(logs)
    print('Hold baseline saved:', out)
    print('Held initial target (degrees):', [round(float(x), 3) for x in initial_deg])
    if args.act_csv:
        with Path(args.act_csv).open('r', newline='', encoding='utf-8-sig') as f:
            act = list(csv.DictReader(f))
        if len(act) != len(logs):
            print(f'WARNING: ACT frames={len(act)} != hold frames={len(logs)}')
        n = min(len(act), len(logs))
        print(f"{'JOINT':16} {'ACT NET deg':>12} {'HOLD NET deg':>13} {'ACT-HOLD FINAL deg':>19}")
        for name in JOINTS[:5]:
            key = f'actual_rad_{name}'
            a = [math.degrees(float(r[key])) for r in act[:n]]
            h = [math.degrees(float(r[key])) for r in logs[:n]]
            if not a:
                continue
            print(f'{name:16} {a[-1]-a[0]:12.3f} {h[-1]-h[0]:13.3f} {a[-1]-h[-1]:19.3f}')
        print('Interpretation: ACT-HOLD FINAL isolates effects relative to passive/hold dynamics, not task success.')
    print('No ACT inference or hardware used in this control baseline.')


if __name__ == '__main__':
    main()

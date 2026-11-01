#!/usr/bin/env python3
"""Replay recorded teacher and precomputed ACT predictions in two MuJoCo arms.

Simulation ONLY. No physical robot, cameras, motor or serial interfaces.
Both panels use the same initial measured state and MuJoCo actuator dynamics.
The ACT panel is OFFLINE teacher-forced precomputed action predictions. It is
NOT a closed-loop ACT simulation, nor evidence of task completion.
Car/goal in the scene are visual-only and cannot be pushed.

Put this beside the existing so101_sim.py in real_to_sim folder.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

import numpy as np
import mujoco

from so101_sim import JOINTS, setup, make_mapping, lr_to_mj


def parse_args():
    p = argparse.ArgumentParser(description="SO-101: teacher replay vs precomputed ACT prediction; MuJoCo only")
    p.add_argument("--csv", type=Path, required=True, help="offline_ep2_all450/predictions.csv")
    p.add_argument("--scene", type=Path, required=True, help="scene_car_goal_visual.xml")
    p.add_argument("--mapping", type=Path, required=True, help="mapping_candidate.json (UNVERIFIED)")
    p.add_argument("--video", type=Path, required=True, help="Output side-by-side mp4")
    p.add_argument("--output-csv", type=Path, help="Detailed per-frame simulator tracking log")
    p.add_argument("--start-frame", type=int, default=70, help="Episode-relative start, not global dataset index")
    p.add_argument("--frames", type=int, default=300, help="Up to 450 from existing CSV; 300 = 20sec @15fps")
    p.add_argument("--max-step-rad", type=float, default=0.10, help="Rate limit per joint target per video frame, same for both panels")
    p.add_argument("--camera", choices=["car_goal_overview", "car_goal_top"], default="car_goal_overview")
    p.add_argument("--fps", type=int, default=15)
    args = p.parse_args()
    if args.frames < 1 or args.frames > 450 or args.fps < 1 or args.fps > 60:
        p.error("--frames must be 1..450 and --fps must be 1..60")
    if not 0 < args.max_step_rad <= 0.25:
        p.error("--max-step-rad must be >0 and <=0.25")
    return args


def load_records(path: Path, start: int, count: int):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"episode_frame", "episode"} | {
            f"{kind}_{joint}" for kind in ("state", "teacher", "pred") for joint in JOINTS
        }
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"predictions CSV missing fields: {sorted(missing)}")
        rows = [r for r in reader if int(r["episode_frame"]) >= start and int(r["episode_frame"]) < start+count]
    rows.sort(key=lambda x: int(x["episode_frame"]))
    actual_indices = [int(r["episode_frame"]) for r in rows]
    expected = list(range(start, start+count))
    if actual_indices != expected:
        raise ValueError(f"Need every frame {start}..{start+count-1}. Found {len(rows)} of {count}. "+
                         "Use the full 450-frame stride-1 predictions.csv")
    if len({int(r["episode"]) for r in rows}) != 1:
        raise ValueError("CSV range contains multiple episodes")
    values = {}
    for k in ("state", "teacher", "pred"):
        values[k] = np.asarray([[float(r[f"{k}_{j}"]) for j in JOINTS] for r in rows], dtype=float)
        if values[k].shape != (count,6) or not np.isfinite(values[k]).all():
            raise ValueError(f"Invalid 6-D {k} data")
    return int(rows[0]["episode"]), values


def initialize(model, data, joints, acts, qpos5, grip_rad):
    for i, j in enumerate(joints[:5]):
        data.qpos[model.jnt_qposadr[j]] = float(qpos5[i])
        data.ctrl[acts[i]] = float(qpos5[i])
    data.qpos[model.jnt_qposadr[joints[5]]] = grip_rad
    data.ctrl[acts[5]] = grip_rad
    mujoco.mj_forward(model, data)


def get_qpos(model,data,joints):
    return np.array([float(data.qpos[model.jnt_qposadr[j]]) for j in joints[:5]])


def control_frame(model,data,joints,acts,desired,rate,nsubsteps,grip_rad):
    clipped_by_range = 0
    for i,(j,a) in enumerate(zip(joints[:5], acts[:5])):
        lo,hi = model.jnt_range[j]
        req = float(desired[i])
        raw = np.clip(req,lo+1e-4,hi-1e-4)
        if abs(float(raw)-req) > 1e-5:
            clipped_by_range += 1
        old = float(data.ctrl[a])
        data.ctrl[a] = float(np.clip(raw,old-rate,old+rate))
    data.ctrl[acts[5]] = grip_rad
    for _ in range(nsubsteps):
        mujoco.mj_step(model,data)
    return clipped_by_range


def add_header(left_rgb, right_rgb, frame_no, fps):
    """Uses Pillow if installed; otherwise simple unlabeled video is still valid.

    The names and disclaimers are embedded in normal output when Pillow exists.
    """
    h,w,_ = left_rgb.shape
    out = np.zeros((h+64,2*w,3),dtype=np.uint8)
    out[:64] = (22,27,34)
    out[64:,:w]=left_rgb
    out[64:,w:]=right_rgb
    try:
        from PIL import Image,ImageDraw,ImageFont
        im=Image.fromarray(out)
        draw=ImageDraw.Draw(im)
        fnt=ImageFont.load_default()
        draw.text((12,9),"LEFT: TEACHER DEMONSTRATION REPLAY",fill=(255,255,255),font=fnt)
        draw.text((w+12,9),"RIGHT: ACT OFFLINE PREDICTION REPLAY",fill=(255,255,255),font=fnt)
        draw.text((12,32),"Both: recorded action targets, NOT autonomous task success",fill=(220,220,220),font=fnt)
        draw.text((w+12,32),f"frame {frame_no:03d}  time={frame_no/fps:.1f}s",fill=(220,220,220),font=fnt)
        out=np.asarray(im)
    except ImportError:
        pass
    return out


def main():
    args = parse_args()
    episode, signal = load_records(args.csv,args.start_frame,args.frames)
    model, data_teacher, joints, acts = setup(args.scene)
    if model.nu != 6 or model.nq != 6:
        raise ValueError("Expected exactly six SO-101 actuated joints")
    if mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_CAMERA,args.camera)<0:
        raise ValueError(f"Camera not found: {args.camera}; use scene_car_goal_visual.xml")
    unit, signs, offsets, grip_rad = make_mapping(args.mapping,model,joints)
    qinit, init_clip = lr_to_mj(signal['state'][0,:5],unit,signs,offsets,model,joints)
    if init_clip:
        raise ValueError("Initial recorded state outside MJCF joint limits; choose a later --start-frame")
    data_act=mujoco.MjData(model)
    initialize(model,data_teacher,joints,acts,qinit,grip_rad)
    initialize(model,data_act,joints,acts,qinit,grip_rad)
    expected_physics_hz=1./model.opt.timestep
    nsubsteps=round(1.0/(args.fps*model.opt.timestep))
    print(f"Episode {episode}, frames {args.start_frame}..{args.start_frame+args.frames-1}; "
          f"duration {args.frames/args.fps:.2f}s at {args.fps} fps")
    print(f"Initial position (rad): {np.round(qinit,4).tolist()}; nominal substeps={nsubsteps}, physics_hz={expected_physics_hz:g}")
    print("Left = teacher recorded joint commands. Right = ACT OFFLINE predictions from teacher observations.")
    print("This is NOT closed-loop ACT control, NOT physical car pushing, and NOT verified robot calibration.")
    print("NO HARDWARE OR COM PORT ACCESS.",flush=True)
    args.video.parent.mkdir(parents=True,exist_ok=True)
    logpath=args.output_csv or args.video.with_name(args.video.stem+'_tracking.csv')
    logpath.parent.mkdir(parents=True,exist_ok=True)
    import imageio.v2 as imageio
    writer=imageio.get_writer(str(args.video),fps=args.fps,codec='libx264',macro_block_size=16)
    renderer=None
    columns=['episode_frame','simulation_t_s','teacher_range_clips','act_range_clips']
    for joint in JOINTS[:5]:
        columns += [f'teacher_command_{joint}',f'act_prediction_{joint}',
                    f'teacher_actual_rad_{joint}',f'act_actual_rad_{joint}']
    try:
        renderer=mujoco.Renderer(model,height=480,width=640)
        with logpath.open('w',encoding='utf-8',newline='') as file:
            log=csv.DictWriter(file,fieldnames=columns)
            log.writeheader()
            nclip_teacher=nclip_act=0
            initial_teacher=get_qpos(model,data_teacher,joints)
            initial_act=get_qpos(model,data_act,joints)
            for k in range(args.frames):
                teacher_signal=signal['teacher'][k,:5]
                act_signal=signal['pred'][k,:5]
                target_teacher,clip_t=lr_to_mj(teacher_signal,unit,signs,offsets,model,joints)
                target_act,clip_a=lr_to_mj(act_signal,unit,signs,offsets,model,joints)
                nclip_teacher+=clip_t
                nclip_act+=clip_a
                control_frame(model,data_teacher,joints,acts,target_teacher,args.max_step_rad,nsubsteps,grip_rad)
                control_frame(model,data_act,joints,acts,target_act,args.max_step_rad,nsubsteps,grip_rad)
                renderer.update_scene(data_teacher,camera=args.camera)
                frame_teacher=renderer.render().copy()
                renderer.update_scene(data_act,camera=args.camera)
                frame_act=renderer.render().copy()
                writer.append_data(add_header(frame_teacher,frame_act,k,args.fps))
                t_q=get_qpos(model,data_teacher,joints)
                a_q=get_qpos(model,data_act,joints)
                line={'episode_frame':args.start_frame+k,'simulation_t_s':float(data_teacher.time),
                      'teacher_range_clips':clip_t,'act_range_clips':clip_a}
                for i,joint in enumerate(JOINTS[:5]):
                    line[f'teacher_command_{joint}']=float(teacher_signal[i])
                    line[f'act_prediction_{joint}']=float(act_signal[i])
                    line[f'teacher_actual_rad_{joint}']=float(t_q[i])
                    line[f'act_actual_rad_{joint}']=float(a_q[i])
                log.writerow(line)
                if k==0 or (k+1)%30==0:
                    print(f"{k+1:03d}/{args.frames}: teacher displacement={np.linalg.norm(np.rad2deg(t_q-initial_teacher)):.2f}deg "
                          f"ACT displacement={np.linalg.norm(np.rad2deg(a_q-initial_act)):.2f}deg",flush=True)
    finally:
        writer.close()
        if renderer is not None:
            renderer.close()
    print(f"Saved: {args.video}\nSaved: {logpath}")
    print(f"Joint range clips: teacher={nclip_teacher}, ACT={nclip_act}. "
          "If high, MJCF/calibration mismatch affects interpretation.")

if __name__=='__main__':
    main()

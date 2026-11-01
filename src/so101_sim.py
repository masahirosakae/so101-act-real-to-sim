#!/usr/bin/env python3
"""SO-101 simulation-only diagnostics, animation, and ACT-driven joint control.

NO hardware / serial / robot code is imported. Assumptions of joint mapping
must be checked against a known real pose before calling --mode act.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def setup(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"MJCF not found: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joints, acts = [], []
    for n in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        if jid < 0 or aid < 0:
            raise RuntimeError(f"Missing joint/actuator {n}: joint={jid}, actuator={aid}")
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
            raise RuntimeError(f"Expected hinge joint: {n}")
        if model.actuator_trnid[aid, 0] != jid:
            raise RuntimeError(f"Actuator {n} does not target joint {n}")
        joints.append(jid)
        acts.append(aid)
    return model, data, joints, acts


def inspect(model, data, joints, acts):
    print(f"SO-101: nq={model.nq}, nv={model.nv}, nu={model.nu}, timestep={model.opt.timestep}")
    print(f"{'JOINT':<16} {'QPOSADR':>7} {'ACTID':>6} {'LOW(rad)':>11} {'HIGH(rad)':>11} {'LOW(deg)':>10} {'HIGH(deg)':>10}")
    for name, j, a in zip(JOINTS, joints, acts):
        lo, hi = model.jnt_range[j]
        print(f"{name:<16} {model.jnt_qposadr[j]:7d} {a:6d} {lo:11.4f} {hi:11.4f} {math.degrees(lo):10.1f} {math.degrees(hi):10.1f}")
    print("No hardware or COM ports accessed.")


def make_mapping(path, model, joints):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if obj.get("joint_order") != JOINTS:
        raise ValueError("mapping joint_order must match SO-101 joint names in declared order")
    unit = obj.get("arm_units")
    if unit not in ("degrees", "range_m100_100"):
        raise ValueError("arm_units must be 'degrees' or 'range_m100_100'; verify dataset units")
    signs = np.asarray(obj.get("arm_signs", []), dtype=float)
    offsets = np.asarray(obj.get("arm_offsets_rad", []), dtype=float)
    if signs.shape != (5,) or not all(abs(abs(s) - 1) < 1e-9 for s in signs):
        raise ValueError("arm_signs must contain exactly five +1/-1 values")
    if offsets.shape != (5,) or not np.isfinite(offsets).all():
        raise ValueError("arm_offsets_rad must contain five finite floats")
    if obj.get("gripper_mode") != "freeze":
        raise ValueError("Only gripper_mode='freeze' is supported until calibration is verified")
    grip_rad = float(obj.get("gripper_fixed_rad", 0.0))
    gl, gh = model.jnt_range[joints[-1]]
    if not gl <= grip_rad <= gh:
        raise ValueError("gripper_fixed_rad is outside MJCF joint range")
    return unit, signs, offsets, grip_rad


def lr_to_mj(pos, unit, signs, offsets, model, joints):
    pos = np.asarray(pos, dtype=float).reshape(-1)
    if pos.size != 5 or not np.isfinite(pos).all():
        raise ValueError("Expected five finite LeRobot arm joint positions")
    out = np.zeros(5)
    clip_count = 0
    for i, j in enumerate(joints[:5]):
        lo, hi = model.jnt_range[j]
        if unit == "degrees":
            rad = math.radians(pos[i])
        else:
            rad = float((pos[i] + 100.0) / 200.0 * (hi - lo) + lo)
        raw = signs[i] * rad + offsets[i]
        safe = np.clip(raw, lo + 1e-4, hi - 1e-4)
        if abs(safe - raw) > 1e-5:
            clip_count += 1
        out[i] = safe
    return out, clip_count


def mj_to_lr(model, data, joints, unit, signs, offsets, fixed_gripper_obs):
    result = []
    for i, j in enumerate(joints[:5]):
        q = float(data.qpos[model.jnt_qposadr[j]])
        raw = (q - offsets[i]) / signs[i]
        if unit == "degrees":
            result.append(math.degrees(raw))
        else:
            lo, hi = model.jnt_range[j]
            result.append(200.0 * (raw - lo) / (hi - lo) - 100.0)
    result.append(float(fixed_gripper_obs))
    return result


def optional_viewer(model, data, enabled):
    if not enabled:
        return None
    import mujoco.viewer
    return mujoco.viewer.launch_passive(model, data)


def do_step(model, data, n, viewer=None):
    for _ in range(n):
        mujoco.mj_step(model, data)
    if viewer is not None and viewer.is_running():
        viewer.sync()


def prepare_video(path, model):
    if not path:
        return None, None
    import imageio.v2 as imageio
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(dst), fps=15, codec="libx264", macro_block_size=16)
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
    except Exception:
        writer.close()
        raise
    return writer, renderer


def close_video(writer, renderer):
    if writer is not None:
        writer.close()
    if renderer is not None:
        renderer.close()


def capture(writer, renderer, data):
    if writer is None:
        return
    renderer.update_scene(data)
    writer.append_data(renderer.render())


def process_reader(proc, payload):
    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("ACT worker pipes are not available")
    proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"ACT worker terminated; exit={proc.poll()} (see stderr above)")
    data = json.loads(line)
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def animate(args, model, data, joints, acts):
    for a, j in zip(acts, joints):
        data.ctrl[a] = np.clip(0.0, *model.jnt_range[j])
    mujoco.mj_forward(model, data)
    viewer = optional_viewer(model, data, args.viewer)
    writer = renderer = None
    nsteps = max(1, round(1.0 / (15 * model.opt.timestep)))
    try:
        writer, renderer = prepare_video(args.video, model)
        for k in range(args.frames):
            wall_start = time.perf_counter()
            t = k / 15
            for i, (j, a) in enumerate(zip(joints[:5], acts[:5])):
                lo, hi = model.jnt_range[j]
                data.ctrl[a] = np.clip(0.15 * np.sin(t * 1.4 + 0.7*i), lo+1e-3, hi-1e-3)
            do_step(model, data, nsteps, viewer)
            capture(writer, renderer, data)
            if k % 15 == 0:
                print(f"SIMULATION DEMO ONLY (not ACT): frame={k}, time={data.time:.2f}")
            if viewer is not None:
                time.sleep(max(0, 1.0/15.0 - (time.perf_counter() - wall_start)))
    finally:
        close_video(writer, renderer)
        if viewer:
            viewer.close()


def drive_act(args, model, data, joints, acts):
    if not args.mapping:
        raise ValueError("--mapping is mandatory for ACT mode. Verify mapping with known real poses first")
    if not args.checkpoint or not args.dataset:
        raise ValueError("--checkpoint and --dataset are required for ACT mode")
    unit, signs, offsets, grip_rad = make_mapping(args.mapping, model, joints)
    worker_script = Path(__file__).with_name("so101_act_worker.py")
    worker_cmd = [str(Path(args.act_python)), "-u", str(worker_script), "--checkpoint", str(args.checkpoint),
                  "--dataset", str(args.dataset), "--frame", str(args.dataset_frame), "--device", args.device]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with subprocess.Popen(worker_cmd, text=True, encoding="utf-8", bufsize=1,
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=None) as proc:
        hello = process_reader(proc, {"op": "hello"})
        initial = hello["initial_state"]
        if len(initial) != 6:
            raise RuntimeError("ACT dataset initial observation is not 6-D")
        print("ACT worker:", hello["model"], "initial state:", initial)
        positions, clipped = lr_to_mj(initial[:5], unit, signs, offsets, model, joints)
        if clipped:
            raise RuntimeError(f"Initial pose clipped on {clipped} joint(s): mapping likely incorrect. Stop and inspect.")
        for i, j in enumerate(joints[:5]):
            data.qpos[model.jnt_qposadr[j]] = positions[i]
            data.ctrl[acts[i]] = positions[i]
        data.qpos[model.jnt_qposadr[joints[5]]] = grip_rad
        data.ctrl[acts[5]] = grip_rad
        mujoco.mj_forward(model, data)
        print("Assumed initial MJCF radians:", [round(float(x), 4) for x in positions])
        viewer = optional_viewer(model, data, args.viewer)
        if args.pose_only:
            print("POSE-ONLY verification. No ACT action issued. Check visually, then end with Ctrl+C or close the viewer.")
            try:
                if viewer:
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
                return
            finally:
                if viewer:
                    viewer.close()
        writer = renderer = None
        nsteps = max(1, round(1.0 / (15.0 * model.opt.timestep)))
        rate = float(args.max_step_rad)
        log_path = out / "act_sim_log.csv"
        columns = ["step", "sim_time_s", "inference_calls", "action_is_from_ACT", "image_source", "num_clipped"]
        columns += [f"obs_{x}" for x in JOINTS]
        columns += [f"act_{x}" for x in JOINTS]
        columns += [f"target_rad_{x}" for x in JOINTS]
        columns += [f"actual_rad_{x}" for x in JOINTS]
        with log_path.open("w", encoding="utf-8", newline="") as file:
            csv_writer = csv.DictWriter(file, fieldnames=columns)
            csv_writer.writeheader()
            try:
                writer, renderer = prepare_video(args.video, model)
                for tick in range(args.frames):
                    wall_start = time.perf_counter()
                    if viewer and not viewer.is_running():
                        break
                    obs = mj_to_lr(model, data, joints, unit, signs, offsets, initial[5])
                    answer = process_reader(proc, {"op": "act", "state": obs})
                    action = answer["action"]
                    if len(action) != 6 or not np.isfinite(action).all():
                        raise ValueError(f"ACT returned nonfinite/wrong-size action: {action}")
                    predicted, clipped = lr_to_mj(action[:5], unit, signs, offsets, model, joints)
                    applied = np.empty(6, dtype=float)
                    for i, (j, a) in enumerate(zip(joints[:5], acts[:5])):
                        old = float(data.ctrl[a])
                        data.ctrl[a] = np.clip(predicted[i], old-rate, old+rate)
                        applied[i] = data.ctrl[a]
                    data.ctrl[acts[5]] = grip_rad
                    applied[5] = grip_rad
                    do_step(model, data, nsteps, viewer)
                    capture(writer, renderer, data)
                    row = dict(step=tick, sim_time_s=float(data.time), inference_calls=int(answer["inference_calls"]),
                               action_is_from_ACT=True, image_source=f"fixed_real_frame_{args.dataset_frame}", num_clipped=clipped)
                    row.update({f"obs_{n}": float(v) for n, v in zip(JOINTS, obs)})
                    row.update({f"act_{n}": float(v) for n, v in zip(JOINTS, action)})
                    row.update({f"target_rad_{n}": float(v) for n, v in zip(JOINTS, applied)})
                    row.update({f"actual_rad_{n}": float(data.qpos[model.jnt_qposadr[j]]) for n, j in zip(JOINTS, joints)})
                    csv_writer.writerow(row)
                    file.flush()
                    if viewer is not None:
                        time.sleep(max(0, 1.0/15.0 - (time.perf_counter() - wall_start)))
                    if tick % 10 == 0:
                        print(f"ACT step={tick:04d} inference={answer['inference_calls']} clipped={clipped} "
                              f"qpos={[round(float(data.qpos[model.jnt_qposadr[j]]), 2) for j in joints[:5]]}")
            finally:
                close_video(writer, renderer)
                if viewer:
                    viewer.close()
                print("Saved CSV:", log_path)
                if args.video:
                    print("Requested video:", args.video)
                try:
                    process_reader(proc, {"op": "quit"})
                except Exception:
                    pass


def main():
    p = argparse.ArgumentParser(description="Simulation-only SO-101. NO serial/hardware access.")
    p.add_argument("--mode", choices=["inspect", "demo", "act"], default="inspect")
    p.add_argument("--scene", required=True)
    p.add_argument("--viewer", action="store_true")
    p.add_argument("--video", default="")
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--mapping")
    p.add_argument("--pose-only", action="store_true")
    p.add_argument("--act-python", default=r"C:\SO101\.venv-act\Scripts\python.exe")
    p.add_argument("--dataset")
    p.add_argument("--checkpoint")
    p.add_argument("--dataset-frame", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--output", default=r"D:\SO101\real_to_sim_results")
    p.add_argument("--max-step-rad", type=float, default=0.08)
    args = p.parse_args()
    if args.frames < 1 or args.frames > 5000:
        p.error("--frames must be 1..5000")
    if not (0 < args.max_step_rad <= 0.2):
        p.error("--max-step-rad must be >0 and <=0.2")
    model, data, joints, acts = setup(Path(args.scene))
    inspect(model, data, joints, acts)
    if args.mode == "demo":
        animate(args, model, data, joints, acts)
    elif args.mode == "act":
        drive_act(args, model, data, joints, acts)


if __name__ == "__main__":
    main()

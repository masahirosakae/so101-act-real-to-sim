#!/usr/bin/env python3
"""Export time-matched *recorded* SO-101 camera images and states.

READ ONLY. No robot, motor, COM, external camera, or simulation access.
Use the existing LeRobot dataset and venv-act on the recording computer.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def parse_args():
    p = argparse.ArgumentParser(description="Export SO-101 REAL recorded frames and same-index measured joint states")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--episode", type=int, default=2)
    p.add_argument("--frames", default="70,120,170,220,270,320,370", help="Comma-separated episode-relative frame indices")
    p.add_argument("--image-key", default="observation.images.top")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        a.frames = [int(s.strip()) for s in a.frames.split(",")]
    except ValueError as ex:
        p.error("--frames must be comma-separated integers, e.g. 70,120,170")
    if not a.frames or len(set(a.frames)) != len(a.frames) or min(a.frames) < 0 or a.episode < 0:
        p.error("Frame indices must be unique nonnegative integers; episode must be >= 0")
    return a


def frame_image(tensor):
    import numpy as np
    from PIL import Image
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().cpu().numpy()
    arr = np.asarray(tensor)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3-D image, got {arr.shape}")
    if arr.shape[0] in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected RGB(A), got {arr.shape}")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all() or arr.min() < -0.02 or arr.max() > 1.02:
            raise ValueError(f"Image values do not look like normalized [0,1]: {arr.min()}..{arr.max()}")
        arr = np.round(np.clip(arr, 0, 1) * 255).astype("uint8")
    else:
        arr = arr.astype("uint8")
    return Image.fromarray(arr[..., :3], "RGB")


def main():
    args = parse_args()
    if not (args.dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset: {args.dataset}")
    import numpy as np
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/so101_pose_calibration", root=args.dataset)
    lookup = {}
    # Scan only tiny scalar metadata; DO NOT decode video to find frame indices.
    meta = ds.hf_dataset.select_columns(["episode_index", "frame_index"])
    wanted = set(args.frames)
    for ix in range(len(meta)):
        e, f = int(meta[ix]["episode_index"]), int(meta[ix]["frame_index"])
        if e == args.episode and f in wanted:
            if f in lookup:
                raise ValueError(f"Duplicate frame {f} in episode {args.episode}")
            lookup[f] = ix
    missing = sorted(wanted.difference(lookup))
    if missing:
        raise ValueError(f"Episode {args.episode} missing requested frames: {missing}")

    args.output.mkdir(parents=True, exist_ok=True)
    report = {"version": 1, "kind": "recorded_so101_real_image_and_state", "episode": args.episode,
              "dataset": str(args.dataset.resolve()), "image_key": args.image_key, "joint_order": JOINTS,
              "notes": "Recorded image and measured state at the SAME dataset index. No hardware or ACT.", "samples": []}
    for frame in sorted(args.frames):
        i = lookup[frame]
        sample = ds[i]
        if args.image_key not in sample:
            raise ValueError(f"Missing image key {args.image_key}; available keys={sorted(sample.keys())}")
        state = sample["observation.state"]
        if hasattr(state, "detach"):
            state = state.detach().cpu().numpy()
        state = np.asarray(state, dtype=float).ravel()
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError(f"Frame {frame}: invalid measured 6D state")
        image = frame_image(sample[args.image_key])
        filename = f"real_ep{args.episode:02d}_frame{frame:04d}.png"
        image.save(args.output / filename)
        report["samples"].append({"episode_frame": frame, "dataset_index": i,
                                  "image_file": filename, "state": state.tolist()})
        print(f"Exported episode={args.episode} frame={frame} dataset_index={i} image={filename}", flush=True)
    path = args.output / "real_pose_samples.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved:", path)
    print("READ ONLY: no robot, COM port, or simulation accessed")


if __name__ == "__main__":
    main()

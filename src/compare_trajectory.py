#!/usr/bin/env python3
"""Compare sim joint observation trace to recorded LeRobot episode states (evaluation only)."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--start-frame", type=int, default=0, help="Frame offset WITHIN the selected episode")
    p.add_argument("--episode", type=int, default=2, help="Dataset episode index")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    from lerobot.datasets import LeRobotDataset
    ds = LeRobotDataset(repo_id="local/so101_real_to_sim", root=Path(a.dataset))
    with Path(a.csv).open(encoding="utf-8", newline="") as f:
        trace = list(csv.DictReader(f))
    index_table = ds.hf_dataset.select_columns(["episode_index", "frame_index"])
    indices = sorted(
        ((int(index_table[i]["frame_index"]), i) for i in range(len(ds))
         if int(index_table[i]["episode_index"]) == a.episode),
        key=lambda x: x[0],
    )
    if a.start_frame < 0 or a.start_frame >= len(indices):
        raise ValueError("start-frame is outside selected episode")
    selected = [idx for frame, idx in indices if frame >= a.start_frame]
    if len(trace) > len(selected):
        raise ValueError("Trajectory exceeds episode boundary; refusing cross-episode comparison")
    count = len(trace)
    if count == 0:
        raise ValueError("Empty trajectory")
    virtual = np.array([[float(r[f"obs_{name}"]) for name in JOINTS] for r in trace[:count]])
    teacher = np.array([ds[selected[i]]["observation.state"].cpu().numpy().reshape(-1) for i in range(count)])
    if teacher.shape != virtual.shape:
        raise ValueError(f"Dataset shape mismatch: {teacher.shape} vs sim {virtual.shape}")
    mae = np.abs(virtual - teacher).mean(axis=0)
    rmse = np.sqrt(np.mean((virtual - teacher)**2, axis=0))
    result = {
        "description": "Same-tick observed joint state comparison; not task success and not visual closed-loop",
        "frames": count,
        "units": "LeRobot dataset position units; verify degrees vs normalized range",
        "joint_mae": dict(zip(JOINTS, mae.tolist())),
        "joint_rmse": dict(zip(JOINTS, rmse.tolist())),
        "mean_mae_first_five_joints": float(mae[:5].mean()),
    }
    dst = Path(a.output)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

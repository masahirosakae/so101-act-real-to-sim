#!/usr/bin/env python3
"""SO-101 ACT teacher-forced, one-step offline evaluation for LeRobot 0.6.1.

Inputs at time t: recorded real image[t] AND real joint observation.state[t].
Comparison: ACT first action for t vs recorded action[t].
No MuJoCo, hardware, COM port, serial, cameras or actuator commands.

IMPORTANT: A low offline MAE on training episodes is not proof of rollout
success/generalization. In particular, a stationary/hold-pose predictor may get
low absolute errors; compare to persistence baseline action=state.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import logging
import math
import statistics
import sys
import time
from pathlib import Path

JOINTS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]


def parse_args():
    p = argparse.ArgumentParser(description="ACT offline imitation quality (NO real/sim robot control)")
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--episode", type=int, default=2)
    p.add_argument("--stride", type=int, default=1, help="1=every frame; 15=one frame/second")
    p.add_argument("--max-samples", type=int, default=0, help="0=all selected")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    if a.episode < 0 or a.stride < 1 or a.max_samples < 0 or a.threads < 1:
        p.error("Invalid episode, stride, max-samples, or threads")
    return a


def scalar(x):
    return float(x)


def summarize(rows):
    import numpy as np

    state = np.array([r["state"] for r in rows], dtype=np.float64)
    teacher = np.array([r["teacher"] for r in rows], dtype=np.float64)
    pred = np.array([r["pred"] for r in rows], dtype=np.float64)
    delta_t = teacher - state
    delta_p = pred - state

    out = []
    for j, name in enumerate(JOINTS):
        err = pred[:, j] - teacher[:, j]
        base = state[:, j] - teacher[:, j]  # hold-position prediction
        base_mae = float(np.mean(np.abs(base)))
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err * err)))
        # Dimensionless; >1 = worse than holding current state, <1 = better.
        relative_mae = (mae / base_mae) if base_mae > 1e-9 else None
        dt = delta_t[:, j]
        dp = delta_p[:, j]
        if len(dt) >= 3 and float(np.std(dt)) > 1e-6 and float(np.std(dp)) > 1e-6:
            delta_corr = float(np.corrcoef(dt, dp)[0, 1])
        else:
            delta_corr = None
        # Deliberately exclude gripper from degree-based criteria.
        threshold = 1.0 if j < 5 else 0.5
        moving = np.abs(dt) >= threshold
        sign_match = float(np.mean(np.sign(dt[moving]) == np.sign(dp[moving]))) if moving.any() else None
        out.append({
            "joint": name,
            "unit": "degree" if j < 5 else "LeRobot gripper scale (not degrees)",
            "samples": len(rows),
            "act_mae": round(mae, 5),
            "act_rmse": round(rmse, 5),
            "hold_state_mae": round(base_mae, 5),
            "mae_ratio_to_hold": None if relative_mae is None else round(relative_mae, 5),
            "teacher_action_std": round(float(np.std(teacher[:, j])), 5),
            "act_action_std": round(float(np.std(pred[:, j])), 5),
            "teacher_mean_abs_delta_from_state": round(float(np.mean(np.abs(dt))), 5),
            "act_mean_abs_delta_from_state": round(float(np.mean(np.abs(dp))), 5),
            "delta_corr": None if delta_corr is None or not math.isfinite(delta_corr) else round(delta_corr, 4),
            "delta_sign_agreement": None if sign_match is None else round(sign_match, 4),
            "delta_sign_samples": int(moving.sum()),
        })
    return out


def point_trace(x_values, y_values, xmin, xmax, ymin, ymax):
    # Draw a per-joint time series as inline SVG; no plotting dependencies.
    xx = (xmax - xmin) or 1
    yy = (ymax - ymin) or 1.0
    return " ".join(
        f"{66 + 660 * ((x - xmin) / xx):.1f},{222 - 180 * ((y - ymin) / yy):.1f}"
        for x, y in zip(x_values, y_values)
    )


def build_html(rows, metrics, episode, outpath):
    import numpy as np

    frames = [r["episode_frame"] for r in rows]
    state = np.array([r["state"] for r in rows], dtype=float)
    teacher = np.array([r["teacher"] for r in rows], dtype=float)
    pred = np.array([r["pred"] for r in rows], dtype=float)
    cards = []
    for i, name in enumerate(JOINTS):
        mn = float(np.min([state[:, i].min(), teacher[:, i].min(), pred[:, i].min()]))
        mx = float(np.max([state[:, i].max(), teacher[:, i].max(), pred[:, i].max()]))
        pad = max((mx - mn) * 0.09, 0.08 if i < 5 else 0.02)
        mn -= pad
        mx += pad
        x1, x2 = frames[0], frames[-1]
        colors = [(state[:, i], "#777777", "Observed state"),
                  (teacher[:, i], "#1864ab", "Teacher action"),
                  (pred[:, i], "#d9480f", "ACT prediction")]
        traces = "\n".join(
            f'<polyline points="{point_trace(frames, yy, x1, x2, mn, mx)}" '
            f'fill="none" stroke="{col}" stroke-width="1.7" />'
            for yy, col, _ in colors
        )
        m = metrics[i]
        chart = f'''<article><h2>{html.escape(name)}</h2>
        <p>MAE: <b>{m['act_mae']:.3f}</b> / hold-state baseline: {m['hold_state_mae']:.3f}
        &nbsp; | &nbsp; ratio: {m['mae_ratio_to_hold'] if m['mae_ratio_to_hold'] is not None else 'N/A'}
        &nbsp; ({html.escape(m['unit'])})</p>
        <svg viewBox="0 0 760 260" role="img" aria-label="{html.escape(name)} observed state, teacher action and ACT prediction">
        <rect x="66" y="42" width="660" height="180" fill="#f8f9fa" stroke="#dce2e8" />
        <text x="5" y="47" font-size="13">{mx:.1f}</text>
        <text x="5" y="226" font-size="13">{mn:.1f}</text>
        <text x="66" y="247" font-size="13">frame {x1}</text>
        <text x="650" y="247" font-size="13">{x2}</text>
        {traces}
        </svg></article>'''
        cards.append(chart)
    text = f'''<!doctype html><html lang="en"><head><meta charset="utf-8" />
    <title>SO-101 ACT offline evaluation — episode {episode}</title>
    <style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:960px;margin:24px auto;padding:0 16px;background:#f2f4f8;color:#111}}
    article{{background:white;margin:18px 0;padding:16px;border-radius:10px;box-shadow:0 1px 4px #bbb}}
    svg{{width:100%;max-width:760px}}h2{{margin-bottom:2px}}.legend span{{margin-right:22px}}
    </style></head><body><h1>SO-101 ACT — Teacher-forced offline evaluation</h1>
    <p>Episode {episode}; {len(rows)} sampled frames. Uses each recorded real image and recorded joint state.
    Predicts one action per frame, then compares with the action recorded for that frame. No simulated rollout.</p>
    <p class="legend"><span style="color:#777777">&#9632; Observed state</span>
    <span style="color:#1864ab">&#9632; Teacher action</span>
    <span style="color:#d9480f">&#9632; ACT prediction</span></p>
    <p><b>Interpretation:</b> MAE ratio below 1 means lower absolute error than predicting current joint position;
    above 1 means worse. Small MAE alone does not demonstrate an ability to move the car. For the gripper,
    its LeRobot unit is not degrees and dataset state/action scale is suspect.</p>
    {''.join(cards)}
    <p>Method: ACT (one-step re-query), saved normalization processors, no online feedback, no MuJoCo,
    no goal-success validation. Episodes used for training must not be treated as held-out test data.</p>
    </body></html>'''
    outpath.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    # Import numerical/ML libraries only when needed; the chart/summary code can
    # be independently tested without requiring LeRobot.
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    torch.set_num_threads(args.threads)
    if not args.dataset.joinpath("meta", "info.json").is_file():
        raise FileNotFoundError(f"Dataset meta/info.json missing: {args.dataset}")
    for f in ("config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"):
        if not (args.checkpoint / f).is_file():
            raise FileNotFoundError(f"Expected checkpoint file {f}: {args.checkpoint}")

    args.output.mkdir(parents=True, exist_ok=True)
    print("Loading metadata/dataset...", flush=True)
    ds = LeRobotDataset(repo_id="local/so101_real_to_sim", root=args.dataset)
    # 'hf_dataset' is a materialized table. Its row indices match ds[i].
    index_table = ds.hf_dataset.select_columns(["episode_index", "frame_index"])
    selected = [
        (int(index_table[i]["frame_index"]), i)
        for i in range(len(ds)) if int(index_table[i]["episode_index"]) == args.episode
    ]
    selected.sort()
    selected = selected[::args.stride]
    if args.max_samples:
        selected = selected[:args.max_samples]
    if not selected:
        raise ValueError(f"Episode {args.episode} not found or no frames selected")
    if len(set(f for f, _ in selected)) != len(selected):
        raise ValueError("Duplicate episode frame indices detected")
    print(f"Selected episode={args.episode}; samples={len(selected)}; stride={args.stride}", flush=True)

    cfg = PreTrainedConfig.from_pretrained(str(args.checkpoint))
    if cfg.type != "act":
        raise ValueError(f"Model must be ACT; got {cfg.type}")
    if cfg.temporal_ensemble_coeff is not None:
        raise ValueError("Temporal ensembling enabled: comparison needs a separate evaluation design")
    cfg.device = args.device
    cfg.n_action_steps = 1  # every teacher-observation gets its own forward pass
    policy = ACTPolicy.from_pretrained(str(args.checkpoint), config=cfg, strict=True)
    policy.eval()
    policy.reset()
    image_keys = list(cfg.image_features)
    if image_keys != ["observation.images.top"]:
        raise ValueError(f"Unexpected image keys: {image_keys}")
    image_key = image_keys[0]
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=str(args.checkpoint),
        dataset_stats=ds.meta.stats,
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    print("ACT checkpoint and saved pre/postprocessors loaded", flush=True)

    output_csv = args.output / "predictions.csv"
    fields = ["episode", "episode_frame", "dataset_index", "inference_s"]
    for prefix in ("state", "teacher", "pred", "teacher_minus_state", "pred_minus_state", "prediction_error"):
        fields.extend([f"{prefix}_{j}" for j in JOINTS])

    records = []
    started = time.perf_counter()
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for count, (episode_frame, idx) in enumerate(selected, start=1):
            item = ds[idx]
            state = torch.as_tensor(item["observation.state"]).float().flatten()
            teacher = torch.as_tensor(item["action"]).float().flatten()
            im = item[image_key]
            if len(state) != 6 or len(teacher) != 6:
                raise ValueError(f"Expected six joints at dataset index={idx}")
            if not isinstance(im, torch.Tensor) or im.ndim != 3 or im.shape[0] != 3:
                raise ValueError(f"Unexpected image tensor at dataset index={idx}: {type(im)}, {getattr(im,'shape',None)}")
            if not torch.isfinite(state).all() or not torch.isfinite(teacher).all():
                raise ValueError(f"NaN/Inf in dataset index={idx}")
            with torch.inference_mode():
                t0 = time.perf_counter()
                processed = pre({"observation.state": state, image_key: im})
                result = policy.select_action(processed)
                prediction = torch.as_tensor(post(result)).detach().cpu().float().flatten()
                infer_s = time.perf_counter() - t0
            if prediction.numel() != 6 or not torch.isfinite(prediction).all():
                raise ValueError(f"Nonfinite/incorrect ACT prediction at index={idx}")
            s, t, p = state.tolist(), teacher.tolist(), prediction.tolist()
            records.append({"episode_frame": episode_frame, "dataset_index": idx, "state": s, "teacher": t, "pred": p})
            row = {"episode": args.episode, "episode_frame": episode_frame, "dataset_index": idx, "inference_s": round(infer_s, 5)}
            for prefix, vv in (
                ("state", s), ("teacher", t), ("pred", p),
                ("teacher_minus_state", [a-b for a,b in zip(t,s)]),
                ("pred_minus_state", [a-b for a,b in zip(p,s)]),
                ("prediction_error", [a-b for a,b in zip(p,t)]),
            ):
                row.update({f"{prefix}_{j}": float(v) for j, v in zip(JOINTS, vv)})
            writer.writerow(row)
            f.flush()  # preserve partial measurements if interrupted
            if count == 1 or count % 10 == 0 or count == len(selected):
                print(f"{count}/{len(selected)}  frame={episode_frame:03d} index={idx}  inference={infer_s:.2f}s", flush=True)

    metrics = summarize(records)
    result = {
        "evaluation": "offline_single_step_teacher_forcing",
        "image_source": "recorded_image_at_each_evaluated_frame",
        "state_source": "recorded_real_state_at_each_evaluated_frame",
        "goal_success_evaluated": False,
        "model_uses_MuJoCo": False,
        "hardware_used": False,
        "episode": args.episode,
        "stride": args.stride,
        "samples": len(records),
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "seconds_total": round(time.perf_counter()-started, 2),
        "per_joint": metrics,
    }
    (args.output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    build_html(records, metrics, args.episode, args.output / "comparison.html")
    with (args.output / "summary.txt").open("w", encoding="utf-8") as f:
        print(f"Episode={args.episode}, samples={len(records)}, stride={args.stride}", file=f)
        print("One-step teacher-forced offline evaluation; no simulator / robot / test success", file=f)
        print(f"{'Joint':16} {'ACT MAE':>11} {'Hold MAE':>11} {'ACT/Hold':>11} {'T Δavg':>10} {'P Δavg':>10}", file=f)
        for m in metrics:
            ratio = m["mae_ratio_to_hold"]
            print(f"{m['joint']:16} {m['act_mae']:11.3f} {m['hold_state_mae']:11.3f} {str(ratio):>11} {m['teacher_mean_abs_delta_from_state']:10.3f} {m['act_mean_abs_delta_from_state']:10.3f}", file=f)
    print("\n=== One-step offline evaluation (units: degrees for first 5 axes) ===")
    print((args.output / "summary.txt").read_text(encoding="utf-8"))
    print("Saved:", output_csv, args.output / "metrics.json", args.output / "comparison.html", flush=True)
    print("No motors, COM ports or MuJoCo accessed.", flush=True)


if __name__ == "__main__":
    main()

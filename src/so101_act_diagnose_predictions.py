#!/usr/bin/env python3
"""Audit SO-101 ACT one-step offline prediction CSV without running models or accessing hardware.

Example:
 python so101_act_diagnose_predictions.py --csv D:/SO101/real_to_sim_results/offline_ep2_all450/predictions.csv --max-lag 15

Comparison notes: precompute common centre reference frames for every lag; best lags
are exploratory, not proof of timestamp error. Dataset is training data, not held-out.
"""
import argparse
import csv
import math
from pathlib import Path

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def average(x):
    return sum(x) / len(x) if x else float("nan")


def std(x):
    mu = average(x)
    return math.sqrt(average([(v - mu) ** 2 for v in x])) if x else float("nan")


def correlation(x, y):
    if not x or len(x) != len(y):
        return float("nan")
    mx, my = average(x), average(y)
    dx = [v - mx for v in x]
    dy = [v - my for v in y]
    denom = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / denom if denom > 1e-12 else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--max-lag", type=int, default=15)
    args = parser.parse_args()
    if args.max_lag < 0:
        parser.error("--max-lag must be >= 0")
    if not args.csv.is_file():
        raise SystemExit(f"File not found: {args.csv}")
    with args.csv.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("No prediction rows")
    needed = {"episode_frame"} | {f"{p}_{joint}" for p in ("state", "teacher", "pred") for joint in JOINTS}
    missing = needed - rows[0].keys()
    if missing:
        raise SystemExit("Missing columns: " + ", ".join(sorted(missing)))

    by_frame = {}
    for row in rows:
        frame = int(row["episode_frame"])
        if frame in by_frame:
            raise SystemExit(f"Duplicate frame: {frame}")
        by_frame[frame] = row
    frames = sorted(by_frame)
    max_lag = args.max_lag
    common = [t for t in frames if all((t + lag) in by_frame for lag in range(-max_lag, max_lag + 1))]
    if len(common) < 10:
        raise SystemExit("Not enough contiguous frames for lag check; pass a full stride=1 episode CSV or reduce --max-lag")

    def col(prefix, joint, times):
        return [float(by_frame[t][f"{prefix}_{joint}"]) for t in times]

    print("SO-101 ACT prediction audit (no inference / simulator / COM-port access)")
    print(f"CSV: {args.csv}")
    print(f"Original frames: {len(rows)}; common comparison frames: {len(common)}; lags: [-{max_lag}, +{max_lag}]")
    print("Lag convention: compare prediction[t] to teacher[t+lag]")
    print("All MAE comparisons use the same common reference frames. Units: degrees.\n")
    print(f"{'Joint':16} {'ACT MAE':>9} {'Hold MAE':>9} {'Bias':>9} {'ActStd':>8} {'TeachStd':>8} {'DirAgree':>9} {'DeltaCorr':>10}")
    for j in JOINTS:
        p = col("pred", j, common)
        t = col("teacher", j, common)
        s = col("state", j, common)
        td = [a - b for a, b in zip(t, s)]
        pd = [a - b for a, b in zip(p, s)]
        valid = [i for i, value in enumerate(td) if abs(value) >= 1.0]
        agree = (sum((td[i] > 0) == (pd[i] > 0) for i in valid) / len(valid)) if valid else float("nan")
        print(f"{j:16} {average([abs(a-b) for a,b in zip(p,t)]):9.3f} {average([abs(a-b) for a,b in zip(s,t)]):9.3f} {average([a-b for a,b in zip(p,t)]):9.3f} {std(p):8.3f} {std(t):8.3f} {agree:9.3f} {correlation(td,pd):10.3f}")

    results = []
    for lag in range(-max_lag, max_lag + 1):
        shifted = [t + lag for t in common]
        per_joint = [average([abs(a-b) for a,b in zip(col("pred",j,common), col("teacher",j,shifted))]) for j in JOINTS]
        results.append((lag, average(per_joint), per_joint))
    sorted_results = sorted(results, key=lambda x: x[1])
    print("\nFive-arm-joint mean MAE by teacher offset (same common frames):")
    print(f"  lag 0: {next(x[1] for x in results if x[0]==0):.3f} deg")
    for lag, value, _ in sorted_results[:7]:
        print(f"  lag {lag:+3d}: {value:.3f} deg")
    best = sorted_results[0]
    if best[0] != 0:
        print("\nA nonzero best lag is an investigation lead, NOT proof of a time alignment bug; smooth motions can create spurious best lags.")
    print("Note: teacher-forced TRAINING episode only; no task success / generalization inferred.")


if __name__ == "__main__":
    main()

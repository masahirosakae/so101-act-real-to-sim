#!/usr/bin/env python3
"""Validate apparent temporal lead in recorded SO-101 ACT predictions.

CSV-only diagnostic. No ML inference, MuJoCo, hardware, camera or COM-port access.
Uses common reference times to compare all lags fairly. Requires stride=1 data.
"""
import argparse
import csv
import math
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def corr(xs, ys):
    if len(xs) < 3 or len(xs) != len(ys):
        return float("nan")
    mx, my = mean(xs), mean(ys)
    a = [x - mx for x in xs]
    b = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / den if den > 1e-12 else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--max-lag", type=int, default=15)
    ap.add_argument("--fps", type=float, default=15)
    args = ap.parse_args()
    if args.max_lag < 1 or args.fps <= 0:
        ap.error("--max-lag must be positive and --fps must be positive")
    if not args.csv.is_file():
        ap.error(f"Missing CSV: {args.csv}")
    with args.csv.open("r", newline="", encoding="utf-8-sig") as f:
        data = list(csv.DictReader(f))
    needed = {"episode_frame"} | {f"{prefix}_{joint}" for joint in JOINTS for prefix in ("pred", "teacher", "state")}
    if not data or not needed.issubset(data[0]):
        ap.error("CSV needs stride=1 evaluation: " + ",".join(sorted(needed - set(data[0] if data else {}))))
    rows = {int(row["episode_frame"]): row for row in data}
    if len(rows) != len(data):
        ap.error("Duplicate episode_frame values")
    # Fix the identical interior reference timestamps for all lags.
    common = [t for t in sorted(rows) if all((t + k) in rows for k in range(-args.max_lag, args.max_lag + 1))]
    if len(common) < 30:
        ap.error("Insufficient contiguous frames; use the all450 stride=1 predictions.csv")

    def val(t, prefix, j):
        return float(rows[t][f"{prefix}_{j}"])

    def mae(refs, lag, predicted=True, j=None):
        joints = JOINTS if j is None else (j,)
        return mean([mean([abs(val(t, "pred" if predicted else "state", jj) - val(t + lag, "teacher", jj)) for t in refs]) for jj in joints])

    def derivative_corr(refs, lag, j):
        times = [t for t in refs if t - 1 in rows and t + lag - 1 in rows]
        dp = [val(t, "pred", j) - val(t - 1, "pred", j) for t in times]
        dt = [val(t + lag, "teacher", j) - val(t + lag - 1, "teacher", j) for t in times]
        return corr(dp, dt)

    lags = list(range(-args.max_lag, args.max_lag + 1))
    scores = {lag: mae(common, lag) for lag in lags}
    best = min(lags, key=scores.get)
    zero = scores[0]
    print("SO-101 ACT temporal lag validation -- CSV ONLY, no inference or hardware")
    print(f"Data: {args.csv}")
    print(f"Samples={len(common)} common reference frames; fps={args.fps:g}; best_lag={best:+d} ({best/args.fps:+.3f}s)")
    print("Prediction[t] versus Teacher[t+lag], fixed reference frames")
    print("\nCOMPARISON ON IDENTICAL FRAMES (five-arm-joint mean MAE)")
    print(f"  lag 0:   ACT {zero:.3f} deg   Hold {mae(common, 0, False):.3f} deg")
    print(f"  lag {best:+d}: ACT {scores[best]:.3f} deg   Hold {mae(common, best, False):.3f} deg")
    if abs(zero) > 1e-12:
        print(f"  apparent relative ACT MAE reduction: {100 * (zero - scores[best])/zero:.1f}%")
    print("\nPER-JOINT TIMING AND FIRST-DIFFERENCE CHECK")
    print(f"{'Joint':16} {'BestLag':>7} {'MAE(0)':>9} {'MAE(best)':>10} {'Hold@best':>10} {'dCorr(0)':>10} {'dCorr(best)':>12}")
    for j in JOINTS:
        j_best = min(lags, key=lambda lag: mae(common, lag, j=j))
        print(f"{j:16} {j_best:+7d} {mae(common,0,j=j):9.3f} {mae(common,j_best,j=j):10.3f} {mae(common,j_best,False,j=j):10.3f} {derivative_corr(common,0,j):10.3f} {derivative_corr(common,j_best,j):12.3f}")
    print("\nBEST LAG BY EPISODE SEGMENT (same global reference subset)")
    length = len(common)
    for n in range(3):
        subset = common[n*length//3 : (n+1)*length//3]
        segment_best = min(lags, key=lambda lag: mae(subset,lag))
        print(f"  segment {n+1}: frames={subset[0]}-{subset[-1]} best_lag={segment_best:+d}, MAE(0)={mae(subset,0):.3f}, MAE(best)={mae(subset,segment_best):.3f}")
    print("\nInterpretation: lag is exploratory; lower shifted MAE alone does NOT prove timestamp misalignment.")
    print("Check whether per-joint and segment lags agree and whether derivative correlations improve.")
    print("Dataset episode is used for training. No rollout success or generalization claim.")

if __name__ == "__main__":
    main()

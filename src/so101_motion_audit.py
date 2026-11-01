#!/usr/bin/env python3
"""Audit an existing ACT rollout video and log, and create a zoomed evidence movie.

No simulation, inference, or hardware activity. The robot frames are cropped and
resized with a fixed camera window; joint angles are taken verbatim from CSV.
Does not amplify or invent physical joint motion.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
COLORS = [(59, 130, 246), (225, 82, 82), (16, 166, 117), (150, 91, 211), (205, 134, 38)]


def robot_bounds(frames: list[np.ndarray]):
    boxes = []
    for frame in frames[::max(1, len(frames) // 18)]:
        rgb = frame[:, :, :3].astype(np.int16)
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        mask = (r > 145) & (g > 95) & (r > b * 1.3) & (g > b * 1.17) & (b < 160)
        yy, xx = np.where(mask)
        if xx.size > 100:
            # Percentiles tolerate isolated color specks in the renderer.
            boxes.append((int(np.percentile(xx, 0.5)), int(np.percentile(yy, 0.5)),
                          int(np.percentile(xx, 99.5)), int(np.percentile(yy, 99.5))))
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def crop_box(bounds, width, height, zoom):
    if bounds is None:
        cx, cy = width / 2, height / 2
        needed_h = height / zoom
    else:
        x0, y0, x1, y1 = bounds
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        needed_h = max((y1 - y0 + 1) * 1.30, (x1 - x0 + 1) * 1.30 * height / width, height / zoom)
    crop_h = min(height, int(math.ceil(needed_h)))
    crop_w = min(width, int(math.ceil(crop_h * width / height)))
    # A single fixed crop for all frames, not object-following camera movement.
    x0 = int(max(0, min(width - crop_w, round(cx - crop_w / 2))))
    y0 = int(max(0, min(height - crop_h, round(cy - crop_h / 2))))
    return x0, y0, x0 + crop_w, y0 + crop_h


def default_font(size):
    for f in ('C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/arial.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_metrics(frame, rows, index, angles, scale, scale_lbl):
    h, w = frame.shape[:2]
    img = Image.new('RGB', (w, h + 212), '#101827')
    img.paste(Image.fromarray(frame[:, :, :3]), (0, 0))
    dr = ImageDraw.Draw(img)
    font = default_font(13)
    bold = default_font(16)
    dr.rectangle([0, 0, w - 1, 28], fill='#101827')
    dr.text((10, 6), 'ACT / MuJoCo | fixed real camera image | joint feedback', fill='white', font=font)
    dr.text((10, h + 5), f'ACT-inferred control | frame {index+1}/{len(rows)} | measured qpos delta since logged step 0',
            fill='white', font=bold)
    cx = round(w * 0.68)
    half = round(w * 0.18)
    dr.text((w - 128, h + 31), f'bar range {scale_lbl}', fill='#d6deeb', font=font)
    # Center is zero, left is negative angle change, right is positive.
    for i, name in enumerate(JOINTS):
        y = h + 55 + i * 30
        delta = angles[index, i] - angles[0, i]
        dr.text((10, y), name, fill='#dce5f3', font=font)
        dr.line((cx - half, y + 9, cx + half, y + 9), fill='#45556b', width=2)
        dr.line((cx, y + 2, cx, y + 17), fill='#b0bbc7', width=1)
        end = cx + int(np.clip(delta / scale, -1, 1) * half)
        if end != cx:
            dr.rectangle((min(cx, end), y + 5, max(cx, end), y + 13), fill=COLORS[i])
        dr.text((w - 95, y), f'{delta:+6.2f} deg', fill='#eef1f5', font=font)
    return np.array(img)


def main():
    p = argparse.ArgumentParser(description='Read-only ACT video audit; creates a fixed-crop diagnostic copy')
    p.add_argument('--video', required=True)
    p.add_argument('--csv', required=True)
    p.add_argument('--out', required=True, help='New output folder, never the original video path')
    p.add_argument('--zoom', type=float, default=1.8)
    a = p.parse_args()
    if not (1 <= a.zoom <= 4):
        p.error('--zoom must be in [1,4]')
    src = Path(a.video)
    if not src.is_file():
        p.error(f'Video not found: {src}')
    with Path(a.csv).open('r', newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        p.error('CSV is empty')
    angles = np.array([[math.degrees(float(row[f'actual_rad_{name}'])) for name in JOINTS] for row in rows])
    if not np.isfinite(angles).all():
        p.error('non-finite CSV values')
    v = imageio.get_reader(str(src), format='ffmpeg')
    try:
        frames = [np.asarray(f)[:, :, :3] for f in v]
    finally:
        v.close()
    n = min(len(rows), len(frames))
    if n < 2:
        p.error(f'Need >=2 video frames and log rows (video={len(frames)}, csv={len(rows)})')
    if n != len(rows) or n != len(frames):
        print(f'WARNING: video frames={len(frames)}, csv rows={len(rows)}; using first {n}')
    frames, rows, angles = frames[:n], rows[:n], angles[:n]
    height, width = frames[0].shape[:2]
    bounds = robot_bounds(frames)
    region = crop_box(bounds, width, height, a.zoom)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    delta = np.abs(frames[0].astype(np.int16) - frames[-1].astype(np.int16))
    pixel_mae = float(delta.mean())
    changed_pct = float((np.max(delta, axis=2) > 8).mean() * 100)
    print('Decoded video frames:', len(frames), 'CSV rows:', len(rows))
    print(f'First-to-last pixel MAE: {pixel_mae:.3f}/255; >8-channel-change pixels: {changed_pct:.2f}%')
    print('Detected yellow robot bbox:', bounds, 'Fixed crop:', region)
    if bounds is None:
        print('WARNING: yellow robot segmentation failed; using center crop')
    print(f"{'JOINT':16} {'NET deg':>10} {'RANGE deg':>12} {'MAX TRACK ERROR deg':>21}")
    for i, name in enumerate(JOINTS):
        vals = angles[:, i]
        targets = np.array([math.degrees(float(r[f'target_rad_{name}'])) for r in rows])
        print(f'{name:16} {vals[-1]-vals[0]:10.3f} {np.ptp(vals):12.3f} {np.max(np.abs(vals-targets)):21.3f}')
    start = Image.fromarray(frames[0]).crop(region).resize((width, height), Image.Resampling.LANCZOS)
    last = Image.fromarray(frames[-1]).crop(region).resize((width, height), Image.Resampling.LANCZOS)
    start.save(out / 'start_zoom.png')
    last.save(out / 'end_zoom.png')
    # Keep diff visualization distinct; contrast is x4 only in this diagnostic PNG.
    enhanced = np.clip(delta * 4, 0, 255).astype(np.uint8)
    imageio.imwrite(str(out / 'pixel_difference_x4_diagnostic.png'), enhanced)
    scale = max(5.0, float(np.max(np.abs(angles - angles[:1]))) * 1.2)
    scale_label = f'+/-{scale:.1f}deg'
    dst = out / 'act_zoom_with_joint_telemetry.mp4'
    writer = imageio.get_writer(str(dst), fps=15, codec='libx264', macro_block_size=16, quality=8)
    try:
        for i, frame in enumerate(frames):
            resized = Image.fromarray(frame).crop(region).resize((width, height), Image.Resampling.LANCZOS)
            panel = add_metrics(np.asarray(resized), rows, i, angles, scale, scale_label)
            writer.append_data(panel)
    finally:
        writer.close()
    print('Wrote:', dst)
    print('Wrote:', out / 'start_zoom.png', 'and', out / 'end_zoom.png')
    print('No joint motion was artificially amplified; zoom is purely camera framing.')


if __name__ == '__main__':
    main()

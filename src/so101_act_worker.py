#!/usr/bin/env python3
"""LeRobot 0.6.1 ACT worker: uses saved real camera frame + changing virtual joint state.

Only called by the simulation Python process via JSON-lines stdin/stdout.
Never connects to a robot, camera, COM port, or motor.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def send(**payload):
    sys.stdout.write(json.dumps(payload, allow_nan=False) + "\n")
    sys.stdout.flush()


def log(*parts):
    print(*parts, file=sys.stderr, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = p.parse_args()

    # Redirect third-party chatter away from stdout (reserved exclusively for JSON).
    real_stdout = sys.stdout
    sys.stdout = sys.stderr
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig

    ckpt = Path(args.checkpoint)
    if not (ckpt / "config.json").is_file() or not (ckpt / "model.safetensors").is_file():
        raise FileNotFoundError(f"Checkpoint must contain config.json and model.safetensors: {ckpt}")
    dataset = Path(args.dataset)
    if not (dataset / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Dataset missing meta/info.json: {dataset}")

    cfg = PreTrainedConfig.from_pretrained(str(ckpt))
    if cfg.type != "act":
        raise ValueError(f"Checkpoint is not ACT: type={cfg.type}")
    cfg.device = args.device
    cfg.n_action_steps = 1  # re-query policy with every new simulated joint observation
    model = ACTPolicy.from_pretrained(str(ckpt), config=cfg, strict=True)
    model.reset()
    imgkeys = list(cfg.image_features)
    if len(imgkeys) != 1:
        raise ValueError(f"Expected exactly one camera for this starter; config has {imgkeys}")
    key = imgkeys[0]
    ds = LeRobotDataset(repo_id="local/so101_real_to_sim", root=dataset)
    if not 0 <= args.frame < len(ds):
        raise IndexError(f"Dataset frame {args.frame} out of bounds 0..{len(ds)-1}")
    sample = ds[args.frame]
    if "observation.state" not in sample or key not in sample:
        raise KeyError(f"Dataset missing image/state needed by ACT; keys={list(sample)}")
    initial_state = sample["observation.state"].detach().cpu().flatten().tolist()
    if len(initial_state) != 6:
        raise ValueError(f"Expected 6-D observation.state; got {initial_state}")
    image = sample[key]
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Unexpected dataset image type {type(image)}")
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image. Got {tuple(image.shape)}")
    available_pre = ckpt / "policy_preprocessor.json"
    available_post = ckpt / "policy_postprocessor.json"
    if available_pre.is_file() and available_post.is_file():
        pre, post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=str(ckpt),
            dataset_stats=ds.meta.stats,
            preprocessor_overrides={"device_processor": {"device": args.device}},
        )
        log("Using saved preprocessors:", available_pre, available_post)
    elif not available_pre.exists() and not available_post.exists():
        pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds.meta.stats)
        log("NOTE: Reconstructed normalization processors from dataset stats; VERIFY training equivalence.")
    else:
        raise RuntimeError("Only one of policy_preprocessor.json / policy_postprocessor.json exists")
    sys.stdout = real_stdout
    log("ACT checkpoint", ckpt, "image", key, tuple(image.shape), "frame", args.frame, "device", args.device)
    calls = 0
    for line in sys.stdin:
        try:
            req = json.loads(line)
            op = req.get("op")
            if op == "hello":
                send(initial_state=initial_state, model="ACTPolicy", image_key=key,
                     image_shape=list(image.shape), image_source="fixed_real_dataset_frame")
            elif op == "act":
                vals = req["state"]
                if len(vals) != 6:
                    raise ValueError("The simulation must provide 6 joint states")
                obs = {"observation.state": torch.tensor(vals, dtype=torch.float32), key: image}
                with torch.inference_mode():
                    batch = pre(obs)
                    action = model.select_action(batch)
                    result = post(action)
                arr = torch.as_tensor(result).detach().cpu().reshape(-1)
                if arr.numel() != 6 or not torch.isfinite(arr).all():
                    raise ValueError(f"ACT output invalid: {arr}")
                calls += 1
                send(action=[float(x) for x in arr.tolist()], inference_calls=calls)
            elif op == "quit":
                send(ok=True)
                break
            else:
                raise ValueError(f"Unknown op: {op}")
        except Exception as exc:
            log(traceback.format_exc())
            send(error=f"{type(exc).__name__}: {exc}")
            break


if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        log(traceback.format_exc())
        sys.exit(1)

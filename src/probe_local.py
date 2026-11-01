#!/usr/bin/env python3
"""Read-only discovery of a LeRobot dataset and pretrained ACT checkpoints."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    d = Path(a.dataset)
    info = d / "meta" / "info.json"
    print("=== DATASET ===")
    if not info.exists():
        print("MISSING:", info)
    else:
        m = json.loads(info.read_text(encoding="utf-8"))
        for k in ("codebase_version", "fps", "total_episodes", "total_frames", "robot_type"):
            print(k, ":", m.get(k, "not specified"))
        print("features:")
        for k, v in m.get("features", {}).items():
            print(" ", k, "shape=", v.get("shape"), "names=", v.get("names"))
    print("\n=== PRETRAINED CHECKPOINTS ===")
    root = Path(a.outputs)
    found = 0
    for conf in sorted(root.glob("**/config.json")):
        model = conf.parent / "model.safetensors"
        if not model.is_file():
            continue
        found += 1
        try:
            cfg = json.loads(conf.read_text(encoding="utf-8"))
            print(f"\n{conf.parent}")
            print("type:", cfg.get("type"), "use_vae:", cfg.get("use_vae"),
                  "n_action_steps:", cfg.get("n_action_steps"), "chunk_size:", cfg.get("chunk_size"))
            print("model size (MB):", round(model.stat().st_size / 1e6, 1))
            print("saved preprocessor:", (conf.parent / "policy_preprocessor.json").exists(),
                  "postprocessor:", (conf.parent / "policy_postprocessor.json").exists())
        except Exception as exc:
            print("  ERROR reading config:", exc)
    if not found:
        print("No pretrained_model found under", root)
        print("Look for checkpoints/6000/pretrained_model or checkpoints/last/pretrained_model.")
    print("\nREAD-ONLY: no motor / serial / COM-port access performed")

if __name__ == "__main__":
    main()

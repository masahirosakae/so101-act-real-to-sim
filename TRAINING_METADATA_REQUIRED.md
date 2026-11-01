# ACT training: verified configuration and remaining unknowns

## Confirmed from the Windows checkpoint and PowerShell run history

- Model: ACT No-VAE, ResNet18; Transformer dim=128, Encoder 2, Decoder 1; chunk_size=20, saved n_action_steps=20.
- Optimizer: AdamW; learning rate 1e-5, weight decay 1e-4, beta=(0.9, 0.999), epsilon=1e-8, clipping norm 10.
- Batch size 1, seed 1000, log_freq 100; W&B disabled; scheduler unset.
- Dataset: 10 episodes at 15 FPS; image_transforms.enable=false; eval_split=0.2.
- `act_fullgoal_novae_fresh_5000_20261101_192814/checkpoints/005000` has use_vae=false, resume=false, no pretrained_path or checkpoint_path.
- Resumed with `--config_path` referencing above and `--resume=true --steps=10000`. Checkpoints 006000..010000 exist; 006000 used in reported results.
- A *different* VAE-enabled run has logs for steps ~2000..5000. Never splice this loss series into the No-VAE loss plot.

## Split interpretation

LeRobot v0.6.1 splits evaluation episodes **per task**: the last ceil(number of episodes for task * 0.2) are held out. If all 10 episodes share a task label, train=Episode 0..7 and eval=Episode 8..9. Dataset task labels and actual logging were not supplied. The reported Episode 2 evaluation must not be presented as held-out testing.

Source: https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/datasets/factory.py

## Still unknown; do not invent

- Exact No-VAE training loss curve for the fresh/resumed runs (no scalar logs found).
- Precise GPU, elapsed training wall time, ResNet18 backbone initialization weights.
- Independent held-out-task/episode metrics and autonomous task success.

The preexisting `src/export_act_config_fields.py` extracts settings from the **local** checkpoint safely; it does not reconstruct missing train-loss history.

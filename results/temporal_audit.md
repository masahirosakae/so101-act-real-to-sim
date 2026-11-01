# Episode 2 temporal alignment audit

Based on 420 common frames at 15 FPS. All quantities are five-arm-joint mean MAE unless specified. Convention: A[t] compared to B[t+lag].

| Comparison | lag 0 | best lag | MAE at best |
|---|---:|---:|---:|
| ACT[t] vs teacher[t+lag] | 4.262° | +9 frames (+0.600 s) | 2.923° |
| Teacher[t] vs measured state[t+lag] | see joint table | +3 frames (+0.200 s) overall | see joint table |

At lag +9, the exact same teacher timestamps produce ACT MAE 2.923°, hold-state baseline MAE 5.040°, and causal 5-frame velocity extrapolation 3.410°. Lag 0 hold-state baseline is 1.784° on the same reference frames. The nonzero optimal lag is **exploratory** and does **not** prove time synchronization defects.

| Servo joint | best lag | MAE before | MAE after |
|---|---:|---:|---:|
| shoulder_pan | +3 | 1.977° | 0.843° |
| shoulder_lift | +3 | 2.374° | 1.208° |
| elbow_flex | +4 | 3.001° | 1.675° |
| wrist_flex | +3 | 1.150° | 0.880° |
| wrist_roll | +3 | 0.417° | 0.328° |

Per-joint optimal ACT prediction lags range from +5 to +10 frames; three time segments show +7/+10/+10 frames. Data are from an episode used in training.

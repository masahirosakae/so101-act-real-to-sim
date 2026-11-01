# ACT checkpoint comparison — Episode 2

Evaluation uses the SAME 45 frames (stride 10) from the **training** episode. The values are mean absolute error in degrees. The sixth gripper channel is excluded from the five-joint mean.

| Joint | 6,000 | 8,000 | 10,000 |
|---|---:|---:|---:|
| shoulder_pan | 4.927 | 4.237 | 4.299 |
| shoulder_lift | 7.739 | 12.969 | 8.334 |
| elbow_flex | 5.898 | 9.378 | 7.253 |
| wrist_flex | 1.987 | 1.937 | 2.176 |
| wrist_roll | 0.829 | 0.840 | 0.769 |
| **Five-axis mean** | **4.276** | 5.872 | 4.566 |

Five-axis hold-position baseline: 1.725 degrees. Only 6,000 was chosen for the subsequent MuJoCo demonstration. These are in-sample offline metrics, not task success rates.

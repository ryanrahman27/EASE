# E25: frozen paired-arm diagnostics

| Seed | Privileged six-joint | Privileged seven-joint |
|---|---:|---:|
| 0 | 6/48 | 48/48 |
| 1 | 12/48 | 48/48 |
| 2 | 2/48 | 48/48 |

| Seed | Probe | Intact | Shuffled actions | Shuffled transitions | Pose trajectory | Initial pose |
|---|---|---:|---:|---:|---:|---:|
| 0 | 930000 | 62/96 | 13/96 | 62/96 | 13/96 | 13/96 |
| 0 | 940000 | 57/96 | 13/96 | 57/96 | 14/96 | 13/96 |
| 0 | 950000 | 61/96 | 23/96 | 60/96 | 11/96 | 13/96 |
| 1 | 930000 | 52/96 | 40/96 | 54/96 | 29/96 | 28/96 |
| 1 | 940000 | 61/96 | 46/96 | 63/96 | 30/96 | 28/96 |
| 1 | 950000 | 62/96 | 67/96 | 60/96 | 28/96 | 28/96 |
| 2 | 930000 | 44/96 | 23/96 | 44/96 | 9/96 | 8/96 |
| 2 | 940000 | 45/96 | 25/96 | 45/96 | 6/96 | 8/96 |
| 2 | 950000 | 48/96 | 27/96 | 52/96 | 9/96 | 8/96 |

| Seed | Model | Initial action RMSE (rad) | Later expert-state RMSE (rad) |
|---|---|---:|---:|
| 0 | teacher | 0.000960 | 0.000512 |
| 0 | reactive | 0.014745 | 0.001059 |
| 1 | teacher | 0.001309 | 0.000561 |
| 1 | reactive | 0.014680 | 0.001127 |
| 2 | teacher | 0.001090 | 0.000588 |
| 2 | reactive | 0.014701 | 0.001037 |

Twin initial-input deterministic squared-action-error floor: 0.000210500496 rad² per valid joint.
This is a supervised first-action ambiguity bound under equal twin weighting, not a lower bound on achievable task success.
Frozen intervention inputs may be out of distribution. Privileged teachers are diagnostics, not performance upper bounds.
All intact replays match prior outcomes exactly. Checkpoints unchanged; final test not evaluated.

## Reset-state coverage

Post-hoc diagnostic: training demonstration joint ranges versus validation home positions (tolerance 1e-6 rad).
- 6-joint validation: out-of-range joints [4, 5].
- 7-joint validation: out-of-range joints [].
The six-joint split changes both kinematics and reset posture; this confounds interpreting failure as kinematic transfer alone.

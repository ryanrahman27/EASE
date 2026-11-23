# E71: fresh-placement validation of extended phase budgets

17/18 prespecified placements pass endpoint IK screening across the same six procedural bodies. All screening rejections are retained without replacement; each accepted placement receives all seven estimators and both conditions (238 trials). Positions, IK seeds and probe seeds are new. Bodies and axis families are reused: this is placement transfer, not new-embodiment transfer or the reserved final test.

| Estimator | Budget | Success | Six-joint | Seven-joint | Collisions | Total steps |
|---|---|---:|---:|---:|---:|---:|
| analytical | baseline | 16/17 | 8/8 | 8/9 | 1 | 4752 |
| analytical | extended | 15/17 | 7/8 | 8/9 | 2 | 5090 |
| parent_0 | baseline | 15/17 | 8/8 | 7/9 | 2 | 5184 |
| parent_0 | extended | 15/17 | 8/8 | 7/9 | 2 | 5471 |
| parent_1 | baseline | 17/17 | 8/8 | 9/9 | 0 | 5030 |
| parent_1 | extended | 15/17 | 8/8 | 7/9 | 2 | 5622 |
| parent_2 | baseline | 15/17 | 8/8 | 7/9 | 2 | 4749 |
| parent_2 | extended | 15/17 | 8/8 | 7/9 | 2 | 5117 |
| candidate_0 | baseline | 14/17 | 7/8 | 7/9 | 3 | 5297 |
| candidate_0 | extended | 15/17 | 6/8 | 9/9 | 2 | 5648 |
| candidate_1 | baseline | 15/17 | 7/8 | 8/9 | 2 | 4907 |
| candidate_1 | extended | 15/17 | 7/8 | 8/9 | 2 | 5233 |
| candidate_2 | baseline | 17/17 | 8/8 | 9/9 | 0 | 4723 |
| candidate_2 | extended | 17/17 | 8/8 | 9/9 | 0 | 5094 |

Pooled learned success: 93/102 baseline versus 92/102 extended; 2 recoveries and 3 regressions. Six-joint: 46 to 45/48; seven-joint: 47 to 47/54. Collisions: 9 to 10. Total control steps including calibration: 29890 to 32185 (+7.7%).

Prespecified aggregate replication rule: FAIL. Per-checkpoint aggregate six-joint retention: False. The E70 improvement does not pass the prespecified replication rule. Keep the original phase budgets and retain extended timing as a failed development candidate.

The frozen E70 implementation adds at most 40 steps per moving phase and 20 to close. It leaves the early-exit rule, physics, waypoints and success/contact criteria unchanged. No strict transition gate is used. No thresholds, parameters or checkpoints were chosen using E71 results. All trials share a cloned measured calibration state within each placement; probes with contacts remain failures rather than being filtered out.

| Family | Budget | Learned success | Collision cases |
|---|---|---:|---:|
| six_amb_train_0 | baseline | 16/18 | 2 |
| six_amb_train_0 | extended | 16/18 | 2 |
| six_amb_train_1 | baseline | 18/18 | 0 |
| six_amb_train_1 | extended | 18/18 | 0 |
| seven_amb_train_0 | baseline | 13/18 | 5 |
| seven_amb_train_0 | extended | 15/18 | 3 |
| seven_amb_train_1 | baseline | 18/18 | 0 |
| seven_amb_train_1 | extended | 17/18 | 1 |
| broad_6_0 | baseline | 12/12 | 0 |
| broad_6_0 | extended | 11/12 | 1 |
| broad_7_8 | baseline | 16/18 | 2 |
| broad_7_8 | extended | 15/18 | 3 |

Screening rejections: candidate 13 (6 joints), failed approach after 0 endpoints.

Verification: all model/input/E70 implementation hashes unchanged; unique trial keys, complete nine-phase rollouts and trace lengths audited. The same 99 EASE-FM tests pass. Full protocols, probes, screening records and trial traces are retained. No model training, native tuning or final-test rollout.

Interpretation is limited by endpoint screening, six reused bodies, dependent training seeds and observed development history. Aggregate six-joint retention does not mean every successful six-joint case is preserved; paired.json lists each regression. Extra time and its cost must remain part of the comparison.

### Decision and next step

The replication rule fails; retain the original phase budgets. Do not promote extended timing. Stop tuning waiting times on these placements. The next distinct experiment should change the path before contact using a shared clearance-aware controller, with identical geometry access for learned and analytical estimators and explicit disclosure of that extra information. The reserved final test remains untouched.

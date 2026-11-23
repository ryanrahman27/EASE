# E99: training-task histories and matched action retention

Collect36 complete parent-driven trajectories on12 existing training placements across four original training bodies. All cases and failures are retained. Save11714 pre-action contexts; fitting uses only post-probe continuation contexts: 5921 training and3419 validation. Third placement per body is validation across every driver, giving24/12 disjoint trajectories. Reuse E98 local-motion data on all eight training bodies (12288 train/4096 validation). No evaluation-arm histories enter training.

Parent data-collection outcomes: training {'n': 24, 'success': 20, 'collisions': 4}; validation {'n': 12, 'success': 7, 'collisions': 5}. These are source-driver outcomes, not trained-student training-task scores. Full phase traces and calibration observations are saved. Original geometry filtering and32-step probe are used; there is no training route planner. Existing IK homes are privileged setup.

Both conditions start from original E66 checkpoints. Each of2000 identical paired batches contains64 synthetic,32 local-motion and32 training-task contexts. AdamW lr2e-5, gradient clip1; analytical ridge prediction target and synthetic parent anchor are identical. Only retained adds0.01 times the mean squared clipped joint-command difference from the frozen parent, divided by0.04 squared, on task contexts. This uses the current measured bias, task error, DLS damping0.015 and gain0.35. It compares commands before geometry filtering and physics; it is not an executed-contact loss. Parent commands from unsuccessful trajectories are also retained, a possible recovery/retention tradeoff.

All six fixed final checkpoints are evaluated without checkpoint/seed selection. Paired batch hashes match within every seed. Local/task targets are independently checked against NumPy solves. Validation holds out trajectories/placements on the same training bodies.

| Seed | Condition | Local MSE before / after | Task MSE before / after | Held task action drift | Synthetic MSE before / after |
|---|---|---:|---:|---:|---:|
| 0 | data_only | 0.0033294 / 0.0023017 | 0.0030902 / 0.0021674 | 0.26684 | 0.0056808 / 0.005066 |
| 0 | retained | 0.0033294 / 0.0026877 | 0.0030902 / 0.0026957 | 0.27655 | 0.0056808 / 0.0052963 |
| 1 | data_only | 0.0031239 / 0.0022316 | 0.0030765 / 0.0021837 | 0.18953 | 0.0057278 / 0.0051118 |
| 1 | retained | 0.0031239 / 0.0027297 | 0.0030765 / 0.0028003 | 0.18315 | 0.0057278 / 0.00539 |
| 2 | data_only | 0.0032851 / 0.0022784 | 0.0030422 / 0.0021779 | 0.20497 | 0.0056559 / 0.0050656 |
| 2 | retained | 0.0032851 / 0.0026706 | 0.0030422 / 0.0027311 | 0.16996 | 0.0056559 / 0.0053294 |

## Full-task evaluation

All16 development tasks x3 seeds x parent/data_only/retained =144 complete rollouts. Same frozen parent approach and probe; only post-probe neural weights change on admitted routes. Every parent replay matches E97, and every handoff/prefix matches the saved references. No-route behavior remains unchanged. Evaluation remains phase-specific and uses privileged route planning.

| Group | Mode | Success | Collisions | Total steps |
|---|---|---:|---:|---:|
| all | parent | 29/48 | 18 | 20835 |
| all | data_only | 31/48 | 17 | 20407 |
| all | retained | 32/48 | 16 | 20716 |
| 6 | parent | 13/24 | 10 | 10250 |
| 6 | data_only | 15/24 | 9 | 10132 |
| 6 | retained | 14/24 | 10 | 10269 |
| 7 | parent | 16/24 | 8 | 10585 |
| 7 | data_only | 16/24 | 8 | 10275 |
| 7 | retained | 18/24 | 6 | 10447 |
| candidate_0 | parent | 8/16 | 8 | 6793 |
| candidate_0 | data_only | 10/16 | 6 | 6541 |
| candidate_0 | retained | 11/16 | 5 | 6736 |
| candidate_1 | parent | 9/16 | 6 | 7163 |
| candidate_1 | data_only | 11/16 | 5 | 6974 |
| candidate_1 | retained | 10/16 | 6 | 7062 |
| candidate_2 | parent | 12/16 | 4 | 6879 |
| candidate_2 | data_only | 10/16 | 6 | 6892 |
| candidate_2 | retained | 11/16 | 5 | 6918 |
| route | parent | 29/39 | 9 | 17033 |
| route | data_only | 31/39 | 8 | 16605 |
| route | retained | 32/39 | 7 | 16914 |
| no_route | parent | 0/9 | 9 | 3802 |
| no_route | data_only | 0/9 | 9 | 3802 |
| no_route | retained | 0/9 | 9 | 3802 |

## All paired changes

| Condition | Control | Case | Seed | DOF | Change | Collision | First collision phase |
|---|---|---:|---|---:|---|---|---|
| data_only | parent | 2 | candidate_0 | 6 | gain | False | None |
| data_only | parent | 2 | candidate_1 | 6 | gain | False | None |
| data_only | parent | 8 | candidate_2 | 7 | loss | True | lower |
| data_only | parent | 10 | candidate_1 | 7 | gain | False | None |
| data_only | parent | 11 | candidate_0 | 7 | gain | False | None |
| data_only | parent | 11 | candidate_2 | 7 | loss | True | retreat |
| data_only | parent | 12 | candidate_0 | 7 | gain | False | None |
| data_only | parent | 12 | candidate_1 | 7 | gain | False | None |
| data_only | parent | 12 | candidate_2 | 7 | loss | True | retreat |
| data_only | parent | 13 | candidate_1 | 7 | loss | True | retreat |
| data_only | parent | 13 | candidate_2 | 7 | gain | False | None |
| data_only | parent | 15 | candidate_0 | 7 | loss | True | lower |
| retained | parent | 2 | candidate_0 | 6 | gain | False | None |
| retained | parent | 8 | candidate_0 | 7 | gain | False | None |
| retained | parent | 10 | candidate_0 | 7 | loss | True | retreat |
| retained | parent | 10 | candidate_1 | 7 | gain | False | None |
| retained | parent | 11 | candidate_0 | 7 | gain | False | None |
| retained | parent | 11 | candidate_2 | 7 | loss | True | lift |
| retained | parent | 12 | candidate_0 | 7 | gain | False | None |
| retained | parent | 12 | candidate_1 | 7 | gain | False | None |
| retained | parent | 13 | candidate_1 | 7 | loss | True | transport |
| retained | data_only | 2 | candidate_1 | 6 | loss | True | retreat |
| retained | data_only | 8 | candidate_0 | 7 | gain | False | None |
| retained | data_only | 8 | candidate_2 | 7 | gain | False | None |
| retained | data_only | 10 | candidate_0 | 7 | loss | True | retreat |
| retained | data_only | 12 | candidate_2 | 7 | gain | False | None |
| retained | data_only | 13 | candidate_2 | 7 | loss | True | retreat |
| retained | data_only | 15 | candidate_0 | 7 | gain | False | None |

## Retention gates

- data_only: **FAIL**. Success improves=True; collisions nonincreasing=True; every parent success retained=False; every seed nondecreasing=False.
- retained: **FAIL**. Success improves=True; collisions nonincreasing=True; every parent success retained=False; every seed nondecreasing=False.
- data_only versus parent: 7 gains, 5 losses.
- retained versus parent: 6 gains, 3 losses.
- retained versus data_only: 4 gains, 3 losses.

## Interpretation and verification

At the fixed coefficient, retained reaches32/48 successes with16 collisions versus parent29/48 with18 and data_only31/48 with17. Retained gains six parent failures but loses three parent successes: case10 seed0 (first collision in retreat), case11 seed2 (lift), and case13 seed1 (transport), all seven-joint. Seed2 declines12/16 to11/16. Both adoption gates fail; no checkpoint replaces the parent. Six-joint retained improves13/24 to14/24 without losing a parent success, but data_only reaches15/24. The action-retention term trades four recoveries against three losses relative to data_only, so its pooled gain is not universal.

Next decision: audit the three retained regressions at matched physical states, measuring parent/student command differences before and after geometry filtering and their accumulation by phase. First use read-only queries without changing executed actions; then predeclare any training-only remedy and repeat the full paired evaluation. Do not tune on these three trajectories or select a successful seed. On-policy student histories on training bodies are a candidate remedy for distribution shift, which this experiment has not established causally.

The matched comparison isolates the added action-retention term at the fixed coefficient. E98 comparisons also change the data mixture, so they do not isolate task-data addition alone. Lower command drift on training-body validation does not guarantee retention on new-body full tasks. No failed gate is overridden by a favorable pooled score, seed selection or post-hoc DOF switch.

129 tests pass, including controller-formula agreement for6/7 joints and finite action-loss gradients. All48 parent full rollouts and144 identical handoffs verify; both trained branches preserve all nine no-route cases. Dataset, parent, paired-batch and final checkpoint hashes are verified. No target-arm training, inference weight updates or reserved-test evaluation occurs. This is not a whole-policy or original-reaching-suite retention claim.

## Videos and figures

Gallery: `outputs/ease_task_retention_training_v1/videos/index.html`. Prespecified case2 candidate_0 and cases10/15 candidate_2, all three modes; first new regression for each condition is appended with its parent if absent, plus a missing outcome category if needed. Outcome selection is disclosed. Full serialized video replays, decoded frame counts and SHA-256 hashes are checked. Scientific figure: `retention.png` and `retention.pdf`.

# E93: fixed recent-history estimation after planned descent

All96 E91 checkpoint/task pairs receive two full rollouts: exact long-history route300 replay and window32. After an accepted planned descent, window32 reconstructs gram/cross from the32 most recent completed transitions using the original initial ridge, forgetting factor and diagonal floor; it repeats this reconstruction after each subsequent measured transition. Adaptive measured bias remains unchanged. No new probe, policy weight update, window sweep or simulator-Jacobian policy input is introduced. Cases without an admitted route retain the original history.

The privileged E89/E91 route planner and300-step descent allowance remain present in both conditions. This evaluates a history intervention in that hybrid controller on reused development cases, not a history-only robot policy or independent transfer test.

| Group | Mode | Success | Collisions | Control steps including calibration | Window applied |
|---|---|---:|---:|---:|---:|
| all | long_history | 57/96 | 33 | 39284 | 0 |
| all | window32 | 58/96 | 34 | 40139 | 78 |
| 6 | long_history | 22/48 | 22 | 19619 | 0 |
| 6 | window32 | 28/48 | 18 | 19459 | 30 |
| 7 | long_history | 35/48 | 11 | 19665 | 0 |
| 7 | window32 | 30/48 | 16 | 20680 | 48 |
| candidate_0 | long_history | 11/16 | 4 | 6128 | 0 |
| candidate_0 | window32 | 11/16 | 5 | 6579 | 13 |
| candidate_1 | long_history | 9/16 | 6 | 6791 | 0 |
| candidate_1 | window32 | 9/16 | 5 | 7052 | 13 |
| candidate_2 | long_history | 9/16 | 6 | 6441 | 0 |
| candidate_2 | window32 | 12/16 | 4 | 6515 | 13 |
| parent_0 | long_history | 11/16 | 5 | 6780 | 0 |
| parent_0 | window32 | 11/16 | 5 | 6482 | 13 |
| parent_1 | long_history | 7/16 | 7 | 7138 | 0 |
| parent_1 | window32 | 7/16 | 8 | 7071 | 13 |
| parent_2 | long_history | 10/16 | 5 | 6006 | 0 |
| parent_2 | window32 | 8/16 | 7 | 6440 | 13 |

| Case | Checkpoint | DOF | Change vs long history | Old collision | New collision |
|---|---|---:|---|---|---|
| 0 | parent_1 | 6 | gain | False | False |
| 0 | parent_2 | 6 | gain | False | False |
| 0 | candidate_0 | 6 | gain | False | False |
| 0 | candidate_2 | 6 | gain | True | False |
| 1 | candidate_2 | 6 | gain | False | False |
| 2 | parent_1 | 6 | loss | False | False |
| 2 | candidate_0 | 6 | gain | True | False |
| 4 | candidate_1 | 6 | gain | True | False |
| 8 | parent_0 | 7 | gain | True | False |
| 10 | parent_2 | 7 | loss | False | True |
| 11 | parent_1 | 7 | loss | False | True |
| 11 | parent_2 | 7 | loss | False | True |
| 11 | candidate_0 | 7 | loss | False | True |
| 11 | candidate_1 | 7 | loss | False | True |
| 12 | parent_0 | 7 | loss | False | True |
| 12 | candidate_2 | 7 | gain | True | False |
| 13 | candidate_0 | 7 | loss | False | True |
| 13 | candidate_2 | 7 | gain | True | False |
| 14 | candidate_2 | 7 | loss | False | True |
| 15 | parent_1 | 7 | gain | False | False |
| 15 | parent_2 | 7 | loss | False | False |

## Decision and interpretation

Do not adopt window32. It produces58/96 successes versus57/96, but increases collisions from33 to34. Six-joint performance improves22/48 to28/48 while seven-joint performance falls35/48 to30/48. The11 gains and10 losses include one lost previously successful six-joint hybrid case. Only five of the ten E91 regressions recover, unlike the privileged exact-Jacobian continuation in E92. These opposing DOF aggregates do not justify selecting a history rule by joint count after observing the results.

The prespecified gate FAILS: success improvement=True, nonincreasing collisions=False, lost successful six-joint hybrid cases=1. There are11 gains and10 losses versus long history. Window32 recovers5/10 E91 regressions diagnosed in E92.

Relative to the original E87 learned geometry-filter baseline, window32 has28 gains and13 losses, including1 six-joint losses. This separate comparison retains the known planner and longer-budget confounds.

A shorter history changes both locality and the amount/directional coverage of excitation. Near a stationary grasp,32 recent transitions may contain very little informative motion; frozen weights may also encounter statistics different from training. Outcomes cannot by themselves isolate stale history, insufficient excitation or distribution shift. Do not interpret aggregate success as estimator calibration or infer a failure threshold from this run.

If this fixed-window intervention fails retention, retain the previous controller and investigate measured excitation/conditioning at handoff before considering another history rule. Any replacement should preserve informative motion rather than blindly discarding older transitions. Predeclare subsequent checks and retain the same successful controls; do not search window sizes on these outcomes and claim independent validation.

Verification:96 exact E91 reference replays and18 exact unavailable-route fallbacks; identical pre-intervention handoff states and statistics; no task simulator Jacobian access; frozen checkpoint/input/source hashes;117 passing tests including exact last32-statistic replay, bias preservation and copied measurement history. Individual trials were saved immediately. No reserved-test evaluation occurred.

## Saved videos

Gallery: `outputs/ease_local_history_v1/videos/index.html`. Prespecified candidate_0 cases0 and4 show both modes; append the first missing success/failure category if needed. A disclosed post-result addition shows the first newly introduced regression (case2 parent_1) in both modes, for six videos total. Executed video replays match the saved results exactly. These illustrative examples do not estimate success rate.

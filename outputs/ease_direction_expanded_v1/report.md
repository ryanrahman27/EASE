# E86: complete-task retention with frozen direction correction

Twelve new parameterizations (same axis families/topology) and two fixed placements per body were saved before screening and policy execution. 21/24 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.

Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 114/126 | 11 | 37010 | 0 | 0 |
| learned | response | 116/126 | 9 | 37080 | 197 | 755034 |
| learned | command | 114/126 | 11 | 37035 | 18 | 827253 |
| learned_6 | baseline | 53/54 | 1 | 15155 | 0 | 0 |
| learned_6 | response | 53/54 | 1 | 15192 | 35 | 292320 |
| learned_6 | command | 53/54 | 1 | 15155 | 0 | 321783 |
| learned_7 | baseline | 61/72 | 10 | 21855 | 0 | 0 |
| learned_7 | response | 63/72 | 8 | 21888 | 162 | 462714 |
| learned_7 | command | 61/72 | 10 | 21880 | 18 | 505470 |
| analytical | baseline | 17/21 | 4 | 6014 | 0 | 0 |
| analytical | response | 18/21 | 3 | 6040 | 23 | 119742 |
| analytical | command | 18/21 | 3 | 5997 | 8 | 126693 |
| candidate_0 | baseline | 21/21 | 0 | 6128 | 0 | 0 |
| candidate_0 | response | 21/21 | 0 | 6128 | 0 | 114576 |
| candidate_0 | command | 21/21 | 0 | 6128 | 0 | 114576 |
| candidate_1 | baseline | 19/21 | 2 | 5969 | 0 | 0 |
| candidate_1 | response | 19/21 | 2 | 6028 | 18 | 117390 |
| candidate_1 | command | 19/21 | 2 | 5969 | 0 | 145719 |
| candidate_2 | baseline | 20/21 | 1 | 5948 | 0 | 0 |
| candidate_2 | response | 20/21 | 1 | 5958 | 4 | 112182 |
| candidate_2 | command | 20/21 | 1 | 5948 | 0 | 111972 |
| parent_0 | baseline | 18/21 | 3 | 6377 | 0 | 0 |
| parent_0 | response | 18/21 | 3 | 6432 | 81 | 146496 |
| parent_0 | command | 18/21 | 3 | 6385 | 1 | 168525 |
| parent_1 | baseline | 16/21 | 4 | 6567 | 0 | 0 |
| parent_1 | response | 17/21 | 3 | 6513 | 93 | 151767 |
| parent_1 | command | 16/21 | 4 | 6584 | 17 | 173250 |
| parent_2 | baseline | 20/21 | 1 | 6021 | 0 | 0 |
| parent_2 | response | 21/21 | 0 | 6021 | 1 | 112623 |
| parent_2 | command | 20/21 | 1 | 6021 | 0 | 113211 |
| body_0 | baseline | 12/12 | 0 | 3238 | 0 | 0 |
| body_0 | response | 12/12 | 0 | 3238 | 0 | 59934 |
| body_0 | command | 12/12 | 0 | 3238 | 0 | 59934 |
| body_1 | baseline | 12/12 | 0 | 3678 | 0 | 0 |
| body_1 | response | 12/12 | 0 | 3678 | 0 | 69174 |
| body_1 | command | 12/12 | 0 | 3678 | 0 | 69174 |
| body_2 | baseline | 12/12 | 0 | 2965 | 0 | 0 |
| body_2 | response | 12/12 | 0 | 2965 | 0 | 54201 |
| body_2 | command | 12/12 | 0 | 2965 | 0 | 54201 |
| body_3 | baseline | 12/12 | 0 | 2927 | 0 | 0 |
| body_3 | response | 12/12 | 0 | 2927 | 0 | 53403 |
| body_3 | command | 12/12 | 0 | 2927 | 0 | 53403 |
| body_4 | baseline | 11/12 | 1 | 3488 | 0 | 0 |
| body_4 | response | 11/12 | 1 | 3488 | 0 | 65772 |
| body_4 | command | 11/12 | 1 | 3488 | 0 | 65772 |
| body_5 | baseline | 11/12 | 1 | 3681 | 0 | 0 |
| body_5 | response | 11/12 | 1 | 3729 | 16 | 76125 |
| body_5 | command | 11/12 | 1 | 3681 | 0 | 93933 |
| body_6 | baseline | 11/12 | 1 | 3705 | 0 | 0 |
| body_6 | response | 12/12 | 0 | 3702 | 1 | 69972 |
| body_6 | command | 11/12 | 1 | 3705 | 0 | 70329 |
| body_7 | baseline | 11/12 | 1 | 3530 | 0 | 0 |
| body_7 | response | 11/12 | 1 | 3551 | 3 | 67389 |
| body_7 | command | 11/12 | 1 | 3530 | 0 | 67536 |
| body_9 | baseline | 5/6 | 1 | 2347 | 0 | 0 |
| body_9 | response | 5/6 | 1 | 2384 | 35 | 55608 |
| body_9 | command | 5/6 | 1 | 2347 | 0 | 85071 |
| body_10 | baseline | 8/12 | 4 | 3910 | 0 | 0 |
| body_10 | response | 8/12 | 4 | 3882 | 70 | 94626 |
| body_10 | command | 8/12 | 4 | 3918 | 10 | 104496 |
| body_11 | baseline | 9/12 | 2 | 3541 | 0 | 0 |
| body_11 | response | 10/12 | 1 | 3536 | 72 | 88830 |
| body_11 | command | 9/12 | 2 | 3558 | 8 | 103404 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- response: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- command: FAIL; improved success=False, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| response | 13 | parent_1 | 7 | gain | True | False |
| response | 16 | analytical | 6 | gain | True | False |
| response | 20 | parent_2 | 7 | gain | True | False |
| command | 16 | analytical | 6 | gain | True | False |

These are prospective development results on new parameterizations derived from six existing body templates. They share existing axis families and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

Both correction modes remain available throughout the task. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline result and filter-log equality against a separately instantiated original GeometryArm, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

Measured-response attribution gate: **PASS**. Requires higher learned success than the command ablation without more collisions. Passing only the baseline-improvement gate does not establish response-model benefit.

Screening records:

- Candidate 0, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 1, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 2, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 3, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 4, body 2, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 5, body 2, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 6, body 3, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 7, body 3, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 8, body 4, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 9, body 4, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 10, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 11, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 12, body 6, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 13, body 6, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 14, body 7, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 15, body 7, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 16, body 8, 6 joints: rejected, 0/9 endpoints; failed phase approach.
- Candidate 17, body 8, 6 joints: rejected, 3/9 endpoints; failed phase lift.
- Candidate 18, body 9, 6 joints: rejected, 0/9 endpoints; failed phase approach.
- Candidate 19, body 9, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 20, body 10, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 21, body 10, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 22, body 11, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 23, body 11, 7 joints: accepted, 9/9 endpoints; failed phase None.

Direct response-versus-command paired effects, by accepted body (checkpoint/placement outcomes are dependent within each body):

| Body | Family | Parameter seed | DOF | Response-only successes | Command-only successes |
|---|---|---:|---:|---:|---:|
| 0 | six_amb_train_0 | 860000 | 6 | 0 | 0 |
| 1 | six_amb_train_0 | 860001 | 6 | 0 | 0 |
| 2 | six_amb_train_1 | 860002 | 6 | 0 | 0 |
| 3 | six_amb_train_1 | 860003 | 6 | 0 | 0 |
| 4 | seven_amb_train_0 | 860004 | 7 | 0 | 0 |
| 5 | seven_amb_train_0 | 860005 | 7 | 0 | 0 |
| 6 | seven_amb_train_1 | 860006 | 7 | 1 | 0 |
| 7 | seven_amb_train_1 | 860007 | 7 | 0 | 0 |
| 9 | broad_6_0 | 860009 | 6 | 0 | 0 |
| 10 | broad_7_8 | 860010 | 7 | 0 | 0 |
| 11 | broad_7_8 | 860011 | 7 | 1 | 0 |

Bodies with no accepted placement are absent from rollout comparisons but retained in screening. A pooled success difference is not evidence of consistent effects across independently sampled bodies; inspect concentration and checkpoint identity before strengthening the claim.

### Decision and scope

Response correction passes baseline retention and matched attribution:116/126 learned successes and nine collisions, versus114/126 and eleven for both baseline and command-only correction. There are two response-only recoveries and no success losses. Both recoveries are seven-joint: body6 (seven_amb_train_1, seed860006), placement case13, parent1; and body11 (broad_7_8, seed860011), case20, parent2. Thus this expansion shows gains on two distinct new parameterizations, families and checkpoints, rather than repetitions of one favorable body. The effect remains only two paired outcomes, not a statistically established broad advantage.

Six-joint results are53/54 in every mode; the expansion preserves strong baseline performance but shows no additional six-joint recovery. Analytical control improves17/21 to18/21 with either correction method, so that analytical gain cannot be attributed to measured response.

The protocol sampled12 new bodies and24 placements. Screening accepted21 placements on11 bodies. Six-joint body8 has both placements rejected (approach and lift); body9 has one approach rejection. These are retained reachability failures, not replacement candidates. Rollout success is conditional on endpoint screening. Joint axes and topology remain unchanged from the six templates.

Response correction adds755,034 geometry queries for197 changed learned actions; combined with filtering,855,780 queries versus102,272 baseline (about8.37x). Command correction adds827,253 queries for18 changes. High search cost is a material limitation; these counts do not measure real-time latency.

Across E85 and E86, response correction yields175/192 learned successes versus172/192 for baseline and command correction, with three response-specific seven-joint recoveries and no losses. This is descriptive aggregation across small, related development studies, not an independent significance test. Six-joint performance across these studies is81/84 in every mode.

Next prioritize the original cross-kinematic question: define an arrangement-disjoint development evaluation using training-family manifests to exclude already trained joint-axis sequences, while leaving reserved final-test data untouched. Freeze controller settings and include matched command-only and analytical controls, all screening failures and per-body outcomes. Do not infer transfer to genuinely new axis arrangements from these parameter-perturbation results, and do not spend additional trials on small perturbations solely to accumulate favorable counts.

Verification: all147 baseline results and filter logs match separately instantiated original controllers; all441 primary evaluations use matched calibration. Input, source and checkpoint hashes and causal update counts pass;108 EASE-FM tests pass. No retraining, outcome-driven replacement or reserved-test evaluation occurred.

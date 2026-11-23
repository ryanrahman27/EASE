# E87: complete-task retention with frozen direction correction

Eight new parameterizations on four unseen procedural axis sequences and two fixed placements per body were saved before screening and policy execution. 16/16 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.

Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 43/96 | 40 | 35473 | 0 | 0 |
| learned | response | 46/96 | 27 | 35191 | 1752 | 1883343 |
| learned | command | 44/96 | 35 | 35283 | 114 | 2531697 |
| learned_6 | baseline | 15/48 | 27 | 17971 | 0 | 0 |
| learned_6 | response | 17/48 | 20 | 17951 | 1453 | 1423023 |
| learned_6 | command | 15/48 | 27 | 17969 | 8 | 2016609 |
| learned_7 | baseline | 28/48 | 13 | 17502 | 0 | 0 |
| learned_7 | response | 29/48 | 7 | 17240 | 299 | 460320 |
| learned_7 | command | 29/48 | 8 | 17314 | 106 | 515088 |
| analytical | baseline | 7/16 | 7 | 5493 | 0 | 0 |
| analytical | response | 9/16 | 4 | 5438 | 171 | 340830 |
| analytical | command | 8/16 | 5 | 5504 | 28 | 373128 |
| candidate_0 | baseline | 6/16 | 8 | 5548 | 0 | 0 |
| candidate_0 | response | 9/16 | 4 | 5564 | 250 | 296898 |
| candidate_0 | command | 7/16 | 6 | 5520 | 31 | 368340 |
| candidate_1 | baseline | 9/16 | 6 | 5998 | 0 | 0 |
| candidate_1 | response | 9/16 | 4 | 5812 | 258 | 278460 |
| candidate_1 | command | 9/16 | 5 | 5812 | 19 | 419076 |
| candidate_2 | baseline | 7/16 | 7 | 5974 | 0 | 0 |
| candidate_2 | response | 7/16 | 4 | 5676 | 434 | 331086 |
| candidate_2 | command | 7/16 | 6 | 5871 | 28 | 534597 |
| parent_0 | baseline | 7/16 | 7 | 6116 | 0 | 0 |
| parent_0 | response | 7/16 | 4 | 6027 | 137 | 309981 |
| parent_0 | command | 7/16 | 6 | 6249 | 23 | 437619 |
| parent_1 | baseline | 8/16 | 6 | 5705 | 0 | 0 |
| parent_1 | response | 8/16 | 6 | 5797 | 284 | 327201 |
| parent_1 | command | 8/16 | 6 | 5705 | 2 | 367353 |
| parent_2 | baseline | 6/16 | 6 | 6132 | 0 | 0 |
| parent_2 | response | 6/16 | 5 | 6315 | 389 | 339717 |
| parent_2 | command | 6/16 | 6 | 6126 | 11 | 404712 |
| body_0 | baseline | 5/12 | 1 | 4625 | 0 | 0 |
| body_0 | response | 5/12 | 1 | 4621 | 8 | 91245 |
| body_0 | command | 5/12 | 1 | 4625 | 0 | 91581 |
| body_1 | baseline | 10/12 | 2 | 3524 | 0 | 0 |
| body_1 | response | 10/12 | 2 | 3524 | 0 | 68964 |
| body_1 | command | 10/12 | 2 | 3524 | 0 | 68964 |
| body_2 | baseline | 0/12 | 12 | 4889 | 0 | 0 |
| body_2 | response | 0/12 | 12 | 4891 | 848 | 830235 |
| body_2 | command | 0/12 | 12 | 4888 | 3 | 948612 |
| body_3 | baseline | 0/12 | 12 | 4933 | 0 | 0 |
| body_3 | response | 2/12 | 5 | 4915 | 597 | 432579 |
| body_3 | command | 0/12 | 12 | 4932 | 5 | 907452 |
| body_4 | baseline | 6/12 | 0 | 4523 | 0 | 0 |
| body_4 | response | 6/12 | 0 | 4523 | 0 | 87213 |
| body_4 | command | 6/12 | 0 | 4523 | 0 | 87213 |
| body_5 | baseline | 9/12 | 3 | 3802 | 0 | 0 |
| body_5 | response | 9/12 | 3 | 3800 | 19 | 79380 |
| body_5 | command | 9/12 | 3 | 3802 | 0 | 78540 |
| body_6 | baseline | 3/12 | 8 | 5560 | 0 | 0 |
| body_6 | response | 4/12 | 2 | 5325 | 246 | 201075 |
| body_6 | command | 4/12 | 3 | 5372 | 106 | 247044 |
| body_7 | baseline | 10/12 | 2 | 3617 | 0 | 0 |
| body_7 | response | 10/12 | 2 | 3592 | 34 | 92652 |
| body_7 | command | 10/12 | 2 | 3617 | 0 | 102291 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- response: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- command: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| response | 6 | candidate_0 | 6 | gain | True | False |
| response | 7 | analytical | 6 | gain | True | False |
| response | 7 | candidate_0 | 6 | gain | True | False |
| response | 12 | candidate_0 | 7 | gain | True | False |
| response | 13 | analytical | 7 | gain | True | False |
| command | 12 | candidate_0 | 7 | gain | True | False |
| command | 13 | analytical | 7 | gain | True | False |

These are prospective development results on new parameterizations using length/dynamics templates with newly assigned axis sequences. Their exact ordered axis sequences are excluded from training and known catalogs and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

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

Direct response-versus-command paired effects, by accepted body (checkpoint/placement outcomes are dependent within each body):

| Body | Family | Parameter seed | DOF | Response-only successes | Command-only successes |
|---|---|---:|---:|---:|---:|
| 0 | axis_dev87_6_ZYXYXY | 870100 | 6 | 0 | 0 |
| 1 | axis_dev87_6_ZYXYXY | 870101 | 6 | 0 | 0 |
| 2 | axis_dev87_6_ZYXYZY | 870102 | 6 | 0 | 0 |
| 3 | axis_dev87_6_ZYXYZY | 870103 | 6 | 2 | 0 |
| 4 | axis_dev87_7_ZYZYXYX | 870104 | 7 | 0 | 0 |
| 5 | axis_dev87_7_ZYZYXYX | 870105 | 7 | 0 | 0 |
| 6 | axis_dev87_7_ZYXZXZY | 870106 | 7 | 0 | 0 |
| 7 | axis_dev87_7_ZYXZXZY | 870107 | 7 | 0 | 0 |

Bodies with no accepted placement are absent from rollout comparisons but retained in screening. A pooled success difference is not evidence of consistent effects across independently sampled bodies; inspect concentration and checkpoint identity before strengthening the claim.

Axis-specific outcomes:

| Sequence | DOF | Estimator group | Mode | Success | Collisions |
|---|---:|---|---|---:|---:|
| ZYXYXY | 6 | learned | baseline | 15/24 | 3 |
| ZYXYXY | 6 | learned | response | 15/24 | 3 |
| ZYXYXY | 6 | learned | command | 15/24 | 3 |
| ZYXYXY | 6 | analytical | baseline | 2/4 | 1 |
| ZYXYXY | 6 | analytical | response | 2/4 | 1 |
| ZYXYXY | 6 | analytical | command | 2/4 | 1 |
| ZYXYZY | 6 | learned | baseline | 0/24 | 24 |
| ZYXYZY | 6 | learned | response | 2/24 | 17 |
| ZYXYZY | 6 | learned | command | 0/24 | 24 |
| ZYXYZY | 6 | analytical | baseline | 0/4 | 4 |
| ZYXYZY | 6 | analytical | response | 1/4 | 3 |
| ZYXYZY | 6 | analytical | command | 0/4 | 4 |
| ZYZYXYX | 7 | learned | baseline | 15/24 | 3 |
| ZYZYXYX | 7 | learned | response | 15/24 | 3 |
| ZYZYXYX | 7 | learned | command | 15/24 | 3 |
| ZYZYXYX | 7 | analytical | baseline | 3/4 | 0 |
| ZYZYXYX | 7 | analytical | response | 3/4 | 0 |
| ZYZYXYX | 7 | analytical | command | 3/4 | 0 |
| ZYXZXZY | 7 | learned | baseline | 13/24 | 10 |
| ZYXZXZY | 7 | learned | response | 14/24 | 4 |
| ZYXZXZY | 7 | learned | command | 14/24 | 5 |
| ZYXZXZY | 7 | analytical | baseline | 2/4 | 2 |
| ZYXZXZY | 7 | analytical | response | 3/4 | 0 |
| ZYXZXZY | 7 | analytical | command | 3/4 | 0 |

Outcome diagnosis (mutually exclusive, collision takes precedence among failures):

| Group | Mode | Success | Collision | No grasp | Grasp without lift | Placement/release failure after lift |
|---|---|---:|---:|---:|---:|---:|
| learned_6 | baseline | 15 | 27 | 1 | 5 | 0 |
| learned_6 | response | 17 | 20 | 3 | 8 | 0 |
| learned_6 | command | 15 | 27 | 1 | 5 | 0 |
| learned_7 | baseline | 28 | 13 | 4 | 2 | 1 |
| learned_7 | response | 29 | 7 | 4 | 2 | 6 |
| learned_7 | command | 29 | 8 | 4 | 2 | 5 |
| analytical | baseline | 7 | 7 | 1 | 1 | 0 |
| analytical | response | 9 | 4 | 1 | 1 | 1 |
| analytical | command | 8 | 5 | 1 | 1 | 1 |

Grasp and lift flags denote whether the event occurred at any point. Placement/release combines the remaining final object-position, stability and finger-contact checks; these records do not isolate which final check failed. Endpoint IK feasibility does not establish successful contact-rich execution.

Baseline phase diagnostics (unreached phases can overlap within a trial and are descriptive, not proven causes):

| Sequence | Estimator group | First contact phase counts | Unreached phase counts |
|---|---|---|---|
| ZYXYXY | learned | {"lower": 3} | {"close": 7, "descend": 7, "lift": 2, "lower": 7, "release": 7, "retreat": 1, "settle": 2, "transport": 3} |
| ZYXYXY | analytical | {"lower": 1} | {"close": 1, "descend": 1, "lower": 1, "release": 1} |
| ZYXYZY | learned | {"close": 3, "descend": 17, "lower": 4} | {"close": 18, "descend": 17, "lift": 2, "lower": 24, "release": 24, "retreat": 5, "settle": 5, "transport": 2} |
| ZYXYZY | analytical | {"close": 1, "descend": 3} | {"close": 3, "descend": 3, "lower": 3, "release": 3, "retreat": 1, "settle": 1} |
| ZYZYXYX | learned | {"descend": 3} | {"close": 6, "descend": 6, "lift": 2, "lower": 4, "release": 4, "retreat": 1, "settle": 1, "transport": 1} |
| ZYZYXYX | analytical | {} | {"close": 1, "descend": 1, "lower": 1, "release": 1} |
| ZYXZXZY | learned | {"descend": 4, "lift": 5, "transport": 1} | {"close": 5, "descend": 4, "lift": 5, "lower": 5, "release": 4, "retreat": 3, "settle": 3, "transport": 9} |
| ZYXZXZY | analytical | {"lift": 1, "retreat": 1} | {"lift": 1, "retreat": 1, "settle": 1, "transport": 1} |

The lineage audit verifies four training axis sequences and excludes 36 exact sequences from the checked training/development catalogs and built-in fixtures. Reserved geometry metadata is used only for exclusion; no reserved task goals or trajectories are evaluated. E30/E35 synthetic random linear-system supervision remains part of training, and E66 measured training contexts come from the eight audited training bodies. Exact sequence disjointness does not prove inequivalent kinematic functions or independence from prior development choices.

## Decision and limitations

The frozen response variant passes both advancement and attribution gates, but absolute task success remains low: 46/96 pooled learned trials, including 17/48 six-joint trials. Baseline achieves 43/96 and command correction 44/96. Response has three task gains and no losses against baseline. Its two gains over command are cases 6 and 7, both candidate_0 on body 3 (ZYXYZY), so the measured-response-specific success effect has not replicated across bodies or checkpoints in this study. The seven-joint gain in case 12 occurs in both correction modes.

Learned collision counts are 40 baseline, 27 response and 35 command. Avoiding a collision often leaves an unsuccessful task: response has 23 noncollision failures, compared with 13 baseline. These mutually exclusive outcome counts do not prove that correction caused new grasp failures; many baseline failures already collide before later task checks.

Response adds 1,883,343 geometry queries, for 2,026,782 total versus 153,357 baseline (13.22 times as many). It changes 1,752 actions. No real-time or efficiency claim is supported. Analytical control improves from 7/16 to 9/16 with response, versus 8/16 command; this also argues against attributing all benefit specifically to the Transformer.

Next: run a separately labeled privileged kinematic-control diagnostic on all 16 fixed cases with the same home states, calibration, task budgets and geometry filter, preserving all outcomes. Compare exact Jacobian substitution with measured-history estimates to distinguish motion-estimation limits from shared path/contact limitations. If exact kinematics still fails, inspect path feasibility and phase transitions before retraining or tuning the correction. This diagnostic is not a deployable policy or a final-test result.

Verification completed: 336 primary task evaluations, 112 independently instantiated exact baseline replays, unchanged checkpoint/input/source hashes and calibration probes, causal response updates, and 108 passing EASE tests. The report generator also passes Python syntax compilation. Artifacts: outputs/ease_direction_novelaxis_v1.

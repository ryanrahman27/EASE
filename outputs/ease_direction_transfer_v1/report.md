# E84: complete-task retention with frozen direction correction

All 17 separate E73 placements and seven estimators are evaluated in two modes (238 full pick-and-place tasks). Sustained correction is the exploratory primary candidate; no pulse comparison or settings retuning is performed. All phase transitions, budgets, collision rules and final success criteria are unchanged. Calibration is identical across modes; all 119 baseline traces, probes and filter logs reproduce exactly.

The correction implementation and numerical settings are frozen from E82. Selection uses causal measured-response predictions, whole-arm geometry and the learned Jacobian task-progress constraint; analytical comparison rows use their corresponding analytical estimator. Geometry and exact current simulator state remain privileged information, and object motion is not predicted. No simulation lookahead selects actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 93/102 | 9 | 29792 | 0 | 0 |
| learned | sustained | 95/102 | 7 | 29868 | 136 | 605010 |
| learned_6 | baseline | 46/48 | 2 | 13545 | 0 | 0 |
| learned_6 | sustained | 47/48 | 1 | 13544 | 1 | 252924 |
| learned_7 | baseline | 47/54 | 7 | 16247 | 0 | 0 |
| learned_7 | sustained | 48/54 | 6 | 16324 | 135 | 352086 |
| analytical | baseline | 16/17 | 1 | 4752 | 0 | 0 |
| analytical | sustained | 16/17 | 1 | 4752 | 4 | 89460 |
| candidate_0 | baseline | 14/17 | 3 | 5303 | 0 | 0 |
| candidate_0 | sustained | 15/17 | 2 | 5291 | 6 | 101955 |
| candidate_1 | baseline | 15/17 | 2 | 4940 | 0 | 0 |
| candidate_1 | sustained | 16/17 | 1 | 4899 | 44 | 105231 |
| candidate_2 | baseline | 17/17 | 0 | 4726 | 0 | 0 |
| candidate_2 | sustained | 17/17 | 0 | 4726 | 0 | 87822 |
| parent_0 | baseline | 15/17 | 2 | 5010 | 0 | 0 |
| parent_0 | sustained | 15/17 | 2 | 5060 | 48 | 112476 |
| parent_1 | baseline | 17/17 | 0 | 5030 | 0 | 0 |
| parent_1 | sustained | 17/17 | 0 | 5030 | 0 | 94206 |
| parent_2 | baseline | 15/17 | 2 | 4783 | 0 | 0 |
| parent_2 | sustained | 15/17 | 2 | 4862 | 38 | 103320 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- sustained: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| sustained | 1 | candidate_1 | 6 | gain | True | False |
| sustained | 6 | candidate_0 | 7 | gain | True | False |

These are pooled results across checkpoints and repeated historical development placements, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

Sustained correction remains available throughout. These placements differ from E83 but reuse the same robot bodies and were previously used in development; this is not independent final-test or new-embodiment validation. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline trace and filter-log equality, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

### Replication decision

The frozen sustained controller passes the retention gate: learned success increases from 93/102 to 95/102 and collisions fall from nine to seven. Six-joint success rises from 46/48 to 47/48 and seven-joint success from 47/54 to 48/54. Two casewise gains (case1/candidate1 and case6/candidate0) occur with no losses. The analytical comparison remains16/17. Gains are pooled across checkpoint seeds; neither a single policy success rate nor independent trials should be inferred.

Across E83 and E84, sustained correction gives166/198 learned successes versus161/198 for the geometry-filter baseline, with25 versus35 collisions and no casewise success losses. This aggregation describes repeated historical development experiments, not an independent statistical test. The two suites share robot bodies and both have been used previously for diagnostics. No new-embodiment generalization claim is established.

Cost remains material: E84 adds605,010 correction geometry queries for136 changed learned actions. Including original filtering, query count is685,192 versus80,912 (about8.47x). Control steps including calibration rise from29,792 to29,868. This is not an efficiency gain, and query count is not a wall-clock latency measurement.

Next freeze this implementation and define a new-body development evaluation before examining outcomes, preserving the reserved final test. Include a matched command-displacement/no-response ablation with identical candidate set, geometry access and progress constraint, plus analytical controls, to distinguish response-model benefit from additional geometry search. Evaluate all cases and seed-level outcomes, retain failures and cost measurements, and do not tune thresholds on the two recovered trajectories.

Verification: all119 baseline trajectories, filter logs and probes reproduce exactly; causal update counts and input/checkpoint/source hashes verified. All107 EASE-FM tests pass. No neural retraining or reserved final-test evaluation occurred.

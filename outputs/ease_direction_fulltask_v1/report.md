# E83: complete-task retention with frozen direction correction

All 16 original placements and seven estimators are evaluated in three modes (336 full pick-and-place tasks). Sustained correction is the exploratory primary candidate; one five-step pulse at the first online warning is secondary. All phase transitions, budgets, collision rules and final success criteria are unchanged. Calibration is identical across modes; all 112 baseline traces, probes and filter logs reproduce exactly.

The correction implementation and numerical settings are frozen from E82. Selection uses causal measured-response predictions, whole-arm geometry and the learned Jacobian task-progress constraint; analytical comparison rows use their corresponding analytical estimator. Geometry and exact current simulator state remain privileged information, and object motion is not predicted. No simulation lookahead selects actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 68/96 | 26 | 31622 | 0 | 0 |
| learned | sustained | 71/96 | 18 | 31079 | 797 | 1106931 |
| learned | pulse | 69/96 | 24 | 31315 | 91 | 455658 |
| learned_6 | baseline | 36/42 | 5 | 12440 | 0 | 0 |
| learned_6 | sustained | 37/42 | 2 | 12266 | 233 | 290094 |
| learned_6 | pulse | 37/42 | 3 | 12280 | 22 | 202230 |
| learned_7 | baseline | 32/54 | 21 | 19182 | 0 | 0 |
| learned_7 | sustained | 34/54 | 16 | 18813 | 564 | 816837 |
| learned_7 | pulse | 32/54 | 21 | 19035 | 69 | 253428 |
| analytical | baseline | 11/16 | 4 | 4930 | 0 | 0 |
| analytical | sustained | 11/16 | 4 | 4922 | 55 | 192570 |
| analytical | pulse | 11/16 | 4 | 5000 | 16 | 67410 |
| candidate_0 | baseline | 11/16 | 5 | 4978 | 0 | 0 |
| candidate_0 | sustained | 11/16 | 5 | 5031 | 76 | 162267 |
| candidate_0 | pulse | 11/16 | 5 | 4932 | 13 | 67977 |
| candidate_1 | baseline | 11/16 | 5 | 5309 | 0 | 0 |
| candidate_1 | sustained | 11/16 | 4 | 5340 | 206 | 239190 |
| candidate_1 | pulse | 11/16 | 5 | 5292 | 19 | 78120 |
| candidate_2 | baseline | 13/16 | 3 | 4806 | 0 | 0 |
| candidate_2 | sustained | 14/16 | 2 | 4731 | 30 | 179067 |
| candidate_2 | pulse | 13/16 | 3 | 4840 | 12 | 72891 |
| parent_0 | baseline | 12/16 | 3 | 5715 | 0 | 0 |
| parent_0 | sustained | 12/16 | 2 | 5533 | 159 | 193473 |
| parent_0 | pulse | 12/16 | 2 | 5680 | 7 | 84609 |
| parent_1 | baseline | 9/16 | 6 | 5856 | 0 | 0 |
| parent_1 | sustained | 9/16 | 3 | 5582 | 107 | 161490 |
| parent_1 | pulse | 9/16 | 6 | 5664 | 24 | 76461 |
| parent_2 | baseline | 12/16 | 4 | 4958 | 0 | 0 |
| parent_2 | sustained | 14/16 | 2 | 4862 | 219 | 171444 |
| parent_2 | pulse | 13/16 | 3 | 4907 | 16 | 75600 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- sustained: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- pulse: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| sustained | 4 | parent_2 | 6 | gain | True | False |
| sustained | 11 | parent_2 | 7 | gain | True | False |
| sustained | 11 | candidate_2 | 7 | gain | True | False |
| pulse | 4 | parent_2 | 6 | gain | True | False |

These are pooled results across checkpoints and repeated historical development placements, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

The pulse triggers once per trajectory and continues for five total steps even if a phase transition occurs; it never retriggers. Sustained correction remains available throughout. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline trace and filter-log equality, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

### Decision

**E83 completed:** Full-task sustained correction yields 71/96 learned successes versus baseline 68/96, with 18 versus 26 collisions. Pulse yields 69/96. Sustained has 3 casewise gains and 0 losses; six-joint success is 37/42. The primary retention gate passes. All 112 baselines reproduce exactly; 107 tests pass. These are historical development tasks, not independent generalization. See Section 4.73.

Freeze the sustained controller and replicate it on the separate E73 placement suite, retaining every previously successful six-joint case and reporting all regressions and query costs. Do not retune on this development gain.

All 107 EASE-FM tests pass. Local avoidance and development task gains do not establish superiority on new embodiments.

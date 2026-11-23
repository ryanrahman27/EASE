# E177: frozen-controller transfer to prospective unseen bodies

New body parameters were sampled after E176 was frozen. Twelve procedural six-joint bodies, three in each of four previously used axis families; four fixed placements per body. The cohort and all controller settings were frozen before initialization screening and before any prospective policy outcome. No replacement, training, tuning or best-checkpoint selection.

Initialization accepted 48/48 candidate tasks across 12/12 bodies. Only the starting posture was screened; no grasp, transport or policy-success screening. Rejections remain in screening.json and the all-candidate denominator.

| Estimator | Success / accepted | Success / all48 candidates | Collision episodes | Total control steps | Median steps |
|---|---:|---:|---:|---:|---:|
| e120_ensemble | 42/48 | 42/48 | 1 | 17275 | 326.0 |
| analytical | 42/48 | 42/48 | 0 | 16614 | 307.5 |

Paired outcomes: {'both_success': 39, 'E120_only': [9, 42, 45], 'analytical_only': [0, 8, 37], 'both_fail': 3}.

## What is frozen and matched

The AST-equivalence audit verifies unchanged geometry/projection helpers, ProtectedArm, phase targets, controller attachment and full rollout/scoring. The estimator evaluator is unchanged except removal of assertions against unrelated archived fitting cases. Both estimators exactly replayed five fitting cases (including successes and failures):10 compatibility episodes before prospective evaluation. E120 is the same reference/correction three-model ensemble with its bounded residual; analytical is the original regularized history solve. Neither is retrained.

Both methods have identical body/placement/home/calibration, frozen calibrated bias,32 calibration steps and32 local probe steps. They share E175/E176 recovery gates, strict grasp-first planner, table/link/gripper and probe protections, causal clearance allowance, grasp guard and265mm release. Both have the same conditional step caps:944 total including calibration with an accepted route,744 otherwise. Descent is100 steps or300 after an accepted route; all other caps and early exits are unchanged. Actual steps differ and are reported above. Planning time/computation is not equalized. Exact geometry, object pose, contact information and IK initialization remain privileged.

The independent audit reconstructs all histories from calibration and saved state transitions, verifies both estimator statistics and frozen bias, reconstructs analytical predictions/actions, checks executed commands and safety constraints, verifies every saved post-state geometry, and audits static paths, grasp guards and budgets. All raw trials, states, actions, histories and planning outcomes are saved.

## Per-body results

| Body seed | Axis family | Accepted tasks | E120 successes | Analytical successes | E120 / analytical collisions |
|---|---|---:|---:|---:|---|
| 1770100 | six_amb_train_0 | 4 | 3 | 4 | 0 / 0 |
| 1770101 | six_amb_train_0 | 4 | 3 | 3 | 0 / 0 |
| 1770102 | six_amb_train_0 | 4 | 3 | 3 | 1 / 0 |
| 1770103 | six_amb_train_1 | 4 | 4 | 4 | 0 / 0 |
| 1770104 | six_amb_train_1 | 4 | 4 | 4 | 0 / 0 |
| 1770105 | six_amb_train_1 | 4 | 4 | 4 | 0 / 0 |
| 1770106 | axis_dev87_6_ZYXYXY | 4 | 4 | 4 | 0 / 0 |
| 1770107 | axis_dev87_6_ZYXYXY | 4 | 4 | 4 | 0 / 0 |
| 1770108 | axis_dev87_6_ZYXYXY | 4 | 4 | 4 | 0 / 0 |
| 1770109 | axis_dev87_6_ZYXYZY | 4 | 2 | 3 | 0 / 0 |
| 1770110 | axis_dev87_6_ZYXYZY | 4 | 4 | 3 | 0 / 0 |
| 1770111 | axis_dev87_6_ZYXYZY | 4 | 3 | 2 | 0 / 0 |

Descriptive paired body-mean success-rate difference (E120 minus analytical): 0.00 percentage points; body-bootstrap percentile95 interval [-8.33, 8.33] points (10,000 resamples, seed1779000). Only12 body clusters and four related axis families; this is not48 independent embodiment trials or a definitive significance claim.

## Layout groups

| Previously used layouts | E120 successes | Analytical successes | E120 / analytical collisions |
|---|---|---|---|
| familiar_axis_layouts | 21/24 | 22/24 | 1 / 0 |
| prior_development_axis_layouts | 21/24 | 20/24 | 0 / 0 |

## Discordant task outcomes

| Case | Body | E120 success | Analytical success | E120 collision | Analytical collision |
|---|---|---|---|---|---|
| 0 | 1770100 | False | True | False | False |
| 8 | 1770102 | False | True | True | False |
| 9 | 1770102 | True | False | False | False |
| 37 | 1770109 | False | True | False | False |
| 42 | 1770110 | True | False | False | False |
| 45 | 1770111 | True | False | False | False |

## Scope and interpretation

These body parameters and policy outcomes are prospective and absent from pre-existing case manifests. The four axis layouts were already used in training/development: two familiar fitting layouts and two prior development layouts. This tests transfer to newly sampled kinematics/dynamics within those layouts, not untouched axis-family generalization, commercial-arm transfer, hardware, or a learned-only policy. The inherited controller was developed around E120; equal frozen rules do not remove that development bias. No test-based adjustments are made. Legacy reserved cohorts remain untouched.

A performance drop is retained as evidence about transfer of this entire privileged control stack. We do not repair failures within this cohort and then call it an untouched test. Task successes require the original grasp/lift/place/no-collision score; static feasibility or reaching a hand waypoint is not success. Videos show both estimators for the first case of each body and every outcome-discordant case, with additional categories if necessary. All cases retain full raw artifacts regardless of video selection.

## Failure categories

Contact episodes are classified by their first forbidden contact, including calibration. Non-contact failures are classified by guard abort, missing grasp, missing lift, or final placement/release. These categories describe outcomes; they do not identify a unique causal estimator defect.

| Estimator | Failure categories | First collision phases |
|---|---|---|
| e120_ensemble | {'no_lift': 2, 'placement_or_release': 2, 'collision': 1, 'guard_abort': 1} | {'close': 1} |
| analytical | {'no_lift': 3, 'no_grasp': 2, 'guard_abort': 1} | {} |

Detailed contacts, grasp guards, recovery triggers and failed phases are in failure_categories.json. Initial calibration is identical between estimators and retains the original E176 behavior. This prospective cohort will not be retuned and relabeled as untouched.

Video coverage was extended after evaluation to include every failed case for both estimators, including shared failures. This changes illustrations only; all raw outcomes and comparison denominators are unchanged. See video_coverage.json.

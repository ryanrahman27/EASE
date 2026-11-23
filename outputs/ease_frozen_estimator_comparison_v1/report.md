# E176: E120 versus analytical estimation under the frozen E175 controller

96 paired fitting episodes,48 per method;5 preflight reference replays. E120 also exactly reproduces all48 complete E175 state sequences and traces. No training, controller tuning or held evaluation.

| Estimator | Success /48 | Collision episodes | Total control steps | Median steps |
|---|---:|---:|---:|---:|
| e120_ensemble | 48 | 0 | 16106 | 309.5 |
| analytical | 46 | 1 | 16525 | 298.5 |

Paired outcomes: {'both_success': 46, 'E120_only': [26, 64], 'analytical_only': [], 'both_fail': 0}.

## What is matched

Both methods receive the same initial state and32-step calibration, frozen measured bias,32-step local probe, state measurements and privileged geometry/object/contact inputs. Recovery triggers, grasp-first planner, tracking, protections, guarded handoff,265mm release, phase targets, action bounds, early-exit rules and success scoring are unchanged from E175. The motion statistics update with forgetting.98; at the local probe both rebuild from the last32 measured transitions. The independent audit reconstructs these statistics from saved physical states for every executed step.

E120 is the established three-model reference/correction ensemble with its bounded action residual. Analytical uses the original regularized solve(gram,cross).T, with no neural prediction or residual. This compares the existing estimator recipes; it does not isolate ensembling, training, or the bounded residual separately. The safety-QP metric uses each estimator prediction under the same formula. Online histories and recovery activation can diverge because actions differ.

## Interaction budget

Maximum total steps are944 with an accepted recovery route and744 otherwise, including32 calibration steps. Both receive the same conditional allowance: descent100, extended to300 only after a successful plan; all other phase caps match. Each performs one task attempt and32 probe steps. We do not force equal realized steps or pad completed phases. Actual steps above include calibration and all task phases. Planning wall-clock time and computation are not equalized or benchmarked.

## Paired diagnostic details

| Case | E120 success | Analytical success | E120 collision | Analytical collision | E120 / analytical steps |
|---|---|---|---|---|---|
| 0 | True | True | False | False | 290 / 281 |
| 26 | True | False | False | False | 406 / 848 |
| 43 | True | True | False | False | 524 / 494 |
| 62 | True | True | False | False | 527 / 520 |
| 64 | True | False | False | True | 319 / 404 |
| 65 | True | True | False | False | 506 / 496 |
| 67 | True | True | False | False | 515 / 406 |

## Failure and recovery details

| Case / estimator | Trigger | Descent observation | Accepted route | Close ready | Final two-pad close steps | Cube descent displacement (mm) | Abort |
|---|---|---:|---|---|---:|---:|---|
| 0 / e120_ensemble | None | None | False | False | 40 | 0.000 | False |
| 0 / analytical | None | None | False | False | 40 | 0.000 | False |
| 26 / e120_ensemble | early_object_tilt | 13 | True | True | 40 | 0.000 | False |
| 26 / analytical | late_stall | 73 | True | False | 0 | 19.378 | True |
| 43 / e120_ensemble | early_object_tilt | 86 | True | True | 40 | 0.000 | False |
| 43 / analytical | late_stall | 69 | True | True | 40 | 12.005 | False |
| 62 / e120_ensemble | late_stall | 70 | True | True | 40 | 0.000 | False |
| 62 / analytical | late_stall | 77 | True | True | 40 | 0.000 | False |
| 64 / e120_ensemble | None | None | False | False | 40 | 0.000 | False |
| 64 / analytical | None | None | False | False | 40 | 0.000 | False |
| 65 / e120_ensemble | late_stall | 64 | True | True | 40 | 0.000 | False |
| 65 / analytical | late_stall | 60 | True | True | 40 | 2.558 | False |
| 67 / e120_ensemble | late_stall | 61 | True | True | 40 | 0.000 | False |
| 67 / analytical | late_stall | 64 | False | False | 40 | 19.670 | False |

Close-ready is only evaluated after an accepted route; false on an ordinary successful episode does not imply a rejected grasp. These are trajectory-level observations, not matched-history causal estimates.

## Scope and interpretation

These are48 six-joint fitting placements across two training families (24 each), used during controller development, not48 untouched embodiments or independent random seeds. E175 was developed around E120, so a difference here may reflect compatibility with that frozen controller. A tie establishes no learned task-success advantage. Any gain is a descriptive fitting result under shared privilege and must be tested prospectively. Full state/action/history traces, failures and independent audits are retained. Historical E174 case26 failure video is context only and excluded from E176 metrics.

The inherited controller applies established grasp-first task-and-motion planning and reactive grasp guards; E176 adds no new literature method or control feature. See the previous experiment’s literature_application.md.

## Frozen-comparison failure review

No controller or estimator settings were changed after observing these outcomes. Distances below use the same conservative geometry measure as the safety controller; the strict physical collision score is recorded separately.

### Case26

First executed-action difference from the same calibrated state: 0.0223938 rad (maximum joint difference). Later histories differ, so downstream observations do not isolate a single estimator error.

e120_ensemble: success=True, collision=False; recovery=early_object_tilt at descent observation13; aborted=False.
Close-entry measurements: XY gap=1.460mm (limit8), Z gap=-0.845mm (absolute limit12), axis error=0.0025 (limit.08), object drift=0.000mm (limit8); ready=True.

analytical: success=False, collision=False; recovery=late_stall at descent observation73; aborted=True.
Close-entry measurements: XY gap=1.528mm (limit8), Z gap=-0.979mm (absolute limit12), axis error=0.0030 (limit.08), object drift=13.502mm (limit8); ready=False.

### Case64

First executed-action difference from the same calibrated state: 0 rad (maximum joint difference). Later histories differ, so downstream observations do not isolate a single estimator error.

e120_ensemble: success=True, collision=False; recovery=None at descent observationNone; aborted=False.

analytical: success=False, collision=True; recovery=None at descent observationNone; aborted=False.
First forbidden contact: step120, phaselift, pairs=[['gripper_right_pad', 'table']]. Safety projection eligible=True, solver success=True, fallback=False. Full pre/predicted/post/substep clearance vectors are in failure_review.json (six links, palm, left pad, right pad). Passing a local linear constraint is not a guarantee of collision-free physical tracking.

## Family breakdown

| Training family | E120 successes | Analytical successes | E120 / analytical collisions |
|---|---:|---:|---|
| six_amb_train_0 | 24/24 | 22/24 | 0 / 1 |
| six_amb_train_1 | 24/24 | 24/24 | 0 / 0 |

E120 uses fewer total control steps in this batch, while analytical has the lower median. Totals include the long aborted/failing trajectories. This supports a fitting-case reliability difference, not a general runtime or sample-efficiency claim. Planner wall-clock timings are excluded from deterministic plan equality. All path values, executed states and E120 traces remain checked exactly.

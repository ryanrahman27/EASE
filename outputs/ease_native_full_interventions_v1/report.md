# E65: continuous full-schedule interventions

All twelve E63 development cases, including successes. Ten estimators (analytical plus nine frozen learned models), crossed with measured additive bias versus zero additive bias for all 1700 steps after shared 32-step calibration. All geometry contacts disabled. Same fixed references, gains, thresholds and evolving history statistics. Analytical substitution from calibration is independent of the unused learned checkpoint, so run it once per case/bias condition rather than counting nine identical copies. All 120 measured-bias trials must exactly replay E63. No tuning, training, phase-dependent switching or early stopping.

All nine phases must hold XYZ error <.006 m and direction norm <.08 for their final eight steps. Report paired gains/losses, robot and seed breakdown. Diagnostic unconstrained tracking, not manipulation or independent embodiment validation.

| Estimator group | Robot | Measured bias | Zero bias | Paired gains | Paired losses |
|---|---|---:|---:|---:|---:|
| analytical | piper | 6/6 | 6/6 | 0 | 0 |
| analytical | panda | 6/6 | 2/6 | 0 | 4 |
| original | piper | 18/18 | 18/18 | 0 | 0 |
| original | panda | 15/18 | 7/18 | 0 | 8 |
| uncorrected | piper | 18/18 | 17/18 | 0 | 1 |
| uncorrected | panda | 14/18 | 5/18 | 0 | 9 |
| corrected | piper | 16/18 | 17/18 | 2 | 1 |
| corrected | panda | 12/18 | 7/18 | 3 | 8 |

## Per-seed complete schedules

| Method | Measured PiPER /6 | Zero PiPER /6 | Measured Panda /6 | Zero Panda /6 |
|---|---:|---:|---:|---:|
| analytical | 6 | 6 | 6 | 2 |
| original_0 | 6 | 6 | 6 | 4 |
| original_1 | 6 | 6 | 3 | 0 |
| original_2 | 6 | 6 | 6 | 3 |
| uncorrected_0 | 6 | 6 | 6 | 3 |
| uncorrected_1 | 6 | 6 | 5 | 2 |
| uncorrected_2 | 6 | 5 | 3 | 0 |
| corrected_0 | 4 | 6 | 4 | 2 |
| corrected_1 | 6 | 5 | 5 | 2 |
| corrected_2 | 6 | 6 | 3 | 3 |

All 240 trial files and phase summaries audited; 120 complete measured-bias replays match E63 exactly. 408,000 control steps, 384 shared physical calibration steps, 32-step calibration cost per trial. Nine checkpoints unchanged. Analytical rows are deduplicated controls, not nine independent seed replicas. No model promotion, training, final-test evaluation or manipulation-success claim.

## Paired phase holds

| Group | Robot | Measured holds | Zero-bias holds | Gains | Losses |
|---|---|---:|---:|---:|---:|
| analytical | piper | 54/54 | 54/54 | 0 | 0 |
| analytical | panda | 54/54 | 45/54 | 0 | 9 |
| original | piper | 162/162 | 162/162 | 0 | 0 |
| original | panda | 155/162 | 126/162 | 1 | 30 |
| uncorrected | piper | 162/162 | 161/162 | 0 | 1 |
| uncorrected | panda | 153/162 | 129/162 | 7 | 31 |
| corrected | piper | 160/162 | 160/162 | 2 | 2 |
| corrected | panda | 143/162 | 140/162 | 17 | 20 |

## Interpretation and decision

Bias removal reduces complete-schedule success in every estimator group: analytical 12/12 to 8/12, E35 33/36 to 25/36, E61 uncorrected 32/36 to 22/36, and corrected 28/36 to 24/36. Across the 108 learned-model cases there are five paired gains and 27 losses. This rejects blanket removal of measured additive bias as a controller improvement on these development schedules.

The two corrected-seed-0 PiPER failures from E64 both recover over complete schedules. However, corrected seed 1 loses another PiPER case, and uncorrected seed 2 also loses a PiPER case. Corrected Panda has three gains and eight losses. E35 retains all 18/18 PiPER successes, but Panda falls from 15/18 to 7/18. The local phase recoveries therefore do not establish a generally better deployed controller.

With bias retained, analytical estimation from calibration passes all twelve schedules and exactly reproduces E63. This is the continuous analytical replacement control; it is independent of the discarded learned checkpoint and is counted once per case, not as nine independent replicas. Compared with E64's late substitutions, the result is consistent with trajectory/history accumulation contributing to unrecovered late failures, but does not isolate that mechanism.

Keep the E35 checkpoints and measured bias. Prioritize improving the learned motion estimator's closed-loop behavior while retaining bias correction. A subsequent training comparison should use procedural data and fixed three-seed budgets, pass the original six-/seven-joint retention gates, and then face the complete native suite. These observed native cases remain diagnostics, not training targets or independent confirmation. No selective per-robot/seed bias switch is adopted.

These results measure contact-disabled tool tracking through scripted waypoints. They do not demonstrate reliable grasping, pick-and-place, novel-embodiment generalization or a publishable learned-method advantage. The reserved final test remains untouched.

## Verification

All 240 saved trials and their nine phase summaries were audited. All 120 original-condition traces match E63 exactly, covering 204,000 replayed control steps. Total execution is 408,000 control steps plus 384 physical calibration steps cloned into 240 starts; each trial is charged 32 calibration steps. All nine checkpoint hashes are unchanged. The EASE-FM regression suite passes 90 tests. The comparison figure was visually inspected. No broader-suite pass is claimed.

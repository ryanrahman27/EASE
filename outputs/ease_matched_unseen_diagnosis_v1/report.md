# E178: why E120 loses cases0,8,37

Retrospective matched-history diagnosis of the E177 cohort. Cases9,42,45, where E120 won, are included as controls. All12 original trajectories were exactly replayed with passive full integration-state logging. No weights, controller settings, task budgets or scoring were changed.

4620 pre-action histories from both source policies were analyzed. At each state, both estimators receive identical full-precision history statistics, calibrated bias, current state, goal and controller context. Six action candidates are evaluated for one40ms step in independent restored simulators. The actual source candidate reproduces every saved post-state to1e-12 and every contact union.

## Matched prediction and action results

Primary summary: states clean before the action in approach, descent, lift, transport, lower or retreat. Prediction error is vector RMSE of predicted position change J*dq against the same observed next position change, using measured joint change dq. Command-to-motion error is separate in summary.json. Post-contact states, probe and grasp phases remain in the raw/per-phase results.

| Group / history source | States | Reference J RMSE (mm) | Correction J RMSE (mm) | Analytical RMSE (mm) | Local true J RMSE (mm) | E120 minus analytical one-step progress (mm) |
|---|---:|---:|---:|---:|---:|---:|
| E120_loss / e120_ensemble | 772 | 1.0375 | 1.0304 | 0.4794 | 0.0432 | -0.0967 |
| E120_loss / analytical | 721 | 1.5055 | 1.4909 | 1.0304 | 0.0466 | 0.5832 |
| E120_win_control / e120_ensemble | 583 | 1.4951 | 1.4724 | 0.7419 | 0.0749 | -0.1820 |
| E120_win_control / analytical | 618 | 1.3113 | 1.2995 | 0.7735 | 0.0640 | -0.0773 |

E120 has no single blended Jacobian: its reference Jacobian supplies the main DLS action and projection metric, while the correction Jacobian contributes a bounded action residual. The two neural prediction columns above are component diagnostics, not two separately deployed policies. The local true Jacobian is an offline privileged derivative; finite-step curvature can give it nonzero error.

## Action attribution

The full E120/analytical swap changes both the nominal action and the metric used by the unchanged safety projection. Crossed candidates separate these on each fixed state: analytical metric held fixed while changing the nominal target, then E120 nominal held fixed while changing the metric. These two progress differences sum exactly to the total one-step difference; the reverse order is also saved to expose interactions. The no-correction candidate removes only the bounded E120 residual.

| Case / history | Reference prediction RMSE | Analytical prediction RMSE | Total E120 progress difference | Nominal-target effect | Metric effect | Bounded-correction effect |
|---|---:|---:|---:|---:|---:|---:|
| 0 / e120_ensemble | 1.4480 | 0.6981 | -0.5627 | 0.7040 | -1.2667 | 0.0172 |
| 0 / analytical | 1.6983 | 1.1073 | 0.2856 | 0.3951 | -0.1095 | 0.0003 |
| 8 / e120_ensemble | 0.9701 | 0.4529 | -1.2552 | -0.4546 | -0.8006 | -0.0198 |
| 8 / analytical | 1.7160 | 1.2874 | 0.2004 | 0.3494 | -0.1490 | -0.0013 |
| 37 / e120_ensemble | 0.7669 | 0.3227 | 0.4497 | -0.7409 | 1.1906 | 0.0216 |
| 37 / analytical | 0.9944 | 0.4357 | 1.2705 | 0.8102 | 0.4603 | 0.0080 |
| 9 / e120_ensemble | 1.6658 | 0.9753 | -0.5398 | 0.3609 | -0.9007 | -0.0254 |
| 9 / analytical | 1.1552 | 0.6275 | -0.1178 | 1.2934 | -1.4112 | 0.0117 |
| 42 / e120_ensemble | 1.4697 | 0.6730 | -0.1722 | -0.2520 | 0.0798 | 0.0115 |
| 42 / analytical | 1.6790 | 1.1492 | 0.0085 | -0.1494 | 0.1579 | 0.0158 |
| 45 / e120_ensemble | 1.3495 | 0.5342 | 0.1336 | 0.3728 | -0.2392 | 0.0220 |
| 45 / analytical | 1.0365 | 0.3297 | -0.1136 | 0.6406 | -0.7542 | -0.0175 |

All prediction/progress quantities above are mm; progress differences are mean per step, not terminal task effects. Distance to a privileged local-Jacobian action (saved separately) is a diagnostic proxy, not an optimal-action ground truth. The original route, handoff/abort decisions, phase, aperture, clearance gradient and error allowance are held fixed for each source context. A different full policy could have arrived with a different context.

## Failure chronology

| Case / source | Success | Grasped | Lifted | Collision | First forbidden contact | First unreached motion phase |
|---|---|---|---|---|---|---|
| 0 / e120_ensemble | False | True | False | False | None | descend |
| 8 / e120_ensemble | False | True | False | True | {'step': 179, 'phase': 'close', 'pairs': [['gripper_right_pad', 'table']]} | descend |
| 37 / e120_ensemble | False | True | True | False | None | lift |
| 9 / e120_ensemble | True | True | True | False | None | None |
| 42 / e120_ensemble | True | True | True | False | None | descend |
| 45 / e120_ensemble | True | True | True | False | None | descend |
| 0 / analytical | True | True | True | False | None | None |
| 8 / analytical | True | True | True | False | None | None |
| 37 / analytical | True | True | True | False | None | descend |
| 9 / analytical | False | False | False | False | None | descend |
| 42 / analytical | False | True | False | False | None | descend |
| 45 / analytical | False | True | False | False | None | descend |

Full phase endpoints, object positions, pad contacts, guarded-handoff states and first contacts are in chronology.json. divergences.json retains the first actual action divergence and prespecified phase-boundary/contact checkpoints with matched model and action diagnostics. per_phase.json separates early prediction/action errors from later failed-trajectory aftermath.

## Interpretation limits

This is retrospective analysis of six outcome-selected cases, not a new held evaluation or a population-level estimator comparison. One-step counterfactuals can identify immediate prediction/action differences, but cannot prove a complete-task rescue or uniquely attribute accumulated failures to one component. Better immediate goal progress may still harm grasp/contact behavior. Collision and object-displacement outcomes are recorded alongside progress. No candidate is deployed, no source outcome is relabeled and no tuning follows this analysis. The full E177 cohort remains42/48 for each estimator, with1 versus0 collisions.

## Case37: posthoc matched lift continuations

The one-step averages did not explain case37, so20-step continuations were added retrospectively from lift entry and40 steps later on both source histories. All candidates share the full starting state and exact32-transition history. The lift goal, aperture, bias, update rule and safety constraints are unchanged. These are phase-local diagnostic branches, not full task evaluations or controller improvements. Four source continuations reproduce all80 saved states to1e-12.

| Source / start | Candidate | Initial error (mm) | Final error (mm) | Final axis error | Collision | Lift phase reached |
|---|---|---:|---:|---:|---|---|
| e120_ensemble / 126 | e120 | 110.48 | 49.27 | 0.112 | False | False |
| e120_ensemble / 126 | analytical | 110.48 | 123.25 | 0.278 | False | False |
| e120_ensemble / 126 | e120_no_correction | 110.48 | 55.97 | 0.130 | False | False |
| e120_ensemble / 126 | e120_analytical_metric | 110.48 | 87.68 | 0.170 | False | False |
| e120_ensemble / 126 | analytical_e120_metric | 110.48 | 81.44 | 0.132 | False | False |
| e120_ensemble / 166 | e120 | 16.37 | 27.68 | 0.112 | False | False |
| e120_ensemble / 166 | analytical | 16.37 | 95.92 | 0.253 | False | False |
| e120_ensemble / 166 | e120_no_correction | 16.37 | 27.17 | 0.115 | False | False |
| e120_ensemble / 166 | e120_analytical_metric | 16.37 | 30.48 | 0.106 | False | False |
| e120_ensemble / 166 | analytical_e120_metric | 16.37 | 54.59 | 0.028 | False | False |
| analytical / 185 | e120 | 101.39 | 75.29 | 0.023 | False | False |
| analytical / 185 | analytical | 101.39 | 77.97 | 0.187 | False | False |
| analytical / 185 | e120_no_correction | 101.39 | 74.95 | 0.023 | False | False |
| analytical / 185 | e120_analytical_metric | 101.39 | 82.21 | 0.023 | False | False |
| analytical / 185 | analytical_e120_metric | 101.39 | 84.44 | 0.251 | False | False |
| analytical / 225 | e120 | 55.26 | 13.85 | 0.014 | False | False |
| analytical / 225 | analytical | 55.26 | 29.35 | 0.105 | False | False |
| analytical / 225 | e120_no_correction | 55.26 | 11.43 | 0.005 | False | False |
| analytical / 225 | e120_analytical_metric | 55.26 | 30.93 | 0.157 | False | False |
| analytical / 225 | analytical_e120_metric | 55.26 | 16.08 | 0.082 | False | False |

## Case37: full original lift-budget follow-up

Posthoc phase-local comparison with a100-step maximum and the original5-step reach rule. All candidates share the full starting state and exact32-transition history. The lift goal, aperture, bias, update rule and safety constraints are unchanged. These are not full task evaluations or controller improvements. 2 source continuations reproduce all173 saved states to1e-12.

| Source / start | Candidate | Initial error (mm) | Final error (mm) | Final axis error | Collision | Lift phase reached |
|---|---|---:|---:|---:|---|---|
| e120_ensemble / 126 | e120 | 110.48 | 45.18 | 0.112 | False | False |
| e120_ensemble / 126 | analytical | 110.48 | 4.06 | 0.020 | False | True |
| e120_ensemble / 126 | e120_no_correction | 110.48 | 95.33 | 0.268 | False | False |
| analytical / 185 | e120 | 101.39 | 24.21 | 0.192 | False | False |
| analytical / 185 | analytical | 101.39 | 4.47 | 0.006 | False | True |
| analytical / 185 | e120_no_correction | 101.39 | 63.15 | 0.121 | False | False |

## Detailed interpretation

See [interpretation.md](interpretation.md) for the matched-history findings, full-budget lift result, and next fitting-only experiment.

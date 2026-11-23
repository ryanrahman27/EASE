# E74: first-collision counterfactual audit

All ten unsuccessful E73 geometry-filter trajectories are replayed exactly: nine learned-controller failures and one analytical failure. Their first forbidden contacts are finger/table contacts in seven descend and three lower phases. The paired E73 unfiltered trajectories fail on the same cases and first-contact control indices. This is outcome-conditioned diagnosis, not a new success-rate evaluation.

| Counterfactual at the pre-contact state | Immediate contact avoided /10 |
|---|---:|
| selected | 0 |
| half_selected | 0 |
| bias_only | 0 |
| analytical_same_history | 5 |
| selected_zero_incoming_velocity | 2 |

| Case | Estimator | Phase | 21-sample minimum (mm) | Actual substep minimum (mm) | Analytical action avoids | Zero-velocity avoids |
|---|---|---|---:|---:|---|---|
| 1 | candidate_0 | descend | -0.042 | -0.476 | False | False |
| 1 | candidate_1 | descend | 2.000 | -0.527 | False | False |
| 6 | analytical | lower | 2.000 | -1.013 | False | False |
| 6 | candidate_0 | lower | 2.000 | -0.271 | True | True |
| 8 | parent_0 | descend | 2.000 | -0.596 | False | False |
| 8 | parent_2 | descend | 2.000 | -1.532 | True | False |
| 8 | candidate_0 | lower | 2.000 | -1.158 | True | False |
| 8 | candidate_1 | descend | 2.000 | -0.974 | True | False |
| 14 | parent_2 | descend | 2.000 | -1.545 | False | False |
| 15 | parent_0 | descend | 2.000 | -0.207 | True | True |

### Interpretation

All ten filtered actions were accepted by the original geometric test. Nine remain at the 2 mm clearance cap even when the commanded joint/finger interpolation is sampled at 21 points, including its start. Reconstructed actual dynamic substep configurations penetrate forbidden geometry in all ten. Thus simply adding interpolation samples does not resolve these nine misses: the commanded straight joint path is not the executed dynamic trajectory. One event already has about 0.042 mm penetration at the start of the diagnosed step; the original contact flag first records it during that step. This contact-timing distinction is retained.

Half motion and bias-only avoid zero immediate collisions. Changing only the estimator used to compute the action—from the same measured Gram/cross/bias and task error—avoids five. This is a controlled local action comparison and implicates the proposed motion direction, but it does not establish full-task recovery or uniquely identify neural-estimation error as the cause. The analytical controller's independent whole-task rollout succeeds on eight of the ten corresponding event cases, with repeated case identities; it is a different trajectory.

Resetting incoming arm/finger velocities avoids two immediate contacts, so incoming velocity matters in those states. This privileged reset also recomputes constraints and is not a realizable braking action. Its limited benefit does not exclude acceleration, unequal joint responses, tracking lag or contact dynamics in the other failures. The analytical-action and zero-velocity successes overlap; they are not additive recoveries.

### Decision

Retain the original controller and keep the static geometry filter unpromoted. Do not perform another geometry-margin or interpolation-density sweep. The next useful comparison should account for the predicted executed joint response and current velocity, while retaining the same-history analytical action as a comparator. Any simulator-dynamics lookahead must be labeled a privileged model-based diagnostic and separated from claims about history-only adaptation. Before claiming a useful fix, require complete-task evaluation and preservation of successful six-joint cases.

### Verification and limitations

Ten full trajectories, filter logs and calibration probes replay exactly; all ten selected-action branches reproduce the original contact-step tool position. Fifty branches run for one 40 ms control step each, with original state/velocity/time restored after each. Checkpoint, input and source hashes are unchanged. All 104 EASE-FM tests pass, including branch state/flag restoration. No weights, phase budgets, collision masks or task success criteria changed.

All chosen states precede an observed failure, so branch counts cannot estimate deployment success. There is no successful-state intervention control or multi-step recovery evaluation here. Actual-path clearance reconstruction uses the current object pose rather than updating it; the first forbidden pair in every event is finger/table, which does not involve the object. Discrete geometry samples do not guarantee continuous clearance. No training, native tuning or reserved final-test evaluation occurred.

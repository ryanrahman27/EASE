# E82: response-based direction corrections

The first causal no-velocity response warning at the unchanged 2 mm threshold triggers each local experiment, regardless of eventual outcome. No warning is selected by counting backward from a known collision. Warning values come from the previously exact-replayed E78 run. Collisions without advance warning remain in coverage counts.

Each triggering state has three matched 20-step branches: baseline, direction correction for five steps then baseline, and correction for all 20 steps. Candidate actions change one arm joint by +/-0.01 rad, clipped to +/-0.04. Frozen no-velocity response predictions score 21-point whole-arm clearance paths. A correction requires >1e-6 clearance improvement and predicted task error no more than 0.001 above baseline, using the learned Jacobian. The fixed 2 mm warning threshold, candidate set and progress bound are not tuned. Geometry is privileged; object pose stays fixed during path prediction. No future simulator outcomes select actions.

Controller and response history update independently from completed measurements in each branch. The triggering phase goal and aperture remain fixed. Both pulse and sustained variants are reported; neither is selected retrospectively as a validated controller.

| DOF | Original collisions | Advance warning | Warning only on collision step | No warning | Successful warning trajectories |
|---|---:|---:|---:|---:|---:|
| all | 26 | 14 | 11 | 1 | 2 |
| 6 | 5 | 3 | 2 | 0 | 2 |
| 7 | 21 | 11 | 9 | 1 | 0 |

| Group | Horizon (steps) | Action | Contacts / branches | Avoided vs matched | New vs matched | Mean task-error change (mm-equivalent) |
|---|---:|---|---:|---:|---:|---:|
| original_collisions | 5 | selected | 20/25 | 0 | 0 | 0.0000 |
| original_collisions | 5 | direction_pulse | 14/25 | 6 | 0 | -0.3054 |
| original_collisions | 5 | direction_all | 14/25 | 6 | 0 | -0.3054 |
| original_collisions | 20 | selected | 24/25 | 0 | 0 | 0.0000 |
| original_collisions | 20 | direction_pulse | 17/25 | 7 | 0 | -0.8317 |
| original_collisions | 20 | direction_all | 15/25 | 9 | 0 | -1.4137 |
| original_collisions_6 | 5 | selected | 3/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 5 | direction_pulse | 2/5 | 1 | 0 | -0.2485 |
| original_collisions_6 | 5 | direction_all | 2/5 | 1 | 0 | -0.2485 |
| original_collisions_6 | 20 | selected | 5/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 20 | direction_pulse | 2/5 | 3 | 0 | -3.0751 |
| original_collisions_6 | 20 | direction_all | 2/5 | 3 | 0 | -3.3547 |
| successful_controls | 5 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | direction_pulse | 0/2 | 0 | 0 | 0.1963 |
| successful_controls | 5 | direction_all | 0/2 | 0 | 0 | 0.1963 |
| successful_controls | 20 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | direction_pulse | 0/2 | 0 | 0 | 0.2659 |
| successful_controls | 20 | direction_all | 0/2 | 0 | 0 | 0.2659 |

Task error uses sqrt(position_error^2 + (0.15 * tool_direction_error)^2). Positive changes mean worse progress. Contact outcomes are cumulative within each horizon; per-step contact and error traces are retained. Avoidance is relative to the matched phase-held baseline, not necessarily the original recorded collision. A contact after the five-step pulse but before step 20 counts against longer-horizon avoidance.

Verification: 27 full reference trajectories and filter logs reproduce exactly; selected-branch first observations and contact labels match their originals. All 20 matched baseline steps also equal E81 exactly. Cloned state restoration and input/source/checkpoint hashes pass. 81 branches contain 1620 simulated control steps.

Limitations: historical development data, outcome reporting on the same bodies, and successful warnings concentrated in two trajectories. The phase-held horizon can itself create contacts in an originally successful task. These results cannot establish full-task recovery, broad retention, or an independent safety guarantee. No neural training or reserved-test evaluation occurred; the original controller and E78 failed gate remain unchanged.

Correction costs (additional geometry queries, excluding the original filter):

- direction_pulse: 89/135 checked actions changed; 35133 geometry queries.
- direction_all: 199/540 checked actions changed; 98322 geometry queries.

All 107 EASE-FM tests pass, including correction bounds/input preservation and rejection of a clearance improvement that violates the progress constraint. These are implementation checks, not evidence of physical safety.

### Decision and next experiment

Direction changes produce retained local avoidance where motion scaling did not. At 20 steps the baseline has 24/27 colliding branches, the five-step correction pulse 17/27, and sustained correction 15/27. These are seven and nine paired avoided contacts with no new contacts, respectively. Both variants avoid three of five six-joint contacts. Both successful warning controls remain clear, with mean terminal task-error increase of 0.266 mm-equivalent; only two such controls are available.

This supports a complete-task development retention experiment, not a success claim. Freeze the candidate set, no-velocity predictor, clearance threshold and progress constraint. Evaluate sustained correction as the exploratory primary candidate and report the pulse variant as a secondary comparison on all original placements and checkpoints, including tasks that previously produced no warnings. Use online triggering on the altered trajectory, original phase transitions and budgets, the unchanged geometry-filter baseline and analytical comparisons. Require improved complete-task success with no increased collisions and preservation of every baseline-successful six-joint case; record all regressions, interaction and query costs. Do not select or retune using individual outcome labels.

Both variants were examined on these same historical development states; choosing sustained correction for the next test is exploratory. Every actual gain must subsequently replicate on separate placements and embodiments before a generalization claim. Original task success and the failed E78 prediction gate remain unchanged.

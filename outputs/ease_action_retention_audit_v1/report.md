# E80: bounded action counterfactuals with successful controls

Compare five actions from identical states at every E78 learned first contact (26 states) and every response warning on successful learned runs (110 states from two six-joint trajectories). Each branch advances one 40 ms control step with original incoming velocity, finger aperture and physics. The selected action reproduces the original observation and contact label exactly. Full reference trajectories and filter logs also replay exactly.

Actions: selected geometry-filter action; half or quarter of its task-motion component while preserving clipped measured bias; bias-only; analytical Jacobian from the same measured Gram/cross history and bias. All commands remain clipped to +/-0.04. Alternatives bypass the geometry filter, so this evaluates raw bounded alternatives, not a completed new policy. Cloned branches do not update the controller or alter the reference rollout.

| State group | Action | Contacts / states | Mean task-error change vs selected (mm-equivalent) | Worse-error states | Lost all pad contacts vs selected |
|---|---|---:|---:|---:|---:|
| collision | selected | 26/26 | 0.0000 | 0 | 0 |
| collision | half | 26/26 | 0.1286 | 9 | 0 |
| collision | quarter | 25/26 | 0.1901 | 9 | 0 |
| collision | bias_only | 22/26 | 0.2477 | 9 | 0 |
| collision | analytical_same_history | 16/26 | -1.8649 | 5 | 0 |
| collision_6 | selected | 5/5 | 0.0000 | 0 | 0 |
| collision_6 | half | 5/5 | 0.1020 | 1 | 0 |
| collision_6 | quarter | 5/5 | 0.1698 | 1 | 0 |
| collision_6 | bias_only | 5/5 | 0.2380 | 1 | 0 |
| collision_6 | analytical_same_history | 2/5 | -2.7625 | 0 | 0 |
| collision_7 | selected | 21/21 | 0.0000 | 0 | 0 |
| collision_7 | half | 21/21 | 0.1349 | 8 | 0 |
| collision_7 | quarter | 20/21 | 0.1950 | 8 | 0 |
| collision_7 | bias_only | 17/21 | 0.2500 | 8 | 0 |
| collision_7 | analytical_same_history | 14/21 | -1.6511 | 5 | 0 |
| successful_controls | selected | 0/110 | 0.0000 | 0 | 0 |
| successful_controls | half | 0/110 | 0.0434 | 2 | 0 |
| successful_controls | quarter | 0/110 | 0.0658 | 2 | 0 |
| successful_controls | bias_only | 0/110 | 0.0884 | 2 | 0 |
| successful_controls | analytical_same_history | 68/110 | -4.6418 | 1 | 0 |

Task error is sqrt(position_error^2 + (0.15 * tool_direction_error)^2), following the controller task scaling. Positive change means worse progress than the selected action. Worse-error counts use only a 1e-9 numerical tolerance; magnitudes, not these counts alone, determine practical relevance. Pad counts are the existing within-step contact union, not proof of a stable retained grasp.

This is outcome-selected, retrospective diagnosis with privileged simulator cloning. The 110 controls are dependent samples from only two successful six-joint trajectories; they do not establish broad retention. A branch avoiding one contact can fail on the next step. No branch is selected by an executable prediction rule here, and no full task is recovered. E78 remains a failed prediction gate.

Next decision must use both collision avoidance and successful-control interference. A promising action family would still need an explicit causal selection rule and complete-task validation against unchanged controls, including all casewise regressions and analytical comparisons. Do not substitute one-step avoided-contact counts for task success.

### Interpretation and next decision

Half motion avoids 0/26 imminent contacts, quarter motion 1/26, and bias-only 4/26; none avoids any of the five six-joint contacts. These three alternatives introduce no contacts in the 110 successful controls. Analytical same-history control avoids 10/26 (3/5 six-joint), but introduces 68/110 contacts in successful controls. Its better average task-direction error therefore does not make it an acceptable universal fallback.

The existing geometry filter already selects scale zero (clipped measured bias only) in 13/26 collision states and 108/110 successful-control states; additional scaling then cannot change the command. Nine collision states and 85 successful controls also have no feasible filter candidate. This is a restricted candidate-action limitation as well as a possible timing limitation; timing causality is not established here.

Next compare bounded interventions at earlier causally available warning states, keeping all successful controls and reporting the number of collisions without advance warning. Use the same fixed warning rule rather than choosing a lookback from each known collision. Evaluate short multi-step branches to determine whether avoidance merely delays contact, before any complete-task rollout. Do not install the analytical fallback or claim task recoveries from this experiment.

Verification: 28 full reference trajectories and their filter logs replay exactly; all 136 selected-action branches match their recorded positions and contact labels. Five branches per state produce 680 one-step comparisons. Clone restoration and input/source/checkpoint hashes pass. No policy weights or original task outcomes change.

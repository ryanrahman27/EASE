# E81: early-warning intervention pulses

The first causal no-velocity response warning at the unchanged 2 mm threshold triggers each local experiment, regardless of eventual outcome. No warning is selected by counting backward from a known collision. Warning values come from the previously exact-replayed E78 run. Collisions without advance warning remain in coverage counts.

Every triggered state has four 20-step branches (0.8 seconds). Half, quarter and bias-only interventions last five steps (0.2 seconds), followed by fifteen steps of baseline control. The selected branch uses baseline control throughout. Controller history and bias update from each branch independently. Phase goal and aperture stay fixed at their triggering values, so these are matched local continuations, not the original phase-switching full task. No future phase information or analytical fallback is used.

| DOF | Original collisions | Advance warning | Warning only on collision step | No warning | Successful warning trajectories |
|---|---:|---:|---:|---:|---:|
| all | 26 | 14 | 11 | 1 | 2 |
| 6 | 5 | 3 | 2 | 0 | 2 |
| 7 | 21 | 11 | 9 | 1 | 0 |

| Group | Horizon (steps) | Action | Contacts / branches | Avoided vs matched | New vs matched | Mean task-error change (mm-equivalent) |
|---|---:|---|---:|---:|---:|---:|
| original_collisions | 5 | selected | 20/25 | 0 | 0 | 0.0000 |
| original_collisions | 5 | half | 20/25 | 0 | 0 | 1.2116 |
| original_collisions | 5 | quarter | 20/25 | 0 | 0 | 1.8571 |
| original_collisions | 5 | bias_only | 21/25 | 0 | 1 | 2.5543 |
| original_collisions | 20 | selected | 24/25 | 0 | 0 | 0.0000 |
| original_collisions | 20 | half | 24/25 | 0 | 0 | -0.8231 |
| original_collisions | 20 | quarter | 24/25 | 0 | 0 | 2.5619 |
| original_collisions | 20 | bias_only | 24/25 | 0 | 0 | 1.8077 |
| original_collisions_6 | 5 | selected | 3/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 5 | half | 3/5 | 0 | 0 | 0.4535 |
| original_collisions_6 | 5 | quarter | 3/5 | 0 | 0 | 0.5991 |
| original_collisions_6 | 5 | bias_only | 3/5 | 0 | 0 | 0.7416 |
| original_collisions_6 | 20 | selected | 5/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 20 | half | 5/5 | 0 | 0 | -1.8676 |
| original_collisions_6 | 20 | quarter | 5/5 | 0 | 0 | 9.9360 |
| original_collisions_6 | 20 | bias_only | 5/5 | 0 | 0 | 10.5417 |
| successful_controls | 5 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | half | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | quarter | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | bias_only | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | half | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | quarter | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | bias_only | 0/2 | 0 | 0 | 0.0000 |

Task error uses sqrt(position_error^2 + (0.15 * tool_direction_error)^2). Positive changes mean worse progress. Contact outcomes are cumulative within each horizon; per-step contact and error traces are retained. Avoidance is relative to the matched phase-held baseline, not necessarily the original recorded collision. A contact after the five-step pulse but before step 20 counts against longer-horizon avoidance.

Verification: 27 full reference trajectories and filter logs reproduce exactly; selected-branch first observations and contact labels match their originals. Cloned state restoration and input/source/checkpoint hashes pass. 108 branches contain 2160 simulated control steps.

Limitations: historical development data, outcome reporting on the same bodies, and successful warnings concentrated in two trajectories. The phase-held horizon can itself create contacts in an originally successful task. These results cannot establish full-task recovery, broad retention, or an independent safety guarantee. No neural training or reserved-test evaluation occurred; the original controller and E78 failed gate remain unchanged.

### Decision

Reject this five-step motion-scaling pulse as a recovery mechanism. Of 26 original collisions, 14 have advance warning, 11 first warn on the collision step, and one has no warning. Among the 27 triggered trajectories (25 original collision runs and two successful controls), matched baseline contact counts at steps 1/5/10/20 are 11/20/22/24. Bias-only gives 7/21/22/24: the four avoided first-step contacts do not survive the five-step horizon, and one additional contact occurs by step five. Every intervention and baseline has the same 24 colliding branches by step 20, including all five six-joint collision cases. Both successful controls remain collision-free, with no terminal task-error difference.

The one triggered original-collision trajectory that stays clear for 20 steps is also clear under the matched baseline, so it is not an intervention recovery. The original collision with no warning is absent from branch comparisons but retained in coverage. No task-success improvement is established.

Next change the candidate action family rather than tune another slowing pulse: evaluate bounded direction corrections based on the frozen measured-response predictor and whole-arm clearance, with explicit progress constraints and matched successful controls. Candidate selection must use only current state/history and predicted geometry, never the outcomes of simulator branches. Keep the existing controller as the reference and require multi-step and then complete-task retention. This is a proposed new development method; its feasibility and benefit are untested.

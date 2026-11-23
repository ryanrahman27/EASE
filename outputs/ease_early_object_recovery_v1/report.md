# E175: earlier object-safe recovery

48 fitting episodes and5 exact E174 guarded-handoff replays. E120, bias, probe protection, release fix, guarded handoff, late recovery and scoring remain unchanged. Simulator object pose and geometry remain privileged. No held cases or training.

| Controller | Success /48 | Collisions |
|---|---:|---:|
| E120_release | 37 | 11 |
| E174_guarded | 47 | 0 |
| early_guard | 48 | 0 |

Paired against E174: {'rescues': [26], 'losses': []}. Exact unchanged episodes: 46. Total descent steps above100 per episode: 488 (19.52 simulated seconds).

## Intervention and evidence

The additional causal guard observes the last5 measured descent states. It triggers when the pad/object separating-axis gap is below12mm, tool-axis error exceeds.15, positional error exceeds15mm, the gap has closed by more than15mm and axis error has increased by more than.05 over4 control steps (160ms). SAT gaps provide a conservative lower bound on box separation, avoiding native box-distance numerical zeros. Thresholds were designed from these fitting reference histories; this is not a generalization test. No case identifiers enter the controller. The reference-state screen activates on cases26 and43; both have strict grasp-first paths before physical intervention.

The planner is unchanged from E174: choose a feasible grasp, reverse a vertical approach on the same IK branch, then connect the current posture to that pregrasp while excluding all robot/object contact. The original300-step routed descent budget and guarded grasp transition remain. This applies the task-and-motion planning principle of jointly checking grasp and approach feasibility, and reactive task-state guards, rather than integrating or reproducing PDDLStream, cuRobo or STORM. See literature_application.md.

The audit independently reconstructs causal gates, paths, executed commands, safety margins, actual pad contacts and guard decisions. Every saved post-state geometry is checked, and non-early episodes must exactly reproduce complete E174 traces/states, including its three rescued cases. Calibration and frozen bias match the reference. Static path checks alone do not establish dynamic safety; task outcomes and full physical contact scoring are retained. Videos retain the prior failure and all newly affected cases.

## Physical execution and interaction cost

| Case | Controller | Success | Descent steps | Max cube XY movement during descent (mm) | Final hand error (mm) | Final two-pad close steps |
|---|---|---|---:|---:|---:|---:|
| 26 | E174 | False | 100 | 44.391 | 67.132 | 0 |
| 26 | early_guard | True | 105 | 0.000 | 0.781 | 40 |
| 43 | E174 | True | 100 | 21.599 | 33.205 | 40 |
| 43 | early_guard | True | 206 | 0.000 | 0.981 | 40 |
| 62 | E174 | True | 240 | 0.000 | 1.374 | 40 |
| 62 | early_guard | True | 240 | 0.000 | 1.374 | 40 |
| 65 | E174 | True | 221 | 0.000 | 0.648 | 40 |
| 65 | early_guard | True | 221 | 0.000 | 0.648 | 40 |
| 67 | E174 | True | 216 | 0.000 | 1.202 | 40 |
| 67 | early_guard | True | 216 | 0.000 | 1.202 | 40 |

Total executed control steps across48 cases: E174=14537, E175=14570; difference=33 steps (1.32 simulated seconds). This includes all task phases, excluding the identical initial calibration. Planning wall-clock latency is not benchmarked.

Cube displacement is measured over saved40ms descent states. Original contact scoring is evaluated at2ms physics substeps. Video playback samples every80ms and can miss brief contacts. No frame selection changes the task score.

## Interpretation and next comparison

The fitting screen passes:48/48 complete tasks, zero forbidden-contact episodes, and no lost E174 success.46 complete trajectories are exactly unchanged. Case26 is the only rescued failure; case43 remains successful with an earlier object-preserving route. Both changed cases finish with40 consecutive end-of-step bilateral pad contacts during closing and zero measured cube displacement during descent. The guard's rejection/abort branch is not exercised in these successful routed episodes.

This supports the diagnosis that the remaining failure was a task-control timing problem: the strict planner was being called after contact had already displaced the cube. It does not establish an advantage of E120 over analytical history estimation. Keep the archived references and freeze this controller before a matched learned-versus-analytical comparison, with the same sensing privileges, probe schedule, recovery rules and interaction budgets. Generalization to untouched embodiments remains a separate evaluation; these48 fitting cases were used in controller development.

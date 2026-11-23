# E168: privileged link-clearance constraints around E120

**PRIVILEGED GEOMETRY DIAGNOSTIC ONLY.** Exact simulator link/table distances and finite-difference gradients constrain the frozen E120 ensemble command. The estimator remains learned, but the geometry layer is privileged; these results are not history-only adaptation, unseen-body transfer, or a safety guarantee.

| Controller | Success /48 | All collision episodes | Link/table collision episodes | Collision-free failures | Rescues / regressions |
|---|---:|---:|---:|---:|---:|
| E120 + release reference | 37 | 11 | 11 | 0 | — |
| joint_projection | 38 | 6 | 4 | 4 | 1 / 0 |
| task_projection | 39 | 7 | 3 | 2 | 2 / 0 |

All48 six-joint fitting placements were executed for each variant (96 new episodes), with four exact disabled-layer baseline replays beforehand. All cases are retained. Calibration, frozen measured bias, original E120 models and bounded residual,265mm release target, task waypoints, phase budgets and strict success/contact scoring remain unchanged. No higher approach or previous descent/centering intervention is included.

The layer enforces linearized constraints G*u ≥ G*bias −0.5*(distance−0.003m), original command bounds and joint-target bounds. Gradients use scratch forward kinematics, never advancing or mutating live state. The objective is either nearest joint command or nearest predicted tool motion using the E120 estimated Jacobian plus0.015² joint regularization. No exact tool Jacobian, future dynamics rollout or object-pose control input is used.

The geometry constraint covers all six arm capsules against the table. Gripper, ground and self-contact remain scored but are not constrained. Calibration and the fixed local probe are intentionally untouched. Constraints therefore may move failure into a probe or another contact type. The assumed displacement is command minus calibrated bias; nonlinear motion and dynamics can violate actual clearance even when the linear QP succeeds. Solver failures use a logged clipped-bias hold, not silent constraint relaxation.

| Variant | Grasped / lifted | QP steps / fallbacks | Link/table contact during probe (episodes) | Gripper/table contact (episodes) | Positive fitting screen |
|---|---:|---:|---:|---:|---|
| joint_projection | 43 / 40 | 2394 / 0 | 4 | 2 | True |
| task_projection | 45 / 44 | 1959 / 0 | 3 | 4 | True |

Success, collision-free failures and failed phase counts are reported separately so that holding safely without completing the task is not counted as improvement. All paired losses and first-contact phases are retained in summary.json, together with actual sampled substep minimum clearances and constraint activity. Contact-type episode counts can overlap.

Verification covers original calibration/frozen bias, unchanged probe commands and release goals, every final executed command, all accepted linear inequalities, all saved post-state geometry distances, one independently reconstructed distance gradient per eligible episode, and comparison with a feasible hold command’s objective cost. Before intervention outcomes, a nominal-action fast-path joint-bound omission was corrected and archived under pre_intervention_review; numerical checks include known QP solutions, target limits, infeasible fallback and an independent directional geometry derivative. No outcome-driven parameter change occurred.

Success/failure videos are labeled by controller privilege and replay actual saved states. Baseline clips are separately labeled E120+release. Contacts are scored at2ms simulation steps; videos sample80ms and can miss brief contacts. Every rendered state matches saved tool/object traces and decoded frame counts. This diagnostic is fitting-only and does not replace E120 or open a held test.

First arm-link/table contact was separately audited, rather than inferred from overlapping episode counts: joint_projection: {'probe': 4}; task_projection: {'probe': 3}. All such first contacts occurred during the fixed probe, where this diagnostic intentionally leaves the original commands unchanged. Contact may persist into later phases after that first failure; this is not a claim that protected phases never contain contact. Full per-case phase sequences are in contact_followthrough.json.

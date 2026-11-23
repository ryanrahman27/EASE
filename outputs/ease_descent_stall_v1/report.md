# E172: selective late-stall descent recovery

Privileged fitting development following rejected E171 variants. One fixed candidate evaluated on all48 six-joint fitting placements;4 exact E170 replays. E120, measured-error allowance, probe protection, release fix, phase budgets and scoring remain unchanged.

| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 |
|---|---:|---:|---:|---|
| E120_release | 37 | 11 | 0 | — |
| E170_protected | 44 | 0 | 4 | — |
| stall_recovery | 44 | 0 | 4 | [] / [] |

E171 direct task optimization and its joint-limit gate each completed43/48 and lost a previous success. They are rejected and their complete results/videos remain in ../ease_descent_progress_v1. E172 is an additional adaptive fitting experiment, not independent validation.

Activation uses measured state only: at least60 descent observations,25–75mm position error, less than1mm progress over the previous19 control intervals, and any joint magnitude at least2.4rad. Recovery stays active for the remainder of descent. Earlier broader stall rules also activated on successful case43; their read-only diagnoses are retained. The local late-stall rule activates only on the four reference failures in the fitting diagnosis. No case identifiers enter the controller.

Recovery minimizes the E120-estimated task residual with a posture regularizer: joints with magnitude above2.35rad prefer a0.02rad command toward the center; others prefer zero relative command. Regularization scale0.03 replaces0.015 only during recovery. The same QP bounds and privileged clearance constraints still apply. Outside recovery, the controller is exactly E170. No extra steps, new task Jacobian, oracle object state, goal changes, or model training are used.

Audit independently reconstructs every activation decision, posture vector, objective center/metric, causal clearance allowance, accepted inequality and executed command. Every post-state geometry is checked, along with full-state gradient reconstruction and exact complete state/trace equality for nonactivated episodes. Four disabled-recovery reference replays passed first. Video replays verify saved tool/object traces and decoded frame counts;80ms sampling can miss contacts scored at2ms.

A positive task-success screen requires more than44 successes, zero collisions and no lost E170 success. More local progress without a completed task is reported separately. E120+release remains the original baseline. All results are fitting-only and geometry-privileged; no safety guarantee or unseen-body claim is made.

## Progress versus completed tasks

| Case | Reference error (mm) | Recovery error (mm) | Reference orientation error | Recovery orientation error | Last20-step recovery progress (mm) |
|---|---:|---:|---:|---:|---:|
| 26 | 67.13 | 19.10 | 0.3827 | 0.0671 | -2.64 |
| 62 | 59.96 | 46.82 | 0.0422 | 0.0904 | 1.40 |
| 65 | 40.07 | 39.03 | 0.0442 | 0.0388 | -0.05 |
| 67 | 48.41 | 40.59 | 0.0127 | 0.0379 | 0.02 |

All 44 nonactivated episodes reproduce the protected reference's complete states and traces exactly. Full-task outcome: 44/48 successes, 0 collisions; rescues/losses versus E170: {'rescues': [], 'losses': []}. No new completion is inferred from a smaller descent error.

The broader E171 variants each lost one success and are rejected. E172 preserves the existing successes by avoiding intervention on their observed reference trajectories. This activation rule was developed using fitting diagnostics, so retention here does not establish gate generalization. The E120 baseline, probe protection and release fix remain unchanged.

The remaining errors and late progress distinguish a shorter approach to a constrained equilibrium from successful grasping. Orientation can trade off against position, and the nonlinear execution/model mismatch remains. Merely increasing the time budget is not established as a solution. Next work should inspect whether a different collision-free arm posture or earlier route can reach the grasp pose, with a clearly separated privileged feasibility diagnostic before changing the learned controller.

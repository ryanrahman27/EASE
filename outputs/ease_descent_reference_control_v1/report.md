# E166: conditional higher approach with descent reference cap

E120 ensemble plus the E161265mm release target is the unchanged reference. All48 six-joint fitting placements were executed, plus4 seven-joint no-op replays;44 seven-joint results were reused unchanged. No held placement or unseen body was evaluated.

| Metric | Release-fixed E120 | Candidate |
|---|---:|---:|
| Six-joint successes | 37/48 | 35/48 |
| Six-joint collision episodes | 11 | 12 |
| First approach/descent contacts | 10 | 9 |
| All fitting successes, including unchanged seven-joint reference | 77/96 | 75/96 |

Paired six-joint outcomes: **0 rescues, 2 regressions**. Meets fixed retention/contact screen: **False**.

The approach target is raised330→400mm only when initial measured tool-to-target lateral distance exceeds8mm, the existing task tolerance. The decision is made once after calibration, without body IDs, geometry or contact signals. This rule was introduced after inspecting fitting E164 outcomes; it is adaptive development, not independent validation. No threshold sweep was performed.

During descent in an active episode, position error used by the DLS command is capped at30mm; orientation error is unchanged. The exact original E120 bounded residual is preserved around the new DLS base. Completion uses the original final target, and the100-step budget is unchanged. All original/new DLS calculations and executed commands were independently recomputed from logged values. Other phases retain the original controller. This was registered after E164 outcomes, not selected by a held result.

Audit-only correction: the initial checker reconstructed the logged float32 Jacobian as float64. The checker now preserves its original dtype, with strict equality retained. Original source, hashes and case12 trajectory were archived; the repeated case12 trial and every state matched exactly. No controller parameter or checkpoint changed. See audit_precision_correction.json. This extra replay is not an independent task trial.

There were44 active and4 inactive six-joint episodes. Every inactive episode exactly reproduced its full baseline state trajectory and task trace. Calibration and frozen bias were verified for all cases; checkpoint/source hashes are unchanged. The original release fix and strict contact/success criteria remain in force. Elevated release permits a short drop, not gentle-contact placement.

First-contact phases: {'lift': 2, 'descend': 9, 'close': 1}. These count first failures; already failed trajectories may collide again later.

This is a task-controller development result on two known six-joint procedural bodies. It does not establish learned-estimator superiority, unseen morphology transfer or a safety guarantee. E120 is not replaced. All changed placements are retained in summary.json. Videos include outcome-selected baseline/candidate pairs, including the earlier rescue and regression cases.

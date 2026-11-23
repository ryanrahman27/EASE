# E165: conditional higher approach

E120 ensemble plus the E161265mm release target is the unchanged reference. All48 six-joint fitting placements were executed, plus4 seven-joint no-op replays;44 seven-joint results were reused unchanged. No held placement or unseen body was evaluated.

| Metric | Release-fixed E120 | Candidate |
|---|---:|---:|
| Six-joint successes | 37/48 | 38/48 |
| Six-joint collision episodes | 11 | 10 |
| First approach/descent contacts | 10 | 9 |
| All fitting successes, including unchanged seven-joint reference | 77/96 | 78/96 |

Paired six-joint outcomes: **2 rescues, 1 regressions**. Meets fixed retention/contact screen: **False**.

The approach target is raised330→400mm only when initial measured tool-to-target lateral distance exceeds8mm, the existing task tolerance. The decision is made once after calibration, without body IDs, geometry or contact signals. This rule was introduced after inspecting fitting E164 outcomes; it is adaptive development, not independent validation. No threshold sweep was performed.

There were44 active and4 inactive six-joint episodes. Every inactive episode exactly reproduced its full baseline state trajectory and task trace. Calibration and frozen bias were verified for all cases; checkpoint/source hashes are unchanged. The original release fix and strict contact/success criteria remain in force. Elevated release permits a short drop, not gentle-contact placement.

First-contact phases: {'descend': 9, 'lift': 1}. These count first failures; already failed trajectories may collide again later.

This is a task-controller development result on two known six-joint procedural bodies. It does not establish learned-estimator superiority, unseen morphology transfer or a safety guarantee. E120 is not replaced. All changed placements are retained in summary.json. Videos include outcome-selected baseline/candidate pairs, including the earlier rescue and regression cases.

# E90: physical execution of frozen descent routes

All 16 fixed cases run in three conditions: E88 exact-Jacobian baseline; frozen E89 joint-waypoint tracking during descent with the original100-step budget; and the same tracker with300 descent steps. All other phases, budgets, measured bias updates, physics, geometry filtering and full-task success criteria are unchanged. The13 cases with a sampled-clear selected route receive the intervention; the other three fall back to the original controller with original budgets. No controller tuning occurred after outcomes.

| Group | Mode | Full-task success | Collisions | Descent target reached | Steps including calibration |
|---|---|---:|---:|---:|---:|
| all | baseline | 8/16 | 6 | 12 | 5115 |
| all | route100 | 11/16 | 4 | 13 | 5929 |
| all | route300 | 13/16 | 3 | 15 | 5818 |
| 6 | baseline | 3/8 | 4 | 5 | 2646 |
| 6 | route100 | 4/8 | 4 | 6 | 3053 |
| 6 | route300 | 5/8 | 3 | 7 | 2877 |
| 7 | baseline | 5/8 | 2 | 7 | 2469 |
| 7 | route100 | 7/8 | 0 | 7 | 2876 |
| 7 | route300 | 8/8 | 0 | 8 | 2941 |

Paired success changes:

- {"case": 1, "mode": "route100", "change": "gain"}
- {"case": 1, "mode": "route300", "change": "gain"}
- {"case": 4, "mode": "route300", "change": "gain"}
- {"case": 8, "mode": "route300", "change": "gain"}
- {"case": 12, "mode": "route100", "change": "gain"}
- {"case": 12, "mode": "route300", "change": "gain"}
- {"case": 13, "mode": "route100", "change": "gain"}
- {"case": 13, "mode": "route300", "change": "gain"}

## Interpretation and next decision

Full-task success increases from8/16 to11/16 under the original descent budget and13/16 with the longer budget. Route100 recovers cases1,12,13; route300 additionally recovers cases4,8. No baseline successes are lost. The difficult-family case4, previously only a static-clear route, now completes physical pick-and-place with300 descent steps. Under100 steps it reaches waypoint50/100, then proceeds through the unchanged phase sequence and fails with contact. Cases5-7 have no selected clear route and remain unchanged failures.

This establishes dynamic executability of the selected case4 route and shows that the specific waypoint tracker needs more than the original budget. It does not show that every possible controller requires that larger budget. Improved downstream outcomes after only a descent intervention support posture/path dependence, but do not isolate a single cause for every recovered case.

These are privileged development results, not improved learned-policy scores: exact Jacobians and offline geometry/IK route search are used, and E89 route selection is informed by these development cases. The longer-budget condition is explicitly not interaction-matched. Neither condition constitutes new held-out generalization evidence. Learned E87 scores remain46/96 with response correction.

Next separate the two requirements: test whether the frozen learned motion estimators can complete the remaining phases after the same privileged route intervention, with the planner contribution clearly disclosed; independently investigate multi-segment routes for cases5-7. A matched learned comparison should preserve original and longer budgets separately and retain all cases. Do not train on these outcomes and then describe these cases as unseen evaluation.

Verification:48 saved full rollouts;16 exact E88 baseline replays; six additional unchanged no-route runs; exact descent-start qpos/qvel across interventions; unchanged input hashes;112 passing tests, including waypoint waiting/clipping and non-descent action preservation. Six executed full-task videos reproduce every saved result and trace exactly; decoded frame counts and video hashes are recorded.

## Saved videos

Gallery: `outputs/ease_novelaxis_path_execution_v1/videos/index.html`. Cases0 and4 in all three modes were selected before evaluation. Labels disclose executed physics, privileged control, final outcome and cumulative collision. These are full-task executions, unlike the E89 static path animations.

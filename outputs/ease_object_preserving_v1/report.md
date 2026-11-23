# E174: object-preserving routes and guarded grasp handoff

Two matched privileged fitting interventions,48 cases each, with4 exact E170 replays. Original E120 weights/frozen bias, probe protection and265mm release are unchanged.

| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 | Extra descent steps |
|---|---:|---:|---:|---|---:|
| E120_release | 37 | 11 | 0 | — | 0 |
| E170_protected | 44 | 0 | 4 | — | 0 |
| object_route | 47 | 0 | 1 | [62, 65, 67] / [] | 377 |
| guarded_handoff | 47 | 0 | 1 | [62, 65, 67] / [] | 377 |

The planner rejects all robot/object contacts in static checks. It samples grasp configurations at current simulator object XY and240mm height, reverses a vertical approach to320mm, then connects to that pregrasp posture. This preserves the grasp/approach IK branch. The initial pregrasp-first search failed and is archived; this correction occurred before physical intervention outcomes. Case26 has an existing robot/object contact at the gate, so no strict route is accepted and the original controller is retained.

Both variants use the same route, gain1 joint tracking, protected32-step probe and up to300 descent steps after accepted planning. Pick/close/lift XY follows the object location observed at planning, while final placement and release height remain unchanged. object_route retains the ordinary close/lift transition. guarded_handoff checks close-entry XY/Z/axis alignment and object drift, holds arm posture during closure, and requires both pads contacting at the END of the final5 close steps before lift. Otherwise it holds with fingers open through remaining scheduled phases and is scored normally as a failure.

Geometry, simulator object pose and contact signals are privileged. These interventions do not train E120 or demonstrate learned-only cross-embodiment adaptation. Static contact-free sampling does not guarantee actual object preservation. Actual object displacement, grasp/lift and complete-task outcomes are retained. No held test is executed.

Independent audit reconstructs gates, current-state plans, strict path contacts, waypoints, clearance allowances, all actual commands and apertures, close readiness, end-step pad contacts, sustained-contact counters and aborts. All saved post-state clearances and representative gradients are checked; accepted linear constraints and real phase budgets are verified. Nonexecuted-route episodes reproduce full E170 traces/states exactly. Both variants have identical trajectories before close. Four exact reference replays passed after correcting and archiving a calibration-time phase-bookkeeping bug before interventions.

See object_handoff_diagnosis.json for object displacement and each remaining failure. Saved videos include every original failure and a retained success; replayed tool/object traces and decoded frame counts are verified. Contact scoring occurs at2ms;80ms video sampling may miss brief contacts. literature_review.md maps established planning/control methods to these failure modes, distinguishes source findings from proposed applications, and states which ideas were actually implemented.

## Actual object motion and grasp handoff

| Variant / case | Route | Task success | Object moved (mm) | Previous route moved (mm) | Close XY gap (mm) | Final two-pad steps | Abort |
|---|---|---|---:|---:|---:|---:|---|
| object_route / 26 | False | False | 44.39 | 153.88 | 40.72 | 0 | False |
| object_route / 62 | True | True | 0.00 | 0.00 | 0.17 | 40 | False |
| object_route / 65 | True | True | 0.00 | 77.47 | 1.40 | 40 | False |
| object_route / 67 | True | True | 0.00 | 72.58 | 1.49 | 40 | False |
| guarded_handoff / 26 | False | False | 44.39 | 153.88 | 40.72 | 0 | False |
| guarded_handoff / 62 | True | True | 0.00 | 0.00 | 0.98 | 40 | False |
| guarded_handoff / 65 | True | True | 0.00 | 77.47 | 2.15 | 40 | False |
| guarded_handoff / 67 | True | True | 0.00 | 72.58 | 2.63 | 40 | False |

Object movement is measured from the first to last executed descent state; it includes pre-trigger motion. Case26 has no accepted strict route and is an exact fallback, so it is not evidence of the new route preserving the object. All endpoints and grasp guards are evaluated from measured states. Lower displacement, a reached hand pose or an aborted lift is not scored as a successful pick-and-place.

The guarded and unguarded variants have identical route/probe prefixes; their divergence starts at closure. This isolates the handoff on these fitting trajectories, while the broader route changes relative to E173 also include grasp-first approach planning, current object XY and gain1 tracking. Do not attribute a between-study improvement to a single component.

The related methods review is in literature_review.md. We implemented staged planning and a guarded task-state transition, not the cited cuRobo/STORM/PDDLStream/contact-MPC algorithms. All findings remain privileged fitting diagnostics with frozen E120; no unseen-embodiment or learned-policy superiority claim is made.

Both conditions complete 47/48 with zero forbidden-contact episodes and rescue cases62,65,67. Their routes produce zero measured object displacement during descent. All44 prior successes and the failed no-route case26 reproduce exactly (45 complete reference trajectories). Added interaction totals377 descent steps per variant, or15.08 simulated seconds. The guarded condition adds no successes over route-only; all three routed grasps pass, so no guard-triggered abort was exercised physically. The gain is a privileged fitting control result, with no held-body test or learned-estimator change.


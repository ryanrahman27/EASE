# E173: alternative postures and route execution

Privileged fitting diagnostic. E120, frozen bias, probe protection and release265mm are unchanged. Static planner found alternative grasp postures and sampled collision-free routes for all four failures. Dynamic execution is evaluated separately below.

| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 | Extra descent steps |
|---|---:|---:|---:|---|---:|
| E120_release | 37 | 11 | 0 | — | 0 |
| E170_protected | 44 | 0 | 4 | — | 0 |
| route_fixed | 44 | 3 | 1 | [] / [] | 0 |
| route_extended | 44 | 0 | 4 | [] / [] | 697 |

All96 complete fitting episodes and4 exact protected-reference replays are retained. Neither variant changes the learned model. The online planner is activated only by the previously specified measured late-local-stall gate. It uses exact simulator geometry, object/jaw state and finite kinematics in scratch data,24 IK starts and bounded path search. No physics advances occur during planning; live state is asserted unchanged. All four accepted routes were direct joint-space interpolations to alternative postures, so no RRT route was required.

The route tracker uses nearest remaining dense path node plus3-node lookahead, original joint command clipping, frozen measured bias and the unchanged E170 clearance/error-allowance QP with E120 metric. It takes over descent only; E120 handles all other phases. Probe schedule/aperture, calibration, lower/release goals and strict contact/task scoring remain unchanged. The route is privileged assistance, not learned adaptation.

route_fixed keeps the original100-step descent budget. route_extended allows up to300 steps only following accepted planning, with unchanged reach termination. Extra interactions and simulated time are reported, not hidden as policy improvement. Static paths hold jaw/object state fixed and check discrete samples; they do not certify continuous or dynamic safety. Full episode collisions and task failures remain scored.

The audit reconstructs every gate, plan source state, waypoint, allowance and executed command; independently checks saved path samples, all post-state geometry and representative gradients; verifies accepted constraints and actual descent budgets; and checks complete trace/state equality for nonactivated episodes. Online routes match the pre-execution static diagnosis. Video replays verify saved tool/object states and frame counts;80ms video samples can miss contacts scored at2ms.

See summary.json for all four route outcomes, failure phases and endpoint errors. Publication-readiness assessment, evidence gaps and primary-source related-work links are saved in publication_readiness.md. This study is fitting-only; no unseen-body result or superiority to a flow-matching policy is claimed.

## Dynamic route and grasp handoff

| Variant / case | Descent reached | Descent steps | Descent error (mm) | Object moved during descent (mm) | Closing tool/object XY gap (mm) | Grasped / lifted | Task success |
|---|---|---:|---:|---:|---:|---|---|
| route_fixed / 26 | False | 100 | 105.34 | 85.78 | 7.19 | True / True | False |
| route_fixed / 62 | False | 100 | 197.24 | 0.00 | 28.62 | False / False | False |
| route_fixed / 65 | False | 100 | 169.73 | 0.00 | 48.50 | False / False | False |
| route_fixed / 67 | False | 100 | 182.75 | 0.00 | 72.22 | False / False | False |
| route_extended / 26 | True | 225 | 7.49 | 153.88 | 156.52 | False / False | False |
| route_extended / 62 | False | 300 | 114.20 | 0.00 | 21.82 | True / False | False |
| route_extended / 65 | False | 300 | 12.36 | 77.47 | 73.75 | False / False | False |
| route_extended / 67 | True | 272 | 6.51 | 72.58 | 72.38 | False / False | False |

The planner preserves the original forbidden-contact definition: finger-pad/cube contact is allowed. Thus an accepted static path may touch the cube, while static checks do not integrate its subsequent displacement. Joint tracking and contact dynamics can also deviate from the sampled path. Actual object motion and full-task scoring above remain decisive; endpoint/path feasibility is not counted as a successful pick-and-place.

The long-budget result should be interpreted together with its extra interactions and grasp outcome. A route can solve the local arm-pose blockage while moving the object away from the fixed grasp target. An object-preserving approach and explicit grasp handoff would be a separate privileged control intervention requiring a new frozen fitting comparison, not a reason to relabel these failures.

Publication implication: these results diagnose where controller/task assumptions fail. They do not establish a new learned-adaptation advantage. The next publication-critical experiment remains a frozen comparison of learned versus analytical history under identical control/privilege on untouched kinematic embodiments. See publication_readiness.md.

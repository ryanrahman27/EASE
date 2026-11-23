# E178 interpretation: matched-history diagnosis

The controller, E120 checkpoints, protections and interaction budgets remain frozen. This retrospective analysis covers the three E177 analytical-only successes (0, 8, 37) and three E120-only successes (9, 42, 45). It does not change the original cohort result: 42/48 successes for each estimator, with one E120 collision and zero analytical collisions.

Twelve full trajectories replay exactly. Across 4,620 pre-action histories, both estimators receive the same state, history statistics, calibrated bias, goal and controller context; six action candidates are evaluated in restored simulators. Source predictions are exact and source physical steps reproduce recorded states to 1e-12. On collision-free-before-action motion states from the three failed E120 trajectories, reference-network position-change RMSE is 1.04 mm versus 0.48 mm for analytical estimation. Analytical also predicts better on the E120-win controls (1.50 versus 0.74 mm), so prediction accuracy alone does not determine task success. These errors use measured joint displacement, not an assumption that commanded motion was executed exactly.

## Cases 0 and 8: descent, projection and grasp handoff

On the same E120 descent states, analytical actions improve immediate goal progress by an average 1.51 mm/step in case 0 and 1.46 mm/step in case 8. Crossed nominal-action/projection-metric tests implicate the learned metric: holding the E120 nominal action fixed and replacing only its projection metric improves progress by 2.67 and 0.90 mm/step respectively. Removing the bounded correction changes progress by only -0.04 and +0.02 mm/step. All six candidates have zero one-step collisions on these descent starts. These are local counterfactual effects, not demonstrated complete-episode rescues.

Case 0 exhausts descent at 19.87 mm tool error, then closes at 18.05 mm error. It registers both pads but loses contact during lift and leaves the object on the table. Analytical closes at 2.02 mm error and succeeds. Case 8 exhausts descent at 65.77 mm error after moving the object about 33 mm; its recovery route is rejected because robot/object contact already exists. Closing later causes right-pad/table contact at step 179. Analytical obtains a route and succeeds.

The frozen grasp-readiness guard is conditional on an accepted recovery route. Neither E120 case has an accepted route, so the phase logic proceeds to closing despite poor alignment. This is a concrete interaction between estimation-dependent descent progress and the existing controller handoff, rather than evidence that the small correction residual alone causes the failures.

## Case 37: longer-horizon lift convergence

E120 reaches descent and establishes two-pad contact, but its lift times out at 45.18 mm tool error. Transport also times out and the final object position misses the target by about 40.6 mm. Analytical reaches lift and ultimately finishes about 4.9 mm from the target. Both actual trajectories lose pad contact during transport; analytical's success is the original final-position score, not evidence of uninterrupted stable grasp through release.

A 20-step continuation initially made an analytical swap appear worse. Extending the diagnostic to the unchanged 100-step lift budget reverses that conclusion. From the identical E120 lift-entry state and history, analytical reaches the lift goal in 83 steps with 4.06 mm error and object height 321.9 mm; E120 times out at 100 steps with 45.18 mm error and object height 280.4 mm. E120 without its correction is worse: 95.33 mm error and object height 229.0 mm. From analytical's lift-entry state, analytical reaches in 73 steps (4.47 mm), while E120 times out (24.21 mm) and the object falls to table height. All six full-budget branches are collision-free. The two source continuations reproduce 173 recorded steps to 1e-12.

This supplies direct phase-level evidence that estimation-dependent control affects lift convergence even after matching initial history and physical state. It does not demonstrate a full pick-and-place rescue: the counterfactuals stop at the lift boundary. The longer follow-up was selected after inspecting the short-horizon result and is explicitly exploratory.

## Consequences for the next experiment

Do not simply disable E120's bounded correction: it is small and mixed in descent, and helps the difficult lift relative to the reference-only condition. The next fitting-only experiment should separate (1) the estimator used to define the clearance-projection metric, (2) grasp-readiness checks for all descent exits, and (3) full-budget lift convergence. Any changes need fitting-case selection before a fresh unseen-body evaluation; these six outcome-selected cases are diagnostic, not a new validation set.

All raw predictions, action candidates, restored branch states, phase continuations, audits and source snapshots are saved in this directory. Twelve videos retain the actual frozen-policy successes and failures, including the E120-win controls. The privileged local true Jacobian is an offline diagnostic and action-distance proxy, not an optimal-action label or deployed component.

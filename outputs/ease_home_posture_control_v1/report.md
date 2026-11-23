# E151/E152: fixed home-posture objective

All controllers receive the same observation-only damped projection toward their supplied starting posture. Gain.05, maximum per-joint correction.01rad, task damping.015. No scene-geometry lookup or physics selector. Exact-map entries remain privileged diagnostics. E152 adds the matched analytical-prior reference after partial E151 outcomes, without changing coefficients or models.

| Estimator | Original successes | Posture successes | Rescues | Lost successes | Posture collisions |
|---|---:|---:|---:|---:|---:|
| uniform32 | 74/96 | 76/96 | 4 | 2 | 17 |
| e120_ensemble | 72/96 | 70/96 | 6 | 8 | 21 |
| learned_prior | 71/96 | 70/96 | 5 | 6 | 17 |
| privileged_exact_map | 76/96 | 77/96 | 1 | 0 | 12 |
| calibration_prior | 67/96 | 68/96 | 8 | 7 | 20 |

E151 frozen fresh-test screen: **False**. The learned-prior controller recovers persistent case13 but loses other successes. It completes70/96, versus matched analytical-prior68 and uniform76. This does not establish a stronger overall learned controller. E120 weights and original baseline records remain unchanged; no fresh/held task was executed.

All480 executed fitting trajectories and posture corrections are retained. Every actual command was independently checked against the logged final action and joint limits; probe actions and measured frozen bias are unchanged. Videos are explicitly outcome-selected comparisons of successes and failures.

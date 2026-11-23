# E33: broader embodiment development evaluation

All three motion-Transformer checkpoints are frozen. This separate development suite contains 16 new axis arrangements (eight per joint count), four crossed geometry/dynamics variants, two starting postures, and eight goals per body/posture: 64 physical bodies and 1,024 cases. Every arrangement is disjoint from the registered prior families, including reserved axes. No final-test policy evaluation occurred.

Axes may begin with X, Y, or Z. Nominal link multipliers are 0.8–1.2 and masses 0.7–1.3; geometry-shift multipliers are 0.55–1.55 with masses 0.5–1.8. Nominal actuator strength/damping are 1/2; shifted strength is sampled from 0.3–2.0 and damping from 0.3–5.0. Each family shares two joint homes and target joint offsets across its four variants. Absolute Cartesian goals differ with geometry; this is not an isolated Cartesian-observation ablation.

Goals are forward-kinematic endpoints, 6–18 cm from the reset position, at collision-free joint configurations within ±2.2 rad. Initial configurations must be contact-free and rank 3. Structural filtering does not establish collision-free paths or closed-loop solvability. No expert or policy success filtering was applied. The privileged expert is an informative reference, not a guaranteed upper bound.

Version 1 stopped during geometry generation on an intrinsically rank-deficient axis family, before controller evaluation. It is preserved. Version 2 adds a recorded generic-pose rank check before accepting a sequence; no controller outcomes informed this change.

## Overall results

All controllers start without a separate calibration. The analytical fixed probe is +0.01 rad on every joint at task step one, charged to its budget. Horizon, action limit, 3 cm / five-step success rule, and collision rule are unchanged. Failures consume 150 steps.

| Controller | Success / 1024 | Six-joint / 512 | Seven-joint / 512 | Collisions | Mean steps | Success by 25 | Success by 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| privileged_reference | 1012 | 500 | 512 | 12 | 13.46 | 994 | 1010 |
| fixed_probe_analytical | 1003 | 491 | 512 | 19 | 19.76 | 881 | 992 |
| analytical_cold | 1003 | 491 | 512 | 19 | 19.48 | 905 | 991 |
| neural_0 | 985 | 477 | 508 | 19 | 27.27 | 759 | 930 |
| neural_1 | 1000 | 489 | 511 | 18 | 21.29 | 859 | 977 |
| neural_2 | 995 | 486 | 509 | 16 | 24.90 | 795 | 950 |

## Crossed conditions

Each cell is successes /128 (64 per joint count).

| Condition | Analytical fixed probe | Neural 0 | Neural 1 | Neural 2 |
|---|---:|---:|---:|---:|
| geometry_nominal__dynamics_nominal__posture_near | 128 | 128 | 127 | 126 |
| geometry_nominal__dynamics_nominal__posture_wide | 126 | 125 | 128 | 128 |
| geometry_nominal__dynamics_shifted__posture_near | 126 | 120 | 126 | 124 |
| geometry_nominal__dynamics_shifted__posture_wide | 120 | 118 | 120 | 119 |
| geometry_shifted__dynamics_nominal__posture_near | 128 | 126 | 126 | 127 |
| geometry_shifted__dynamics_nominal__posture_wide | 128 | 123 | 127 | 127 |
| geometry_shifted__dynamics_shifted__posture_near | 127 | 126 | 126 | 124 |
| geometry_shifted__dynamics_shifted__posture_wide | 120 | 119 | 120 | 120 |

## Paired task outcomes

| Model seed | Neural only | Analytical only | Both successful |
|---:|---:|---:|---:|
| 0 | 4 | 22 | 981 |
| 1 | 5 | 8 | 995 |
| 2 | 4 | 12 | 991 |

## Per-family outcomes

Each cell is successes /64.

| Axis sequence | Joint count | Analytical fixed probe | Neural 0 | Neural 1 | Neural 2 |
|---|---:|---:|---:|---:|---:|
| YXYXXZ | 6 | 64 | 63 | 64 | 64 |
| ZXZXZZ | 6 | 64 | 64 | 64 | 64 |
| XXZZXY | 6 | 64 | 61 | 60 | 63 |
| XYYZYY | 6 | 64 | 64 | 64 | 64 |
| ZYYYXZ | 6 | 64 | 60 | 62 | 63 |
| ZZYYZY | 6 | 64 | 59 | 64 | 63 |
| XXXZYZ | 6 | 43 | 42 | 48 | 42 |
| YXYZYZ | 6 | 64 | 64 | 63 | 63 |
| ZXXZYYX | 7 | 64 | 63 | 64 | 63 |
| YZYXXZZ | 7 | 64 | 64 | 64 | 64 |
| ZZYZYZZ | 7 | 64 | 64 | 64 | 63 |
| XZXZXZX | 7 | 64 | 62 | 64 | 64 |
| XYZXYZZ | 7 | 64 | 64 | 63 | 64 |
| YXXZYXX | 7 | 64 | 63 | 64 | 63 |
| ZZYXXZZ | 7 | 64 | 64 | 64 | 64 |
| YXZYZZY | 7 | 64 | 64 | 64 | 64 |

Structural exclusions: {'axis': 3, 'home': 0, 'goal': 333}. All rejected candidates are retained in the structural audit. Independent verification reproduces all 1,024 target endpoints, checks 128 reset configurations, and validates crossed physical parameters, nonoverlapping axes, case keys, outcome consistency, and unchanged checkpoint hashes. All 6,144 policy/reference records are hashed.

This tests transfer to a broader but still procedural family of six/seven-joint serial arms under clean state observations and position-only reaching. Models, gains, and budgets were not tuned on these results. Repeated goals and physical variants are clustered by 16 axis families; training-seed repeats are not independent morphology samples. No broad real-world, manipulation, noise-robustness, or learned-superiority claim should be inferred from pooled task counts alone.

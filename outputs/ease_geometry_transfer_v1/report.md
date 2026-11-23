# E73: geometry-filter transfer and fixed slowing control

All 17 accepted E71 placements on six existing bodies; all seven frozen estimators and three conditions, 357 trials. The 17/18 screening record and rejected case remain unchanged. This set was not used to design the frozen E72 filter, but its baseline outcomes were already observed in E71; it is development validation, not the reserved final test or new-embodiment transfer.

| Estimator | Mode | Success /17 | Six /8 | Seven /9 | Collisions | Steps | Geometry queries |
|---|---|---:|---:|---:|---:|---:|---:|
| analytical | baseline | 16 | 8 | 8 | 1 | 4752 | 0 |
| analytical | geometry | 16 | 8 | 8 | 1 | 4752 | 12644 |
| analytical | slow | 16 | 8 | 8 | 1 | 5654 | 0 |
| parent_0 | baseline | 15 | 8 | 7 | 2 | 5184 | 0 |
| parent_0 | geometry | 15 | 8 | 7 | 2 | 5010 | 13730 |
| parent_0 | slow | 15 | 6 | 9 | 2 | 5901 | 0 |
| parent_1 | baseline | 17 | 8 | 9 | 0 | 5030 | 0 |
| parent_1 | geometry | 17 | 8 | 9 | 0 | 5030 | 13458 |
| parent_1 | slow | 13 | 8 | 5 | 4 | 6189 | 0 |
| parent_2 | baseline | 15 | 8 | 7 | 2 | 4749 | 0 |
| parent_2 | geometry | 15 | 8 | 7 | 2 | 4783 | 13175 |
| parent_2 | slow | 14 | 8 | 6 | 3 | 6116 | 0 |
| candidate_0 | baseline | 14 | 7 | 7 | 3 | 5297 | 0 |
| candidate_0 | geometry | 14 | 7 | 7 | 3 | 5303 | 14281 |
| candidate_0 | slow | 16 | 7 | 9 | 1 | 5789 | 0 |
| candidate_1 | baseline | 15 | 7 | 8 | 2 | 4907 | 0 |
| candidate_1 | geometry | 15 | 7 | 8 | 2 | 4940 | 13720 |
| candidate_1 | slow | 14 | 7 | 7 | 2 | 5751 | 0 |
| candidate_2 | baseline | 17 | 8 | 9 | 0 | 4723 | 0 |
| candidate_2 | geometry | 17 | 8 | 9 | 0 | 4726 | 12548 |
| candidate_2 | slow | 15 | 8 | 7 | 1 | 5691 | 0 |

| Learned models pooled | Success /102 | Six /48 | Seven /54 | Collisions | Steps |
|---|---:|---:|---:|---:|---:|
| baseline | 93 | 46 | 47 | 9 | 29890 |
| geometry | 93 | 46 | 47 | 9 | 29792 |
| slow | 87 | 44 | 43 | 13 | 35437 |

Prespecified geometry replication rule: FAIL. Per-checkpoint aggregate six-joint retention=True. Geometry success exceeds this fixed slowing control=True.

- geometry versus baseline: 0 paired recoveries, 0 regressions, including 0 six-joint regressions.
- slow versus baseline: 5 paired recoveries, 11 regressions, including 3 six-joint regressions.
- geometry versus slow: 11 paired recoveries, 5 regressions, including 1 six-joint regressions.

Geometry control uses -0.3% control steps versus baseline, 80912 geometry queries, 238 changed actions and 40 no-feasible-candidate fallbacks. Fixed slowing uses +18.6% control steps and zero geometry queries.

The filter is unchanged E72v2: known geometry and exact current simulator pose, 2 mm margin, midpoint/endpoint kinematic checks, action scales 1/.5/.25/0 with measured bias preserved. The slowing control uses a fixed .5 scale on the same motion component, preserves bias and leaves finger commands unchanged. All conditions use the original phase budgets and physics. Slowing is not matched to the filter's per-step action distribution or total interaction time; one constant cannot rule out all geometry-free schedules.

Geometry remains extra privileged information shared across all estimators. A result above the slowing control is not learned-identifier superiority, and a filtered task failure may reflect dynamics outside the sampled kinematic checks. These experiments do not evaluate hardware, latency compliance or a safety guarantee.

The filter changes 238 learned-model actions (about 0.9% of task steps) and performs 80,912 geometry queries, yet all learned casewise success outcomes remain identical to baseline. Its roughly 0.3% reduction in simulated control steps is not evidence of lower computation or real-time latency. The slowing comparison has five recoveries and eleven regressions, so its net loss also masks mixed casewise effects.

Verification: 119 baseline traces and calibration probes exactly reproduce E71. E72 code, model and input hashes are unchanged; physical margins remain unchanged. Geometry access is blocked in both baseline and slowing controls. Trial coverage, nine-phase trajectories and filter/query logs are audited. All 103 EASE-FM tests pass. All failures retained; no training, native tuning or reserved final-test rollout.

### Interpretation and next decision

The frozen geometry filter fails the prespecified replication rule. Retain the original controller; no promotion of the filter. Keep the full slowing comparison as evidence and inspect the paired geometry failures before proposing a different mechanism. Do not retune the filter margin or scale list on this validation set.

The geometry intervention still depends on a known robot model and exact simulator scene state. No outcome here establishes a learned advantage over analytical estimation, broad unseen-embodiment manipulation or publication readiness. The slowing control tests one fixed value, not all geometry-free alternatives.

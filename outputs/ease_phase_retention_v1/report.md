# E70: bounded phase-transition retention

All 16 observed procedural placements, seven frozen estimators and three conditions: 336 trials. Calibration, weights, dynamics, task waypoints and forbidden-contact definitions are unchanged. This experiment isolates phase budget and transition decisions; it does not introduce a collision-clearance planner.

| Estimator | Mode | Success /16 | Six /7 | Seven /9 | Collisions | Aborts | Steps including calibration |
|---|---|---:|---:|---:|---:|---:|---:|
| analytical | baseline | 11 | 7 | 4 | 5 | 0 | 4927 |
| analytical | extended | 11 | 7 | 4 | 5 | 0 | 5224 |
| analytical | gated | 11 | 7 | 4 | 5 | 5 | 3829 |
| parent_0 | baseline | 10 | 4 | 6 | 5 | 0 | 5226 |
| parent_0 | extended | 10 | 4 | 6 | 5 | 0 | 5395 |
| parent_0 | gated | 9 | 4 | 5 | 5 | 7 | 4085 |
| parent_1 | baseline | 9 | 6 | 3 | 6 | 0 | 5610 |
| parent_1 | extended | 9 | 6 | 3 | 6 | 0 | 6220 |
| parent_1 | gated | 9 | 6 | 3 | 6 | 7 | 4314 |
| parent_2 | baseline | 12 | 6 | 6 | 4 | 0 | 4674 |
| parent_2 | extended | 12 | 7 | 5 | 4 | 0 | 5279 |
| parent_2 | gated | 11 | 6 | 5 | 4 | 5 | 4387 |
| candidate_0 | baseline | 11 | 6 | 5 | 5 | 0 | 4853 |
| candidate_0 | extended | 11 | 6 | 5 | 4 | 0 | 5322 |
| candidate_0 | gated | 11 | 6 | 5 | 4 | 5 | 4061 |
| candidate_1 | baseline | 11 | 6 | 5 | 5 | 0 | 5131 |
| candidate_1 | extended | 12 | 6 | 6 | 4 | 0 | 5633 |
| candidate_1 | gated | 10 | 6 | 4 | 4 | 6 | 4017 |
| candidate_2 | baseline | 12 | 6 | 6 | 4 | 0 | 4613 |
| candidate_2 | extended | 13 | 6 | 7 | 3 | 0 | 5401 |
| candidate_2 | gated | 12 | 6 | 6 | 3 | 4 | 4299 |

Extended budgets add at most 40 steps to each moving phase and 20 to close (260 maximum additional steps). Original early pose exit remains active. Gated uses identical budgets but aborts at phase end on accumulated collision, missing five final in-tolerance moving-phase samples, or missing eight final in-tolerance bilateral close samples. Aborts always count as failures. This gate does not stop a collision during a phase, nor avoid contact geometrically.

| Learned models pooled | Success /96 | Six /42 | Seven /54 | Collisions | Aborts | Steps |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 65 | 34 | 31 | 29 | 0 | 30107 |
| extended | 67 | 35 | 32 | 26 | 0 | 33250 |
| gated | 62 | 34 | 28 | 26 | 34 | 25163 |

Prespecified retention requires improved pooled success, no pooled collision increase, and no six-joint success loss in any checkpoint.
- extended: PASS; 5 paired recoveries and 3 regressions; per-checkpoint six-joint retention=True.
- gated: FAIL; 4 paired recoveries and 7 regressions; per-checkpoint six-joint retention=True.

Retention refers to per-checkpoint aggregate six-joint counts, not preservation of every successful case. Extended candidate_0 loses familiar six-joint case 4 while recovering unseen six-joint case 12. Five extended recoveries and three losses therefore remain a mixed result despite passing this aggregate rule. Extended control uses 33,250 versus 30,107 total learned control steps, a 10.4% increase.

Learned gated aborts: collision:lift=6, collision:lower=2, collision:release=1, transition:retreat=1, collision:transport=3, transition:transport=2, collision:descend=13, transition:descend=3, transition:close=2, collision:close=1.

Verification: all 112 baseline traces and calibration probes reproduce E68 exactly. Each of 112 gated traces exactly matches the corresponding extended trajectory up to its stopping point. Model/input hashes are unchanged; 99 EASE-FM tests pass. All failures and stopped trajectories are retained. No training, native evaluation or reserved final-test evaluation.

These dependent repeated seeds and small, IK-screened development placements do not establish population-level transfer. Fewer collision cases can result from ending a trial earlier; task success and paired regressions remain essential. A retention pass would justify further validation, not immediate deployment or a paper-ready claim.

### Interpretation and next decision

Retain these as development candidates for broader validation: extended. Keep the baseline until that validation passes.

First freeze the extended-budget candidate and validate it on fresh procedural placements, including casewise six-joint regressions and cost. Stopping a trial at a failed transition changes exposure to later contacts, so lower collision counts cannot be presented as successful collision avoidance. Extended budgets change interaction duration and can alter contact outcomes even without a gate. These trials test timing and stopping; they do not test geometry-aware motion.

A separate subsequent intervention is a shared path-clearance controller on procedural cases, with an explicit accounting of any geometric information it receives. It should change motion before contact, preserve task success and six-joint retention, and be compared with the original estimator/controller and an analytical estimator under the same geometry access. Avoid another threshold or waiting-time sweep on these observed cases. Native transfer and the reserved final test remain deferred.

# E97: paired post-probe continuation

All 96 checkpoint/task pairs receive learned, analytical and exact continuation (288 full tasks). Every branch has the same learned approach, privileged route descent and fixed32-step local probe. At the first actual close prediction, after all probe measurements update statistics, switch only the motion estimator. The78 admitted-route pairs switch;18 no-route pairs retain original behavior. Bias updates, recent32 history updates, DLS controller, geometry filter, contacts, phase budgets and success criteria remain identical. No state reset or training occurs.

| Group | Continuation | Success | Collisions | Grasped | Lifted | Total steps |
|---|---|---:|---:|---:|---:|---:|
| all | learned | 57/96 | 37 | 85 | 82 | 41896 |
| all | analytical | 68/96 | 27 | 85 | 83 | 39474 |
| all | exact | 78/96 | 18 | 85 | 83 | 38291 |
| 6 | learned | 26/48 | 21 | 37 | 35 | 20438 |
| 6 | analytical | 30/48 | 18 | 37 | 35 | 19552 |
| 6 | exact | 30/48 | 18 | 37 | 35 | 19243 |
| 7 | learned | 31/48 | 16 | 48 | 47 | 21458 |
| 7 | analytical | 38/48 | 9 | 48 | 48 | 19922 |
| 7 | exact | 48/48 | 0 | 48 | 48 | 19048 |
| route | learned | 57/78 | 19 | 78 | 77 | 34533 |
| route | analytical | 68/78 | 9 | 78 | 78 | 32111 |
| route | exact | 78/78 | 0 | 78 | 78 | 30928 |
| no_route | learned | 0/18 | 18 | 7 | 5 | 7363 |
| no_route | analytical | 0/18 | 18 | 7 | 5 | 7363 |
| no_route | exact | 0/18 | 18 | 7 | 5 | 7363 |
| candidate_0 | learned | 8/16 | 8 | 15 | 15 | 6793 |
| candidate_0 | analytical | 10/16 | 6 | 15 | 15 | 6616 |
| candidate_0 | exact | 13/16 | 3 | 15 | 15 | 6291 |
| candidate_1 | learned | 9/16 | 6 | 14 | 12 | 7163 |
| candidate_1 | analytical | 12/16 | 4 | 14 | 13 | 6536 |
| candidate_1 | exact | 13/16 | 3 | 14 | 13 | 6457 |
| candidate_2 | learned | 12/16 | 4 | 14 | 13 | 6879 |
| candidate_2 | analytical | 12/16 | 4 | 14 | 13 | 6622 |
| candidate_2 | exact | 13/16 | 3 | 14 | 13 | 6523 |
| parent_0 | learned | 9/16 | 7 | 13 | 13 | 7340 |
| parent_0 | analytical | 12/16 | 3 | 13 | 13 | 6626 |
| parent_0 | exact | 13/16 | 3 | 13 | 13 | 6485 |
| parent_1 | learned | 9/16 | 6 | 14 | 14 | 6951 |
| parent_1 | analytical | 10/16 | 6 | 14 | 14 | 6521 |
| parent_1 | exact | 13/16 | 3 | 14 | 14 | 6272 |
| parent_2 | learned | 10/16 | 6 | 15 | 15 | 6770 |
| parent_2 | analytical | 12/16 | 4 | 15 | 15 | 6553 |
| parent_2 | exact | 13/16 | 3 | 15 | 15 | 6263 |

## Paired changes against learned continuation

| Case | Checkpoint | DOF | Continuation | Change | Learned collision | Branch collision |
|---|---|---:|---|---|---|---|
| 0 | parent_1 | 6 | analytical | gain | True | False |
| 0 | parent_1 | 6 | exact | gain | True | False |
| 2 | parent_1 | 6 | analytical | gain | True | False |
| 2 | parent_1 | 6 | exact | gain | True | False |
| 2 | candidate_0 | 6 | analytical | gain | True | False |
| 2 | candidate_0 | 6 | exact | gain | True | False |
| 2 | candidate_1 | 6 | analytical | gain | False | False |
| 2 | candidate_1 | 6 | exact | gain | False | False |
| 8 | parent_0 | 7 | analytical | gain | True | False |
| 8 | parent_0 | 7 | exact | gain | True | False |
| 8 | parent_1 | 7 | analytical | gain | False | False |
| 8 | parent_1 | 7 | exact | gain | False | False |
| 8 | candidate_0 | 7 | analytical | gain | True | False |
| 8 | candidate_0 | 7 | exact | gain | True | False |
| 10 | candidate_0 | 7 | analytical | loss | False | True |
| 10 | candidate_1 | 7 | analytical | gain | True | False |
| 10 | candidate_1 | 7 | exact | gain | True | False |
| 11 | parent_1 | 7 | exact | gain | True | False |
| 11 | parent_2 | 7 | exact | gain | True | False |
| 11 | candidate_0 | 7 | exact | gain | True | False |
| 11 | candidate_1 | 7 | analytical | gain | True | False |
| 11 | candidate_1 | 7 | exact | gain | True | False |
| 12 | parent_0 | 7 | analytical | gain | True | False |
| 12 | parent_0 | 7 | exact | gain | True | False |
| 12 | parent_1 | 7 | analytical | loss | False | True |
| 12 | parent_2 | 7 | analytical | gain | True | False |
| 12 | parent_2 | 7 | exact | gain | True | False |
| 12 | candidate_0 | 7 | analytical | gain | True | False |
| 12 | candidate_0 | 7 | exact | gain | True | False |
| 12 | candidate_1 | 7 | analytical | gain | True | False |
| 12 | candidate_1 | 7 | exact | gain | True | False |
| 13 | parent_0 | 7 | analytical | gain | True | False |
| 13 | parent_0 | 7 | exact | gain | True | False |
| 13 | parent_1 | 7 | analytical | loss | False | True |
| 13 | parent_2 | 7 | analytical | gain | True | False |
| 13 | parent_2 | 7 | exact | gain | True | False |
| 13 | candidate_0 | 7 | exact | gain | True | False |
| 13 | candidate_1 | 7 | analytical | loss | False | True |
| 13 | candidate_2 | 7 | exact | gain | True | False |
| 15 | parent_0 | 7 | exact | gain | True | False |

## Gate and retention

Analytical diagnostic gate: **PASS**. Improves success=True; nonincreasing collisions=True; lost learned-successful six-joint cases=0.

- analytical versus learned: 15 gains, 4 losses.
- analytical versus E95 no_probe: 13 gains, 3 losses; 0 six-joint losses.
- analytical versus E95 hold32: 14 gains, 5 losses; 0 six-joint losses.
- exact versus learned: 21 gains, 0 losses.
- exact versus E95 no_probe: 20 gains, 0 losses; 0 six-joint losses.
- exact versus E95 hold32: 19 gains, 0 losses; 0 six-joint losses.
- Exact versus analytical: 10 gains, 0 losses.

## First forbidden-contact phase

| Phase | Learned | Analytical | Exact |
|---|---:|---:|---:|
| close | 3 | 3 | 3 |
| descend | 11 | 11 | 11 |
| lift | 8 | 2 | 0 |
| lower | 9 | 5 | 4 |
| release | 0 | 1 | 0 |
| retreat | 2 | 4 | 0 |
| settle | 3 | 1 | 0 |
| transport | 1 | 0 | 0 |

All paired control changes are saved in summary.json. E95 reference totals remain no-probe58/96 (34 collisions), hold59/96 (35), and learned probe57/96 (37). Analytical/exact branches retain the32-step probe cost; comparisons with no-probe do not isolate estimator quality alone.

## Interpretation and scope

Measured outcome: learned57/96 with37 collisions, analytical68/96 with27, exact78/96 with18. Analytical gains15 and loses4 versus learned; all four losses are seven-joint cases (10 candidate_0,12 parent_1,13 parent_1,13 candidate_1). Both lost six-joint probe cases from E95 recover. Six-joint success is26/48 learned versus30/48 analytical/exact; seven-joint is31/48 versus38/48 versus48/48. The diagnostic gate passes, but four seven-joint regressions prevent an unqualified replacement recommendation.

On the78 route-assisted pairs, analytical reaches68/78 with9 collisions and exact78/78 with0. All18 no-route pairs remain failed and unchanged, so78/96 is the result of this exact-continuation branch, not a universal ceiling or proof that other tasks are infeasible. All ten remaining analytical failures are seven-joint cases. The exact branch demonstrates that successful full-task continuation is possible from every evaluated probe handoff under the same remaining budgets and controller structure.

Next decision: use the analytical estimator as a stronger measured-history training target on training embodiments only, preserving the frozen evaluation split and all observed regressions. A trained replacement must be tested on complete tasks, not selected by handoff matrix error alone. In parallel within the research sequence, inspect estimator drift through the ten analytical failures before claiming that distillation alone closes the exact-reference gap. No weights or deployed controller are changed by E97.

This is a development diagnostic with privileged route planning and repeated checkpoint/task pairs. The analytical branch uses measured history for its motion fit; the exact branch additionally uses privileged simulator Jacobians. Neither branch is a new learned-policy transfer result. Identical prefixes isolate the post-probe estimator choice, but subsequent observations and trajectories diverge. Outcomes do not prove that a single handoff matrix error or stale history is the sole cause. No thresholds or probe parameters were tuned after outcomes.

Verification:96 learned full results and traces match E95;36 no-route fallbacks match; all288 handoffs match E96 physical state, controller statistics and32-transition histories. Branch prefixes through the probe match E95 exactly. Exact Jacobian access is scoped to the post-switch exact branch. All input and checkpoint hashes remain unchanged. All127 EASE tests pass, including switch timing and restoration of the Jacobian access guard. No reserved-test evaluation occurs.

## Videos

Gallery: `outputs/ease_probe_continuation_v1/videos/index.html`. Prespecified case0 parent_1 and case2 candidate_0 (E95 regressions) and case4 candidate_0 (retained success), all three branches. If needed, append first sorted missing success/failure category and first analytical regression against learned with all branches. Selection is disclosed; these are illustrative rather than representative. Every video replays the complete saved result exactly, checks decoded frame count and records SHA-256 hashes.

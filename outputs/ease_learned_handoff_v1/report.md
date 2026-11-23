# E92: same-state post-descent Jacobian intervention

Selection is explicitly outcome-informed: all ten E91 route300 regressions relative to the original learned baseline, plus the earliest retained success in each checkpoint/DOF stratum represented by those regressions (six controls). These16 contexts are not a representative benchmark. Each is run twice with identical learned approach and planned descent, then learned or exact-Jacobian continuation starting at close. No reset or extra probe occurs at the handoff. Adaptive measured bias, controller statistic updates, physics, geometry filter, phase budgets and success criteria remain unchanged.

| Selected group | Continuation | Success | Collisions | Total steps | Median relative Jacobian error at handoff | Error range |
|---|---|---:|---:|---:|---:|---|
| loss | learned | 0/10 | 7 | 5039 | 0.6738 | 0.5927-0.8236 |
| loss | exact_after_descent | 10/10 | 0 | 3509 | 0.6738 | 0.5927-0.8236 |
| retained_success | learned | 6/6 | 0 | 2268 | 0.5899 | 0.3577-0.9686 |
| retained_success | exact_after_descent | 6/6 | 0 | 2102 | 0.5899 | 0.3577-0.9686 |

| Group | Case | Checkpoint | DOF | Learned success | Exact success | Learned collision | Exact collision | Relative error |
|---|---:|---|---:|---|---|---|---|---:|
| loss | 0 | parent_2 | 6 | False | True | False | False | 0.6835 |
| loss | 0 | candidate_0 | 6 | False | True | False | False | 0.6641 |
| loss | 0 | candidate_2 | 6 | False | True | True | False | 0.5927 |
| loss | 2 | candidate_0 | 6 | False | True | True | False | 0.6035 |
| loss | 2 | candidate_1 | 6 | False | True | True | False | 0.6174 |
| loss | 9 | parent_1 | 7 | False | True | True | False | 0.6907 |
| loss | 12 | parent_1 | 7 | False | True | True | False | 0.7246 |
| loss | 13 | parent_1 | 7 | False | True | True | False | 0.6468 |
| loss | 15 | parent_1 | 7 | False | True | False | False | 0.8236 |
| loss | 15 | candidate_1 | 7 | False | True | True | False | 0.7582 |
| retained_success | 3 | candidate_0 | 6 | True | True | False | False | 0.3577 |
| retained_success | 0 | candidate_1 | 6 | True | True | False | False | 0.5934 |
| retained_success | 8 | candidate_1 | 7 | True | True | False | False | 0.9686 |
| retained_success | 2 | candidate_2 | 6 | True | True | False | False | 0.5270 |
| retained_success | 10 | parent_1 | 7 | True | True | False | False | 0.7914 |
| retained_success | 2 | parent_2 | 6 | True | True | False | False | 0.5863 |

## Interpretation

All ten regressed cases recover, and all six retained-success controls remain successful under exact continuation. Seven collisions among the regressions become zero. Since the branches share the same prefix, physical state, measured bias and sufficient statistics at handoff, this is strong controlled evidence that changing the local kinematic estimate can resolve these selected failures. The intervention uses exact Jacobians throughout the remaining task, not only at one instant, and changes subsequent actions, observations and bias estimates.

It does not establish that accumulated history is the specific cause, or that a deployable estimator repair will reproduce these recoveries. Model approximation, insufficient local excitation, nonlinear motion accumulated across the route and subsequent estimation errors remain possible mechanisms. Relative Frobenius error of the6-by-n position/scaled-tool-axis Jacobian is descriptive: the loss group median is0.674 and control median0.590, with overlapping ranges (controls extend to0.969). There is no validated failure threshold or independent statistical inference.

No learned score is improved by this diagnostic. Do not report16/16 as generalization performance: the set was selected using E91 outcomes and the successful continuation is privileged. E91 hybrid scores and failed retention gates remain unchanged.

## Next decision

Investigate a nonprivileged local-history repair with frozen weights. First reconstruct recent motion statistics at the recorded handoff states and compare fixed short-window estimates against the current exponentially accumulated history, without using exact Jacobians as policy inputs. Predeclare any candidate intervention and evaluate full rollouts on all E91 cases, including successful controls, under explicit interaction budgets. Do not tune a success/failure threshold on these16 selected contexts or claim fresh-test performance.

Verification:32 complete saved rollouts;16 exact E91 learned replays; exact approach/descent trace equality; identical handoff qpos/qvel/ctrl/time, measured gram/cross/bias, counters and task flags across branches; unchanged input/checkpoint/source hashes; guarded simulator Jacobian access before handoff except the disclosed read-only audit. All115 EASE tests pass. Four executed video replays match complete saved trajectories and decoded frame counts; manifests store hashes.

## Videos

Gallery: `outputs/ease_learned_handoff_v1/videos/index.html`. Case0 candidate_0 and candidate_2 show matched learned failures and exact-continuation successes. They were prespecified illustrative regressions and are not representative. Labels disclose privileged continuation.

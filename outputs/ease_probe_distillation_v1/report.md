# E98: training-only local-history neural distillation

Fixed three-seed fine tuning from E66 analytical checkpoints using only the eight original training bodies. Generate16 physical trajectories per body:32 settling,32 E95-style probe and96 rank-one sinusoidal motion steps. Save128 pre-action contexts per trajectory. All contacts and trajectories are retained. Whole trajectories0..11 train;12..15 validate on the same bodies. There are16384 contexts:12288 train and4096 validation;23 physical control steps contain contacts. Validation is not unseen-body evaluation.

2000 AdamW updates per seed, learning rate2e-5, batch128, gradient clip1. Each batch mixes64 measured contexts and64 synthetic replay contexts. Measured targets use the existing float64 analytical ridge solve of the input Gram/cross; every target is independently verified in NumPy. Prediction MSE covers both halves; frozen-parent anchor is applied only to synthetic replay (six-joint weight2, seven-joint0.5). Final checkpoints are used without model selection. No target-embodiment history, Jacobian label or outcome enters training.

| Seed | Held-trajectory MSE before | After | Synthetic MSE before | After |
|---|---:|---:|---:|---:|
| 0 | 0.00331488 | 0.00230932 | 0.00651875 | 0.00596832 |
| 1 | 0.00310682 | 0.00218303 | 0.00662603 | 0.00605862 |
| 2 | 0.00338864 | 0.00232708 | 0.00641641 | 0.00590381 |

## Full-task development evaluation

All16 existing E97 tasks x3 E66 parents x2 conditions =96 full tasks. Parent weights control the identical approach and physical probe; only after the probe on admitted routes do student weights replace the motion estimator. No-route behavior is unchanged. This is phase-specific neural substitution with privileged route planning, not a newly trained end-to-end policy. All48 parent results replay E97 exactly; every student shares the original prefix and E96 handoff.

| Group | Mode | Success | Collisions | Total steps |
|---|---|---:|---:|---:|
| all | parent | 29/48 | 18 | 20835 |
| all | student | 31/48 | 17 | 20905 |
| 6 | parent | 13/24 | 10 | 10250 |
| 6 | student | 15/24 | 9 | 10057 |
| 7 | parent | 16/24 | 8 | 10585 |
| 7 | student | 16/24 | 8 | 10848 |
| candidate_0 | parent | 8/16 | 8 | 6793 |
| candidate_0 | student | 12/16 | 4 | 6614 |
| candidate_1 | parent | 9/16 | 6 | 7163 |
| candidate_1 | student | 11/16 | 5 | 7082 |
| candidate_2 | parent | 12/16 | 4 | 6879 |
| candidate_2 | student | 8/16 | 8 | 7209 |
| route | parent | 29/39 | 9 | 17033 |
| route | student | 31/39 | 8 | 17103 |
| no_route | parent | 0/9 | 9 | 3802 |
| no_route | student | 0/9 | 9 | 3802 |

## Matched E97 reference controllers

These saved reference results use the same three candidate-parent prefixes and16 tasks. They are reused context, not new E98 rollouts. The exact reference is privileged.

| Group | Continuation | Success | Collisions |
|---|---|---:|---:|
| all | analytical | 34/48 | 14 |
| all | exact | 39/48 | 9 |
| 6 | analytical | 15/24 | 9 |
| 6 | exact | 15/24 | 9 |
| 7 | analytical | 19/24 | 5 |
| 7 | exact | 24/24 | 0 |

## Paired gains and losses

| Case | Parent seed | DOF | Change | Parent collision | Student collision |
|---|---|---:|---|---|---|
| 2 | candidate_0 | 6 | gain | True | False |
| 2 | candidate_1 | 6 | gain | False | False |
| 8 | candidate_0 | 7 | gain | True | False |
| 10 | candidate_1 | 7 | gain | True | False |
| 10 | candidate_2 | 7 | loss | False | True |
| 11 | candidate_0 | 7 | gain | True | False |
| 11 | candidate_2 | 7 | loss | False | True |
| 12 | candidate_2 | 7 | loss | False | True |
| 13 | candidate_0 | 7 | gain | True | False |
| 15 | candidate_2 | 7 | loss | False | True |

## Retention decision

Gate: **FAIL**. Strict pooled success improvement=True; nonincreasing collisions=True; every parent success retained=False; every seed nondecreasing=False.

There are 6 gains and 4 losses. All three seeds and all failed tasks are retained. Lower held-trajectory MSE does not establish better closed-loop control. Do not promote a checkpoint that fails the prespecified full-task retention gate; no outcome-driven retuning occurs in this experiment.

Observed decision: do not adopt these checkpoints. Students improve29/48 to31/48 and reduce collisions18 to17, but gains6 and losses4 are uneven across seeds. Seed0 improves8/16 to12/16; seed1 improves9/16 to11/16; seed2 drops12/16 to8/16. All four losses are seven-joint seed2 cases10,11,12,15. Six-joint success improves13/24 to15/24 with no losses; this is15/15 on admitted-route six-joint pairs, while nine no-route pairs remain unchanged failures. Seven-joint pooled success remains16/24 because four gains offset four losses. The matched analytical reference remains stronger at34/48; exact reaches39/48.

Held-trajectory prediction MSE improves29.7-31.3 percent in all seeds, and synthetic replay MSE improves too. This does not prevent full-task regression in seed2. The current evidence supports training-only local-history transfer gains for the six-joint subset, with seed-dependent seven-joint control instability. Do not select the best seed or introduce a DOF switch based on these outcomes.

Next: add complete task-trajectory histories from training embodiments and an explicit parent-behavior retention objective, then evaluate every fixed final seed again. Preserve the current four seven-joint failures as development checks and reserve fresh evaluation for confirmation after the method is fixed. This targets the gap between scripted local-motion fitting and closed-loop task behavior; it is a hypothesis to test, not an established cause.

Scientific figure: `training_and_tasks.png` and `training_and_tasks.pdf` show training fit and full-task outcomes. All contexts, provenance, losses, final weights and failures remain saved.

This evaluates only the post-probe use of the new weights. Synthetic replay loss is a retention surrogate, not a full original reaching-suite check. No whole-policy or original-reaching retention claim is made. Training histories are scripted local motions rather than successful pick/place trajectories. The repeated development benchmark remains separate from training bodies, but is not an independent final test.

Verification:127 existing regression tests pass. Training/validation trajectories are disjoint (96/32 trajectories); all target solves agree with independent NumPy; frozen evaluation weight hashes and input dataset hashes verify. No reserved-test evaluation or target-arm adaptation occurs. Raw contexts, provenance, losses, final weights and full trial traces are saved.

## Videos

Gallery: `outputs/ease_probe_distillation_v1/videos/index.html`. Candidate_0 cases2,4,10 are prespecified parent/student pairs; append first new student regression with both modes and first missing outcome category if necessary. Disclosed illustrations, not representative samples. Executed video replays must match complete serialized results; decoded frame counts and hashes are checked.

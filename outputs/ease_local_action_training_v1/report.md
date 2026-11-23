# E56: matched local-action estimator training

Six final checkpoints: three E35 initializations, each with prediction-only and local-action fine tuning on exactly shared synthetic batches and action contexts. Both use clean-ridge MSE plus the E35 anchor; control adds 0.01 times normalized local residual with gain .35, damping .015, clipping .04, bias compensation and joint limits +/-2.5. Each receives 2,000 updates at learning rate 2e-5, batch 128. No native data or geometry enters training. The state-batch distribution and RNG match E52; action contexts use a separate RNG. Native evaluation uses the original tool-Z controller.

The local residual is divided by squared task error plus .002^2, then averaged. Synthetic errors use J*dq; bias and joint positions are independently sampled. This models clipping and approximate bias compensation, but not evolving kinematics, contact or exact actuator dynamics. Score validation does not guarantee optimization improves control. The .01 coefficient is fixed, with no sweep.

## Fixed synthetic validation

| Seed | Condition | Initial prediction MSE | Final prediction MSE | Initial normalized action loss | Final normalized action loss |
|---|---|---:|---:|---:|---:|
| 0 | prediction | 0.008075 | 0.005577 | 1.2466 | 1.1260 |
| 0 | control | 0.008075 | 0.006274 | 1.2466 | 1.0603 |
| 1 | prediction | 0.008560 | 0.005671 | 1.3158 | 1.2713 |
| 1 | control | 0.008560 | 0.006459 | 1.3158 | 0.9522 |
| 2 | prediction | 0.008180 | 0.005676 | 1.3797 | 1.1690 |
| 2 | control | 0.008180 | 0.006501 | 1.3797 | 0.9567 |

Prediction-only checkpoints, state-batch hashes and fit/anchor loss traces exactly reproduce E52 for all three seeds. Its 768 training-gate episodes and 36 native trials are reused with source/hash checks, not counted as new independent evaluations. The local-action arm receives 768 new training-gate episodes and 36 new native trials.

## Original training-control gate

| Condition | Seed | Success /256 |
|---|---:|---:|
| prediction | 0 | 256 |
| prediction | 1 | 256 |
| prediction | 2 | 256 |
| control | 0 | 256 |
| control | 1 | 256 |
| control | 2 | 256 |

## Observed native manipulation

| Condition | Robot | Seed 0 /6 | Seed 1 /6 | Seed 2 /6 | Collisions across seeds |
|---|---|---:|---:|---:|---:|
| original | piper | 6 | 6 | 6 | 0 |
| original | panda | 5 | 1 | 5 | 2 |
| prediction | piper | 5 | 6 | 5 | 0 |
| prediction | panda | 3 | 3 | 4 | 1 |
| control | piper | 5 | 5 | 5 | 2 |
| control | panda | 4 | 1 | 5 | 3 |

Analytical reference solves 6/6 on each robot. Original and analytical scores reuse the E46 records, already exactly replayed in E50; they are not new independent trials. All twelve placements are observed development cases.

## Retention and paired changes

Candidates passing preliminary training/success/collision retention: []. No automatic promotion; any candidate still needs native reaching and broader retention, then independent validation.

## Failures and interaction cost

| Condition | Failures | Mean steps, including calibration |
|---|---|---:|
| prediction | {'no_bilateral_grasp': 1, 'lost_grasp:transport': 3, 'forbidden_contact': 1, 'final_placement': 2, 'lost_grasp:lift': 3} | 546.6 |
| control | {'final_placement': 2, 'forbidden_contact': 5, 'lost_grasp:lift': 4} | 533.2 |

## Direct matched objective comparison

| Robot | Seed | Control gains over prediction | Control losses versus prediction |
|---|---:|---|---|
| piper | 0 | [4] | [2] |
| piper | 1 | [] | [4] |
| piper | 2 | [5] | [1] |
| panda | 0 | [8, 9] | [11] |
| panda | 1 | [] | [9, 11] |
| panda | 2 | [7, 9] | [8] |

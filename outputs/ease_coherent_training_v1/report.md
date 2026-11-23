# E57: matched training on coherent procedural histories

Six final checkpoints: three E35 initializations, paired prediction-only/local-action training with identical mixed batches. Each batch has 64 synthetic E56 contexts and 64 fixed measured procedural contexts. Both receive Jacobian fit loss and the E35 anchor; control adds the unchanged .01 normalized local-action term. Fixed 2,000 updates, lr 2e-5, batch128. No native data or geometry. Measured labels are current geometric Jacobians; synthetic labels remain clean-ridge targets. Cross-E56 changes include context and label distributions, not coherence alone.

The local-action term and anchor weights match E56. Half the batch now has measured histories, errors, biases and joint positions from the same procedural state; the other half retains independent synthetic contexts. The loss still approximates actuator response and omits contact/changes in Jacobians. No coefficient sweep or best-seed selection. All gates and native trials are new because both candidate conditions changed.

## Fixed synthetic validation

| Seed | Condition | Initial prediction MSE | Final prediction MSE | Initial normalized action loss | Final normalized action loss |
|---|---|---:|---:|---:|---:|
| 0 | prediction | 0.008075 | 0.005795 | 1.2466 | 1.1007 |
| 0 | control | 0.008075 | 0.006425 | 1.2466 | 1.1173 |
| 1 | prediction | 0.008560 | 0.005840 | 1.3158 | 1.1984 |
| 1 | control | 0.008560 | 0.006748 | 1.3158 | 1.0975 |
| 2 | prediction | 0.008180 | 0.005875 | 1.3797 | 1.1722 |
| 2 | control | 0.008180 | 0.006449 | 1.3797 | 1.1104 |

## Coherent trajectory holdout

Dataset: 12000 measured states, 8400 training and 3600 held-out trajectory states, from 80 exact E55 replays on eight training bodies. Validation reserves all seed-2 driver trajectories. Same bodies and goals occur in training, so this is not independent embodiment validation. A fixed 512-state validation subset is used for diagnostics.

| Seed | Condition | Initial geometric-J MSE | Final geometric-J MSE | Initial local loss | Final local loss |
|---|---|---:|---:|---:|---:|
| 0 | prediction | 0.019501 | 0.008262 | 1.1746 | 0.7461 |
| 0 | control | 0.019501 | 0.009040 | 1.1746 | 0.6124 |
| 1 | prediction | 0.020534 | 0.008262 | 1.3173 | 0.7974 |
| 1 | control | 0.020534 | 0.010360 | 1.3173 | 0.7409 |
| 2 | prediction | 0.019262 | 0.008125 | 1.0236 | 0.7705 |
| 2 | control | 0.019262 | 0.009768 | 1.0236 | 0.6452 |

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
| prediction | piper | 6 | 6 | 4 | 0 |
| prediction | panda | 3 | 4 | 2 | 1 |
| control | piper | 3 | 4 | 4 | 4 |
| control | panda | 3 | 3 | 1 | 1 |

Analytical reference solves 6/6 on each robot. Original and analytical scores reuse the E46 records, already exactly replayed in E50; they are not new independent trials. All twelve placements are observed development cases.

## Retention and paired changes

Candidates passing preliminary training/success/collision retention: []. No automatic promotion; any candidate still needs native reaching and broader retention, then independent validation.

## Failures and interaction cost

| Condition | Failures | Mean steps, including calibration |
|---|---|---:|
| prediction | {'lost_grasp:lift': 4, 'lost_grasp:transport': 3, 'no_bilateral_grasp': 2, 'forbidden_contact': 1, 'final_placement': 1} | 515.4 |
| control | {'forbidden_contact': 5, 'lost_grasp:lift': 7, 'lost_grasp:transport': 2, 'no_bilateral_grasp': 2, 'phase_timeout:close': 2} | 427.5 |

## Direct matched objective comparison

| Robot | Seed | Control gains over prediction | Control losses versus prediction |
|---|---:|---|---|
| piper | 0 | [] | [1, 3, 5] |
| piper | 1 | [] | [2, 3] |
| piper | 2 | [5] | [1] |
| panda | 0 | [8] | [6] |
| panda | 1 | [9] | [6, 11] |
| panda | 2 | [9] | [8, 10] |

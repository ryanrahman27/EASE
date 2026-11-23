# E52: matched control-aware estimator training

Six final checkpoints: three E35 initializations, each with prediction-only and control-aware fine tuning on exactly the same synthetic batches. Both use clean-ridge MSE and a frozen-parent anchor; control adds 0.1 times damped task-response error. Each receives 2,000 updates at learning rate 2e-5 and batch size 128. No native data, geometry or simulator Jacobian enters training. Native evaluation uses the original tool-Z controller, not E51 full angular control.

The objective is an unclipped linear local-control surrogate. It does not model bias compensation, action clipping, evolving kinematics or contact. Its numerical scale differs from prediction MSE. Better surrogate loss alone is not evidence of task improvement.

## Fixed synthetic validation

| Seed | Condition | Initial prediction MSE | Final prediction MSE | Initial response loss | Final response loss |
|---|---|---:|---:|---:|---:|
| 0 | prediction | 0.008075 | 0.005577 | 5.3750 | 4.1361 |
| 0 | control | 0.008075 | 0.022785 | 5.3750 | 1.5026 |
| 1 | prediction | 0.008560 | 0.005671 | 5.3033 | 4.1754 |
| 1 | control | 0.008560 | 0.023442 | 5.3033 | 1.5295 |
| 2 | prediction | 0.008180 | 0.005676 | 5.7340 | 4.2673 |
| 2 | control | 0.008180 | 0.022630 | 5.7340 | 1.7764 |

## Response-strength diagnostic

| Condition | DoF | Mean response trace across seeds | Ideal trace | Mean action-operator norm ratio |
|---|---:|---:|---:|
| original | 6 | 9.984 | 4.937 | 3.309 |
| original | 7 | 10.734 | 4.971 | 4.479 |
| prediction | 6 | 9.115 | 4.937 | 3.151 |
| prediction | 7 | 9.329 | 4.971 | 4.082 |
| control | 6 | 3.645 | 4.937 | 2.041 |
| control | 7 | 3.908 | 4.971 | 1.887 |

These fixed synthetic measurements show a substantial response-strength shift, not pure improvement in Jacobian accuracy. Trace below the ideal is consistent with attenuated average response; action-operator norms remain above ideal. Neither statistic alone establishes the cause of native failures.

## Original training-control gate

| Condition | Seed | Success /256 |
|---|---:|---:|
| prediction | 0 | 256 |
| prediction | 1 | 256 |
| prediction | 2 | 256 |
| control | 0 | 255 |
| control | 1 | 256 |
| control | 2 | 256 |

## Observed native manipulation

| Condition | Robot | Seed 0 /6 | Seed 1 /6 | Seed 2 /6 | Collisions across seeds |
|---|---|---:|---:|---:|---:|
| original | piper | 6 | 6 | 6 | 0 |
| original | panda | 5 | 1 | 5 | 2 |
| prediction | piper | 5 | 6 | 5 | 0 |
| prediction | panda | 3 | 3 | 4 | 1 |
| control | piper | 0 | 0 | 1 | 16 |
| control | panda | 1 | 0 | 1 | 12 |

Analytical reference solves 6/6 on each robot. Original and analytical scores reuse the E46 records, already exactly replayed in E50; they are not new independent trials. All twelve placements are observed development cases.

## Retention and paired changes

Candidates passing preliminary training/success/collision retention: []. No automatic promotion; any candidate still needs native reaching and broader retention, then independent validation.

## Failures and interaction cost

| Condition | Failures | Mean steps, including calibration |
|---|---|---:|
| prediction | {'no_bilateral_grasp': 1, 'lost_grasp:transport': 3, 'forbidden_contact': 1, 'final_placement': 2, 'lost_grasp:lift': 3} | 546.6 |
| control | {'forbidden_contact': 28, 'lost_grasp:lift': 3, 'phase_timeout:close': 1, 'phase_timeout:descend': 1} | 166.2 |

## Direct matched objective comparison

| Robot | Seed | Control gains over prediction | Control losses versus prediction |
|---|---:|---|---|
| piper | 0 | [] | [0, 1, 2, 3, 5] |
| piper | 1 | [] | [0, 1, 2, 3, 4, 5] |
| piper | 2 | [] | [0, 1, 2, 3] |
| panda | 0 | [] | [6, 11] |
| panda | 1 | [] | [9, 10, 11] |
| panda | 2 | [] | [6, 10, 11] |

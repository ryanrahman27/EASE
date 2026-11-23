# E47: local pre-grasp refresh

Frozen original E35 models on all twelve now-observed E46 development placements. No weight updates. Each intervention adds a shared gated hover 4 cm above grasp and up to 32 physical steps. Probe uses +/- .006 rad targets; frozen probe has identical motion but skips history and bias updates. Wait holds joints and updates history. All initial probes and pre-intervention trajectories match; all 60 historical baseline replays are exact. Total interaction cost includes calibration, hover, probing and failed trials.

| Condition | Robot | Controller | Success | Collisions | Mean steps | Reached refresh |
|---|---|---|---:|---:|---:|---:|
| baseline | piper | privileged | 6/6 | 0 | 950.2 | 0/6 |
| baseline | piper | analytical | 6/6 | 0 | 750.0 | 0/6 |
| baseline | piper | neural_0 | 6/6 | 0 | 736.3 | 0/6 |
| baseline | piper | neural_1 | 6/6 | 0 | 759.8 | 0/6 |
| baseline | piper | neural_2 | 6/6 | 0 | 721.7 | 0/6 |
| baseline | panda | privileged | 6/6 | 0 | 449.8 | 0/6 |
| baseline | panda | analytical | 6/6 | 0 | 435.2 | 0/6 |
| baseline | panda | neural_0 | 5/6 | 0 | 395.2 | 0/6 |
| baseline | panda | neural_1 | 1/6 | 1 | 325.5 | 0/6 |
| baseline | panda | neural_2 | 5/6 | 1 | 404.5 | 0/6 |
| wait | piper | privileged | 6/6 | 0 | 1023.8 | 6/6 |
| wait | piper | analytical | 6/6 | 0 | 780.3 | 6/6 |
| wait | piper | neural_0 | 6/6 | 0 | 769.2 | 6/6 |
| wait | piper | neural_1 | 5/6 | 0 | 737.5 | 6/6 |
| wait | piper | neural_2 | 6/6 | 0 | 746.2 | 6/6 |
| wait | panda | privileged | 6/6 | 0 | 499.3 | 6/6 |
| wait | panda | analytical | 5/6 | 1 | 443.0 | 6/6 |
| wait | panda | neural_0 | 3/6 | 0 | 392.7 | 6/6 |
| wait | panda | neural_1 | 0/6 | 3 | 317.8 | 6/6 |
| wait | panda | neural_2 | 4/6 | 1 | 426.2 | 6/6 |
| probe | piper | privileged | 6/6 | 0 | 1024.3 | 6/6 |
| probe | piper | analytical | 6/6 | 0 | 782.2 | 6/6 |
| probe | piper | neural_0 | 6/6 | 0 | 772.7 | 6/6 |
| probe | piper | neural_1 | 6/6 | 0 | 797.7 | 6/6 |
| probe | piper | neural_2 | 6/6 | 0 | 755.0 | 6/6 |
| probe | panda | privileged | 6/6 | 0 | 498.8 | 6/6 |
| probe | panda | analytical | 6/6 | 0 | 478.5 | 6/6 |
| probe | panda | neural_0 | 4/6 | 1 | 397.5 | 6/6 |
| probe | panda | neural_1 | 1/6 | 1 | 397.2 | 6/6 |
| probe | panda | neural_2 | 2/6 | 0 | 366.5 | 6/6 |
| frozen_probe | piper | privileged | 6/6 | 0 | 1024.3 | 6/6 |
| frozen_probe | piper | analytical | 6/6 | 0 | 778.2 | 6/6 |
| frozen_probe | piper | neural_0 | 6/6 | 0 | 789.0 | 6/6 |
| frozen_probe | piper | neural_1 | 6/6 | 0 | 793.3 | 6/6 |
| frozen_probe | piper | neural_2 | 6/6 | 0 | 761.2 | 6/6 |
| frozen_probe | panda | privileged | 6/6 | 0 | 498.8 | 6/6 |
| frozen_probe | panda | analytical | 4/6 | 0 | 407.5 | 6/6 |
| frozen_probe | panda | neural_0 | 3/6 | 1 | 410.0 | 6/6 |
| frozen_probe | panda | neural_1 | 1/6 | 1 | 388.3 | 6/6 |
| frozen_probe | panda | neural_2 | 4/6 | 0 | 423.2 | 6/6 |

Baseline lacks the added hover, so comparisons to it combine waypoint and calibration effects. The matched intervention comparisons isolate excitation versus waiting and updating versus freezing the estimator. Freezing also freezes bias adaptation. Small, repeatedly observed tasks are diagnostic; these are not independent generalization estimates. Native mechanics and scripted state-based task sequencing remain unchanged. Per-case failures are retained in summary.json and complete traces in records.json.

## Neural outcomes by seed

| Condition | PiPER (each /6) | Panda (each /6) |
|---|---|---|
| baseline | 6 / 6 / 6 | 5 / 1 / 5 |
| wait | 6 / 5 / 6 | 3 / 0 / 4 |
| probe | 6 / 6 / 6 | 4 / 1 / 2 |
| frozen_probe | 6 / 6 / 6 | 3 / 1 / 4 |

## Paired neural effects of updating during probing

| Robot | Seed | Versus | Gained cases | Lost cases | Mean step change |
|---|---|---|---|---|---:|
| piper | neural_0 | baseline | [] | [] | +36.3 |
| piper | neural_0 | wait | [] | [] | +3.5 |
| piper | neural_0 | frozen_probe | [] | [] | -16.3 |
| piper | neural_1 | baseline | [] | [] | +37.8 |
| piper | neural_1 | wait | [0] | [] | +60.2 |
| piper | neural_1 | frozen_probe | [] | [] | +4.3 |
| piper | neural_2 | baseline | [] | [] | +33.3 |
| piper | neural_2 | wait | [] | [] | +8.8 |
| piper | neural_2 | frozen_probe | [] | [] | -6.2 |
| panda | neural_0 | baseline | [] | [7] | +2.3 |
| panda | neural_0 | wait | [6] | [] | +4.8 |
| panda | neural_0 | frozen_probe | [10] | [] | -12.5 |
| panda | neural_1 | baseline | [7] | [10] | +71.7 |
| panda | neural_1 | wait | [7] | [] | +79.3 |
| panda | neural_1 | frozen_probe | [7] | [10] | +8.8 |
| panda | neural_2 | baseline | [] | [6, 9, 11] | -38.0 |
| panda | neural_2 | wait | [] | [6, 11] | -59.7 |
| panda | neural_2 | frozen_probe | [] | [6, 11] | -56.7 |

No promotion. Per-task gains and losses are reported rather than treating multiple seeds on the same six placements as independent trials. Step differences include early termination, so a shorter failed episode is not an efficiency win. The frozen-probe comparison changes both motion-statistic and actuator-bias updates; it cannot attribute an effect exclusively to the Transformer’s Jacobian estimate. The baseline comparison includes the added hover waypoint. All 240 records and 60 exact baseline replays are retained.

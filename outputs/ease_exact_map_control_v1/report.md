# E149: complete fitting control

PRIVILEGED REFERENCE ONLY. Exact current kinematics completes76/96 with16 collision episodes:35/48 six-joint,41/48 seven-joint. This is the same fixed task controller and measured frozen bias, not an upper bound on all possible policies. Exact kinematics improves seven-joint performance but does not recover the six-joint failures of the calibration-prior method. It is never used by a learned estimator or training objective.

All96 placements are known-body fitting/development cases. Exact calibration and initial statistics, fixed bias, checkpoint and source hashes were verified. No future outcomes select a seed or checkpoint. E120 remains immutable.

| Controller | All success | Six joints | Seven joints | Collisions |
|---|---:|---:|---:|---:|
| privileged_exact_map | 76/96 | 35/48 | 41/48 | 16 |
| learned_prior | 71/96 | 35/48 | 36/48 | 22 |
| e120_ensemble | 72/96 | 36/48 | 36/48 | 21 |
| uniform32 | 74/96 | 36/48 | 38/48 | 20 |

Paired gains/losses and all changed placements are preserved in paired_comparison.json. Gallery clips are outcome-selected examples of successes and failures, not rate estimates.

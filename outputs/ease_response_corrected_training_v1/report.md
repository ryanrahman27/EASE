# E61: response-corrected versus uncorrected local-action training

Three E35 initializations, two conditions, 2,000 updates each. Identical batches, lr 2e-5, batch128, fixed .01 local coefficient, fit loss and frozen-parent anchor. Both use local-action loss; prediction directories mean uncorrected, not prediction-only. Control directories mean response-corrected. Frozen leave-one-body-out E59 affine response replaces command-minus-bias only for the measured half. Synthetic half is unchanged.

Measured data: analytical/E35 subset of E57, 3,600 train and 1,200 original_2 trajectory-validation states on eight bodies. Fixed 512-state validation subset. Same bodies/goals occur in policy training and validation; response fits exclude each queried body but the policy is not held out from these bodies. Changes versus E57 include removing E52-driver trajectories, so only the within-E61 pair isolates response correction. No native data, coefficient sweep or seed selection.

## Measured trajectory validation

| Seed | Condition | Initial uncorrected loss | Final uncorrected loss | Initial corrected loss | Final corrected loss |
|---|---|---:|---:|---:|---:|
| 0 | uncorrected | 1.4385 | 0.7646 | 0.8355 | 0.6938 |
| 0 | response-corrected | 1.4385 | 0.7691 | 0.8355 | 0.6934 |
| 1 | uncorrected | 1.4174 | 0.7223 | 0.8223 | 0.6635 |
| 1 | response-corrected | 1.4174 | 1.0627 | 0.8223 | 0.7715 |
| 2 | uncorrected | 0.8353 | 0.6698 | 0.7242 | 0.6608 |
| 2 | response-corrected | 0.8353 | 0.7888 | 0.7242 | 0.7137 |

## Original reaching gate

| Condition | Seed | Success /256 |
|---|---:|---:|
| uncorrected | 0 | 256 |
| uncorrected | 1 | 256 |
| uncorrected | 2 | 256 |
| response-corrected | 0 | 256 |
| response-corrected | 1 | 256 |
| response-corrected | 2 | 256 |

## Native development manipulation

| Condition | Robot | Seed 0 /6 | Seed 1 /6 | Seed 2 /6 | Contacts |
|---|---|---:|---:|---:|---:|
| original E35 | piper | 6 | 6 | 6 | 0 |
| original E35 | panda | 5 | 1 | 5 | 2 |
| uncorrected | piper | 4 | 4 | 3 | 3 |
| uncorrected | panda | 4 | 5 | 2 | 4 |
| response-corrected | piper | 2 | 6 | 6 | 2 |
| response-corrected | panda | 1 | 1 | 3 | 5 |

Preliminary retention candidates: []. No automatic promotion; any passing condition still needs broader native reaching/placement retention and independent validation.

All 1,536 reaching episodes and 72 candidate native trials are fresh. E35 and analytical references reuse audited E46 records; analytical solves all six cases per robot. Twelve placements are observed development cases. Final test untouched.

## Failures and costs

uncorrected: 22/36 successes; failures {'forbidden_contact': 7, 'lost_grasp:lift': 3, 'no_bilateral_grasp': 2, 'phase_timeout:release': 1, 'phase_timeout:lift': 1}; mean steps including calibration 439.2. Early failures can reduce cost.
response-corrected: 19/36 successes; failures {'forbidden_contact': 7, 'phase_timeout:settle': 1, 'no_bilateral_grasp': 1, 'final_placement': 1, 'lost_grasp:lift': 4, 'phase_timeout:lift': 2, 'lost_grasp:transport': 1}; mean steps including calibration 487.0. Early failures can reduce cost.

Direct paired change: corrected gains 7 cases and loses 10 versus uncorrected. Gains (seed, case): [[1, 1], [1, 2], [2, 0], [2, 1], [2, 4], [2, 10], [2, 11]]; losses: [[0, 4], [0, 5], [0, 8], [0, 9], [0, 11], [1, 6], [1, 7], [1, 10], [1, 11], [2, 9]].

84 regression tests pass; 156,000 affine response comparisons agree within 3.4e-9. Paired batch/context hashes, immutable response fits and parents, all reaching/native records and calibration are audited. Response coefficients remain frozen; this experiment trains only the motion estimator.

## Decision

Neither condition passes retention. Keep E35. Response correction gives 19/36 versus matched uncorrected 22/36 and original 29/36. Corrected validation loss is also worse in two seeds; the training benefit is not established even offline. E59/E60 prediction/ranking improvements do not establish a successful policy objective under this fixed recipe. Diagnose free-space native tracking separately from grasp/contact failures before another training intervention. No favorable-seed selection or final-test evaluation.

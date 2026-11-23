# E53: does the objective rank actual actions correctly?

No training. Frozen original/E52 models and analytical estimation receive identical measured histories from all twelve analytical E46 development trajectories. Samples are the first and twentieth step of each reached phase. At each sample, ten controllers branch from exact copied simulator data, with the current reference and aperture fixed for five steps. Online estimator/bias updates, actuator dynamics, clipping, joint limits and contacts remain active.

All twelve complete driver trajectories replay exactly. 202 sampled states produce 2020 counterfactual branches plus 202 exact analytical repeat checks. These are local probes, not independent tasks or success evaluations. No checkpoints changed; the final test is untouched.

The three proxies use a privileged instantaneous geometric Jacobian for diagnostic scoring only: E52 global response-operator MSE; local unclipped target residual; and local executed-command residual after clipping, joint limits and subtracting measured bias. None of these scores enters candidate commands. The executed-command proxy still assumes measured bias predicts actuator response, and omits contact effects.

## Pairwise ordering agreement

Higher is better. Each row averages agreement within each case and then across cases, avoiding domination by longer trajectories. Exact/numerical ties at 1e-12 are excluded. Cases, phases and branches remain dependent; no independent-sample significance claim is made. All-pair results include analytical; neural-only excludes it.

| Robot | Horizon | Candidates | Proxy | Case-mean agreement | Non-tied pairs |
|---|---:|---|---|---:|---:|
| piper | 1 | all | operator_mse | 0.450 | 4589 |
| piper | 1 | all | local_unclipped | 0.659 | 4589 |
| piper | 1 | all | local_executed | 0.734 | 4589 |
| piper | 1 | neural | operator_mse | 0.453 | 3671 |
| piper | 1 | neural | local_unclipped | 0.664 | 3671 |
| piper | 1 | neural | local_executed | 0.731 | 3671 |
| piper | 5 | all | operator_mse | 0.478 | 4590 |
| piper | 5 | all | local_unclipped | 0.624 | 4590 |
| piper | 5 | all | local_executed | 0.679 | 4590 |
| piper | 5 | neural | operator_mse | 0.478 | 3672 |
| piper | 5 | neural | local_unclipped | 0.627 | 3672 |
| piper | 5 | neural | local_executed | 0.676 | 3672 |
| panda | 1 | all | operator_mse | 0.473 | 4500 |
| panda | 1 | all | local_unclipped | 0.734 | 4500 |
| panda | 1 | all | local_executed | 0.740 | 4500 |
| panda | 1 | neural | operator_mse | 0.458 | 3600 |
| panda | 1 | neural | local_unclipped | 0.740 | 3600 |
| panda | 1 | neural | local_executed | 0.747 | 3600 |
| panda | 5 | all | operator_mse | 0.466 | 4500 |
| panda | 5 | all | local_unclipped | 0.586 | 4500 |
| panda | 5 | all | local_executed | 0.586 | 4500 |
| panda | 5 | neural | operator_mse | 0.419 | 3600 |
| panda | 5 | neural | local_unclipped | 0.572 | 3600 |
| panda | 5 | neural | local_executed | 0.571 | 3600 |

## Phase-specific agreement (all candidates)

| Robot | Phase | States | Operator, one step | Executed, one step | Executed, five steps |
|---|---|---:|---:|---:|---:|
| piper | approach | 6 | 0.304 | 0.800 | 0.781 |
| piper | descend | 12 | 0.367 | 0.698 | 0.600 |
| piper | close | 12 | 0.523 | 0.768 | 0.794 |
| piper | lift | 12 | 0.648 | 0.852 | 0.772 |
| piper | transport | 12 | 0.356 | 0.685 | 0.498 |
| piper | lower | 12 | 0.417 | 0.750 | 0.657 |
| piper | release | 12 | 0.383 | 0.611 | 0.644 |
| piper | retreat | 12 | 0.469 | 0.741 | 0.726 |
| piper | settle | 12 | 0.511 | 0.735 | 0.687 |
| panda | approach | 6 | 0.444 | 0.822 | 0.726 |
| panda | descend | 12 | 0.459 | 0.785 | 0.680 |
| panda | close | 12 | 0.509 | 0.796 | 0.702 |
| panda | lift | 12 | 0.463 | 0.713 | 0.617 |
| panda | transport | 12 | 0.431 | 0.781 | 0.507 |
| panda | lower | 10 | 0.435 | 0.678 | 0.419 |
| panda | release | 12 | 0.559 | 0.707 | 0.469 |
| panda | retreat | 12 | 0.504 | 0.757 | 0.583 |
| panda | settle | 12 | 0.426 | 0.646 | 0.604 |

## Local error and contacts

| Robot | Method | Branches | Mean one-step scaled error | Mean five-step scaled error | Five-step contact failures | First-command clipped branches |
|---|---|---:|---:|---:|---:|---:|
| piper | analytical | 102 | 0.008782 | 0.009345 | 0 | 47 |
| piper | original_0 | 102 | 0.008610 | 0.008183 | 0 | 47 |
| piper | original_1 | 102 | 0.008625 | 0.008479 | 0 | 47 |
| piper | original_2 | 102 | 0.008604 | 0.007780 | 0 | 45 |
| piper | prediction_0 | 102 | 0.008724 | 0.008499 | 0 | 48 |
| piper | prediction_1 | 102 | 0.008736 | 0.009120 | 0 | 47 |
| piper | prediction_2 | 102 | 0.008626 | 0.008108 | 0 | 46 |
| piper | control_0 | 102 | 0.008674 | 0.007914 | 0 | 44 |
| piper | control_1 | 102 | 0.008792 | 0.010107 | 0 | 41 |
| piper | control_2 | 102 | 0.008666 | 0.008305 | 0 | 45 |
| panda | analytical | 100 | 0.003093 | 0.002456 | 0 | 12 |
| panda | original_0 | 100 | 0.003145 | 0.002670 | 0 | 13 |
| panda | original_1 | 100 | 0.003130 | 0.002842 | 0 | 9 |
| panda | original_2 | 100 | 0.003317 | 0.002902 | 0 | 16 |
| panda | prediction_0 | 100 | 0.003086 | 0.002751 | 0 | 11 |
| panda | prediction_1 | 100 | 0.003148 | 0.002852 | 0 | 9 |
| panda | prediction_2 | 100 | 0.003276 | 0.002704 | 0 | 10 |
| panda | control_0 | 100 | 0.003246 | 0.003323 | 0 | 10 |
| panda | control_1 | 100 | 0.003157 | 0.003059 | 0 | 5 |
| panda | control_2 | 100 | 0.003250 | 0.003242 | 0 | 8 |

Scaled error is the norm of position residual in metres concatenated with 0.15 times tool-Z direction residual. These means combine phases and are not task-success or safety guarantees.

## Matched control-aware versus prediction-only rankings

| Robot | Horizon | Seed | Operator favors control | Measured residual favors control | Comparable states |
|---|---:|---:|---:|---:|---:|
| piper | 1 | 0 | 62 | 44 | 102 |
| piper | 1 | 1 | 80 | 43 | 102 |
| piper | 1 | 2 | 94 | 31 | 102 |
| piper | 5 | 0 | 62 | 54 | 102 |
| piper | 5 | 1 | 80 | 44 | 102 |
| piper | 5 | 2 | 94 | 45 | 102 |
| panda | 1 | 0 | 89 | 30 | 100 |
| panda | 1 | 1 | 89 | 51 | 100 |
| panda | 1 | 2 | 92 | 46 | 100 |
| panda | 5 | 0 | 89 | 30 | 100 |
| panda | 5 | 1 | 89 | 43 | 100 |
| panda | 5 | 2 | 92 | 30 | 100 |

## Limits and decision

This audit changes neither training nor deployment. It tests local ranking validity on analytical-history development states; it does not reproduce candidate-specific histories, long-horizon grasp losses or final task scores. The five-step reference is held fixed, unlike the continuing sequencer. Operator and action-specific scores answer different questions, so weak agreement identifies objective mismatch but does not by itself prove the cause of E52 failures. Even stronger agreement here would require validation on procedurally generated training arms before defining a new learning objective. Keep E35; no model or loss is promoted. Collision-free ranking details and per-case rates are retained in summary.json.

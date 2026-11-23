# E64: matched-state phase interventions

All 15 failed E63 schedules; all 37 failed phases. Outcome-conditioned diagnostic, not a performance estimate.

Replay each full original schedule exactly. At each failed phase entry clone full simulator state and measured controller statistics for four full-phase branches. Swap learned Jacobian for ridge analytical estimate from the same evolving statistics and/or remove additive measured bias. Online history updates continue in every branch. No privileged dynamics or Jacobian. Zero bias is an ablation, not an oracle. Original reference, thresholds, gains, contact disabling and phase budget retained. Later branches start from original history, not preceding interventions.

| Mode | Holds recovered /37 | PiPER /2 | Panda /35 | Mean clipped-step fraction | Mean motion prediction error |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 0 | 0.8598 | 0.001551 |
| analytical_motion | 24 | 2 | 22 | 0.4807 | 0.000346 |
| zero_bias | 15 | 2 | 13 | 0.6442 | 0.001100 |
| analytical_zero_bias | 20 | 2 | 18 | 0.4499 | 0.000275 |

All 25,500 original replay steps and 37 baseline phase branches match E63 exactly. Telemetry includes estimated singular values, additive bias, motion, executed command, realized joint displacement, command-minus-displacement-minus-bias residual and realized task-delta prediction error. Prediction error uses realized displacement; it is not a causal Jacobian-error oracle. Phase averages weight phases equally.

No training, checkpoint promotion, final-test evaluation or manipulation success claim. Branches are dependent and selected for baseline failure; recovery does not establish whole-task reliability.

## Paired recovery attribution

| Group | Failed phases | Analytical motion | Zero bias | Both |
|---|---:|---:|---:|---:|
| original | 7 | 5 | 3 | 3 |
| uncorrected | 9 | 6 | 4 | 6 |
| corrected | 21 | 13 | 8 | 11 |

Verification: all 148 saved trial files, 37 exact baseline branches, recomputed phase summaries, source and checkpoint hashes pass. EASE-FM regression suite: 90 passed. The repository-wide suite could not collect eight unrelated recovery modules because openpi was not installed in this WSL environment.

## Interpretation and next decision

Replacing only the motion estimate recovers 24/37 failed holds, compared with 15/37 for removing bias and 20/37 for both. The retained E35 subset is 5/7, 3/7 and 3/7 respectively. Both six-joint PiPER failures recover under every intervention; these two cases alone do not identify a unique cause. Analytical substitution reduces mean clipped-step fraction from 0.8598 to 0.4807. These are phase-weighted, failure-conditioned observations, not an estimate of improvement over all tasks.

The learned estimate contributes to recoverable failures, but disabling bias is not a general fix. Analytical substitution still misses 13 phases: five release, five settle, two close and one lower. Twelve of these have only 60 steps, and all inherit the original learned trajectory's state/history. Thus late replacement cannot separate poor initialization from insufficient recovery time or estimation/control error. All-analytical E63 already passes every complete schedule from calibration.

Next: compare continuous interventions from calibration over all 12 full schedules, including previously successful cases, before selecting any estimator/bias redesign. Retain E35; do not train on these native cases or promote an intervention from failure-only recovery. A learned-only improvement still requires procedural training, six-joint retention, matched analytical baselines and independent embodiment validation.

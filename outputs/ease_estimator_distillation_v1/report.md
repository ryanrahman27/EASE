# E66: geometric versus analytical-target estimator training

Matched final-checkpoint fine tuning from E35. Each batch has 64 synthetic E52 contexts plus 64 measured procedural contexts, restricted to analytical/original_0/original_1 training trajectories. Original_2 trajectories are validation (same eight bodies and goals). 3600 training, 1200 validation rows. Identical batches and optimizer budgets. Only measured target changes: current geometric Jacobian versus analytical solve of the observed Gram/cross statistics. Synthetic clean-ridge labels unchanged. No native data, response fit, control surrogate or privileged information at inference.

Masked six-output mean squared fit plus fixed E35 anchor (six-joint weight2, seven-joint .5). AdamW lr2e-5, gradient norm clip1, 2000 updates per condition/seed. No best-checkpoint or seed selection. Analytical targets generated in float64 with padded inactive diagonals filled; network inference remains inverse-free.

First all original 256 reaching tasks per checkpoint with existing bias correction. Then all twelve observed native common schedules and all twelve contact-enabled manipulation cases for all candidates, regardless of failures, reported diagnostically. Full six-/seven-joint retention required before promotion. Fixed E63/E46 E35 and analytical baselines; no final test.

The analytical-target candidates are learned Transformers. They do not call the analytical solver at inference. The analytical controller below is a separate, untrained reference.

## Held-trajectory prediction MSE

| Seed | Target | Geometric MSE | Analytical-estimate MSE | Synthetic MSE |
|---|---|---:|---:|---:|
| 0 | geometric | 0.008404 | 0.005620 | 0.005784 |
| 0 | analytical | 0.009024 | 0.005985 | 0.005695 |
| 1 | geometric | 0.008755 | 0.005973 | 0.005839 |
| 1 | analytical | 0.009434 | 0.006372 | 0.005712 |
| 2 | geometric | 0.008400 | 0.005517 | 0.005849 |
| 2 | analytical | 0.009133 | 0.005956 | 0.005767 |

## Original reaching retention

| Target | Seed | Six joints | Seven joints | Total |
|---|---:|---:|---:|---:|
| geometric | 0 | 128/128 | 128/128 | 256/256 |
| geometric | 1 | 128/128 | 128/128 | 256/256 |
| geometric | 2 | 128/128 | 128/128 | 256/256 |
| analytical | 0 | 128/128 | 128/128 | 256/256 |
| analytical | 1 | 128/128 | 128/128 | 256/256 |
| analytical | 2 | 128/128 | 128/128 | 256/256 |

## common: all twelve observed native cases

| Controller | PiPER per seed /6 | Panda per seed /6 | Total | Gains vs E35 | Losses vs E35 | Contacts |
|---|---|---|---:|---:|---:|---:|
| Analytical controller | [6] | [6] | 12/12 | — | — | 0 |
| E35 Transformer | [6, 6, 6] | [6, 3, 6] | 33/36 | — | — | 0 |
| Geometric-target Transformer | [6, 6, 6] | [5, 6, 3] | 32/36 | 3 | 4 | 0 |
| Analytical-target Transformer | [6, 6, 6] | [6, 6, 6] | 36/36 | 3 | 0 | 0 |

## manipulation: all twelve observed native cases

| Controller | PiPER per seed /6 | Panda per seed /6 | Total | Gains vs E35 | Losses vs E35 | Contacts |
|---|---|---|---:|---:|---:|---:|
| Analytical controller | [6] | [6] | 12/12 | — | — | 0 |
| E35 Transformer | [6, 6, 6] | [5, 1, 5] | 29/36 | — | — | 2 |
| Geometric-target Transformer | [6, 5, 6] | [3, 4, 3] | 27/36 | 3 | 5 | 2 |
| Analytical-target Transformer | [6, 6, 6] | [5, 4, 5] | 32/36 | 4 | 1 | 1 |

All six 2000-update runs verified; paired batch hashes match. All 4800 analytical targets were independently checked against NumPy (maximum absolute difference 2.98e-8). All 1536 reaching records and 144 native records audited; hashes unchanged after evaluation. Regression suite: 92 passed.

Validation trajectories share procedural training bodies/goals; native cases are repeatedly observed development cases. Fixed E35 and analytical reference records are reused from audited E63/E46, not fresh independent baseline trials. No native training or final-test evaluation. A lower fitting error alone does not establish better control.

## Interpretation and next decision

Analytical-target distillation produces the stronger candidate in this matched comparison. It passes all 36/36 common schedules (E35 33/36; geometric-target 32/36), recovering all three E35 failures without a tracking regression. Native manipulation improves to 32/36 (E35 29/36; geometric-target 27/36), with four paired recoveries and one regression. Both tasks retain all 18/18 six-joint PiPER successes. All six trained models pass the original reaching gate at 256/256 each (128/128 per joint count). Bias correction stays enabled; inference remains a learned Transformer plus the existing shared controller, with no analytical estimator solve.

The manipulation improvement is not uniform: Panda per-seed scores are 5/4/5 of six versus E35's 5/1/5. The aggregate gain is concentrated in seed 1. Analytical-target seed 0 loses case 7 during lift and recovers case 9; seed 1 recovers cases 8, 9 and 11. Seed 2 has the same success outcomes as its parent. One forbidden-contact trial remains (E35 has two). The separate analytical controller still passes every native case. These results do not demonstrate a learned advantage over that baseline.

Held-trajectory analytical-estimate MSE is worse for analytical-target training than geometric-target training in all three seeds, despite better native control. Both improve over their parents on those prediction metrics. Average fitting error is therefore not a reliable selection rule for these controllers. The validation split shares training bodies and goals; it cannot establish embodiment generalization.

Keep the E66 analytical-target checkpoints as frozen candidates and preserve E35 as the retained reference. Do not claim complete manipulation retention or promote a uniformly better controller while the paired regression remains. The next experiment should evaluate these fixed checkpoints against E35 and the analytical baseline on prespecified additional procedural embodiments and native placements, without tuning on the observed failures. Keep the reserved final test untouched until the evaluation plan is fixed.

The comparison plot was inspected. A recovered Panda trial (analytical-target seed 1, case 8) and the regression (seed 0, case 7) were rendered and replayed exactly: 411 and 320 total steps including calibration. The preview's first invocation failed before execution because its script-directory import path excluded the repository; explicitly setting PYTHONPATH resolved it without changing simulation or checkpoints. Both successful replay audits and images are retained.

This is an improvement on observed development tasks, not evidence of broad real-robot transfer, hardware performance, end-to-end learned grasping or publication readiness. No native-robot samples entered training. Training budgets, paired batches, source hashes, failed trials and all final checkpoints are retained.

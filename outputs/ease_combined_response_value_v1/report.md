# E136: combined local-response and task-value gate

## Frozen candidate

E120 is unchanged. At the first clean task-phase entry, require both a scalar-calibrated history-Jacobian prediction of >10 micrometers one-step goal progress and a regularized linear paired task-advantage prediction >0.05. If accepted, disable the E120 residual for up to 8 actions within that phase, then resume E120. Otherwise retain E120. At most one intervention. No physics selector, privileged object/contact state, robot IDs, or future observations are gate inputs.

The response scalar is fitted to measured paired Cartesian displacement; the 32-feature ridge model is fitted to final paired success differences. Four family-excluded fitting models were used for validation. Final models fit all fitting rows only. The prespecified primary conjunction also won the 20-rule validation grid. All weights, thresholds, source files and protocol were hashed before fresh fitting rollouts and held evaluation.

## Results

| Split | E120 | Candidate | Rescues | Lost successes |
|---|---:|---:|---:|---:|
| fitting | 97/144 | 102/144 | 5 | 0 |
| held | 15/24 | 15/24 | 1 | 1 |

| Split | Joints | E120 successes | Candidate successes | E120/candidate collisions |
|---|---:|---:|---:|---:|
| fitting | 6 | 54/72 | 54/72 | 16/16 |
| fitting | 7 | 43/72 | 48/72 | 23/18 |
| held | 6 | 11/12 | 11/12 | 1/1 |
| held | 7 | 4/12 | 4/12 | 8/8 |

## Interpretation and limits

All 144 fresh fitting trajectories exactly reproduced the previously executed selected counterfactuals, including full state arrays and task traces. Held episodes were evaluated only after this confirmation, with the frozen all-fitting models. The original held placements have been used in earlier development; they are not a fresh independent final test. All bodies here are known training bodies, so this is placement/family generalization, not new evidence of unseen-embodiment transfer.

This conjunction was motivated by earlier adaptive experiments on these fitting families. Even though its primary rule succeeded before threshold selection, the development history matters. Small paired gains are preliminary, and seeds sharing a placement/body are dependent. Do not present this as a statistically established paper result or an end-to-end flow-matching policy.

Earlier iterations are retained: E133 improved one-step prediction but its eight-step gate lost 8 successes; E134 one-step execution lost 10; E135 compact task value lost 4. E136 combines the two signals, rather than claiming either alone solves the task.

All full trials/actions/states and checkpoints are saved. Media include every changed outcome pair and representative unchanged success/failure cases from both joint counts and both splits. Videos replay executed state traces.

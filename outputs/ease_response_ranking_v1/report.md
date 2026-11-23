# E60: held-out response models rank counterfactual actions

Use immutable E59 leave-one-body-out folds: no response fit or normalization includes this body. Score original candidate commands without changing actions. Full history and command/velocity residual predictions combine with geometric or estimated Jacobians. Geometry is diagnostic only. No native data, tuning or policy training. Drivers/estimators previously saw these bodies; holdout is for response fitting only.

288 fixed states, 2,880 five-step branches; all branch records and original scores exactly reproduce E58. All 32 full driver trajectories and 288 matching-driver first steps replay exactly. No actions change; new predictors score the same alternatives.

## Body-macro pairwise agreement

| DoF | Horizon | Bias + true J | Command/velocity + true J | History + true J | Bias + estimated J | History + estimated J | Realized motion + true J |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6dof | 1 | 78.24% | 80.00% | 88.59% | 58.52% | 57.78% | 99.41% |
| 6dof | 5 | 63.45% | 61.22% | 62.91% | 56.86% | 55.57% | 63.85% |
| 7dof | 1 | 76.38% | 71.88% | 88.50% | 51.85% | 55.69% | 99.48% |
| 7dof | 5 | 66.18% | 60.54% | 69.86% | 50.99% | 56.31% | 68.79% |

True J and realized motion are diagnostic oracles. Estimated J is each candidate estimator; therefore its scores include estimator error as well as response error. Ties at 1e-12 are excluded. Five-step outcomes use continuing reference ramps and online updates; a first-step prediction does not model this entire rollout.

## Counterfactual joint-response RMSE (rad, body macro)

| DoF | Bias baseline | Command/velocity | History |
|---|---:|---:|---:|
| 6dof | 0.009273 | 0.003895 | 0.002178 |
| 7dof | 0.008617 | 0.005069 | 0.001445 |

## Per-body first-step geometric ranking

| Body | Bias baseline | History predictor |
|---|---:|---:|
| 84000 | 74.07% | 84.63% |
| 84001 | 80.80% | 91.54% |
| 84002 | 73.89% | 86.91% |
| 84003 | 84.18% | 91.29% |
| 84004 | 78.49% | 89.31% |
| 84005 | 74.69% | 89.26% |
| 84006 | 74.38% | 88.21% |
| 84007 | 77.96% | 87.22% |

Contacts: 0/2880 branches. Bodies are omitted from response fitting, but were used upstream by the motion estimators/drivers; this is not a full unseen-embodiment policy test. Histories overlap earlier diagnostics. Eight dependent body clusters, no significance or native manipulation claim. Preserve E35; no policy training or promotion.

## Interpretation and decision

The held-out response predictor improves one-step geometric action ranking on all eight bodies: 78.24% to 88.59% for six joints, 76.38% to 88.50% for seven. Counterfactual joint-motion RMSE also falls from 0.009273 to 0.002178 rad and from 0.008617 to 0.001445 rad. This extends E59 beyond the commands executed by the data-collection drivers. The full predictor includes position/bias as well as history, so these gains do not isolate history as a causal factor.

The deployable estimated-J composition is not reliably better: six-joint first-step ranking declines from 58.52% to 57.78%, while seven-joint ranking rises from 51.85% to 55.69%. Five-step true-J ranking declines slightly for six joints (63.45% to 62.91%) and improves for seven (66.18% to 69.86%). Thus a better one-step response surrogate does not establish a uniformly better long-horizon objective or controller. Candidate-specific estimated Jacobians also differ between scores; this is a test of the composed score, not an isolated causal measurement of Jacobian error.

Keep E35 and leave policy training unchanged in this experiment. A bounded next experiment can compare a fixed response-corrected training objective against its matched uncorrected objective, with cross-fitted procedural response models and the original reaching/six-joint/native retention gates. Do not deploy the estimated-J score as an action selector on this evidence. No positive manipulation or independent unseen-embodiment claim follows.

All 83 regression tests pass. Independent verification confirms all 2,880 E58 branches and original scores, 576 matching-driver response predictions against saved E59 held-body predictions, identical trial archives, and unchanged nine policy checkpoints and sixteen response fits. Full traces and failures are retained. No new weights were fitted and the final test is untouched.

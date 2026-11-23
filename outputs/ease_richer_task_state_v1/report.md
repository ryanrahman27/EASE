# E130: Richer measured task state with the two-head objective

E120 remains the baseline. The richer measured-state model gains two and loses two E120 fitting-validation episodes under the fixed rule. Equal aggregate success is not preservation of successful episodes. Both zero-loss-selected rules abstain.

## Controlled feature ablation

**design:** Same E126 two-success-head BCE objective,4 ancestral fitting-family folds,3 seeds126100..126102,1500 full-batch AdamW updates, lr.001 weight_decay.001, no weighting/resampling/model selection. Same64/32 tanh hidden layers and2 output logits. Input grows139 to162.

**initialization:** Copy original139-column first-layer weights, all original biases, hidden and output weights from the same-seed baseline initialization. Added23 input columns start at zero. Thus existing pathways and initial behavior match; only new input weights/capacity are added.

**features:** Append tool position3, tool orientation error3, latest measured joint increment padded7, latest measured Cartesian/rotation increment6, two gripper encoder positions, two preceding gripper actuator commands. Reconstruct from the pre-action E120-on integration/history snapshot. No post-action or future state.

**observability:** Ideal noise-free simulated observations. Tool pose and joint history already available to controller; gripper positions model encoder readings, prior commands are known controller signals. No object pose, contact/grasp/lift/bad flags, object velocity, privileged Jacobian, or physical parameters. Gripper sensing is not a tested hardware sensor pipeline.

**normalization:** Original139 columns retain exact original fold-local normalization; added23 mean/std from that fold training families only, scale floor1e-6; all features clipped to[-10,10].

**primary:** Same predeclared minimum-advantage0.1/max-spread2 first-clean-phase rule on144 fitting-placement/estimator episodes. Secondary unchanged E126 grid and zero-loss selection. No original held-placement or unseen-body evaluation.

**outcomes:** Reconstruct success using existing exact paired continuations: first accepted clean phase gets up-to8 off-pulse steps, then E120 resumes. Post-collision starts omitted because success cannot recover. No fresh online-policy rollout or post-collision trajectory equivalence claim.

**baseline:** E120 unchanged; no promotion on fitting-only results. Same success/failure-video provenance workflow.

The original139 inputs are unchanged. Twenty-three extra columns expand the input layer; hidden widths64/32, two success outputs and paired BCE objective remain fixed. The new input columns start at zero; all original weights and biases match the same-seed baseline initialization. No class weighting, resampling, checkpoint selection, or training-data expansion.

Four fixed ancestral-family folds hold out whole original/shifted/descendant families, across all phases and estimator seeds. Twelve gate models are trained: three seeds per fold. Both models use1500 full-batch AdamW updates at lr0.001, weight decay0.001. Original normalization is preserved exactly; additional-feature normalization uses only training rows within each fold.

The added gripper signals are ideal simulated encoders and known previous commands. Tool pose and measured interaction history already enter the controller. No object pose, contact/grasp/lift/collision labels, privileged dynamics parameters, or future state are features. This is not a real-hardware sensing validation. Object-centric observability remains untested.

## Feature provenance

| Columns | Signal | Timing/source |
|---|---|---|
| 139–141 | Tool position | Pre-action grasp-center pose |
| 142–144 | Tool orientation error | Pre-action tool rotation versus fixed upright orientation |
| 145–151 | Latest joint increment | Already measured controller history, padded to7 |
| 152–157 | Latest Cartesian/rotation increment | Already measured controller history |
| 158–159 | Left/right gripper positions | Ideal pre-action encoder readings |
| 160–161 | Left/right previous actuator commands | Stored controller commands before current decision |

All1108 rows reconstructed from saved integration/history snapshots before the intervention. The original139-vector is independently reconstructed and checked for exact equality, confirming decision-time alignment. No post-step trace entry supplies a new input.

## Out-of-fold prediction quality

| Model | Advantage MSE | Decisive-pair MSE | Correct decisive sign | Beneficial reversed /65 | Harmful reversed /54 |
|---|---:|---:|---:|---:|---:|
| two_head | 0.1996 | 1.1226 | 50.4% | 34 | 25 |
| richer | 0.2020 | 1.0743 | 52.1% | 28 | 29 |

| Model | On/off accuracy | On/off Brier |
|---|---|---|
| two_head | 0.6029/0.5975 | 0.3472/0.3518 |
| richer | 0.6065/0.5984 | 0.3448/0.3489 |

There are989 neutral and119 decisive pairs. These phase-entry rows share trajectories and are not independent episodes. The small decisive-sign improvement does not establish generalizable task improvement.

## Fixed-rule complete-task validation

Same prespecified minimum advantage0.1 and maximum inter-model spread2.0, requiring all three model advantages to pass. At most one short off pulse. These144 expanded fitting-placement/estimator outcomes are reconstructed from saved exact counterfactual continuations, not fresh online-policy rollouts. They are a different set from the original72-episode held-placement benchmark.

| Method | Success /144 | Gains vs E120 | Losses vs E120 | Switches |
|---|---:|---:|---:|---:|
| E120 | 97 | 0 | 0 | 0 |
| two_head | 96 | 1 | 2 | 23 |
| richer | 97 | 2 | 2 | 33 |

| Richer-state case/seed | Selected phase | E120 success | Richer success |
|---|---|---|---|
| 0/0 | lift | True | False |
| 22/2 | descend | False | True |
| 35/0 | descend | True | False |
| 38/2 | descend | False | True |

Richer state rescues22/2 and38/2, but loses0/0 and35/0. It retains the known35/0 harmful descent decision and introduces a lift regression at0/0. These fitting-validation cases do not support promotion over E120.

## Fold and seed variability

| Model | Fold | Advantage MSE | Correct decisive sign |
|---|---:|---:|---:|
| two_head | 0 | 0.3244 | 48.0% |
| two_head | 1 | 0.2345 | 52.0% |
| two_head | 2 | 0.1582 | 53.6% |
| two_head | 3 | 0.1331 | 50.0% |
| richer | 0 | 0.3121 | 54.0% |
| richer | 1 | 0.1983 | 68.0% |
| richer | 2 | 0.2118 | 39.3% |
| richer | 3 | 0.1296 | 43.8% |

| Model | Gate seed | Advantage MSE | Correct decisive sign |
|---|---:|---:|---:|
| two_head | 126100 | 0.2415 | 47.1% |
| two_head | 126101 | 0.2860 | 49.6% |
| two_head | 126102 | 0.2547 | 58.8% |
| richer | 126100 | 0.2533 | 53.8% |
| richer | 126101 | 0.2698 | 57.1% |
| richer | 126102 | 0.2404 | 58.8% |

## Unchanged secondary selection grid

| Minimum advantage | Maximum spread | Original gains/losses/switches | Richer gains/losses/switches |
|---:|---:|---|---|
| 0.1 | 0.05 | 1/2/4 | 0/0/6 |
| 0.1 | 0.1 | 1/2/6 | 0/0/8 |
| 0.1 | 0.25 | 1/2/7 | 1/1/13 |
| 0.1 | 0.5 | 1/2/13 | 1/2/22 |
| 0.1 | 2.0 | 1/2/23 | 2/2/33 |
| 0.25 | 0.05 | 1/2/4 | 0/0/6 |
| 0.25 | 0.1 | 1/2/6 | 0/0/8 |
| 0.25 | 0.25 | 1/2/7 | 0/1/12 |
| 0.25 | 0.5 | 1/2/13 | 0/2/20 |
| 0.25 | 2.0 | 1/2/19 | 1/3/26 |
| 0.5 | 0.05 | 1/2/4 | 0/0/6 |
| 0.5 | 0.1 | 1/2/6 | 0/0/8 |
| 0.5 | 0.25 | 1/2/7 | 0/1/12 |
| 0.5 | 0.5 | 1/2/13 | 0/2/19 |
| 0.5 | 2.0 | 1/2/13 | 0/2/19 |
| 0.75 | 0.05 | 1/2/4 | 0/0/6 |
| 0.75 | 0.1 | 1/2/6 | 0/0/8 |
| 0.75 | 0.25 | 1/2/7 | 0/1/12 |
| 0.75 | 0.5 | 1/2/7 | 0/1/12 |
| 0.75 | 2.0 | 1/2/7 | 0/1/12 |
| 0.9 | 0.05 | 1/2/4 | 0/0/5 |
| 0.9 | 0.1 | 1/2/6 | 0/0/6 |
| 0.9 | 0.25 | 1/2/6 | 0/0/6 |
| 0.9 | 0.5 | 1/2/6 | 0/0/6 |
| 0.9 | 2.0 | 1/2/6 | 0/0/6 |
| 2.0 | 0.0 | 0/0/0 | 0/0/0 |

Both zero-E120-loss selections choose full abstention. These are tuning-validation scores, not an independent confirmation. No original held-placement evaluation, unseen-body test, or full-fitting deployment model follows this result.

## Interpretation

Additional measured proprioceptive and tool state changes decisions but does not solve the task-value ranking problem. Advantage MSE is essentially unchanged (0.200 to0.202), with a small decisive-sign improvement but no net gain over E120. The earlier hypothesis about object/grasp/contact information has not been tested because those signals are not exposed as measurements in this setup.

A next experiment could explicitly separate an object-state sensing implementation from a simulator-ground-truth upper bound. Any privileged upper bound must be labeled as such. Keep this measured-only result, the two-head objective and fixed family splits as controls. E120 remains unchanged.

Verification: all1108 measured-feature rows independently reconstructed from pre-action snapshots; original139 inputs exact; fitting-only normalization and family isolation checked; both objectives' OOF predictions and all rule outcomes reproduced. Model/data hashes unchanged for E120 and the original gate. Saved17 trace-verified success/failure videos.

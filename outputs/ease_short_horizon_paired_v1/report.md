# E132: Dense short-horizon paired supervision

**PRIVILEGED input study.** The learned short-target model does not beat E120. A future-informed short-target diagnostic shows useful headroom but is not a deployable policy, an achieved learned score, or a guarantee of no losses.

## Controlled task and supervision

**inputs_label:** Same182 privileged E131 inputs, explicitly simulator-only upper-information reference; not measured/deployable sensing.

**target:** At1,2,4,8 steps after identical phase-entry states: (entry-goal distance on minus off), meters, and any forbidden collision during that prefix separately for on/off. Entry goal frozen for all horizons even if the scripted phase advances. At most8-step off pulse then E120 resumes. No final success labels in training.

**dense:** Four continuous paired progress labels plus eight prefix-collision labels per original phase entry. Same1108 fitting rows; no new independent starts or simulator rollouts. Phase transitions within the window are recorded rather than filtered using future information.

**training:** Same four ancestral-family splits,3 seeds126100..126102,1500 full-batch AdamW steps, lr.001 weight_decay.001.182-64-32-12 tanh-hidden network:4 linear standardized progress outputs,8 collision logits. Sum of mean normalized-progress MSE and mean collision BCE, equal task weights. Input normalization fixed to E131 fold; progress mean/std training-fold only, floor1e-6m. No resampling, early stopping or checkpoint selection.

**initialization:** Original139-column and hidden32-layer weights/biases match same-seed baseline; extra43 input columns zero. Output head changes to12 targets; objective and output parameterization change together.

**primary:** Off selected at first clean phase iff all3 predictions show8-step progress benefit >0.00025m and every predicted off collision probability at1/2/4/8 steps is <0.1. At most one pulse. Oracle diagnostic uses actual short progress/collision under the same thresholds; it uses future data and is NOT an inference policy.

**secondary:** Same fixed progress-margin/risk-cap grid for learned short targets. Choose zero E120 final-success losses, most gains, fewer switches, higher progress margin, lower risk cap. Final outcomes used only for tuning-validation rule selection, never model training. Full abstention included.

**evaluation:** 144 fitting placement/estimator full-task successes reconstructed from matched continuations. Compare E120, E131 long-target primary, learned short primary and true-short-target diagnostic. Original held-placement benchmark untouched. No fresh online-policy rollout or post-collision trajectory equivalence claim.

**baseline:** E120 immutable. Short-horizon prediction improvement alone cannot justify promotion.

The original long-target control is E131: same182 privileged features and family splits, with two final-success heads. E132 predicts four local progress differences and eight on/off collision-prefix labels. Thus target horizon, output dimensions and supervised loss change together; no claim is made that one of those components alone causes the result.

Each original clean phase-entry pair now supplies12 labels: four continuous effects and eight collision labels. This gives4432 continuous and8864 binary targets, not13296 independent examples. All on/off continuations start from identical integration/history states. No new rollouts or new placements were collected.

A positive progress target means the off-pulse ends closer to the phase-entry goal than E120-on. The goal remains fixed for this label even if the actual scripted controller advances to another phase. The window is1/2/4/8 control steps (0.04/0.08/0.16/0.32 seconds); E120 resumes after the off-pulse. All branches had sufficient saved horizon.

Rows crossing a phase boundary within each horizon: [0, 0, 0, 31]. No rows are dropped based on this future information. Final task success remains the control-evaluation outcome, separate from the local progress target.

## Out-of-fold short-horizon prediction

| Steps | Nonzero progress pairs (>1 micrometer) | Learned RMSE (mm) | Zero-effect RMSE (mm) | Train-mean RMSE (mm) | Correct nonzero sign |
|---:|---:|---:|---:|---:|---:|
| 1 | 974 | 0.0986 | 0.0507 | 0.0507 | 53.6% |
| 2 | 1069 | 0.6947 | 0.2619 | 0.2618 | 50.3% |
| 4 | 1085 | 3.2808 | 1.7049 | 1.7051 | 50.6% |
| 8 | 1099 | 7.2765 | 3.3578 | 3.3603 | 50.1% |

The zero-effect reference predicts identical on/off progress. The train-mean reference is estimated separately inside each training fold. The learned model is worse than both at all four horizons, despite denser continuous supervision.

| Steps | On/off collision-positive rows | Learned on/off Brier | Train-frequency on/off Brier |
|---:|---|---|---|
| 1 | 2/2 | 0.0618/0.0627 | 0.0018/0.0018 |
| 2 | 5/5 | 0.0664/0.0666 | 0.0045/0.0045 |
| 4 | 12/16 | 0.0788/0.0828 | 0.0109/0.0146 |
| 8 | 23/23 | 0.0909/0.0901 | 0.0210/0.0210 |

Immediate collision labels remain sparse; denser timestamps do not make collision events common. Learned collision probabilities also underperform training-frequency references and should not be interpreted as calibrated safety guarantees.

## Complete-task choices on144 fitting validation episodes

Primary learned rule requires every model's predicted8-step improvement >0.25mm and all predicted off-collision probabilities across horizons/models <0.1. Only the first accepted clean phase receives a pulse. Complete-task successes are reconstructed from previously executed paired continuations. Post-collision entries are omitted because task success is already irreversibly false.

**TRUE-SHORT DIAGNOSTIC:** applies that rule to actual future progress/collision labels. It tests the usefulness of this proxy with perfect short-target knowledge. It is not an inference-time physics selector or a newly trained policy, and its106/144 score must not be presented as learned performance.

| Condition | Success /144 | Rescues vs E120 | Lost successes | Switches |
|---|---:|---:|---:|---:|
| E120 | 97 | 0 | 0 | 0 |
| long_target | 96 | 1 | 2 | 46 |
| learned_short | 96 | 2 | 3 | 63 |
| true_short_diagnostic | 106 | 13 | 4 | 113 |

The true-short diagnostic improves aggregate task success but still loses four E120 successes. Thus there are two separate limitations: the learned short model predicts poorly, and the short proxy itself is not sufficient for the no-loss objective. Neither the learned96/144 nor the diagnostic106/144 justifies replacing E120.

| Learned-short case/seed | Selected phase | E120 success | Learned-short success |
|---|---|---|---|
| 3/1 | approach | True | False |
| 19/0 | lift | True | False |
| 33/2 | lift | False | True |
| 35/0 | close | True | False |
| 50/1 | approach | False | True |

## Predeclared secondary margin/risk grid

| Progress margin (mm) | Risk cap | Success /144 | Gains | Losses | Switches |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.05 | 96 | 3 | 4 | 67 |
| 0.00 | 0.1 | 97 | 4 | 4 | 71 |
| 0.00 | 0.25 | 97 | 4 | 4 | 76 |
| 0.10 | 0.05 | 95 | 2 | 4 | 66 |
| 0.10 | 0.1 | 96 | 3 | 4 | 70 |
| 0.10 | 0.25 | 96 | 3 | 4 | 75 |
| 0.25 | 0.05 | 95 | 1 | 3 | 58 |
| 0.25 | 0.1 | 96 | 2 | 3 | 63 |
| 0.25 | 0.25 | 96 | 2 | 3 | 69 |
| 0.50 | 0.05 | 95 | 1 | 3 | 56 |
| 0.50 | 0.1 | 96 | 2 | 3 | 61 |
| 0.50 | 0.25 | 96 | 2 | 3 | 67 |
| 1.00 | 0.05 | 95 | 1 | 3 | 49 |
| 1.00 | 0.1 | 96 | 2 | 3 | 55 |
| 1.00 | 0.25 | 96 | 2 | 3 | 61 |
| abstain | 0.0 | 97 | 0 | 0 | 0 |

Selected zero-loss rule: `{'progress_margin': 1000000000.0, 'risk_cap': 0.0}`; 97/144, 0 gains/0 losses. It abstains completely. These are fitting-validation selection scores, not independent final-test results.

## Interpretation and next controlled comparison

Shortening the horizon supplies more nonzero paired labels but does not solve out-of-family prediction with this MLP. At the same time, the future-informed diagnostic shows that accurate local predictions could guide some useful choices. This separates a concrete prediction-generalization failure from the additional proxy-alignment problem.

A next controlled comparison should benchmark a structured action-response predictor against this generic MLP at one step before extending the horizon. Keep the same inputs, families and dense labels, report zero-effect and fitting-only references, and continue scoring final-task losses. No such alternative model was trained here.

All inputs remain explicitly privileged simulator data. E120, original held-placement results, and prior checkpoints stay unchanged. No new held-placement, unseen-body, hardware, or sensor-deployment claim is made. Videos are outcome-selected counterfactual continuations: all learned/long-target outcome changes, representative true-short diagnostic changes, and earlier diagnosis pairs.

Verification: all dense labels independently regenerated; all1108 paired prefixes/snapshots exact; fold/target normalizations training-only; all OOF predictions and outcome reconstructions reproduced; E120/prior models unchanged. Saved27 trace-verified success/failure videos.

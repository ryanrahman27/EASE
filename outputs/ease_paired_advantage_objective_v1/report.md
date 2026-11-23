# E129: Direct paired advantage versus two success heads

The tested direct signed-advantage objective generalizes worse than the two-head baseline on the same fitting-family folds. E120 remains unchanged. No original held-placement benchmark or unseen bodies were evaluated.

## Controlled comparison

Four identical ancestral-placement folds,1108 clean paired phase entries,48 fitting placements and three frozen motion-estimator seeds. Each complete source family (original, shifted, and descendants) stays in one fold. Reuse the E126 two-head models; train12 direct models (three gate seeds per fold). All same139 inputs, hidden widths64/32 with tanh, identical hidden initialization by seed, fold-local normalization/clipping,1500 full-batch AdamW updates, lr0.001, weight decay0.001. No class weighting, resampling, additional data, early stopping, or held-driven selection.

Two-head objective: paired binary cross-entropy for final on/off success, with advantage computed as sigmoid(off)-sigmoid(on). Direct objective: one tanh-bounded scalar trained by MSE to observed off-minus-on success (-1,0,+1). The output layer and objective necessarily differ; this comparison does not cover every possible direct-ranking objective or output parameterization.

The primary rule was fixed before training: every gate model predicts advantage >0.1, with inter-model spread <=2.0. First accepted clean phase gets up to8 correction-off steps, then E120 resumes. Both objectives are scored with the same rule. Secondary analysis uses the unchanged E126 threshold/spread grid and zero-E120-loss selection criterion; those selected scores use tuning-validation data.

Complete-task outcomes are reconstructed from previously executed, exact-state paired continuations. Starts already contaminated by collision are omitted because success is irreversibly false. This is not a newly executed online-policy benchmark, and does not assert trajectory equivalence for post-collision decisions. All selected video branches were actually simulated previously. No physics selector is introduced.

The144 validation episodes use the expanded fitting-placement set. Their97/144 E120 baseline must not be compared directly with52/72 on the different original evaluation set. Rows/phases share trajectories and are statistically dependent.

## Out-of-fold prediction quality

| Objective | Advantage MSE | Decisive-pair MSE | Neutral-pair MSE | Correct decisive sign | Beneficial reversed /65 | Harmful reversed /54 |
|---|---:|---:|---:|---:|---:|---:|
| two_head | 0.1996 | 1.1226 | 0.0886 | 50.4% | 34 | 25 |
| direct | 0.5349 | 1.5460 | 0.4133 | 47.9% | 34 | 28 |

A constant zero-advantage reference has MSE0.1074, decisive-pair MSE1.0, and neutral-pair MSE0. It abstains rather than ranking choices. There are989 neutral and119 decisive pairs. Report decisive metrics separately to avoid rewarding only neutral predictions.

## Fixed-rule complete-task validation

| Method | Success /144 | Gains vs E120 | Losses vs E120 | Interventions |
|---|---:|---:|---:|---:|
| E120 | 97 | 0 | 0 | 0 |
| two_head | 96 | 1 | 2 | 23 |
| direct | 94 | 7 | 10 | 107 |

The direct model changes more decisions:107 interventions versus23. It rescues seven E120 failures but breaks ten successes. The two-head model rescues one and breaks two. More rescues alone are not a better policy.

## Every fold and gate seed

| Method | Fold | MSE | Correct decisive sign |
|---|---:|---:|---:|
| two_head | 0 | 0.3244 | 48.0% |
| two_head | 1 | 0.2345 | 52.0% |
| two_head | 2 | 0.1582 | 53.6% |
| two_head | 3 | 0.1331 | 50.0% |
| direct | 0 | 0.5884 | 54.0% |
| direct | 1 | 0.5735 | 44.0% |
| direct | 2 | 0.5081 | 39.3% |
| direct | 3 | 0.4940 | 50.0% |

Direct regression has higher OOF advantage MSE in all four folds.

| Method | Gate seed | MSE | Correct decisive sign |
|---|---:|---:|---:|
| two_head | 126100 | 0.2415 | 47.1% |
| two_head | 126101 | 0.2860 | 49.6% |
| two_head | 126102 | 0.2547 | 58.8% |
| direct | 126100 | 0.7781 | 42.0% |
| direct | 126101 | 0.6748 | 48.7% |
| direct | 126102 | 0.7560 | 48.7% |

| Direct training fold | Last optimization MSE by seed |
|---:|---|
| 0 | 0.00410, 0.00435, 0.00418 |
| 1 | 0.00739, 0.00783, 0.00732 |
| 2 | 0.00780, 0.00666, 0.00739 |
| 3 | 0.00634, 0.01139, 0.00632 |

The direct model fits its training targets closely (last-step MSE about0.004–0.011), while excluded-family MSE is0.535. Switching the objective did not remove the training-to-family generalization gap.

## Secondary identical-grid comparison

| Minimum advantage | Maximum spread | Two-head gains/losses/switches | Direct gains/losses/switches |
|---:|---:|---|---|
| 0.1 | 0.05 | 1/2/4 | 3/5/51 |
| 0.1 | 0.1 | 1/2/6 | 5/9/60 |
| 0.1 | 0.25 | 1/2/7 | 6/9/84 |
| 0.1 | 0.5 | 1/2/13 | 6/9/97 |
| 0.1 | 2.0 | 1/2/23 | 7/10/107 |
| 0.25 | 0.05 | 1/2/4 | 3/5/51 |
| 0.25 | 0.1 | 1/2/6 | 5/9/60 |
| 0.25 | 0.25 | 1/2/7 | 6/9/81 |
| 0.25 | 0.5 | 1/2/13 | 6/9/93 |
| 0.25 | 2.0 | 1/2/19 | 7/10/99 |
| 0.5 | 0.05 | 1/2/4 | 3/5/51 |
| 0.5 | 0.1 | 1/2/6 | 5/9/60 |
| 0.5 | 0.25 | 1/2/7 | 6/9/81 |
| 0.5 | 0.5 | 1/2/13 | 6/9/90 |
| 0.5 | 2.0 | 1/2/13 | 6/9/90 |
| 0.75 | 0.05 | 1/2/4 | 3/5/51 |
| 0.75 | 0.1 | 1/2/6 | 5/9/60 |
| 0.75 | 0.25 | 1/2/7 | 6/10/80 |
| 0.75 | 0.5 | 1/2/7 | 6/10/80 |
| 0.75 | 2.0 | 1/2/7 | 6/10/80 |
| 0.9 | 0.05 | 1/2/4 | 3/5/51 |
| 0.9 | 0.1 | 1/2/6 | 5/8/59 |
| 0.9 | 0.25 | 1/2/6 | 5/8/59 |
| 0.9 | 0.5 | 1/2/6 | 5/8/59 |
| 0.9 | 2.0 | 1/2/6 | 5/8/59 |
| 2.0 | 0.0 | 0/0/0 | 0/0/0 |

Both objectives select full abstention under the zero-loss criterion, giving97/144 and zero rescues. This is not an improvement over E120. No full-fitting final model was promoted or tested on original held placements.

## Interpretation and next controlled test

This direct tanh-MSE formulation makes stronger, less reliable intervention predictions outside its training families. The objective change alone did not fix task-value generalization. The result does not rule out alternative ranking losses, but provides no reason to replace E120 or promote this direct model.

The next separate ablation should test richer measured task-state information under the same family splits while retaining the two-head objective as the comparison. Object-relative pose, grasp/contact state where observable, and orientation error are candidate inputs; simulator-only quantities must be labeled privileged. Do not combine representation changes with a new loss and then attribute the result to one factor.

All outcomes, fold/seed metrics, selected rules, checkpoints, normalization, input hashes and diagnostic videos are saved. Media include every fixed-rule outcome change and the earlier E128 diagnostic pairs. Videos are outcome-selected counterfactual continuations, not an unbiased video sample.

Verification: all1108 labels and both objectives' OOF predictions reproduced; ancestral splits and normalization checked; all rule outcomes reconstructed identically; saved video-pair snapshots/prefixes exact. E120 data and prior two-head checkpoints unchanged. Saved40 trace-verified success/failure videos.

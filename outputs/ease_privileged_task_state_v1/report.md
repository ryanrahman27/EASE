# E131: PRIVILEGED simulator task-state information upper bound

**Privileged simulator ground truth. Not a measured-sensor, real-hardware, or deployable-policy result.** This is an upper bound on the tested information access, not a mathematical upper bound on achievable performance. A finite learned model can fail to exploit extra information.

The privileged variant improves decisive ranking accuracy modestly but does not improve fixed-rule task success over E120. All zero-loss selections abstain. E120 remains unchanged.

## Fixed experiment

**design:** Same two-head BCE,4 ancestral-family fitting folds,3 seeds126100..126102,1500 full-batch AdamW steps, lr.001 weight_decay.001, widths64/32. Input expands measured162 to182. No reweighting/resampling/early stopping or held-driven selection.

**initialization:** Same-seed original139-column weights, biases, hidden and2-output weights copied exactly; all43 extra measured/privileged columns start at zero, matching E130 initial function. Input capacity grows with features.

**features:** Measured162 unchanged. Append20 privileged inputs: object-to-tool position3, world object rotation matrix9, destination-minus-object position3 (task landing height0.215m), instantaneous object contacts with left pad/right pad/table3, pre-action cumulative grasp flag1, instantaneous bilateral finger contact1.

**timing:** All added values use E120-on pre-action integration/history snapshot and same-time branch metadata. Grasp flag is from entry metadata, never final trial success/grasp/lift. No future state, final label, collision/bad flag, physical parameters, or true Jacobian in features.

**normalization:** Measured162 normalization retained exactly per fold from E130. Added20 mean/std fitted only on training families, scale floor1e-6; clipping[-10,10].

**primary:** Same min-advantage0.1/max-spread2 first-clean-phase intervention rule,144 fitting-placement/estimator episodes. Compare original139, measured162 and privileged182. Secondary same E126 zero-loss grid, tuning-validation only.

**outcomes:** Reconstruct success from existing exact paired full-task continuations: one up-to8-step correction-off pulse then E120. Post-collision starts omitted because success irreversibly false. No fresh online-policy rollout or post-collision trajectory equivalence claim.

**baseline:** E120 unchanged. No original held-placement benchmark, unseen bodies, sensing deployment, or promotion from this comparison.

The two-head BCE objective, hidden widths64/32, three seeds,1500 AdamW updates, data weighting, first-phase intervention rule and four ancestral fitting-family folds are unchanged. No original held-placement tuning or evaluation. The input layer grows, so this is a feature-access ablation with corresponding input parameters, not a claim of identical parameter count.

## Privileged input definition and causal timing

| Columns | Simulator-only input | Source |
|---|---|---|
| 162–164 | Object position minus tool position | Pre-action body/site pose |
| 165–173 | World object rotation matrix | Pre-action body orientation |
| 174–176 | Destination position minus object position | Known destination XY and landing height0.215m, pre-action object |
| 177–179 | Left-pad/object, right-pad/object, table/object contacts | Instantaneous contact pairs at the snapshot |
| 180 | Has grasped before this decision | Cumulative grasp flag from phase-entry metadata |
| 181 | Current bilateral pad/object contact | Both instantaneous finger contact flags |

Measured columns0–161 remain exact E130 copies. The extra state is reconstructed from E120-on integration snapshots before the candidate action. It never reads final success, final grasp/lift outcomes, future states, collision/bad flag, privileged Jacobians, or physical parameters into the network. Instantaneous contacts differ from contacts accumulated across a control interval.

Quality checks: all1108 rotations are valid; all contact/grasp values are binary; bilateral contact matches both pad flags; grasp values match pre-action metadata. In964 noninitial phase rows, reconstructed object position matches the previous saved observation. Initial entries use the pre-action snapshot directly. Positive counts for left/right/table/grasp-history/bilateral inputs: [452.0, 457.0, 835.0, 686.0, 451.0].

## Out-of-fold prediction quality

| Model | Advantage MSE | Decisive-pair MSE | Correct decisive sign | Beneficial reversed /65 | Harmful reversed /54 |
|---|---:|---:|---:|---:|---:|
| two_head | 0.1996 | 1.1226 | 50.4% | 34 | 25 |
| measured | 0.2020 | 1.0743 | 52.1% | 28 | 29 |
| privileged | 0.2037 | 1.0680 | 56.3% | 26 | 26 |

| Model | On/off head accuracy | On/off Brier |
|---|---|---|
| two_head | 0.6029/0.5975 | 0.3472/0.3518 |
| measured | 0.6065/0.5984 | 0.3448/0.3489 |
| privileged | 0.6318/0.6029 | 0.3233/0.3420 |

Decisive sign accuracy rises from52.1% measured to56.3% privileged, while total advantage MSE remains around0.20. With119 decisive and989 neutral, dependent phase-pair examples, this modest sign change does not establish robust policy improvement.

## Fixed-rule complete-task outcome reconstruction

Minimum advantage0.1, maximum ensemble spread2.0; all three predictions must pass. First accepted clean phase gets one up-to8-step correction-off pulse, then E120 resumes. The144 expanded fitting-validation episodes differ from the original72 benchmark. Outcomes come from existing matched full-task continuations, not fresh online-policy rollout or a new held benchmark. Post-collision starts are omitted because success cannot recover.

| Method | Success /144 | Gains vs E120 | Losses vs E120 | Switches |
|---|---:|---:|---:|---:|
| E120 | 97 | 0 | 0 | 0 |
| two_head | 96 | 1 | 2 | 23 |
| measured | 97 | 2 | 2 | 33 |
| privileged | 96 | 1 | 2 | 46 |

| Privileged case/seed | Selected phase | E120 success | Privileged success |
|---|---|---|---|
| 1/2 | descend | True | False |
| 7/2 | lift | True | False |
| 50/1 | descend | False | True |

Privileged input rescues50/1 but loses1/2 and7/2. It does not preserve all E120 successes, and it does not retain the measured model's22/2 and38/2 rescues. No baseline replacement is warranted.

## Fold and seed variability

| Model | Fold | MSE | Correct decisive sign |
|---|---:|---:|---:|
| two_head | 0 | 0.3244 | 48.0% |
| two_head | 1 | 0.2345 | 52.0% |
| two_head | 2 | 0.1582 | 53.6% |
| two_head | 3 | 0.1331 | 50.0% |
| measured | 0 | 0.3121 | 54.0% |
| measured | 1 | 0.1983 | 68.0% |
| measured | 2 | 0.2118 | 39.3% |
| measured | 3 | 0.1296 | 43.8% |
| privileged | 0 | 0.3653 | 54.0% |
| privileged | 1 | 0.1631 | 72.0% |
| privileged | 2 | 0.2224 | 46.4% |
| privileged | 3 | 0.1250 | 56.2% |

| Model | Gate seed | MSE | Correct decisive sign |
|---|---:|---:|---:|
| two_head | 126100 | 0.2415 | 47.1% |
| two_head | 126101 | 0.2860 | 49.6% |
| two_head | 126102 | 0.2547 | 58.8% |
| measured | 126100 | 0.2533 | 53.8% |
| measured | 126101 | 0.2698 | 57.1% |
| measured | 126102 | 0.2404 | 58.8% |
| privileged | 126100 | 0.2417 | 46.2% |
| privileged | 126101 | 0.2683 | 56.3% |
| privileged | 126102 | 0.2451 | 58.0% |

## Same secondary selection grid

| Min advantage | Max spread | Original gains/losses/switches | Measured | Privileged |
|---:|---:|---|---|---|
| 0.1 | 0.05 | 1/2/4 | 0/0/6 | 0/0/3 |
| 0.1 | 0.1 | 1/2/6 | 0/0/8 | 0/0/6 |
| 0.1 | 0.25 | 1/2/7 | 1/1/13 | 0/1/18 |
| 0.1 | 0.5 | 1/2/13 | 1/2/22 | 1/1/29 |
| 0.1 | 2.0 | 1/2/23 | 2/2/33 | 1/2/46 |
| 0.25 | 0.05 | 1/2/4 | 0/0/6 | 0/0/3 |
| 0.25 | 0.1 | 1/2/6 | 0/0/8 | 0/0/6 |
| 0.25 | 0.25 | 1/2/7 | 0/1/12 | 0/1/18 |
| 0.25 | 0.5 | 1/2/13 | 0/2/20 | 1/1/28 |
| 0.25 | 2.0 | 1/2/19 | 1/3/26 | 1/1/35 |
| 0.5 | 0.05 | 1/2/4 | 0/0/6 | 0/0/3 |
| 0.5 | 0.1 | 1/2/6 | 0/0/8 | 0/0/6 |
| 0.5 | 0.25 | 1/2/7 | 0/1/12 | 0/1/18 |
| 0.5 | 0.5 | 1/2/13 | 0/2/19 | 1/1/27 |
| 0.5 | 2.0 | 1/2/13 | 0/2/19 | 1/1/27 |
| 0.75 | 0.05 | 1/2/4 | 0/0/6 | 0/0/3 |
| 0.75 | 0.1 | 1/2/6 | 0/0/8 | 0/0/6 |
| 0.75 | 0.25 | 1/2/7 | 0/1/12 | 0/1/17 |
| 0.75 | 0.5 | 1/2/7 | 0/1/12 | 0/1/17 |
| 0.75 | 2.0 | 1/2/7 | 0/1/12 | 0/1/17 |
| 0.9 | 0.05 | 1/2/4 | 0/0/5 | 0/0/3 |
| 0.9 | 0.1 | 1/2/6 | 0/0/6 | 0/0/4 |
| 0.9 | 0.25 | 1/2/6 | 0/0/6 | 0/0/4 |
| 0.9 | 0.5 | 1/2/6 | 0/0/6 | 0/0/4 |
| 0.9 | 2.0 | 1/2/6 | 0/0/6 | 0/0/4 |
| 2.0 | 0.0 | 0/0/0 | 0/0/0 | 0/0/0 |

All three zero-E120-loss selections choose full abstention. Selected scores use fitting validation for tuning and are not an independent final test. There is no additional full-fitting model, original held-placement retest, or deployment.

## Interpretation

Providing exact object pose and contact/grasp state is insufficient to fix this gate under the tested architecture, objective, data and training budget. The result weakens the hypothesis that missing these inputs alone explains the failures. It does not prove that object state is unimportant or establish a performance ceiling for privileged control.

Before further sensing or confidence-rule work, test whether the value learner can generalize a simpler target with denser paired supervision under the same family splits. Keep information access explicit and change one factor at a time. Preserve E120 as the task-success baseline.

Verification: all1108 privileged rows independently reconstructed; measured162 unchanged; valid rotations and binary contact flags; pre-action grasp metadata and previous-observation pose timing checked. Fitting-only normalization, ancestral isolation, model predictions and outcome reconstruction verified. E120 and both control models/data unchanged. Saved22 trace-verified videos explicitly labeled PRIVILEGED.

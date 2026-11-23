# E128: Task-value ranking errors on excluded fitting families

Diagnosis only: no policy, estimator, confidence rule or held-placement score changed. E120 stays the baseline. These are out-of-fold predictions on fitting families, not the original held-placement benchmark.

## Prediction errors

1108 clean phase-entry pairs: 65 beneficial off-pulses, 54 harmful, 989 unchanged. Phase rows share trajectories and are not independent trials.

Mean-ensemble on-head accuracy 0.603, off-head accuracy 0.597; Brier scores 0.347 and0.352. Rankings reverse on 34/65 beneficial and 25/54 harmful pairs. Consensus advantage >0.1 accepts 1 beneficial and 2 harmful phase pairs. These pair counts differ from first-choice complete-episode results.

The fixed E126 first-choice rule (minimum advantage0.1, spread2) gains one episode and loses two among144 fitting validation episodes. Its complete-episode choices, rather than arbitrary later phase rows, are included below.

| Fold | Training head accuracy (three models) | OOF on accuracy | OOF off accuracy |
|---:|---:|---:|---:|
| 0 | 1.0000 | 0.5648 | 0.5751 |
| 1 | 1.0000 | 0.6829 | 0.6551 |
| 2 | 1.0000 | 0.8632 | 0.8599 |
| 3 | 1.0000 | 0.3053 | 0.3084 |

## Paired diagnostic cases

Each off branch starts from an identical integration/history snapshot, disables the bounded residual for up to8 steps, and then resumes E120. Videos show both complete continuations. These are outcome-selected examples.

| Case/seed | Phase | On/off success | Predicted on/off (mean) | First command difference (rad) | On/off first collision phase |
|---|---|---|---|---:|---|
| 35/0 | descend | 1/0 | 0.0006/0.9747 | 0.000118 | none/lower |
| 39/0 | release | 1/0 | 0.0103/0.9990 | 0.000437 | none/retreat |
| 50/1 | descend | 0/1 | 0.0100/0.9815 | 0.001000 | retreat/none |
| 24/0 | release | 0/1 | 0.9986/0.0733 | 0.000450 | retreat/none |
| 25/0 | release | 1/0 | 0.9982/0.0008 | 0.000049 | none/retreat |

## Training neighborhoods and input sensitivity

Neighbors come only from that model's training families, with matching phase, joint count and local-history flag. Distances use fold-normalized, clipped features. Full, history-only, and state/action neighborhoods are saved separately. The three returned neighbors are from distinct ancestral families; labels do not determine neighbor selection. Feature swaps below use the nearest full-feature neighbor and are neural sensitivity probes, not physical counterfactuals or causal explanations.

### Case35, seed0, descend

Reasons: first-choice outcome change. Observed labels [1.0, 0.0]; predicted mean advantage 0.9741.

| Neighborhood | Neighbor case/seed | Observed on/off | Distance |
|---|---|---|---:|
| full | 54/0 | 0/0 | 2.4492 |
| full | 36/1 | 1/1 | 2.4536 |
| full | 49/2 | 0/1 | 2.5441 |
| history | 54/0 | 0/0 | 2.8044 |
| history | 49/2 | 0/1 | 2.8613 |
| history | 36/1 | 1/1 | 2.8945 |
| state_action | 53/1 | 1/1 | 1.4414 |
| state_action | 38/1 | 0/0 | 1.7178 |
| state_action | 48/2 | 1/1 | 2.0980 |

| Replaced input block | New predicted advantage | Change |
|---|---:|---:|
| joint_position | 0.2697 | -0.7044 |
| bias | 0.8829 | -0.0912 |
| goal_error | 0.9801 | +0.0059 |
| history | 0.4085 | -0.5656 |
| proposed_actions | 0.9728 | -0.0013 |
### Case39, seed0, release

Reasons: first-choice outcome change, strongest harmful ranking reversal. Observed labels [1.0, 0.0]; predicted mean advantage 0.9887.

| Neighborhood | Neighbor case/seed | Observed on/off | Distance |
|---|---|---|---:|
| full | 52/1 | 1/1 | 1.8537 |
| full | 35/0 | 1/0 | 2.0071 |
| full | 33/0 | 1/1 | 2.0486 |
| history | 52/1 | 1/1 | 2.1720 |
| history | 35/0 | 1/0 | 2.4029 |
| history | 33/0 | 1/1 | 2.4448 |
| state_action | 50/1 | 0/0 | 0.8937 |
| state_action | 49/0 | 1/1 | 1.0066 |
| state_action | 37/1 | 1/1 | 1.1373 |

| Replaced input block | New predicted advantage | Change |
|---|---:|---:|
| joint_position | 0.3326 | -0.6561 |
| bias | 0.6469 | -0.3417 |
| goal_error | 0.9891 | +0.0004 |
| history | -0.2711 | -1.2597 |
| proposed_actions | 0.7352 | -0.2535 |
### Case50, seed1, descend

Reasons: first-choice outcome change, strongest correctly ranked beneficial pulse. Observed labels [0.0, 1.0]; predicted mean advantage 0.9715.

| Neighborhood | Neighbor case/seed | Observed on/off | Distance |
|---|---|---|---:|
| full | 39/2 | 0/0 | 2.9666 |
| full | 37/2 | 1/1 | 2.9858 |
| full | 49/2 | 0/1 | 3.1381 |
| history | 39/2 | 0/0 | 3.5439 |
| history | 37/2 | 1/1 | 3.6013 |
| history | 49/2 | 0/1 | 3.6925 |
| state_action | 37/2 | 1/1 | 1.3791 |
| state_action | 39/0 | 1/0 | 1.5047 |
| state_action | 49/1 | 0/1 | 1.9256 |

| Replaced input block | New predicted advantage | Change |
|---|---:|---:|
| joint_position | 0.7456 | -0.2259 |
| bias | 0.9154 | -0.0561 |
| goal_error | 0.9733 | +0.0018 |
| history | 0.3435 | -0.6280 |
| proposed_actions | 0.9832 | +0.0116 |
### Case24, seed0, release

Reasons: strongest beneficial ranking reversal. Observed labels [0.0, 1.0]; predicted mean advantage -0.9253.

| Neighborhood | Neighbor case/seed | Observed on/off | Distance |
|---|---|---|---:|
| full | 47/1 | 0/1 | 1.5277 |
| full | 44/1 | 1/1 | 1.5442 |
| full | 1/2 | 1/1 | 1.8336 |
| history | 47/1 | 0/1 | 1.7982 |
| history | 44/1 | 1/1 | 1.8503 |
| history | 1/2 | 1/1 | 2.1762 |
| state_action | 3/0 | 1/1 | 0.7649 |
| state_action | 4/1 | 1/1 | 0.7920 |
| state_action | 1/0 | 1/0 | 1.0819 |

| Replaced input block | New predicted advantage | Change |
|---|---:|---:|
| joint_position | -0.9919 | -0.0666 |
| bias | -0.9586 | -0.0333 |
| goal_error | -0.9427 | -0.0173 |
| history | 0.7352 | +1.6605 |
| proposed_actions | -0.8702 | +0.0551 |
### Case25, seed0, release

Reasons: strongest correctly ranked harmful pulse. Observed labels [1.0, 0.0]; predicted mean advantage -0.9973.

| Neighborhood | Neighbor case/seed | Observed on/off | Distance |
|---|---|---|---:|
| full | 47/1 | 0/1 | 1.4802 |
| full | 44/1 | 1/1 | 1.4911 |
| full | 1/2 | 1/1 | 1.7148 |
| history | 47/1 | 0/1 | 1.7495 |
| history | 44/1 | 1/1 | 1.7822 |
| history | 1/2 | 1/1 | 2.0271 |
| state_action | 16/1 | 1/1 | 0.7752 |
| state_action | 44/1 | 1/1 | 0.8039 |
| state_action | 1/2 | 1/1 | 1.0595 |

| Replaced input block | New predicted advantage | Change |
|---|---:|---:|
| joint_position | -0.9695 | +0.0279 |
| bias | -0.9964 | +0.0009 |
| goal_error | -0.9972 | +0.0001 |
| history | 0.8128 | +1.8101 |
| proposed_actions | -0.9578 | +0.0396 |

## What the gate observes

The139 input features contain phase, joint mask, measured q/bias, Cartesian goal error, Gram/cross history statistics, two proposed joint actions, and local-history flag. They do not explicitly include object pose, grasp state, gripper contact, or tool orientation error. Saved simulation traces include some of this information for diagnosis. Its omission is an architectural limitation and a hypothesis for further testing; this experiment does not prove that adding it fixes ranking errors.

Do not infer policy improvement from sensitivity probes or selected videos. No new gate was fitted and no confidence threshold was selected. Before another abstention heuristic, test whether the value representation can predict paired outcomes on whole excluded families, including the errors of both heads. E120 remains unchanged.

## Class-frequency reference and delayed effects

Training-fold outcome-frequency references achieve on/off accuracy 0.788/0.798 and Brier 0.186/0.178. Neural OOF accuracy is0.603/0.597 with Brier0.347/0.352. These references use no validation labels for estimation and are not proposed control policies.

| Fold | Train on/off success frequency | Excluded-family on/off frequency |
|---:|---|---|
| 0 | [0.8163934350013733, 0.8306010961532593] | [0.6528497338294983, 0.6424870491027832] |
| 1 | [0.8550547957420349, 0.8574908375740051] | [0.595818817615509, 0.6271777153015137] |
| 2 | [0.7415730357170105, 0.750312089920044] | [0.9087947607040405, 0.9218240976333618] |
| 3 | [0.7318932414054871, 0.7458704113960266] | [0.9252336621284485, 0.9252336621284485] |

| Case/seed | Pulse phase | Failed branch | First collision phase | Steps after pulse entry | Contact pair |
|---|---|---|---|---:|---|
| 35/0 | descend | off_descend_0 | lower | 221 | [['gripper_left_pad', 'table']] |
| 39/0 | release | off_release_0 | retreat | 110 | [['6', 'gripper_palm']] |
| 50/1 | descend | on_1 | retreat | 246 | [['gripper_left_pad', 'table']] |
| 24/0 | release | on_0 | retreat | 44 | [['gripper_left_pad', 'table']] |
| 25/0 | release | off_release_0 | retreat | 97 | [['gripper_right_pad', 'table']] |

## Diagnosis and next controlled experiment

1. There is a large training-to-excluded-family prediction gap, including confidently wrong predictions from both value heads. All training heads fit their labels perfectly; excluded-family prediction quality and ranking collapse. More confidence filtering cannot repair a confidently wrong value representation by itself.
2. Most phase-pair labels (989/1108) have identical on/off outcomes. Only119 directly distinguish the two choices. This is sparse ranking supervision, with dependent repeated phase labels; it is not1108 independent intervention experiments.
3. The selected physical outcomes are delayed collision changes,44 to246 control steps after pulse entry. Nearest training neighborhoods include mixed outcomes. Single-block swaps show sensitivity to history and joint configuration, but do not establish a physical cause.
4. A next controlled fitting-family experiment should compare direct paired-advantage supervision against the two-success-head objective, and separately test additional measured task-state inputs (object-relative pose and grasp/contact state where observable). Keep the same ancestral splits and E120 fallback. No such model or new rule was trained in E128.

Additional audit: all1108 labels checked against saved collection outcomes; all five selected raw feature vectors exactly reconstructed from intervention snapshots; 3470 on/off action steps match the defined bounded-E120/pulse rule.

Verification: all OOF predictions reproduced, fold ancestry disjoint, fitting-only normalization checked, all selected intervention snapshots/prefixes exact. Source weights, data and E120 records unchanged. Saved10 trace-verified success/failure videos.

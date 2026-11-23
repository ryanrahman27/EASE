# E182 | How online history updates changed pick-and-place outcomes

Retrospective analysis of all six E181 placements with a success disagreement among the three estimators. On 8,321 histories from 18 actual episodes, each estimator was given the same current state, goal and controller context. Online statistics were rebuilt from the source trajectory's measured transitions, including on frozen-policy trajectories. The frozen candidate used its calibration history throughout. We checked the original estimator prediction, protected action, next physical state and forbidden contacts at every step. The E180 recovery rule and all checkpoints were held fixed.

## Where the outcomes changed

| Case | Online E120 | Frozen E120 | Analytical | Observed difference |
|---|---|---|---|---|
| 6 | Success | Failure | Success | Close-entry tool error was 1.8 mm online and 343 mm frozen. Frozen E120 spent all 12 alignment commands and aborted. |
| 38 | Success | Failure | Success | Online E120 and analytical accepted recovery routes and reached close entry within 7.5 mm. Frozen E120 reached close entry 360 mm away and exhausted alignment. |
| 43 | Success | Failure | Success | All three passed readiness and sustained two-pad contact through closing. Only frozen E120 failed to register a lift. |
| 25 | Failure | Success | Success | Online E120 displaced the cube during descent and subsequently failed the object-drift check. |
| 9 | Failure | Failure | Success | Cube drift at close entry was 29.9 mm online and 54.0 mm frozen. Analytical preserved the initial cube position. |
| 35 | Success | Success | Failure | Analytical reached close entry 34.3 mm from the tool goal and exhausted 12 alignment commands. Both E120 variants secured the grasp. |

## Case 25

The online and frozen E120 executed commands first differed at post-calibration step 1 (zero based). The online approach reached the goal. Descent finished with 2.8 mm tool error and 15.6 mm horizontal cube drift. By close entry the drift was 16.9 mm, exceeding the 8 mm recovery limit. No route was accepted, alignment was locked out and the robot aborted without a forbidden contact. Both comparison trajectories kept the cube at its starting XY position through the probe and then scored successful placements.

On the same 24 online descent histories, reference-Jacobian position-change RMSE was 2.75 mm online, 3.60 mm frozen and 0.84 mm analytical. The respective protected actions made 3.58, 5.80 and 4.33 mm of average one-step progress toward the tool goal. RMSE uses measured joint displacement and observed tool displacement. It measures local kinematic prediction accuracy. The branch progress measures the effect of applying the alternative protected joint command for 40 ms.

At step 24, cube drift was already 5.44 mm. From this exact state, one online command moved the cube another 4.52 mm, the frozen command moved it 6.87 mm and analytical moved it 4.68 mm. The successful comparison episodes approached along different trajectories. Substituting their actions at this contact state also moved the cube. The local branches identify a path-dependent failure and do not establish which earlier multi-step intervention would prevent it.

## Matched descent comparison on each online E120 trajectory

| Case | Position-change RMSE in mm, online / frozen / analytical | Mean one-step tool progress in mm, same order |
|---|---:|---:|
| 25 | 2.75 / 3.60 / 0.84 | 3.58 / 5.80 / 4.33 |
| 6 | 1.31 / 4.27 / 0.40 | 5.39 / 5.62 / 5.53 |
| 38 | 5.40 / 9.88 / 5.48 | 0.31 / 0.00 / -0.03 |
| 43 | 1.42 / 2.38 / 0.66 | 3.79 / 4.00 / 4.49 |
| 9 | 1.02 / 2.11 / 0.79 | 0.34 / 1.04 / 3.35 |
| 35 | 0.73 / 1.77 / 0.30 | 1.35 / 1.04 / 0.55 |

These cases were selected from the completed cohort by outcome. The original prospective scores remain 34/48 online E120, 32/48 frozen E120 and 35/48 analytical. The six cases distinguish grasp acquisition, lift retention, and object displacement as separate failure mechanisms. Per-phase metrics cover every source trajectory. Counterfactual branches inherit its route, aperture and guard decisions for that step.

[Per-phase data](per_phase.json) · [Case 25 contact audit](case25_contact.json) · [First action differences](first_online_frozen_command_difference.json)

# E59: held-out-body joint-response prediction

Eight leave-one-body-out folds, no tuning. Fit on analytical/original_0/original_1 from seven other bodies; original_2 trajectories from those bodies form separate trajectory validation. All four drivers of the omitted body are body validation. Training normalization only. No selected checkpoint or refit on evaluation rows.

Shared per-joint linear ridge residual correction to command-minus-bias, fixed lambda .001 in mean-squared objective. Inputs: current limited command, pre-step qvel, q, sin(q), cos(q), measured bias, previous command and previous realized dq. No body/joint ID, geometry, gains, Jacobian or future state. First history inputs zero. Baselines: command-minus-bias and a command/velocity-only residual ridge.

32 exact E55 analytical/E35 trajectories, all 150 transitions, eight procedural training bodies. No native data or benchmark final-test use. Body holdout applies to response fitting only: upstream E35 drivers previously trained on these bodies.

| Held-out body | DoF | Joint rows | Bias baseline RMSE (rad) | Command/velocity RMSE | History RMSE |
|---|---:|---:|---:|---:|---:|
| 84000 | 6 | 3600 | 0.004668 | 0.003185 | 0.000629 |
| 84001 | 6 | 3600 | 0.005321 | 0.003676 | 0.000625 |
| 84002 | 6 | 3600 | 0.004417 | 0.003170 | 0.001244 |
| 84003 | 6 | 3600 | 0.005648 | 0.003725 | 0.001677 |
| 84004 | 7 | 4200 | 0.004229 | 0.003342 | 0.000662 |
| 84005 | 7 | 4200 | 0.006001 | 0.004290 | 0.000856 |
| 84006 | 7 | 4200 | 0.005117 | 0.004195 | 0.000773 |
| 84007 | 7 | 4200 | 0.008522 | 0.006438 | 0.001574 |

One-step regression on driver trajectories, no counterfactual actions, no task-success or closed-loop stability claim. All transitions retained including contacts. Fold estimates share training bodies and are dependent.

The held-out body excludes all its trajectories from fitting. Other-body original_2 trajectories are also excluded; their metrics are retained in scores.json. Eight bodies provide few independent units, and no novel axis-family holdout is claimed. Logged executed commands are float32; velocities, positions and realized displacement are measured in float64. No policy/controller changes or promotion.

## Audited aggregate results

Body-macro mean RMSE; relative reduction compares these means, not independent joint-level samples.

| DoF | Bias baseline | Command/velocity | History | History reduction versus baseline |
|---|---:|---:|---:|---:|
| 6 | 0.005013 | 0.003439 | 0.001044 | 79.2% |
| 7 | 0.005967 | 0.004567 | 0.000966 | 83.8% |

History beats the bias baseline in 32/32 held-out body/driver strata. All 16 saved folds reproduce using only permitted fit rows; all 32 body/trajectory scores and saved body predictions match recomputation. No data from the omitted body enters normalization or fitting.

83 regression tests pass. One initial test demanded bitwise equality across BLAS batch shapes; it was corrected to 1e-14 tolerance after a 1.4e-17 roundoff difference. Experiment code and fits were unchanged.

Decision: promising response prediction, no policy promotion. The history model also includes pose and bias, so this comparison does not isolate history as the source of improvement. Next validate its counterfactual action rankings from cloned simulator states on held-out bodies, before optimizing actions or using it as a policy-training loss. Preserve E35 and keep native data diagnostic-only.

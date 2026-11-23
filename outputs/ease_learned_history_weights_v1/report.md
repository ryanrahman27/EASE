# E141: learned history weights for kinematic estimation

This is a prediction-only study, not a new task-success result. The learned model predicts a local motion map from 32 past measured joint/tool transitions. For supervised evaluation, that map is multiplied by the actual next joint increment to predict tool displacement. The future joint increment is a query used in the loss, not an input to the history-weight network. This tests kinematic estimation, not complete command-to-motion forecasting.

The Transformer outputs positive transition weights inside an explicit least-squares solve. It is a learned/analytical hybrid, distinct from the older inverse-free MotionTransformer. No object/contact truth, body IDs or privileged Jacobian enters its input. Four fitting-family folds and three fixed seeds; no final task-success labels or checkpoint selection.

| Estimator | Position RMSE (mm) | Scaled orientation RMSE (mm) |
|---|---:|---:|
| uniform32 | 1.4681 | 0.5348 |
| analytical | 1.3923 | 0.5246 |
| learned | 0.9041 | 0.3483 |
| zero | 4.0273 | 1.3944 |
| axis_scaled_uniform | 1.4321 | 0.5271 |
| tuned_constant_kernel | 0.8739 | 0.3542 |

Orientation error is measured in ROT_SCALE times the tool-up-vector increment, expressed in millimeter-equivalent units; it is not an angle in millimeters. Paired queries share histories and are not independent episodes.

The learned model reduces position MSE by about62% versus uniform32 and improves both joint counts. However, a stronger training-fold-tuned constant kernel reaches0.874mm, slightly better than learned0.904mm. Axis scaling alone does not explain the improvement. These reference controls were added after the primary result and are labeled diagnostic; their parameters use fitting folds only, never validation targets.

All four reference folds selected forgetting1 and ridge1e-7, the low-ridge boundary. This motivates E142: expand that boundary and initialize learned, mass-normalized relative weights at the stronger kernel. Do not credit the constant-kernel gain to attention. No controller, held-placement or fresh-test rollout was performed in E141.

Independent verification reproduced all predictions exactly, checked family exclusions and training-only scales, and confirmed zero-initialized weights reproduce uniform32. Source data and E120 are unchanged. Videos show recorded E120 success/failure trajectories supplying the prediction histories; they are NOT demonstrations of learned-estimator control.

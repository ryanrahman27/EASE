# E150: observed-only versus dense directional fine-tuning

The same738 histories,96 fitting placements, four ancestry-family folds and three initial E146 checkpoints are used in both conditions. Continued observed-only learning controls for extra updates and context selection; the primary mixed objective adds equal-weight measured signed-joint probes. All final seeds are retained and averaged. No exact-Jacobian training labels.

| Condition | Observed-next position RMSE (mm) | Directional position RMSE (mm) | Directional rotation RMSE (mm) |
|---|---:|---:|---:|
| E146 | 0.8601 | 0.1768 | 0.0775 |
| observed | 0.8433 | 0.1752 | 0.0776 |
| mixed | 0.8436 | 0.1762 | 0.0776 |

The mixed model does not reach the prespecified5% directional position-MSE improvement and slightly worsens seven-joint directional error. Observed-only fine-tuning performs slightly better directionally than mixed. **No complete controller or fresh evaluation is authorized by this result.** All weights, predictions and losses are saved, including the negative result.

These are repeatedly used development folds, not an unbiased final generalization estimate. Collection-policy kernels were cross-fitted once; the entire policy/data-collection pipeline was not nested separately inside each later fold. Network fitting excludes validation-family transition labels, but this does not make the whole research pipeline independent. Reserved fresh placements remain unopened. E120 unchanged.

Source-context success/failure videos: [E144](../ease_history_estimator_control_v1/videos/index.html). The one-step training branches have no full-task success label; before/after states are saved in E148. No new policy-performance video is claimed for this failed prediction-stage candidate.

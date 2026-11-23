# E153: fixed-uniform learned history weights

The uniform32 baseline remains the strongest nonprivileged task reference so far. This study preserves its.98recency and zero-centered1e-6ridge exactly, learning only mass-normalized relative weights over32 measured transitions. No measured-calibration prior term or prior-derived feature is used. The8-feature Transformer and explicit weighted analytical solve are unchanged from the earlier weighting architecture.

Same6706 observed transitions,96 fitting placements, four ancestry-family folds, three fixed seeds and placement-balanced training. The whole research/data-collection pipeline is adaptive development, not a nested final-test estimate. No task-success labels, privileged Jacobians or future queries enter network features.

| Estimator | All position RMSE (mm) | Six joints | Seven joints | Scaled rotation RMSE (mm) |
|---|---:|---:|---:|---:|
| reference | 0.4148 | 0.4440 | 0.3927 | 0.1838 |
| ensemble | 0.3877 | 0.4194 | 0.3635 | 0.1665 |
| individual | 0.3904 | 0.4212 | 0.3670 | 0.1677 |

Fixed all-three ensemble position MSE improves by 12.6%. Every individual seed improves overall and both joint counts improve. Prediction screen passes: True. This is a smaller prediction gain than the calibration-prior model, but preserves the stronger task baseline structure. E154 must establish whether it improves complete tasks; no task gain is claimed here.

Independent NumPy reference solves, uniform initialization, exact prediction reproduction, fixed ridge/forgetting, mass normalization, family exclusion and source hashes were verified. E120 remains unchanged; reserved fresh placements remain unopened.

Source successes/failures: [E144 gallery](../ease_history_estimator_control_v1/videos/index.html). Those are context videos, not E153 controller demonstrations. E154 saves actual learned fixed-kernel trajectories separately.

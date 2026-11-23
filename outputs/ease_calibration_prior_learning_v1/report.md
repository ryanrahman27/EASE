# E146: learned weighting around measured calibration

The model learns relative weights for32 measured transitions while retaining a fixed measured calibration prior in weakly excited directions. This is a Transformer plus explicit weighted least-squares solve, not a flow-matching policy. No exact robot Jacobian, object/contact truth or task-success label is used for training.

6706 correlated transitions from 96 fitting placements; four original ancestry-family folds and three fixed seeds. Sampling balances placements; all three Jacobians are averaged for the primary ensemble. No seed or checkpoint selection.

| Estimator | All position RMSE (mm) | Six joints | Seven joints | Scaled rotation RMSE (mm) |
|---|---:|---:|---:|---:|
| reference | 0.2752 | 0.2589 | 0.2862 | 0.1242 |
| ensemble | 0.2365 | 0.2269 | 0.2432 | 0.1027 |
| individual | 0.2401 | 0.2307 | 0.2466 | 0.1040 |

Primary ensemble lowers position MSE by 26.1%; all three individual models also improve overall, and both joint counts improve. Fixed prediction screen passes: True. This only authorizes a complete fitting-controller test. Earlier studies showed lower prediction error can still reduce task success; no task gain is claimed here.

Actual future joint motion is a supervised query and never a network input. The network observes only past history and residuals against measured initial calibration. Orientation errors use .15 times tool-axis-vector increments, not angle units. Priors, independent NumPy reference solves, family exclusion, placement-balanced sampling, training-only scales, normalized weight mass, all predictions and source hashes were independently verified. E120 remains unchanged.

Source-context successes and failures are in [E144](../ease_history_estimator_control_v1/videos/index.html); the matched analytical-prior control is in [E145](../ease_calibration_prior_control_v1/videos/index.html). These are not learned-prior controller demonstrations. E147 saves those separately.

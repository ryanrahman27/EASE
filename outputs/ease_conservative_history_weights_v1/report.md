# E157: conservative history weighting

One fixed.25 power on E153 relative weights followed by recency-mass normalization reduces the possible max/min weight ratio from25 to sqrt(5). Network checkpoints, kernel.98 and zero-centered ridge1e-6 are unchanged. No temperature sweep or retraining.

Ensemble measured position RMSE0.404mm versus uniform0.415 (about5.2% lowerMSE), improving both joint counts. This deliberately gives up some of the full model's0.388mm accuracy to test a smaller control perturbation. Weight bounds, normalized mass and unchanged weights were checked. No task-success gain is claimed at this stage. Actual control is evaluated separately in E158; fresh set remains closed.

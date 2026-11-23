# E148: measured directional probe supervision

Collected **9614** single40ms branches from **738** clean phase-entry histories across all96 fitting placements. Each active joint receives both signs of a fixed.01rad perturbation around measured holding bias. These are training-only reconstructed simulation states, not full task episodes or inference-time probes.

Histories remain fixed across each branch group. Labels are observed joint and tool displacement; no exact Jacobian label or simulator parameter enters the learned model. Full integrator state was not saved in E144, so these branches explicitly reconstruct qpos/qvel/preceding commands and reset warmstart; they are not claimed to be exact continuations of the original trajectories.

All9,614 labels were reconstructed from saved branch states, and all signed-joint branches for eight representative contexts (one per fold/joint-count cell) were rerun exactly. Every phase-entry selection and history alignment was checked. No held/fresh placement was used.

| Frozen estimator | Position RMSE (mm) | Six joints | Seven joints |
|---|---:|---:|---:|
| uniform | 0.3701 | 0.3850 | 0.3576 |
| constant | 0.2558 | 0.2303 | 0.2747 |
| prior | 0.1919 | 0.1986 | 0.1863 |
| E143 | 0.2542 | 0.2173 | 0.2805 |
| E146 | 0.1768 | 0.1810 | 0.1733 |

These are new-direction prediction measurements, not task success. E146 remains better than its analytical prior on this label set, although its full task performance remains below E120 ensemble. E150 compares observed-only and mixed directional fine-tuning from identical initial weights on identical histories and splits.

Source successes and failures are saved in the [E144 video gallery](../ease_history_estimator_control_v1/videos/index.html). Single-step training branches have no success/failure task label; all before/after states are saved under branches/.

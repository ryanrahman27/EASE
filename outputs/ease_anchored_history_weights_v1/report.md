# E142: measured-history motion prediction

Mass-normalized Transformer weights are anchored to a training-fold-selected constant kernel; ridge and forgetting are frozen. Four family splits and all three prespecified seeds are retained. Pooled position RMSE improves from 0.864 to 0.838 mm, but one seed is slightly worse than the reference, so the individual-model robustness screen fails. No task episodes were executed here. The reference grid is finite and includes a boundary solution, not a globally optimal analytical estimator.

| Estimator | All position RMSE (mm) | Six joints | Seven joints |
|---|---:|---:|---:|
| constant | 0.8637 | 1.0390 | 0.6576 |
| uniform64 | 1.4681 | 1.7296 | 1.1700 |
| learned | 0.8380 | 1.0114 | 0.6330 |

The network sees past measured joint/tool transitions only. Actual next joint motion is the supervised query, not a network input. This is a learned-weighted analytical solve, not a direct flow-matching action policy. The 4,416 paired labels share 2,208 histories and are not independent episodes. Source data, family splits, weight normalization, predictions, and immutable E120 checkpoint hashes were verified.

Success/failure context videos are the already saved E120 trajectories in [the E141 gallery](../ease_learned_history_weights_v1/videos/index.html). Those clips are source-data examples, not control by this estimator. Full estimator-control videos belong to E144.

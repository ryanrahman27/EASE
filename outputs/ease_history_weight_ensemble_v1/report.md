# E143: measured-history motion prediction

The fixed mean of all three E142 Jacobians reaches 0.817 mm versus constant-kernel 0.864 mm: 10.5% lower position MSE. Both joint counts improve; each leave-one-model-out diagnostic also beats the constant reference overall. All three models remain in the primary ensemble; no best seed or pair is selected. This passes the prediction screen for a full-controller experiment, not evidence of higher task success.

| Estimator | All position RMSE (mm) | Six joints | Seven joints |
|---|---:|---:|---:|
| constant | 0.8637 | 1.0390 | 0.6576 |
| all_three | 0.8171 | 0.9911 | 0.6096 |
| without_0 | 0.8362 | 1.0250 | 0.6074 |
| without_1 | 0.8077 | 0.9748 | 0.6101 |
| without_2 | 0.8229 | 0.9882 | 0.6290 |

The network sees past measured joint/tool transitions only. Actual next joint motion is the supervised query, not a network input. This is a learned-weighted analytical solve, not a direct flow-matching action policy. The 4,416 paired labels share 2,208 histories and are not independent episodes. Source data, family splits, weight normalization, predictions, and immutable E120 checkpoint hashes were verified.

Success/failure context videos are the already saved E120 trajectories in [the E141 gallery](../ease_learned_history_weights_v1/videos/index.html). Those clips are source-data examples, not control by this estimator. Full estimator-control videos belong to E144.

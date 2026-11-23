# E48: separate motion and actuator-bias updates

Four update combinations during the identical E47 pre-grasp probe only. Normal updates resume immediately afterward. All twelve observed E46 placements; original E35 models; no training or tuning. Re-run analytical plus three neural seeds for all conditions (192 trials). Privileged E47 references are reused, since they do not consume either estimator state.

Exact E47 hover, +/- .006 rad eight targets, four steps each; open gripper; no reset after initial calibration. All physical steps and failures counted. Freeze motion means freeze Gram and cross including decay/ridge; freeze bias means retain pre-refresh bias.

One-step probe motion residual J(previous statistics) @ measured dq - measured dx, separately XYZ and .15-scaled tool-Z. Bias residual command - previous bias - measured dq. Computed before updating with the transition, diagnostic only. All four conditions share actual probe motions; normal task trajectories may diverge after refresh.

| Updates during probe | Robot | Analytical /6 | Neural seeds 0 / 1 / 2 (each /6) |
|---|---|---:|---|
| both | piper | 6 | 6 / 6 / 6 |
| both | panda | 6 | 4 / 1 / 2 |
| motion_only | piper | 6 | 5 / 6 / 6 |
| motion_only | panda | 6 | 3 / 2 / 5 |
| bias_only | piper | 6 | 6 / 5 / 6 |
| bias_only | panda | 6 | 4 / 2 / 5 |
| neither | piper | 6 | 6 / 6 / 6 |
| neither | panda | 4 | 3 / 1 / 4 |

## Matched probe prediction residuals and total cost

Errors are mean Euclidean norms per transition. XYZ is in millimetres; direction is the residual in .15-scaled tool-Z coordinates multiplied by 1000, not a physical displacement or angle. Bias residual is in radians. These are conditional motion predictions using measured dq, not complete action-to-pose predictions.

| Updates | Robot | Method | XYZ error (mm) | Scaled direction error ×1000 | Bias error (rad) | Collisions | Mean total steps |
|---|---|---|---:|---:|---:|---:|---:|
| both | piper | analytical | 0.0195 | 0.0094 | 0.01136 | 0 | 782.2 |
| both | piper | neural_0 | 0.0673 | 0.0395 | 0.01252 | 0 | 772.7 |
| both | piper | neural_1 | 0.0687 | 0.0449 | 0.01264 | 0 | 797.7 |
| both | piper | neural_2 | 0.0576 | 0.0308 | 0.01069 | 0 | 755.0 |
| both | panda | analytical | 0.0763 | 0.0260 | 0.00645 | 0 | 478.5 |
| both | panda | neural_0 | 0.4940 | 0.1264 | 0.00611 | 1 | 397.5 |
| both | panda | neural_1 | 0.4706 | 0.1470 | 0.00761 | 1 | 397.2 |
| both | panda | neural_2 | 0.4692 | 0.1287 | 0.00600 | 0 | 366.5 |
| motion_only | piper | analytical | 0.0195 | 0.0094 | 0.04568 | 0 | 775.2 |
| motion_only | piper | neural_0 | 0.0673 | 0.0395 | 0.05378 | 1 | 726.3 |
| motion_only | piper | neural_1 | 0.0687 | 0.0449 | 0.05445 | 0 | 780.5 |
| motion_only | piper | neural_2 | 0.0576 | 0.0308 | 0.04015 | 0 | 751.7 |
| motion_only | panda | analytical | 0.0763 | 0.0260 | 0.02386 | 0 | 463.7 |
| motion_only | panda | neural_0 | 0.4940 | 0.1264 | 0.02101 | 1 | 389.2 |
| motion_only | panda | neural_1 | 0.4706 | 0.1470 | 0.03196 | 2 | 349.7 |
| motion_only | panda | neural_2 | 0.4692 | 0.1287 | 0.02043 | 0 | 428.7 |
| bias_only | piper | analytical | 0.0236 | 0.0111 | 0.01136 | 0 | 791.5 |
| bias_only | piper | neural_0 | 0.0700 | 0.0401 | 0.01252 | 0 | 769.8 |
| bias_only | piper | neural_1 | 0.0717 | 0.0475 | 0.01264 | 0 | 726.5 |
| bias_only | piper | neural_2 | 0.0601 | 0.0319 | 0.01069 | 0 | 766.0 |
| bias_only | panda | analytical | 0.1274 | 0.0428 | 0.00645 | 0 | 486.2 |
| bias_only | panda | neural_0 | 0.5580 | 0.1421 | 0.00611 | 0 | 426.7 |
| bias_only | panda | neural_1 | 0.5050 | 0.1553 | 0.00761 | 1 | 404.3 |
| bias_only | panda | neural_2 | 0.5444 | 0.1460 | 0.00600 | 1 | 454.8 |
| neither | piper | analytical | 0.0236 | 0.0111 | 0.04568 | 0 | 778.2 |
| neither | piper | neural_0 | 0.0700 | 0.0401 | 0.05378 | 0 | 789.0 |
| neither | piper | neural_1 | 0.0717 | 0.0475 | 0.05445 | 0 | 793.3 |
| neither | piper | neural_2 | 0.0601 | 0.0319 | 0.04015 | 0 | 761.2 |
| neither | panda | analytical | 0.1274 | 0.0428 | 0.02386 | 0 | 407.5 |
| neither | panda | neural_0 | 0.5580 | 0.1421 | 0.02101 | 1 | 410.0 |
| neither | panda | neural_1 | 0.5050 | 0.1553 | 0.03196 | 1 | 388.3 |
| neither | panda | neural_2 | 0.5444 | 0.1460 | 0.02043 | 0 | 423.2 |

## Paired successes gained and lost

| Robot | Method | Target − control | Gained cases | Lost cases |
|---|---|---|---|---|
| piper | analytical | both − bias_only | [] | [] |
| piper | analytical | motion_only − neither | [] | [] |
| piper | analytical | both − motion_only | [] | [] |
| piper | analytical | bias_only − neither | [] | [] |
| piper | neural_0 | both − bias_only | [] | [] |
| piper | neural_0 | motion_only − neither | [] | [0] |
| piper | neural_0 | both − motion_only | [0] | [] |
| piper | neural_0 | bias_only − neither | [] | [] |
| piper | neural_1 | both − bias_only | [0] | [] |
| piper | neural_1 | motion_only − neither | [] | [] |
| piper | neural_1 | both − motion_only | [] | [] |
| piper | neural_1 | bias_only − neither | [] | [0] |
| piper | neural_2 | both − bias_only | [] | [] |
| piper | neural_2 | motion_only − neither | [] | [] |
| piper | neural_2 | both − motion_only | [] | [] |
| piper | neural_2 | bias_only − neither | [] | [] |
| panda | analytical | both − bias_only | [] | [] |
| panda | analytical | motion_only − neither | [6, 7] | [] |
| panda | analytical | both − motion_only | [] | [] |
| panda | analytical | bias_only − neither | [6, 7] | [] |
| panda | neural_0 | both − bias_only | [] | [] |
| panda | neural_0 | motion_only − neither | [10] | [8] |
| panda | neural_0 | both − motion_only | [8] | [] |
| panda | neural_0 | bias_only − neither | [10] | [] |
| panda | neural_1 | both − bias_only | [7] | [8, 10] |
| panda | neural_1 | motion_only − neither | [8, 11] | [10] |
| panda | neural_1 | both − motion_only | [7] | [8, 11] |
| panda | neural_1 | bias_only − neither | [8] | [] |
| panda | neural_2 | both − bias_only | [] | [6, 9, 11] |
| panda | neural_2 | motion_only − neither | [7] | [] |
| panda | neural_2 | both − motion_only | [] | [6, 7, 11] |
| panda | neural_2 | bias_only − neither | [9] | [] |

All 96 both/neither controls exactly reproduce E47. All 48 four-condition groups have identical initial calibration and physical trajectories through refresh. Enabled state components match the both-update state; frozen components equal their pre-refresh state. Normal online updates resume after refresh, so downstream differences include their subsequent closed-loop effects. The intervention tests the finite refresh interval, not permanently disabling bias or identification.

No promotion. Six placements per robot are repeatedly observed development data, and seeds share tasks. Preserve per-case failures; early failure can shorten mean cost. Privileged references and no-hover baselines are retained in E47, not rerun here. No simulator Jacobian enters learned/analytical control. No checkpoint training or final-test evaluation.

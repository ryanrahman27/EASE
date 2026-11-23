# E115: Motion adaptation with frozen command bias

Motion-only adaptation succeeds in 27/48 learned checkpoint-task runs and 13/16 analytical tasks. Learned both-updates is27/48 and neither-updates is30/48; analytical references are12/16 and13/16 respectively. Freezing bias while retaining learned response updates does not improve aggregate success over updating both. It also does not establish a benefit of learned online response updating over the fully calibrated frozen control.

64 new episodes complete the response/bias update factorial. The192 E114 controls are reused and explicitly not counted as new experiments. Same16 previously studied unseen-body development tasks, eight bodies, four axis families and two placements per body. Learned counts pool three frozen checkpoint seeds on the same16 tasks; analytical counts are16 distinct tasks. No physics selector, geometry filter, route planner or privileged Jacobian.

All conditions receive32 measured home-probe transitions. The new frozen_bias condition retains that calibrated bias exactly while updating the motion estimate from subsequent measured transitions, including the existing recent32-history switch and local32-step probe. It uses one E99 retained checkpoint throughout. Bias is checked for exact equality at every update. It is not zero bias.

| Estimator | Condition | DOF | Success | Collision episodes |
|---|---|---|---:|---:|
| learned | online | all | 27/48 | 17 |
| learned | online | 6 | 11/24 | 12 |
| learned | online | 7 | 16/24 | 5 |
| learned | frozen_response | all | 22/48 | 24 |
| learned | frozen_response | 6 | 10/24 | 14 |
| learned | frozen_response | 7 | 12/24 | 10 |
| learned | frozen_all | all | 30/48 | 16 |
| learned | frozen_all | 6 | 14/24 | 10 |
| learned | frozen_all | 7 | 16/24 | 6 |
| learned | frozen_bias | all | 27/48 | 13 |
| learned | frozen_bias | 6 | 10/24 | 12 |
| learned | frozen_bias | 7 | 17/24 | 1 |
| analytical | online | all | 12/16 | 3 |
| analytical | online | 6 | 6/8 | 2 |
| analytical | online | 7 | 6/8 | 1 |
| analytical | frozen_response | all | 11/16 | 4 |
| analytical | frozen_response | 6 | 5/8 | 3 |
| analytical | frozen_response | 7 | 6/8 | 1 |
| analytical | frozen_all | all | 13/16 | 1 |
| analytical | frozen_all | 6 | 7/8 | 1 |
| analytical | frozen_all | 7 | 6/8 | 0 |
| analytical | frozen_bias | all | 13/16 | 1 |
| analytical | frozen_bias | 6 | 7/8 | 1 |
| analytical | frozen_bias | 7 | 6/8 | 0 |

`online`: both updates. `frozen_response`: bias only. `frozen_all`: neither. `frozen_bias`: motion only.

## Paired changes: motion-only versus each control

| Estimator | Control | DOF | Gains | Losses |
|---|---|---|---:|---:|
| analytical | online | all | 2 | 1 |
| analytical | online | 6 | 1 | 0 |
| analytical | online | 7 | 1 | 1 |
| analytical | frozen_response | all | 3 | 1 |
| analytical | frozen_response | 6 | 2 | 0 |
| analytical | frozen_response | 7 | 1 | 1 |
| analytical | frozen_all | all | 0 | 0 |
| analytical | frozen_all | 6 | 0 | 0 |
| analytical | frozen_all | 7 | 0 | 0 |
| learned_0 | online | all | 2 | 4 |
| learned_0 | online | 6 | 1 | 2 |
| learned_0 | online | 7 | 1 | 2 |
| learned_0 | frozen_response | all | 3 | 2 |
| learned_0 | frozen_response | 6 | 1 | 1 |
| learned_0 | frozen_response | 7 | 2 | 1 |
| learned_0 | frozen_all | all | 2 | 4 |
| learned_0 | frozen_all | 6 | 1 | 2 |
| learned_0 | frozen_all | 7 | 1 | 2 |
| learned_0 | analytical_frozen_bias | all | 0 | 6 |
| learned_0 | analytical_frozen_bias | 6 | 0 | 4 |
| learned_0 | analytical_frozen_bias | 7 | 0 | 2 |
| learned_1 | online | all | 1 | 2 |
| learned_1 | online | 6 | 1 | 2 |
| learned_1 | online | 7 | 0 | 0 |
| learned_1 | frozen_response | all | 5 | 3 |
| learned_1 | frozen_response | 6 | 2 | 2 |
| learned_1 | frozen_response | 7 | 3 | 1 |
| learned_1 | frozen_all | all | 3 | 3 |
| learned_1 | frozen_all | 6 | 1 | 2 |
| learned_1 | frozen_all | 7 | 2 | 1 |
| learned_1 | analytical_frozen_bias | all | 1 | 5 |
| learned_1 | analytical_frozen_bias | 6 | 0 | 4 |
| learned_1 | analytical_frozen_bias | 7 | 1 | 1 |
| learned_2 | online | all | 5 | 2 |
| learned_2 | online | 6 | 3 | 2 |
| learned_2 | online | 7 | 2 | 0 |
| learned_2 | frozen_response | all | 4 | 2 |
| learned_2 | frozen_response | 6 | 2 | 2 |
| learned_2 | frozen_response | 7 | 2 | 0 |
| learned_2 | frozen_all | all | 1 | 2 |
| learned_2 | frozen_all | 6 | 0 | 2 |
| learned_2 | frozen_all | 7 | 1 | 0 |
| learned_2 | analytical_frozen_bias | all | 1 | 3 |
| learned_2 | analytical_frozen_bias | 6 | 0 | 3 |
| learned_2 | analytical_frozen_bias | 7 | 1 | 0 |

## Interpretation

Motion-only versus neither isolates response/history updating with calibrated bias held fixed. Motion-only versus both isolates bias updating with response updating active. Both versus bias-only tests response updating under active bias adaptation. These effects may interact; pooled success differences do not imply universal benefit across bodies or checkpoints.

All policies use measured joint/tool pose, a common reset configuration and fixed task-phase goals. Later trajectories and probe commands can diverge across conditions; no additional interaction budget is granted. This is a reused development-set attribution test, not a new confirmatory test, native-arm result or flow-matching comparison. No weights are trained or promoted.

## Verification and saved artifacts

Twelve old-condition episodes (cases0/8, analytical/learned_0, all three controls) are re-executed through the extended evaluator and match every old trace and state exactly. All64 new initial estimators/calibrations match E114. Bias invariance is asserted on every new task update. Checkpoint and source-result hashes remain unchanged. Complete traces, states, models, body/seed outcomes and outcome-selected success/failure videos are saved.

Final audit: 12 exact old-control replays and 6 exact new-condition replays; 15 trace-verified success/failure videos saved.

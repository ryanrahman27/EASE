# E114: Learned adaptation without privileged action selection

Learned online succeeds in 27/48 checkpoint-task runs, versus 22/48 with response frozen and 30/48 with all post-calibration adaptation frozen. Analytical online succeeds in 12/16 distinct tasks. Learned counts pool three checkpoint seeds on the same16 tasks; they are not48 independent body/task samples. The result depends on which component is frozen; it does not support a blanket benefit from all online adaptation. The learned online estimator does not exceed analytical online task success.

16 reused unseen-body development tasks: eight bodies, four previously studied axis families, two placements per body. Three frozen learned checkpoints and one analytical estimator, each in three update conditions, produce 192 complete episodes. The analytical episodes are not replicated as independent seed trials.

Every condition receives the same 32 measured calibration transitions. Online updates both response and command bias; frozen response updates only bias; frozen all updates neither. The same 32-step local probe is executed before grasp, but frozen conditions do not incorporate its motion observations. Online switches to the existing 32-transition recent-history estimator at that point. All learned conditions use a single E99 retained checkpoint throughout the episode, eliminating the earlier parent/student switch.

No geometry filter, physics selector, planner, privileged response Jacobian or gravity feedforward is used. Estimator initialization is 1e-6 I / zero cross statistics, not a model-derived Jacobian. The common simulator reset pose and phase goals are supplied; observations include measured joint positions and tool pose. Physics only executes the commanded actions and scores the outcome.

| Estimator | Updates | DOF | Successful | Collision episodes |
|---|---|---|---:|---:|
| analytical | online | all | 12/16 | 3 |
| analytical | online | 6 | 6/8 | 2 |
| analytical | frozen_response | all | 11/16 | 4 |
| analytical | frozen_response | 6 | 5/8 | 3 |
| analytical | frozen_all | all | 13/16 | 1 |
| analytical | frozen_all | 6 | 7/8 | 1 |
| learned | online | all | 27/48 | 17 |
| learned | online | 6 | 11/24 | 12 |
| learned | frozen_response | all | 22/48 | 24 |
| learned | frozen_response | 6 | 10/24 | 14 |
| learned | frozen_all | all | 30/48 | 16 |
| learned | frozen_all | 6 | 14/24 | 10 |
| analytical | online | 7 | 6/8 | 1 |
| analytical | frozen_response | 7 | 6/8 | 1 |
| analytical | frozen_all | 7 | 6/8 | 0 |
| learned | online | 7 | 16/24 | 5 |
| learned | frozen_response | 7 | 12/24 | 10 |
| learned | frozen_all | 7 | 16/24 | 6 |

## Paired attribution

| Estimator online | Compared with | DOF | Gains | Losses |
|---|---|---|---:|---:|
| analytical | frozen_response | all | 2 | 1 |
| analytical | frozen_response | 6 | 1 | 0 |
| analytical | frozen_response | 7 | 1 | 1 |
| analytical | frozen_all | all | 1 | 2 |
| analytical | frozen_all | 6 | 0 | 1 |
| analytical | frozen_all | 7 | 1 | 1 |
| learned_0 | frozen_response | all | 3 | 0 |
| learned_0 | frozen_response | 6 | 1 | 0 |
| learned_0 | frozen_response | 7 | 2 | 0 |
| learned_0 | frozen_all | all | 3 | 3 |
| learned_0 | frozen_all | 6 | 1 | 1 |
| learned_0 | frozen_all | 7 | 2 | 2 |
| learned_0 | analytical_online | all | 1 | 4 |
| learned_0 | analytical_online | 6 | 0 | 2 |
| learned_0 | analytical_online | 7 | 1 | 2 |
| learned_1 | frozen_response | all | 5 | 2 |
| learned_1 | frozen_response | 6 | 2 | 1 |
| learned_1 | frozen_response | 7 | 3 | 1 |
| learned_1 | frozen_all | all | 4 | 3 |
| learned_1 | frozen_all | 6 | 2 | 2 |
| learned_1 | frozen_all | 7 | 2 | 1 |
| learned_1 | analytical_online | all | 2 | 4 |
| learned_1 | analytical_online | 6 | 0 | 2 |
| learned_1 | analytical_online | 7 | 2 | 2 |
| learned_2 | frozen_response | all | 3 | 4 |
| learned_2 | frozen_response | 6 | 2 | 3 |
| learned_2 | frozen_response | 7 | 1 | 1 |
| learned_2 | frozen_all | all | 2 | 6 |
| learned_2 | frozen_all | 6 | 1 | 4 |
| learned_2 | frozen_all | 7 | 1 | 2 |
| learned_2 | analytical_online | all | 1 | 5 |
| learned_2 | analytical_online | 6 | 0 | 3 |
| learned_2 | analytical_online | 7 | 1 | 2 |

## Interpretation and scope

Online versus frozen response isolates the value of updating the motion response/history pipeline while retaining measured command-bias adaptation in both conditions. Frozen response versus frozen all exposes bias adaptation. Learned versus analytical online compares estimators under matched observations and controller structure. Online versus frozen all bundles both response and bias adaptation; it cannot alone establish a learned-estimator advantage.

Frozen controls retain a calibrated estimate; these are not zero-history or untrained controls. Later trajectories differ, so prediction residuals are own-trajectory diagnostics, not matched-transition prediction benchmarks. The local probe actions also depend on each controller’s current bias; interaction counts/budgets are matched, not post-divergence state trajectories.

Bodies are excluded from estimator training, but these tasks and axis families have already been examined during development. This is an attribution diagnostic, not a fresh confirmatory generalization test. Scores are not directly comparable to E112 because this removes its geometry filter and phase-dependent checkpoint handoff. No flow-matching policy was evaluated. No checkpoint is promoted or tuned from these results.

## Verification and artifacts

All 192 trials have identical per-case initial sufficient statistics and calibration, frozen-state assertions, saved state trajectories and exact scene XML. Checkpoint/input hashes are unchanged. Task Jacobian queries are blocked during calibration and execution. Full traces, per-step prediction diagnostics, phase results and body/seed breakdowns are retained. Videos are selected diagnostics rather than a representative success-rate sample.

Final verification: six relevant existing tests passed; eight independent full re-executions matched every saved trace and state exactly under forbidden-selector/geometry guards. Saved 30 labeled success/failure videos; all replayed state positions match their recorded traces.

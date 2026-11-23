# E159: one-time fresh-placement validation

This is the first evaluation on the48 reserved E137 placements: new placements on four known training bodies, not unseen-embodiment transfer. All48 placements and all192 episodes are retained, including collisions and failures. One shared frozen learned ensemble controls both joint counts without body/family input or target-body weight training.

| Frozen controller | Success | Six joints | Seven joints | Collision episodes |
|---|---:|---:|---:|---:|
| learned_final | 41/48 (85.4%) | 21/24 | 20/24 | 7 |
| uniform32 | 41/48 (85.4%) | 21/24 | 20/24 | 6 |
| e120_ensemble | 39/48 (81.2%) | 22/24 | 17/24 | 9 |
| analytical | 39/48 (81.2%) | 20/24 | 19/24 | 8 |

| Learned compared with | Rescued tasks | Lost successes | Net tasks | No regressions? |
|---|---:|---:|---:|---|
| uniform32 | 2 | 2 | +0 | False |
| analytical | 4 | 2 | +2 | False |
| e120_ensemble | 5 | 3 | +2 | False |

The fitting improvement does not meet the prospective net-improvement pattern. The fresh outcomes remain a negative or mixed validation result; no model was retuned or substituted.

The earlier E137/E154 retention gates were **not passed**. Before any fresh outcome, E159 explicitly amended the question from loss-free retention to prospective net improvement. E154 standard had77/96 fitting successes versus uniform74/E12072, but lost two uniform successes and failed legacy individual-seed E120 case13. Its matched E120 ensemble also failed that legacy case. All original failed-gate flags remain unchanged. This amended test does not retroactively satisfy them or authorize replacing E120. See [the frozen pre-test protocol](protocol.json).

The selected model is the full-strength E153 relative-weight model with fixed.98recency and1e-6zero-centered ridge, refit once on all6706 fitting transitions/96 placements. All three predetermined final checkpoints are averaged. No calibration-prior term, posture correction, conservative weight attenuation or physics selector is used. Task calibration, fixed close probe, frozen measured bias, DLS, waypoint schedule, budgets and success/contact criteria match the references. The architecture is a learned/analytical hybrid, not a flow-matching action policy; no conventional flow-matching baseline was evaluated here.

The fitting77/96 score came from family-excluded development models; E159 uses the prespecified full-fitting refit for prospective evaluation. Training normalization and weights were frozen before test execution. All measured calibration trajectories and initial statistics match exactly across methods, and all executed-history/query alignments, frozen bias and checkpoint/source hashes were verified.

There are only four known body clusters. Descriptive paired counts and exploratory body-cluster intervals are not evidence of broad unseen-morphology generalization or a conference-ready result. All interval assumptions and changed placements are in summary.json. The held placements are not added to training, and this candidate is not retuned or repeatedly tested on them.

| Body | Learned | Uniform | E120 ensemble | Analytical |
|---|---:|---:|---:|---:|
| 84000 | 9/12 | 9/12 | 10/12 | 8/12 |
| 84001 | 12/12 | 12/12 | 12/12 | 12/12 |
| 84004 | 12/12 | 12/12 | 9/12 | 10/12 |
| 84005 | 8/12 | 8/12 | 8/12 | 9/12 |

Videos are outcome-selected executed-state replays, not estimates of success rates. Both successes and failures are included, with matched baseline clips for representative rescues and regressions. Every rendered state is checked against the saved task trace and decoded frame count.

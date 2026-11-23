# E162: matched release-margin comparison

This fitting-only extension was registered after partial E161 outcomes, with the release height unchanged at 265 mm. All 96 fitting placements are included for each estimator. The 96 E120 elevated-release episodes are reused from E161; 192 analytical/learned episodes are new. These are development results, not independent held validation.

| Estimator | Original success | Elevated release | Six joints (original → elevated) | Seven joints | Collisions | Rescues / regressions |
|---|---:|---:|---:|---:|---:|---:|
| e120 | 72/96 | 77/96 | 36 → 37 /48 | 36 → 40 /48 | 21 → 16 | 5 / 0 |
| uniform32 | 74/96 | 79/96 | 36 → 37 /48 | 38 → 42 /48 | 20 → 15 | 5 / 0 |
| learned | 77/96 | 81/96 | 37 → 38 /48 | 40 → 43 /48 | 17 → 14 | 4 / 0 |

With the same elevated-release controller, learned versus E120: 7 rescues / 3 regressions. Learned versus uniform analytical: 4 rescues / 2 regressions.

The learned model is the unchanged E153/E154 ensemble, using all three seeds and the same four family-excluded fitting models. No retraining, checkpoint selection, online weight update, new model inputs, object/contact signal or physics selector was introduced. The analytical condition retains its 32-transition, .98-recency, 1e-6-ridge estimate. E120 retains its original bounded correction and checkpoints.

Only lower/release target height changes, by 20 mm. This permits short-drop placement and is not a gentle-contact manipulation result. All success criteria, phase budgets, probe motions and measured bias rules remain unchanged. Every saved state and trace before lowering matches its own original estimator/controller reference exactly. Thus within-estimator changes isolate the release-stage controller modification; cross-estimator comparisons still contain different earlier trajectories.

E159 is not retested or used for fitting. No unseen bodies are evaluated. The original E120 results remain immutable and separately reported. Any controller gains should not be attributed to learned adaptation without the matched analytical comparison.

[E120 paired videos](../ease_release_clearance_control_v1/videos/index.html). This study saves additional paired analytical/learned successes and failures. Clips are selected by outcome for diagnosis, not for estimating rates. All changed placements and losses are retained in summary.json.

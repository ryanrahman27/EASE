# E179 interpretation

The fitting preservation screen failed for all three candidates. Keep E120 and the original protected controller as the baseline. No unseen bodies were evaluated and no weights were updated.

| Condition | Successes | Collision episodes | Lost baseline successes |
|---|---:|---:|---|
| E120 baseline | 48/48 | 0 | None |
| Analytical projection metric | 43/48 | 3 | 27, 43, 62, 63, 67 |
| Universal readiness and existing handoff | 46/48 | 1 | 40, 42 |
| Both | 41/48 | 2 | 26, 27, 43, 47, 62, 63, 67 |

## What changed, and what this means

E120's nominal action, bounded correction, learned weights and frozen calibrated bias were retained in every condition. The metric intervention replaces only the Jacobian used by the clearance projection with the analytical estimate of the current measured history. It is applied wherever the projection is eligible, including probes. Although this improved immediate descent progress on the retrospective unseen failures in E178, it loses five complete fitting episodes here. A local progress improvement is therefore insufficient evidence for a global metric replacement.

For analytical-metric losses 27, 62 and 63, the first forbidden contact is right-pad/table during closing. Their close-entry orientations fail the inherited .08 readiness bound; cases 62 and 63 also exceed the 8 mm horizontal-position bound. Cases 43 and 67 accept recovery routes but fail the existing object-displacement readiness check (29.70 and 38.58 mm drift), then abort without a collision. The metric intervention changes earlier trajectories, route availability and subsequent histories even though the nominal policy implementation is unchanged.

Universal readiness extends the entire existing routed handoff: it checks readiness, holds the close-entry joint posture, and aborts/opens when not ready or without five consecutive final two-pad contacts. This is not a pure threshold-only ablation. All its trajectories match baseline exactly until closing. Case 40 passes readiness and retains two-pad contact for all 40 close steps, but subsequently collides with the table during lift (step 121). It cannot be explained as an overly strict rejection; the changed closing behavior leads to a different lift state/history. Case 42 instead fails readiness at close entry (16.27 mm horizontal error, .1906 axis error) and aborts, although baseline later succeeds. Rejecting that state without a recovery opportunity loses a recoverable success.

Combining the interventions avoids case 63's collision by aborting, but does not recover its task success. It still collides in cases 27 and 62 despite opening/holding; that fallback is not a guaranteed safe stop under the frozen dynamics/projection. It adds collision-free losses 26 and 47 after both passed readiness and accumulated 39/40 final two-pad steps, respectively. These outcomes do not support deploying the combination.

## Next fitting-only work

Separate readiness checks from posture holding: preserve normal closing behavior for ready, non-routed grasps, and diagnose an alignment recovery for rejected states within the original budget. Any such follow-up must also address the dynamic safety of open-and-hold rather than treating an abort flag as a safety guarantee. Independently, investigate a phase-limited projection-metric change instead of replacing the metric throughout the episode. Neither follow-up has been run or validated in E179. Require retention of the 48 fitting successes and zero collisions before another fresh unseen-body evaluation.

## Evidence and scope

All 192 full episodes and five exact preflight replays completed. Every baseline trajectory exactly reproduces the original protected E120 run. The audit checks 61,215 post-calibration steps for metric construction, bias, readiness/abort behavior, identical calibration, guard-only pre-close equality, and phase budgets. Outcomes and executed states/actions/histories are saved for all cases; videos include every discordant case across all four conditions plus fixed comparison cases. These are deterministic, previously used fitting cases with privileged simulator geometry/object/contact inputs, not independent held-body evidence or hardware results. E120 remains the baseline; no candidate is promoted.

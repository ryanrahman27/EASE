# E179: projection metric and universal grasp-readiness on fitting cases

Frozen E120 nominal policy and correction weights in every condition. Existing 48 six-joint fitting cases; privileged geometry/object/contact protections and release fix retained. A prespecified 2x2 ablation, not a new unseen-body evaluation. Baseline exactly reproduces all 48 original successes.

| Condition | Success | Collision episodes | Lost baseline successes | Abort episodes | Total steps | Median steps |
|---|---:|---:|---|---:|---:|---:|
| baseline | 48/48 | 0 | [] | 0 | 16106 | 309.5 |
| analytical_metric | 43/48 | 3 | [27, 43, 62, 63, 67] | 2 | 16997 | 312.5 |
| universal_readiness | 46/48 | 1 | [40, 42] | 1 | 16664 | 316.5 |
| combined | 41/48 | 2 | [26, 27, 43, 47, 62, 63, 67] | 5 | 17592 | 309.5 |

The analytical-metric conditions change only J used in the clearance projection metric, using current measured-history gram/cross statistics; E120 nominal actions remain neural. The universal-readiness conditions extend the existing routed close-anchor/ready/abort behavior to every grasp. This includes holding the close-entry joint posture, not merely adding a boolean check. Thresholds and interaction caps are inherited unchanged; no retries are introduced.

All conditions retain calibration32 and probe32. Remaining original phase caps and early exits are unchanged; only an accepted route permits descent up to300 rather than100 steps. Actual interaction totals include calibration. Aborts still consume the existing later-phase budgets while holding/opening, and can therefore cost more steps.

The full cohort and raw trajectories are retained, including failures. Baseline success is already48/48, so candidates can only preserve or regress this fitting score; this experiment cannot establish improved generalization. E178 unseen outcomes motivated these candidate mechanisms, but no unseen episode is executed here.

## Audits and artifacts

61215 steps audited for the requested metric, frozen bias, guard behavior and original phase limits. Five exact preflight replays and48 exact full baseline replays. Guard-only trajectories match baseline exactly up to closing. Results: summary.json and details.json; protocol.json specifies the intervention before outcomes; each condition retains trials, states, actions, histories, clearance and planning artifacts. Videos are executed-state replays, verified against saved position/object traces.

## Interpretation

See [interpretation.md](interpretation.md) for failure mechanisms and the next fitting-only tests. All three candidates fail the zero-loss preservation screen; E120 remains the baseline.

# E164: early six-joint clearance interventions

E120 ensemble plus the fixed265mm E161 release height is the baseline. Both controller changes are restricted to six-joint bodies. All48 six-joint fitting placements are evaluated per variant; seven-joint results are unchanged references, with four exact no-op replays per variant. This totals96 new intervention episodes and8 exact replays, not192 new independent episodes.

| Variant | Six-joint success | Six-joint collisions | First approach/descent contacts | Rescues / regressions |
|---|---:|---:|---:|---:|
| E120 + release baseline | 37/48 | 11 | 10 | — |
| high_approach | 37/48 | 11 | 10 | 2 / 2 |
| centered_early | 36/48 | 11 | 11 | 0 / 1 |

Variant meeting the fixed retention/contact screen: **None**. Every regression remains recorded; no baseline replacement or held test is performed.

**High approach:** raise only the approach target330→400mm; descent remains240mm with its original100-step budget. **Centered early:** add a measured-Jacobian projected joint-limit centering term during approach/descent only, with comfort band ±2rad and maximum0.005rad correction. The original E120 bounded action correction remains present. These are separate interventions, never combined.

Both preserve the original calibration, local probe, frozen measured bias, contact/success criteria and all phase budgets. The elevated release allows a short drop; it is not gentle-contact placement. Estimator weights are frozen. Neither variant uses simulator geometry, exact Jacobians, body IDs, contact signals or future physics. E163 geometry measurements were offline diagnosis only.

The approach and centering changes alter earlier histories, so later task differences are downstream effects of those interventions, unlike E161’s exact pre-lowering intervention. All actual final commands are checked against the saved command log. Seven-joint no-op replay states/traces match E161 exactly.

These are fitting-only studies on two procedural six-joint bodies. All original six-joint baseline failures lie on body84000. Results do not establish unseen-body transfer, learned-estimator superiority, or a general safety guarantee. First-contact counts do not imply that subsequent contact disappears in already failed episodes.

The geometry IDs3/4 in old traces attach to link1/link2; see [E163 diagnosis](../ease_early_contact_diagnosis_v1/report.md). All outcomes, ordered histories, actions, controls and states are saved. Videos show actual saved-state replays, selected to include successes, failures, rescues/regressions and diagnosed cases13/26.

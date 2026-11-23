# E140: paired effect classifier

The paired labels contain 1,994 neutral, 121 beneficial and 93 harmful outcomes. Compare balanced three-class fitting, conditional benefit-versus-harm fitting, and a phase-by-joint-count frequency reference. Conditional classifier losses omit neutral rows; their input scaling uses all training rows, never validation rows. Balanced classifier votes are not calibrated success probabilities. Both feature representations remain near chance on changed-outcome rankings; no positive-gain zero-loss rule qualified.

| Condition | Successes | Rescues | Lost successes | Switches |
|---|---:|---:|---:|---:|
| compact_conditional_0.65 | 195/288 | 0 | 0 | 1 |
| compact_conditional_0.75 | 195/288 | 0 | 0 | 0 |
| compact_conditional_0.85 | 195/288 | 0 | 0 | 0 |
| compact_conditional_0.95 | 195/288 | 0 | 0 | 0 |
| rich_conditional_0.65 | 195/288 | 0 | 0 | 0 |
| rich_conditional_0.75 | 195/288 | 0 | 0 | 0 |
| rich_conditional_0.85 | 195/288 | 0 | 0 | 0 |
| rich_conditional_0.95 | 195/288 | 0 | 0 | 0 |
| phase_conditional_0.65 | 195/288 | 7 | 7 | 91 |
| phase_conditional_0.75 | 195/288 | 0 | 0 | 0 |
| phase_conditional_0.85 | 195/288 | 0 | 0 | 0 |
| phase_conditional_0.95 | 195/288 | 0 | 0 | 0 |
| compact_three_cap_0.1 | 195/288 | 0 | 0 | 0 |
| compact_three_cap_0.2 | 194/288 | 0 | 1 | 6 |
| rich_three_cap_0.1 | 195/288 | 0 | 0 | 0 |
| rich_three_cap_0.2 | 195/288 | 0 | 0 | 2 |
| abstain | 195/288 | 0 | 0 | 0 |

E120 baseline: 195/288. Selected rule: abstain, 0 gains/0 losses. These are adaptive fitting-validation reconstructions from executed counterfactuals. The reserved fresh test and old held placements were not evaluated. E120 remains unchanged.

All predictions, family exclusions and full-task reconstructions were independently verified. Saved outcome-selected success/failure replays include negative results; these are not newly executed policy episodes.

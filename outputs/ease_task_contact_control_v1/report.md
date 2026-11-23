# E160: task-control rate limits on fitting cases

All 96 fitting placements were evaluated for each controller. E120 checkpoints, reference results, calibration, fixed probes, task targets, time budgets and success/contact criteria remain unchanged. No held placements or unseen bodies were evaluated.

| Controller | Original success | Limited success | Six joints (original → limited) | Seven joints | Collision episodes | Rescues / regressions |
|---|---:|---:|---:|---:|---:|---:|
| e120_ensemble | 72/96 | 46/96 | 36 → 31 /48 | 36 → 15 /48 | 21 → 30 | 4 / 30 |
| uniform32 | 74/96 | 53/96 | 36 → 31 /48 | 38 → 22 /48 | 20 → 26 | 1 / 22 |

The intervention scales the complete proposed joint-motion vector after removing frozen measured bias, then restores bias. Limits per 40 ms step: 4 mm predicted position displacement, 0.0045 scaled tool-axis displacement, and 0.02 rad maximum joint displacement. The estimate comes from measured history. No exact kinematics, geometry clearance, contact signal, future simulator prediction or physics selector enters the controller. Thus this is measured-motion rate limiting, not collision avoidance or a safety guarantee.

The baseline audit found arm-link/table contacts during approach/descent and pad/table contacts during lowering. These diagnoses motivated testing whether aggressive motion contributes to failures. Contacts remain scored with the original strict definition, including transient contacts; no failure is filtered out. Equal budgets intentionally expose lost progress from slowing down.

All 16 prespecified baseline replays matched saved states and traces exactly before intervention. All 192 intervention episodes have verified calibration, frozen bias, actual executed commands and unchanged probe commands. Every candidate success/failure clip is paired with the same original reference placement; clips are outcome-selected and do not estimate success rates.

This is fitting-only controller development, not evidence of new-morphology transfer or of a learned estimator outperforming analytical estimation. E120 remains the baseline; any gains and all regressions are reported separately. No automatic held evaluation is authorized by the fitting screen.

- e120_ensemble: positive fitting screen = False; first forbidden-contact counts = {'6:descend': 9, '7:descend': 6, '6:approach': 3, '7:settle': 3, '7:retreat': 3, '7:transport': 3, '7:lower': 3}; unreached phase counts = {'lower': 23, 'retreat': 48, 'descend': 23, 'probe': 15, 'close': 23, 'lift': 31, 'settle': 25, 'transport': 13, 'release': 23, 'approach': 15}. These counts include later phases after an earlier failure.
- uniform32: positive fitting screen = False; first forbidden-contact counts = {'7:descend': 5, '6:lift': 1, '7:retreat': 8, '6:approach': 5, '6:descend': 5, '7:lower': 2}; unreached phase counts = {'retreat': 41, 'lower': 21, 'descend': 18, 'probe': 14, 'lift': 22, 'transport': 11, 'release': 18, 'settle': 22, 'close': 16, 'approach': 11}. These counts include later phases after an earlier failure.
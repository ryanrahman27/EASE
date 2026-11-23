# E144: complete fitting episodes for history estimation

All five methods execute the same96 fitting placements with identical calibration, task schedule and collision definitions. Neural/kernel models exclude each placement ancestry family. This is development validation on known bodies, not a new embodiment test.

| Controller | All successes | Six-joint successes | Seven-joint successes | Collision episodes |
|---|---:|---:|---:|---:|
| analytical | 70/96 | 36/48 | 34/48 | 22 |
| uniform32 | 74/96 | 36/48 | 38/48 | 20 |
| constant | 70/96 | 37/48 | 33/48 | 22 |
| e120_ensemble | 72/96 | 36/48 | 36/48 | 21 |
| learned_ensemble | 67/96 | 37/48 | 30/48 | 26 |

E120 ensemble is a new matched three-model reference, not the original195/288 individual-seed baseline. Learned estimator uses a fixed average of all three history-weight models inside an explicit analytical solve. No physics selector, exact Jacobian, object/contact input, task-success gate or online gradient update.

| Learned versus | Rescues | Lost successes |
|---|---:|---:|
| e120_ensemble | 6 | 11 |
| constant | 4 | 7 |
| analytical | 3 | 6 |
| uniform32 | 2 | 9 |

Passed frozen advancement screen: **False**. E120 fitting gain placement retention: {'1': True, '13': False, '18': False}. All per-case outcomes and regressions are in summary.json. No fresh/held placements were executed. E120 remains unchanged.

All480 full state/action/history records are saved. History ordering, actual motion-query alignment, frozen bias, source and baseline hashes were checked. Gallery is outcome-selected and includes both successes and failures, not an unbiased sample of rates.

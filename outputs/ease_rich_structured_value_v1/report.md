# E139: rich structured value

Expanded measured-state features from 32 to 115 on the same 2,208 paired starts and four family folds. Primary richer trees retain the E138 hyperparameters and thresholds. Added tree-family consensus is reported separately. Extra state information did not create a positive-gain zero-loss candidate. No object/contact truth or simulator geometry enters inference.

| Condition | Successes | Rescues | Lost successes | Switches |
|---|---:|---:|---:|---:|
| ridge | 194/288 | 2 | 3 | 65 |
| consensus | 195/288 | 0 | 0 | 4 |
| trees | 196/288 | 5 | 4 | 77 |
| tree_consensus | 195/288 | 0 | 0 | 2 |
| abstain | 195/288 | 0 | 0 | 0 |

E120 baseline: 195/288. Selected rule: abstain, 0 gains/0 losses. These are adaptive fitting-validation reconstructions from executed counterfactuals. The reserved fresh test and old held placements were not evaluated. E120 remains unchanged.

All predictions, family exclusions and full-task reconstructions were independently verified. Saved outcome-selected success/failure replays include negative results; these are not newly executed policy episodes.

# E170: descent execution-error allowances

Privileged fitting diagnostic. E120, frozen bias, probe protection and the265mm release fix remain unchanged. All96 new episodes and4 exact protected-reference replays completed; no held evaluation or retraining.

| Controller | Success /48 | Collisions | Collision-free failures | Rescues / losses vs protected reference |
|---|---:|---:|---:|---|
| E120_release | 37 | 11 | 0 | — |
| E169_protected | 41 | 4 | 3 | — |
| fixed_allowance | 42 | 2 | 4 | [64] / [] |
| history_allowance | 44 | 0 | 4 | [42, 43, 64] / [] |

Only descent gripper rows receive extra allowance. Fixed:3mm one-step error; history:1mm plus maximum positive clearance prediction error over the previous8 eligible descent steps (error capped20mm). Same settings for every case, frozen before outcomes. The allowance is subtracted from the linear constraint slack, equivalently from distance divided by barrier gain0.5. Real geometry and strict scoring remain unchanged. Non-descent phases retain E169 both constraints. Activation uses effective distance to engage earlier.

The history allowance is based exclusively on already-executed measured errors and resets per episode. It does not update E120 weights or its frozen bias. Full simulator gripper geometry remains privileged. The conservative infinite tabletop plane and arm finite-difference gradients are retained; no future dynamics rollout is used. This is not learned-only transfer or guaranteed safety.

Reference fitting errors were diagnosed before interventions: maximum downward clearance overprediction4.74mm,99th percentile4.20mm. Four exact E169 replays passed. All post-state geometry, actual executed commands, causal history windows, linear constraints, fallback actions, full-state gradient reconstructions and unchanged pre-descent prefixes were verified. Every failure is retained. summary.json contains paired outcomes against E120+release and E169, progress/contact categories, allowances and first-contact prediction errors.

Videos replay saved states and verify tool/object traces and decoded frame counts.80ms video sampling may miss contacts scored at2ms. No candidate automatically replaces the baseline.

## Interpretation

The history-based allowance completes44/48 fitting episodes (91.7%), compared with41/48 (85.4%) for E169 protection and37/48 (77.1%) for E120+release. It rescues cases42,43 and64 without losing any previous success. All four previous collision episodes are eliminated across complete episodes, not merely delayed beyond descent; there are zero forbidden-contact episodes and zero solver fallbacks. This is an observed result on these48 fitting cases, not a general safety guarantee or independent transfer result.

The fixed allowance completes42/48 and reduces collision episodes to2, but those remaining contacts occur during the later probe. This distinguishes postponing a collision from avoiding it throughout the episode. The history-based allowance ranges from1mm to6.65mm in this run; its90th percentile is4.88mm. The history mechanism is an explicit measured-error rule around frozen E120, not a newly trained adaptation model.

Four history-allowance cases remain unsuccessful:26,62,65 and67. All are collision-free but fail the descent reach criterion and later task criteria. Their terminal descent position/rotation errors and grasp/lift outcomes are preserved in remaining_failures.json. Next development should target descent pose/orientation progress while retaining this protection and all44 successes. The baseline and release fix remain untouched; no held test has been opened.

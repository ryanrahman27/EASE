# E26: crossed kinematics/posture and shared recovery

Recovery union: 3211 accepted expert chunks from two frozen source policies.

Single fresh training seed. Each action policy receives 15,000 updates. No final-test rollout.

| Data | Policy | Training success | Training gate |
|---|---|---:|---|
| original | reactive | 154/256 | False |
| original | history | 256/256 | True |
| shared_recovery | reactive | 246/256 | False |
| shared_recovery | history | 256/256 | True |

| Data | Policy/context | Seen axes / seen posture | New axes / seen posture | Seen axes / shifted posture | New axes / shifted posture |
|---|---|---:|---:|---:|---:|
| original | reactive/reactive | 54/96 | 39/96 | 51/96 | 31/96 |
| original | history/correct | 96/96 | 59/96 | 95/96 | 35/96 |
| original | history/empty | 49/96 | 28/96 | 54/96 | 21/96 |
| original | history/twin | 12/96 | 21/96 | 41/96 | 12/96 |
| shared_recovery | reactive/reactive | 93/96 | 74/96 | 96/96 | 48/96 |
| shared_recovery | history/correct | 96/96 | 70/96 | 95/96 | 51/96 |
| shared_recovery | history/empty | 96/96 | 75/96 | 88/96 | 51/96 |
| shared_recovery | history/twin | 49/96 | 57/96 | 94/96 | 47/96 |

| Frozen old teacher seed | Seen/seen | New/seen | Seen/shifted | New/shifted |
|---|---:|---:|---:|---:|
| 0 | 96/96 | 52/96 | 96/96 | 48/96 |
| 1 | 96/96 | 82/96 | 76/96 | 50/96 |
| 2 | 96/96 | 51/96 | 92/96 | 50/96 |

Privileged references were trained on the previous dataset, not on these new demonstrations.
Every failed gate is retained. Validation is diagnostic when training control fails.
Goal offsets and physical parameters are matched across factors; Cartesian states necessarily vary with posture/axes.
One recovery round and one training seed do not establish general adaptation.

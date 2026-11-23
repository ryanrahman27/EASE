# E50: measured contact diagnostic

48 exact full original E35 replays: analytical plus three neural seeds, all twelve observed E46 placements. No local refresh. Log current-step cube/finger normal forces, opening, relative object/tool pose, cube-symmetric yaw mismatch, and controller motion/bias/action. Telemetry never enters control. No success-rule, mechanics or weight change.

| Robot | Method | Case | Success | Failure | Close yaw mismatch (deg) | Final close normal forces (N) |
|---|---|---:|---|---|---:|---|
| piper | analytical | 0 | True | None | 14.38 | [0.354, 0.352] |
| piper | analytical | 1 | True | None | 23.56 | [0.382, 0.383] |
| piper | analytical | 2 | True | None | 12.86 | [0.325, 0.314] |
| piper | analytical | 3 | True | None | 0.11 | [0.292, 0.291] |
| piper | analytical | 4 | True | None | 10.43 | [0.341, 0.339] |
| piper | analytical | 5 | True | None | 5.85 | [0.29, 0.295] |
| panda | analytical | 6 | True | None | 0.02 | [0.747, 0.748] |
| panda | analytical | 7 | True | None | 0.01 | [0.748, 0.748] |
| panda | analytical | 8 | True | None | 0.00 | [0.748, 0.748] |
| panda | analytical | 9 | True | None | 42.82 | [0.981, 0.993] |
| panda | analytical | 10 | True | None | 0.00 | [0.748, 0.748] |
| panda | analytical | 11 | True | None | 14.01 | [0.773, 1.059] |
| piper | neural_0 | 0 | True | None | 18.12 | [0.235, 0.497] |
| piper | neural_0 | 1 | True | None | 0.32 | [0.29, 0.289] |
| piper | neural_0 | 2 | True | None | 20.00 | [0.371, 0.373] |
| piper | neural_0 | 3 | True | None | 1.32 | [0.272, 0.308] |
| piper | neural_0 | 4 | True | None | 12.82 | [0.057, 0.636] |
| piper | neural_0 | 5 | True | None | 0.15 | [0.291, 0.292] |
| panda | neural_0 | 6 | True | None | 0.00 | [0.748, 0.748] |
| panda | neural_0 | 7 | True | None | 34.09 | [1.002, 1.077] |
| panda | neural_0 | 8 | True | None | 0.02 | [0.748, 0.748] |
| panda | neural_0 | 9 | False | lost_grasp:transport | 42.78 | [0.989, 0.987] |
| panda | neural_0 | 10 | True | None | 0.00 | [0.748, 0.748] |
| panda | neural_0 | 11 | True | None | 0.01 | [0.748, 0.748] |
| piper | neural_1 | 0 | True | None | 22.81 | [0.381, 0.38] |
| piper | neural_1 | 1 | True | None | 0.13 | [0.291, 0.293] |
| piper | neural_1 | 2 | True | None | 20.84 | [0.374, 0.375] |
| piper | neural_1 | 3 | True | None | 0.06 | [0.292, 0.292] |
| piper | neural_1 | 4 | True | None | 13.24 | [0.289, 0.34] |
| piper | neural_1 | 5 | True | None | 0.02 | [0.29, 0.293] |
| panda | neural_1 | 6 | False | forbidden_contact | not reached | not reached |
| panda | neural_1 | 7 | False | lost_grasp:lift | 0.06 | [0.748, 0.748] |
| panda | neural_1 | 8 | False | lost_grasp:lift | 0.02 | [0.748, 0.748] |
| panda | neural_1 | 9 | False | lost_grasp:lift | 0.00 | [0.741, 0.755] |
| panda | neural_1 | 10 | True | None | 0.03 | [0.746, 0.75] |
| panda | neural_1 | 11 | False | phase_timeout:release | 0.00 | [0.751, 0.747] |
| piper | neural_2 | 0 | True | None | 11.46 | [0.362, 0.384] |
| piper | neural_2 | 1 | True | None | 20.49 | [0.366, 0.382] |
| piper | neural_2 | 2 | True | None | 0.16 | [0.277, 0.287] |
| piper | neural_2 | 3 | True | None | 23.55 | [0.526, 0.243] |
| piper | neural_2 | 4 | True | None | 1.71 | [0.234, 0.345] |
| piper | neural_2 | 5 | True | None | 7.44 | [0.235, 0.333] |
| panda | neural_2 | 6 | True | None | 0.01 | [0.748, 0.748] |
| panda | neural_2 | 7 | False | forbidden_contact | 0.00 | [0.748, 0.748] |
| panda | neural_2 | 8 | True | None | 0.02 | [0.748, 0.748] |
| panda | neural_2 | 9 | True | None | 0.00 | [0.748, 0.748] |
| panda | neural_2 | 10 | True | None | 0.00 | [0.748, 0.748] |
| panda | neural_2 | 11 | True | None | 0.02 | [0.769, 0.727] |

All 48 full traces reproduce E46 exactly. Normal forces are instantaneous final-physics-state measurements, not minima over physics substeps. Yaw mismatch is modulo cube symmetry (90 degrees); it is a diagnostic, not a success criterion. Corner-oriented grips can succeed, and face-aligned grips can fail; these observations motivate but do not prove an orientation cause.

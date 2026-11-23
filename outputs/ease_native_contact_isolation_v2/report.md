# E62: native cube-contact isolation

All 12 observed E46 cases, analytical + E35 three + E61 uncorrected three + corrected three. Exact original native rollout first. Record ramped target each step, aperture and phase before action. Restore identical calibrated simulator/controller state. Replay same finite reference/aperture schedule with only cube collision contype/conaffinity zero. Table, arm, gripper mechanics unchanged. Stop at forbidden contact. No model fitting or privileged control.

Paired prefixes, not object-free full-task success. Original failures truncate schedules and models produce different schedules; comparisons between models are not common-task success comparisons. Cube ablation removes grasp load/contact but table/self-contact remain. Recorded object-dependent goals are frozen, avoiding alternate goal feedback. These are observed native diagnostics, no policy promotion or final test.

| Group | Robot | Prefixes | Original task successes | Original contacts | Cube-disabled contacts | Cube-disabled endpoint tracks | Original XYZ tracking RMS | Cube-disabled XYZ tracking RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| analytical | piper | 6 | 6 | 0 | 6 | 0 | 0.015784 | 0.015908 |
| analytical | panda | 6 | 6 | 0 | 0 | 6 | 0.005924 | 0.006417 |
| original | piper | 18 | 18 | 0 | 18 | 0 | 0.016646 | 0.016595 |
| original | panda | 18 | 11 | 2 | 1 | 14 | 0.010872 | 0.011854 |
| uncorrected | piper | 18 | 11 | 3 | 15 | 1 | 0.015123 | 0.015183 |
| uncorrected | panda | 18 | 11 | 4 | 3 | 14 | 0.008982 | 0.008831 |
| corrected | piper | 18 | 14 | 2 | 18 | 0 | 0.019477 | 0.019739 |
| corrected | panda | 18 | 5 | 5 | 2 | 9 | 0.018660 | 0.017621 |

Endpoint tracks means only the final reference of the recorded prefix is within 6 mm and 0.08 tool-Z error, without forbidden contact. It is not completion of a full task or an eight-step hold. RMS uses each pair’s shared prefix, then equal-weight trial RMS aggregation; differing original failures produce different horizons.

All original outcomes reproduce exactly; all intact fixed-reference replays match position/rotation to 1e-10. Cube-disabled replays retain table/self-collisions. No checkpoint changes or final-test evaluation.

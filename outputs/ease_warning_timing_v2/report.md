# E77: warning timing and six-joint contact coverage

Read-only diagnosis of all 96 learned E76 trajectories. No policy, fitting rule, threshold, action or original scoring label changes. E76 remains a failed prediction gate.

The response predictor has eight one-step false alarms. Seven occur one or two control steps before first contact; one occurs 83 steps before first contact. Four response-only warnings occur one or two steps (40-80 ms) before contact; one command-only warning offsets these to a net increase of three, all on placement 15 with different checkpoints. None occurs on a successful task. These are correlated observations on unchanged trajectories, not demonstrated useful interventions. A longer-horizon metric chosen after seeing this pattern would be exploratory.

| Six-joint case | Estimator | First-contact phase | Step within phase (zero-based) | Contact pairs |
|---|---|---|---:|---|
| 1 | parent_0 | lift | 25 | [['3', 'table'], ['4', 'table']] |
| 1 | parent_1 | lift | 33 | [['3', 'table'], ['4', 'table']] |
| 2 | candidate_1 | transport | 71 | [['4', 'table'], ['5', 'table']] |
| 4 | parent_2 | lower | 17 | [['gripper_right_pad', 'table']] |
| 12 | candidate_0 | lift | 3 | [['gripper_left_pad', 'table']] |

Only one of five six-joint first contacts is in the scored descent/lowering phases. Of the four omitted contacts, three involve arm-link/table contact during lift or transport; one involves a finger/table contact during lift. Exact geometry-to-body mappings and preceding phase completion flags are saved in contacts.json. End-effector clearance alone cannot cover these failures.

Next: run the frozen predictor across every pre-contact phase with successful-run controls. Keep the original one-step metric and fixed threshold, report phase-specific coverage and full warning lead times separately, and require whole-arm geometry. This is a new development diagnostic, not a revision of the failed E76 gate. Only after that audit should a bounded action intervention be tested for full-task recoveries and regressions.

No new simulation or training occurred. All 112 E76 trajectories had previously replayed exactly; this audit reads all 96 learned traces and verifies source hashes. It neither improves success rates nor establishes independent generalization.

Audit execution: the incomplete v1 directory is preserved. Its assertion incorrectly expected three response-only warnings from the net false-alarm increase; v2 correctly records four response-only warnings and one command-only warning. No experiment inputs or thresholds changed.

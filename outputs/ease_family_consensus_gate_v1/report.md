# E137: family-deletion confidence

Same structured response/task-value gate as E136, with six leave-one-training-family-out models inside each outer fitting fold. Require every model to exceed the fixed local progress threshold and task-value margin. Eight family-deletion final models were also saved. No original held or fresh test was executed.

| Task-value margin | Success /144 | Rescues | Lost successes | Switches |
|---|---:|---:|---:|---:|
| 0.0 | 100 | 9 | 6 | 90 |
| 0.025 | 101 | 5 | 1 | 37 |
| 0.05 | 97 | 0 | 0 | 9 |
| 0.1 | 97 | 0 | 0 | 1 |
| 1000000000.0 | 97 | 0 | 0 | 0 |

Primary margin 0.05 yields 97/144, zero rescues/losses, nine interventions that do not change success. Zero-loss selection fully abstains. Confidence reduces intervention but does not preserve the gains; E120 remains unchanged. All OOF models and predictions reproduced. Videos include representative successes and failures, including unchanged outcomes.

The 48 fresh randomized placements remain reserved and unexecuted. Next, expand fitting data independently (E138 seed 138000), without converting this test set into training data.

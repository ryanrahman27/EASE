# E156: outcome-informed fitting diagnostic

Only cases13,1,6,10, selected because of previous outcomes. These12 episodes are diagnostic and do not estimate general success rates. The projection uses measured calibration/history only; there is no physics selector or exact geometry.

| Case | Uniform | Learned | E120 ensemble |
|---|---:|---:|---:|
| 13 | success | failure | failure |
| 1 | success | success | success |
| 6 | success | failure | success |
| 10 | failure | success | failure |

Lift-only projection does not recover learned case13 and loses case6. Every pre-lift trace and saved state matches the original standard controller exactly; all non-lift actions are unchanged. Diagnostic gate fails. No full96-placement or fresh evaluation was performed. All trials/states and success/failure videos are retained; E120 unchanged.

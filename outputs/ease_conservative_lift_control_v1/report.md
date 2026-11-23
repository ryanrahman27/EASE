# E158: conservative weights with lift-only posture correction

One fixed.25-power attenuation of E153 history weights, with the unchanged E156 lift-only projection. No retraining or parameter sweep. Four outcome-selected fitting cases13,1,6,10 are diagnostics, not a success-rate benchmark. Four additional conservative standard rollouts supply exact matched pre-lift references.

| Case | Uniform lift correction | Conservative learned lift correction | E120 ensemble lift correction |
|---|---:|---:|---:|
| 13 | success | failure | failure |
| 1 | success | success | success |
| 6 | success | success | success |
| 10 | failure | failure | failure |

The conservative model retains case6 but still fails case13. The diagnostic gate fails; no full fitting or fresh evaluation of this variant follows. All pre-lift states match their corresponding standard controller exactly. All16 trajectories, including failures, remain saved.

E159 instead selects the unattenuated standard fixed-kernel model that reached77/96 in E154. E159 records an explicit amendment before fresh outcomes to test net improvement; it does not claim these failed retention gates passed. E120 remains unchanged.

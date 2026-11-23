# E171: descent progress under active constraints

Privileged fitting-only control diagnostic. Frozen E120, measured-error descent allowance, probe protection,265mm release fix, original phase budgets and success criteria are retained.96 intervention episodes and4 exact E170 reference replays completed. No held evaluation or retraining.

| Controller | Success /48 | Collisions | Collision-free failures | Rescues / losses vs E170 |
|---|---:|---:|---:|---|
| E120_release | 37 | 11 | 0 | — |
| E170_protected | 44 | 0 | 4 | — |
| direct_task | 43 | 0 | 5 | [] / [43] |
| limit_gated | 43 | 0 | 5 | [] / [47] |

Diagnosis: all four remaining reference failures spend100 descent steps and end with joint5 at its upper bound, active clearance constraints, and little or negative late progress. Case26 also has large orientation error. Full48-case saved-state diagnosis is retained in diagnosis.json.

The original controller clips each damped least-squares joint command before projecting it into constraints. The new objective minimizes the E120-estimated task residual directly under the same command, joint-target and clearance constraints. Its quadratic center is the unclipped damped least-squares action; its metric remains J.T J +0.015 squared I. direct_task applies during every descent step; limit_gated applies only when any measured absolute joint coordinate is at least2.4rad. These settings were frozen before outcomes. No added steps, oracle task Jacobian, goal change, retraining or relaxation of safety/scoring is included.

The E170 causal measured-error allowance and all probe/palm/link protections remain in both variants. E120 weights and calibrated bias are unchanged. Full simulator geometry is still privileged, and the local model can be inaccurate. This is not evidence of learned-only adaptation or independent transfer.

Verification covers four exact reference replays including stalled cases, every saved geometry state and executed command, full-state finite-difference gradients, causal error allowances, accepted constraint inequalities, target/gate reconstruction, frozen calibration and identical pre-descent prefixes. The quadratic task-objective equivalence was checked before interventions. All failures, collisions and solver fallbacks are retained. Videos replay actual saved states with trace/frame-count checks;80ms video sampling can miss contacts scored at2ms.

A positive screen requires more than44 successes, no collisions and no loss of any E170 success. Outcomes are reported against both E170 and original E120+release. No candidate automatically replaces the baseline.

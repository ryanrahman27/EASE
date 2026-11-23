# E67: frozen transfer to additional embodiments and placements

Use E33 structural generator with seed670000; exclude all registered prior paired and E33 broad axis sequences. Sixteen fresh axis arrangements, four geometry/dynamics variants, two homes and eight goals per home: 64 bodies, 1024 cases. Same 150-step reaching horizon, no separate calibration; analytical first action +.01 rad, neural cold start. All structural rejections and policy outcomes retained. Body IDs are local to this manifest.

Twelve placements fixed before IK and policy evaluation. Same PiPER/Panda robot designs, new XY positions, same 4cm Y transfer. Nine nominal endpoint IK checks, bounded seed67000+candidate; no policy-outcome filtering. Retain rejected placements and do not replace them. One 32-step measured probe per accepted case, clone full simulator state/statistics for every method. Both complete 1700-step contact-disabled tracking and contact-enabled original manipulation. Bias retained throughout.

Prospective development evaluation after E66, not the reserved final test. Native transfer changes placements, not robot designs. Structural screening limits scope. Seeds share bodies/goals; report per-seed, per-DoF, per-family and paired gains/losses with collision and interaction costs.

The E33 generator retains its original schema/experiment labels inside the data manifest; the parent E67 protocol, generation seed and manifest hash identify this new evaluation.

## Procedural reaching: 16 new axis arrangements, 64 bodies

| Method | Six joints /512 | Seven joints /512 | Total /1024 | Collisions | Mean steps, all cases | Paired gains vs parent | Paired losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytical | 503 | 448 | 951 | 30 | 31.18 | — | — |
| parent_0 | 508 | 451 | 959 | 24 | 31.95 | — | — |
| parent_1 | 505 | 448 | 953 | 30 | 31.20 | — | — |
| parent_2 | 509 | 450 | 959 | 30 | 29.01 | — | — |
| candidate_0 | 506 | 450 | 956 | 27 | 30.07 | 14 | 17 |
| candidate_1 | 507 | 454 | 961 | 32 | 29.28 | 19 | 11 |
| candidate_2 | 507 | 460 | 967 | 26 | 27.55 | 21 | 13 |

## Native placements

12/12 candidates pass the prespecified nine-endpoint IK screen. Calibration contact occurs in 0 accepted cases. Rejected candidates are retained in screening.json; no replacements.

### common

| Method | PiPER | Panda | Total | Contacts | Gains vs parent | Losses |
|---|---:|---:|---:|---:|---:|---:|
| analytical | 6/6 | 6/6 | 12/12 | 0 | — | — |
| parent_0 | 6/6 | 6/6 | 12/12 | 0 | — | — |
| parent_1 | 6/6 | 5/6 | 11/12 | 0 | — | — |
| parent_2 | 6/6 | 5/6 | 11/12 | 0 | — | — |
| candidate_0 | 6/6 | 6/6 | 12/12 | 0 | 0 | 0 |
| candidate_1 | 6/6 | 6/6 | 12/12 | 0 | 1 | 0 |
| candidate_2 | 6/6 | 4/6 | 10/12 | 0 | 0 | 1 |
### manipulation

| Method | PiPER | Panda | Total | Contacts | Gains vs parent | Losses |
|---|---:|---:|---:|---:|---:|---:|
| analytical | 6/6 | 6/6 | 12/12 | 0 | — | — |
| parent_0 | 6/6 | 5/6 | 11/12 | 0 | — | — |
| parent_1 | 6/6 | 2/6 | 8/12 | 1 | — | — |
| parent_2 | 5/6 | 4/6 | 9/12 | 1 | — | — |
| candidate_0 | 6/6 | 4/6 | 10/12 | 0 | 0 | 1 |
| candidate_1 | 5/6 | 5/6 | 10/12 | 0 | 3 | 1 |
| candidate_2 | 5/6 | 3/6 | 8/12 | 0 | 0 | 1 |

All 7,168 procedural records, native trial files and common-phase summaries audited. Six checkpoint hashes, both prior manifests and generated data hashes unchanged. Detailed paired, axis-family and physical-condition results are retained. Regression suite: 92 passed. All evaluations are prospective development comparisons, not final-test or hardware results.

## Interpretation and decision

The E66 native improvement does not replicate as an aggregate gain on these new placements. E35 and E66 both pass 34/36 complete tracking schedules and 28/36 manipulation trials. Tracking has one paired gain and one loss; manipulation has three gains and three losses. PiPER manipulation drops from 17/18 to 16/18, while Panda rises from 11/18 to 12/18. The E66 seed-1 PiPER case-2 lift regression is new; seed 2 shares its parent's PiPER case-3 missing-bilateral-grasp failure. E66 has zero forbidden-contact trials here versus two for E35, but equal overall success and the six-joint regression prevent a broad improvement claim. Analytical native control passes 12/12 in both tasks.

Procedural reaching shows a small, mixed improvement: E35 2871/3072 versus E66 2884/3072 (+13; 54 paired gains, 41 losses). Per-seed changes are -3/+8/+8. Six-joint success changes 1522/1536 to 1520/1536; seven-joint success changes 1349/1536 to 1364/1536. Collisions increase from 84 to 85 across the three seeds. Mean interaction counts fall for every candidate seed, but include both successful and failed trials and are not a safety metric. The analytical controller scores 951/1024; all six neural checkpoints exceed that total on this suite, so this is not a unique advantage introduced by E66. Shared families, bodies, goals and seeds are dependent observations, not thousands of independent embodiments.

Structural screening accepted all twelve new native candidates; all probes were contact-free. These scenarios use new placements, IK homes and calibration random seeds on the same two native designs. They are not new native embodiments and do not isolate placement from initialization effects. Procedural generation yields sixteen new axis arrangements, eight per joint count, with no overlap against the registered paired/E33 catalogs. Rejected structural proposals are retained; goals are endpoint-screened rather than policy-success-filtered. The reserved final test is untouched.

All eight E66 native manipulation failures are grasp-contact failures: five lost bilateral contact after lift, two after transport, and one lacked bilateral contact after closing. This describes the termination criteria, not proof that the estimator is uninvolved or that every object physically dropped. The common tracking schedule has different references and transitions from manipulation. A useful next experiment is procedural grasp-stability/recovery with frozen E35/E66/analytical estimators and a shared, prespecified contact-aware intervention, followed by retention checks. Do not select per-seed settings or tune on these newly observed native failures.

Retain E35 as the reference and preserve E66 as an experimental candidate. E67 narrows E66's earlier promise: tracking/manipulation gains on the original placements did not establish reliable transfer. No model promotion, retraining, hardware claim, or publication-readiness claim is supported here.

## Verification and artifacts

All 7168 procedural records and 168 native records, common phase summaries, source manifests, generated assets and six frozen checkpoint hashes were audited. All 92 EASE-FM regression tests pass. The comparison figure was visually inspected. Rendered candidate-seed-1 replays exactly match the saved Panda case-7 recovery (497 total steps) and PiPER case-2 regression (351 total steps), including their calibration records. Per-family, physical-condition and per-DoF paired results are saved in family_results.json, condition_results.json and paired.json. Both earlier comparison suites and the new failures remain preserved.

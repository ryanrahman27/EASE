# E116: Diagnosing six-joint learned motion adaptation

All eight six-joint E115 tasks and all three learned checkpoints are retained, including successes.32 baseline executions reproduce E115 exactly (24 learned,8 analytical);48 additional complete episodes substitute the analytical response estimator from either probe or lift onward. Bias stays frozen at the same post-calibration value. No physics selector, geometry filter, planner or exact Jacobian is used.

## Same-history response diagnostic

Before first forbidden collision, the episode-macro mean XYZ prediction error is 0.9143 mm for the learned estimator and 0.3614 mm for analytical ridge on the SAME measured transitions and pre-update statistics. Analytical prediction is better in 24/24 learned-driven episodes.

The comparator receives the identical gram/cross history and predicts the actual next joint/tool transition before its update. No analytic driver trajectory is substituted for this metric. The first colliding transition and every subsequent transition are excluded from clean diagnostics. Phase aggregates weight transitions; the headline averages per-episode means. Scaled tool-axis errors use the existing0.15 response scale, not angular degrees.

| Phase | Clean steps | Learned XYZ mm | Analytical XYZ mm | Median history condition number |
|---|---:|---:|---:|---:|
| approach | 120 | 0.2211 | 0.0218 | 42.5 |
| descend | 724 | 0.9003 | 0.2838 | 90.3 |
| probe | 672 | 0.2511 | 0.0808 | 153.1 |
| close | 840 | 0.1893 | 0.0699 | 58.0 |
| lift | 616 | 2.2680 | 1.1553 | 318.9 |
| transport | 495 | 1.1598 | 0.3689 | 122.8 |
| lower | 560 | 1.5548 | 0.6989 | 139.1 |
| release | 487 | 0.2341 | 0.0718 | 266.0 |
| retreat | 433 | 2.0886 | 0.7450 | 144.4 |
| settle | 480 | 0.2480 | 0.0965 | 298.8 |

## Controlled estimator substitution

| Condition | Success | Collision episodes |
|---|---:|---:|
| baseline | 10/24 | 12 |
| probe | 17/24 | 5 |
| lift | 18/24 | 4 |
| Analytical throughout (distinct tasks) | 7/8 | 1 |

Switch at probe: 9 paired gains and 2 losses relative to learned baseline. 3/24 prefixes already contain a disqualifying collision before handoff.

Switch at lift: 9 paired gains and 1 losses relative to learned baseline. 3/24 prefixes already contain a disqualifying collision before handoff.

## Interpretation boundaries

The supported mechanism is a response-estimation error that persists even when the measured history is held identical. The learned mapping has about2.53 times the clean next-step XYZ prediction error, and its replacement after grasp recovers nine runs while losing one. This localizes a substantial part of the failure to the estimator output used for subsequent joint commands. The mean nominal joint-command vector difference during lift is0.0478rad; the median relative response-matrix difference is56.1%. These are comparisons against the same-history analytical estimate, not ground-truth Jacobian errors.

History becomes less evenly excited around lift (median regularized Gram condition number318.9, versus58.0 during close), which is a plausible contributor to the learned approximation gap. This association does not establish poor excitation as the cause: analytical estimation performs better from exactly those same statistics. The next training diagnosis should examine estimator accuracy and action fidelity on matched training-body histories before changing the task controller or adding privileged safety machinery.

An analytical substitution keeps all observations, sufficient statistics, history updates, frozen bias and downstream controller unchanged at handoff. Improvements therefore support an estimator-output contribution to failure. They do not identify a unique training defect. Poor excitation/conditioning and action differences are diagnostic associations; this experiment does not manipulate history excitation or retrain the transformer. Saved action-disagreement/saturation fields compare nominal DLS commands; during the explicit probe these are not the executed probe commands.

The probe switch occurs after the existing recent32-history rebuild, before local probing. The lift switch retains the complete learned descent, probe and grasp. Once a baseline has already collided or dropped the object, later substitution may be unable to recover success; lack of recovery then does not exonerate earlier prediction errors. All selected switch points are fixed before outcomes. No best-state or best-seed selection.

These are reused development bodies and task placements. Scores are not fresh generalization estimates; no hardware or flow-matching comparison. A hybrid used for diagnosis is not promoted as a learned-adaptation result.

## Artifacts and verification

Full observations, predicted-motion diagnostics, state sequences and exact scene XML are saved for all80 episodes.32 baselines match every E115 trace/state exactly;48 substitutions match the full pre-handoff trace. Input and checkpoint hashes remain unchanged. Outcome-labeled success/failure videos show diagnostic examples, not a representative success-rate sample.

Final audit: 4 independent full intervention replays matched exactly; 14 success/failure videos verified against all stored tool/object state positions.

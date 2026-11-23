# E181 prospective pick-and-place transfer

All 12 fresh six-joint bodies passed initial-posture screening. Four placements per body give 48 paired cases in four previously used axis families. Body seeds and physical signatures were checked against archived case manifests before scoring. Each condition uses the E180 bounded recovery rule, privileged geometry and contact protections, and the same interaction caps.

| Estimator | Successes | Episodes with collision | Median control steps including calibration | Episodes using alignment |
|---|---:|---:|---:|---:|
| Online E120 | 34/48 | 6 | 316 | 4 |
| Online analytical | 35/48 | 5 | 305.5 | 4 |
| Calibration-only E120 | 32/48 | 7 | 323 | 5 |

## Paired comparisons

Online E120 versus Online analytical had 1 unique successes and 2 unique losses. Paired difference -2.1 percentage points (body-bootstrap 95% interval -8.3 to +4.2).

Online E120 versus Calibration-only E120 had 3 unique successes and 1 unique losses. Paired difference +4.2 percentage points (body-bootstrap 95% interval -4.2 to +12.5).

Online history updates gained placements 6, 38 and 43 and lost placement 25 relative to calibration-only E120. Online E120 uniquely completed placement 35 against analytical, which uniquely completed 9 and 25. Both body-level intervals include zero. This cohort demonstrates transfer with the protected controllers and leaves the comparative benefit of learned updates uncertain.

The frozen-history condition uses both E120 networks and current state feedback. Its initial 32 calibration transitions determine its motion estimate for the entire episode. Its 32 pre-close probe actions still count toward the budget. Audits verified identical calibration and initial state across all three methods, identical first actions for the two E120 conditions, zero post-calibration updates for the frozen condition, one update per step for both online conditions, and the original phase limits.

Intervals resample 12 complete bodies with replacement, including all four placements of each sampled body. The inference applies to this simulated perturbation distribution and the four sampled axis families.

| Body | Axis family | Online E120 | Analytical | Frozen E120 |
|---|---|---:|---:|---:|
| 0 | six_amb_train_0 | 0/4 | 0/4 | 0/4 |
| 1 | six_amb_train_0 | 3/4 | 3/4 | 2/4 |
| 2 | six_amb_train_0 | 1/4 | 2/4 | 1/4 |
| 3 | six_amb_train_1 | 4/4 | 4/4 | 4/4 |
| 4 | six_amb_train_1 | 4/4 | 4/4 | 4/4 |
| 5 | six_amb_train_1 | 4/4 | 4/4 | 4/4 |
| 6 | axis_dev87_6_ZYXYXY | 2/4 | 3/4 | 3/4 |
| 7 | axis_dev87_6_ZYXYXY | 4/4 | 4/4 | 4/4 |
| 8 | axis_dev87_6_ZYXYXY | 4/4 | 3/4 | 4/4 |
| 9 | axis_dev87_6_ZYXYZY | 2/4 | 2/4 | 1/4 |
| 10 | axis_dev87_6_ZYXYZY | 3/4 | 3/4 | 2/4 |
| 11 | axis_dev87_6_ZYXYZY | 3/4 | 3/4 | 3/4 |

![Paired task successes by body](body_results.png)

[Video gallery](videos/index.html) · [All paired results](paired.json) · [Cohort and protocol](protocol.json)

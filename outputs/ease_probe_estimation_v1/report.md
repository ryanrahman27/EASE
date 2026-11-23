# E96: same-state post-probe estimation audit

All 288 E95 full-task rollouts reproduce their saved result and trace exactly. At the first actual close-phase prediction, learned, ridge analytical and exact simulator Jacobians are compared at the same physical state using the same measured history. The diagnostic never changes the policy prediction. Primary summaries include 78 admitted-route pairs in each mode (30 six-joint and 48 seven-joint); 18 no-route pairs per mode are reported separately. These are repeated checkpoints and tasks, not independent samples.

Relative Jacobian error is ||J_hat - J_exact||_F / ||J_exact||_F. Historical residual is ||DQ J_hat^T - DX||_F / ||DX||_F over the 32 measured transitions (unweighted). Linearized task residual is ||0.35 e - J_exact u||_2, where u is the bias-free DLS correction clipped to +/-0.04 rad. Position and scaled tool-Z component errors, matrices, histories and actions are retained in each snapshot.

| Group | Mode | Estimator | N | Median relative Jacobian error | Median in-history residual | Median linearized task residual |
|---|---|---|---:|---:|---:|---:|
| all | no_probe | learned | 78 | 0.98125 | 0.28667 | 0.001836 |
| all | no_probe | analytical | 78 | 0.96537 | 0.04425 | 0.000229 |
| all | no_probe | exact | 78 | 0.00000 | 0.08729 | 0.000009 |
| all | hold32 | learned | 78 | 0.95504 | 0.24870 | 0.001422 |
| all | hold32 | analytical | 78 | 0.95262 | 0.14587 | 0.000334 |
| all | hold32 | exact | 78 | 0.00000 | 0.00191 | 0.000009 |
| all | probe32 | learned | 78 | 0.41939 | 0.38322 | 0.000640 |
| all | probe32 | analytical | 78 | 0.07895 | 0.07720 | 0.000076 |
| all | probe32 | exact | 78 | 0.00000 | 0.00182 | 0.000009 |
| 6 | no_probe | learned | 30 | 0.98251 | 0.30101 | 0.002426 |
| 6 | no_probe | analytical | 30 | 0.96210 | 0.06969 | 0.000320 |
| 6 | no_probe | exact | 30 | 0.00000 | 0.13758 | 0.000009 |
| 6 | hold32 | learned | 30 | 0.95488 | 0.28463 | 0.001621 |
| 6 | hold32 | analytical | 30 | 0.95255 | 0.18761 | 0.000380 |
| 6 | hold32 | exact | 30 | 0.00000 | 0.00248 | 0.000009 |
| 6 | probe32 | learned | 30 | 0.41939 | 0.38777 | 0.000664 |
| 6 | probe32 | analytical | 30 | 0.07013 | 0.06266 | 0.000071 |
| 6 | probe32 | exact | 30 | 0.00000 | 0.00179 | 0.000009 |
| 7 | no_probe | learned | 48 | 0.97982 | 0.28011 | 0.001584 |
| 7 | no_probe | analytical | 48 | 0.96857 | 0.03934 | 0.000097 |
| 7 | no_probe | exact | 48 | 0.00000 | 0.07758 | 0.000010 |
| 7 | hold32 | learned | 48 | 0.95523 | 0.22214 | 0.001398 |
| 7 | hold32 | analytical | 48 | 0.95868 | 0.14530 | 0.000198 |
| 7 | hold32 | exact | 48 | 0.00000 | 0.00172 | 0.000010 |
| 7 | probe32 | learned | 48 | 0.41042 | 0.36876 | 0.000630 |
| 7 | probe32 | analytical | 48 | 0.08285 | 0.07947 | 0.000091 |
| 7 | probe32 | exact | 48 | 0.00000 | 0.00182 | 0.000011 |
| no_route | no_probe | learned | 18 | 0.59231 | 0.42446 | 0.004686 |
| no_route | no_probe | analytical | 18 | 0.18687 | 0.08594 | 0.001375 |
| no_route | no_probe | exact | 18 | 0.00000 | 0.00275 | 0.000114 |
| no_route | hold32 | learned | 18 | 0.59231 | 0.42446 | 0.004686 |
| no_route | hold32 | analytical | 18 | 0.18687 | 0.08594 | 0.001375 |
| no_route | hold32 | exact | 18 | 0.00000 | 0.00275 | 0.000114 |
| no_route | probe32 | learned | 18 | 0.59231 | 0.42446 | 0.004686 |
| no_route | probe32 | analytical | 18 | 0.18687 | 0.08594 | 0.001375 |
| no_route | probe32 | exact | 18 | 0.00000 | 0.00275 | 0.000114 |
| candidate_0 | no_probe | learned | 13 | 0.97909 | 0.30636 | 0.001839 |
| candidate_0 | no_probe | analytical | 13 | 0.96938 | 0.04449 | 0.000229 |
| candidate_0 | no_probe | exact | 13 | 0.00000 | 0.08776 | 0.000009 |
| candidate_0 | hold32 | learned | 13 | 0.95398 | 0.18128 | 0.001404 |
| candidate_0 | hold32 | analytical | 13 | 0.95295 | 0.14580 | 0.000334 |
| candidate_0 | hold32 | exact | 13 | 0.00000 | 0.00192 | 0.000009 |
| candidate_0 | probe32 | learned | 13 | 0.34518 | 0.33410 | 0.000545 |
| candidate_0 | probe32 | analytical | 13 | 0.07891 | 0.07709 | 0.000076 |
| candidate_0 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| candidate_1 | no_probe | learned | 13 | 0.97536 | 0.28342 | 0.002386 |
| candidate_1 | no_probe | analytical | 13 | 0.96208 | 0.04431 | 0.000201 |
| candidate_1 | no_probe | exact | 13 | 0.00000 | 0.08743 | 0.000009 |
| candidate_1 | hold32 | learned | 13 | 0.95598 | 0.25365 | 0.001290 |
| candidate_1 | hold32 | analytical | 13 | 0.95271 | 0.17969 | 0.000358 |
| candidate_1 | hold32 | exact | 13 | 0.00000 | 0.00191 | 0.000009 |
| candidate_1 | probe32 | learned | 13 | 0.35361 | 0.32779 | 0.000541 |
| candidate_1 | probe32 | analytical | 13 | 0.07890 | 0.07737 | 0.000076 |
| candidate_1 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| candidate_2 | no_probe | learned | 13 | 0.98468 | 0.28214 | 0.001664 |
| candidate_2 | no_probe | analytical | 13 | 0.96215 | 0.04426 | 0.000202 |
| candidate_2 | no_probe | exact | 13 | 0.00000 | 0.08732 | 0.000009 |
| candidate_2 | hold32 | learned | 13 | 0.95393 | 0.24055 | 0.001232 |
| candidate_2 | hold32 | analytical | 13 | 0.95246 | 0.18041 | 0.000337 |
| candidate_2 | hold32 | exact | 13 | 0.00000 | 0.00191 | 0.000009 |
| candidate_2 | probe32 | learned | 13 | 0.36572 | 0.34846 | 0.000489 |
| candidate_2 | probe32 | analytical | 13 | 0.07890 | 0.07735 | 0.000076 |
| candidate_2 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| parent_0 | no_probe | learned | 13 | 1.01033 | 0.34547 | 0.001602 |
| parent_0 | no_probe | analytical | 13 | 0.96914 | 0.04424 | 0.000229 |
| parent_0 | no_probe | exact | 13 | 0.00000 | 0.08726 | 0.000009 |
| parent_0 | hold32 | learned | 13 | 0.95098 | 0.23527 | 0.001573 |
| parent_0 | hold32 | analytical | 13 | 0.95264 | 0.14549 | 0.000319 |
| parent_0 | hold32 | exact | 13 | 0.00000 | 0.00191 | 0.000009 |
| parent_0 | probe32 | learned | 13 | 0.45793 | 0.43507 | 0.000686 |
| parent_0 | probe32 | analytical | 13 | 0.07899 | 0.07714 | 0.000076 |
| parent_0 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| parent_1 | no_probe | learned | 13 | 0.98227 | 0.27966 | 0.002211 |
| parent_1 | no_probe | analytical | 13 | 0.96847 | 0.04228 | 0.000229 |
| parent_1 | no_probe | exact | 13 | 0.00000 | 0.08338 | 0.000009 |
| parent_1 | hold32 | learned | 13 | 0.95231 | 0.27745 | 0.001887 |
| parent_1 | hold32 | analytical | 13 | 0.95241 | 0.14488 | 0.000326 |
| parent_1 | hold32 | exact | 13 | 0.00000 | 0.00184 | 0.000009 |
| parent_1 | probe32 | learned | 13 | 0.46751 | 0.45567 | 0.000861 |
| parent_1 | probe32 | analytical | 13 | 0.07902 | 0.07718 | 0.000078 |
| parent_1 | probe32 | exact | 13 | 0.00000 | 0.00181 | 0.000009 |
| parent_2 | no_probe | learned | 13 | 0.98912 | 0.23494 | 0.001518 |
| parent_2 | no_probe | analytical | 13 | 0.96197 | 0.04199 | 0.000229 |
| parent_2 | no_probe | exact | 13 | 0.00000 | 0.08280 | 0.000009 |
| parent_2 | hold32 | learned | 13 | 0.96786 | 0.28001 | 0.001412 |
| parent_2 | hold32 | analytical | 13 | 0.95207 | 0.14510 | 0.000328 |
| parent_2 | hold32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| parent_2 | probe32 | learned | 13 | 0.49970 | 0.46707 | 0.000869 |
| parent_2 | probe32 | analytical | 13 | 0.07905 | 0.07723 | 0.000077 |
| parent_2 | probe32 | exact | 13 | 0.00000 | 0.00181 | 0.000009 |

## Paired estimator comparisons

| DOF | Mode | N | Analytical lower Jacobian error | Analytical lower task residual |
|---|---|---:|---:|---:|
| 6 | no_probe | 30 | 21 | 30 |
| 6 | hold32 | 30 | 19 | 29 |
| 6 | probe32 | 30 | 30 | 30 |
| 7 | no_probe | 48 | 39 | 48 |
| 7 | hold32 | 48 | 27 | 48 |
| 7 | probe32 | 48 | 48 | 48 |

## Paired probe-versus-control error changes

Negative delta means probe error is lower. Different intervention modes have different physical states; only comparisons between estimators within a mode hold state fixed.

| DOF | Estimator | Control | N | Improved | Median error delta |
|---|---|---|---:|---:|---:|
| 6 | learned | no_probe | 30 | 30 | -0.56558 |
| 6 | learned | hold32 | 30 | 30 | -0.52131 |
| 6 | analytical | no_probe | 30 | 30 | -0.89614 |
| 6 | analytical | hold32 | 30 | 30 | -0.88242 |
| 7 | learned | no_probe | 48 | 48 | -0.56168 |
| 7 | learned | hold32 | 48 | 48 | -0.52579 |
| 7 | analytical | no_probe | 48 | 48 | -0.87590 |
| 7 | analytical | hold32 | 48 | 48 | -0.87034 |

## Descriptive outcome strata

These strata use the original E95 outcomes, with no outcome selection for the audit. They do not define a failure threshold.

| DOF | Mode | Success | N | Median learned error | Median analytical error |
|---|---|---|---:|---:|---:|
| 6 | no_probe | False | 2 | 1.01122 | 0.97166 |
| 6 | no_probe | True | 28 | 0.98251 | 0.96203 |
| 6 | hold32 | False | 5 | 0.95669 | 0.95271 |
| 6 | hold32 | True | 25 | 0.95486 | 0.95241 |
| 6 | probe32 | False | 4 | 0.41158 | 0.07565 |
| 6 | probe32 | True | 26 | 0.42103 | 0.07013 |
| 7 | no_probe | False | 18 | 0.97968 | 0.96887 |
| 7 | no_probe | True | 30 | 0.97982 | 0.96367 |
| 7 | hold32 | False | 14 | 0.95991 | 0.96526 |
| 7 | hold32 | True | 34 | 0.95490 | 0.95228 |
| 7 | probe32 | False | 17 | 0.44506 | 0.08314 |
| 7 | probe32 | True | 31 | 0.38301 | 0.08257 |

## Six-joint regressions retained from E95

Both cases newly lost against both controls are shown below. Better instantaneous matrix and task-direction estimates do not guarantee successful subsequent contact-rich control.

| Case | Checkpoint | Mode | Success | Learned relative error | Learned task residual |
|---|---|---|---|---:|---:|
| 0 | parent_1 | no_probe | True | 0.98775 | 0.002641 |
| 0 | parent_1 | hold32 | True | 0.95231 | 0.001887 |
| 0 | parent_1 | probe32 | False | 0.42006 | 0.000749 |
| 2 | candidate_0 | no_probe | True | 0.98133 | 0.001734 |
| 2 | candidate_0 | hold32 | True | 0.95398 | 0.000929 |
| 2 | candidate_0 | probe32 | False | 0.40309 | 0.000889 |

## Finding and next decision

The probe supplies measurements from which the existing analytical estimator obtains a substantially more accurate local motion map. On all 78 route-assisted post-probe states, analytical error and linearized task residual are lower than the learned estimator at that same state. Median relative Jacobian errors are 0.419 versus 0.070 for six joints (30 pairs) and 0.410 versus 0.083 for seven joints (48 pairs). The pooled medians are 0.419 learned and 0.079 analytical. Every route-assisted learned and analytical estimate also improves its relative Jacobian error after probing versus either control, although those mode comparisons involve different states.

This identifies unused estimation accuracy in the frozen model on these measured histories. It does not show that estimation mismatch alone causes failures: both lost six-joint cases have improved instantaneous matrix and task-direction measures after probing. Do not adopt the probe or retrain on these evaluation histories. Next predeclare a paired continuation diagnostic with an identical learned/planner/probe prefix and learned versus analytical versus exact post-probe prediction, preserving measured bias, updates, contacts and full-task budgets. This will test whether better estimation translates into task recovery before changing training.

Figure: `estimation.png` and `estimation.pdf`. Each point compares estimators at one identical physical state; points below the diagonal favor analytical error. All snapshots and repeated-checkpoint results remain descriptive development evidence.

## Scope and verification

Exact Jacobian is a read-only reference and never an action input. Full replay equality verifies no behavior change. Linearized task residual uses the exact Jacobian applied to the clipped, bias-free DLS correction; it excludes dynamics, contact and geometry filtering. In-history residual is an in-sample fit, not future prediction accuracy. The exact local Jacobian can itself have nonzero historical residual because history spans different postures. Error differences do not establish full-task recovery or a causal failure threshold.

All input and checkpoint hashes are unchanged. Every snapshot retains 32 transitions; admitted-route snapshot excitation matches E95 post-intervention statistics, and no-route snapshots match across modes. No training or reserved-test evaluation occurs. The 288 replay checks are in addition to 125 passing tests, including known-linear-map and zero-motion diagnostic checks. E95 remains a failed intervention: probe57/96, hold59/96, no-probe58/96. The benchmark uses privileged route planning on reused development cases.

## Video evidence

All ten original E95 videos are linked in `videos/index.html` after SHA-256 verification. They show unchanged successes and failures, not new outcomes from this read-only diagnostic. Selection and playback disclosures are copied from E95.

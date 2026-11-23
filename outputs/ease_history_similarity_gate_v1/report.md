# E127: History similarity as an abstention signal

Candidate: 52/72 successes, 18 collision episodes. E120 remains the immutable baseline and fallback.

## Fixed protocol

**folds:** Four folds, each holding two complete source families from different bodies. Eight ancestral source families total: original placement, prior shifted placement, and all four added descendants stay together, across every phase and estimator seed. Twelve placements per fold. All four bodies remain represented in each training fold. Held placements excluded.

**model:** Reuse E126 four fold ensembles and final three models exactly; no neural retraining. Frozen seeds126100..126102. All inherited normalizations and OOF predictions reused by hash.

**similarity:** Nearest standardized RMS distance in measured Gram/cross history only (features33:124), active joint dimensions only. Same phase, joint count, and local-history flag. No positions, goals, actions, IDs, or outcome labels in metric. No clipping of standardized history distances.

**calibration:** Per-fold history mean/std from training rows only, std floor1e-6. For each training row compute nearest distance excluding its complete ancestral family. Per phase/DOF/local threshold is a fixed quantile of finite leave-family-out training distances. Empty support rejects. Final bank/calibration rebuilt from all fitting rows. Runtime has no family or case ID.

**selection:** 18 predeclared minimum-advantage/history-quantile rules plus full abstention. Consensus of all three model advantages and history support required. Reconstruct first accepted clean phase using saved full-task counterfactual outcomes. Require zero E120 regressions; maximize gains, then fewer switches, larger advantage, smaller support quantile. No held optimization. Pooled fitting validation is selection data, not an unbiased test estimate.

**inference:** At most one up-to8-step correction-off pulse, then resume E120. Otherwise E120 throughout. No physics selector. Histories from all clean fitting outcomes enter support, not just successes. Similarity is not a calibrated success probability.

**evaluation:** All72 original episodes after freezing; fitting and reused held-development placements reported separately. Existing training bodies only. E120 unchanged. Compare prior gate gains and losses as well.

Reuse1108 clean fitting pairs and1440 saved collection episodes. No new neural fitting or training rollouts. Fitting-bank membership is independent of outcome labels; nearest-neighbor history distance is not a success probability. All placements from an ancestral family stay together. Final fitting evaluations are in-sample and can match their own history-bank entries; they do not measure generalization.

Validation outcomes reconstruct the first accepted clean phase using matched complete-episode continuations. Phase starts after an earlier collision are excluded because success is already irreversibly false; runtime receives no collision flag. The same zero-regression validation criterion as E126 is used. Full abstention is feasible by construction. Selected rules are not guaranteed to preserve held success.

## Fitting-family validation

| Min advantage | History quantile | Success /144 | Gains | Losses | Switches |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.1 | 97 | 0 | 0 | 3 |
| 0.1 | 0.25 | 97 | 0 | 0 | 7 |
| 0.1 | 0.5 | 97 | 0 | 0 | 9 |
| 0.1 | 0.75 | 97 | 0 | 0 | 10 |
| 0.1 | 0.9 | 96 | 0 | 1 | 14 |
| 0.1 | 1.0 | 96 | 0 | 1 | 19 |
| 0.5 | 0.1 | 97 | 0 | 0 | 1 |
| 0.5 | 0.25 | 97 | 0 | 0 | 3 |
| 0.5 | 0.5 | 97 | 0 | 0 | 5 |
| 0.5 | 0.75 | 97 | 0 | 0 | 6 |
| 0.5 | 0.9 | 96 | 0 | 1 | 8 |
| 0.5 | 1.0 | 96 | 0 | 1 | 10 |
| 0.9 | 0.1 | 97 | 0 | 0 | 0 |
| 0.9 | 0.25 | 97 | 0 | 0 | 1 |
| 0.9 | 0.5 | 97 | 0 | 0 | 1 |
| 0.9 | 0.75 | 97 | 0 | 0 | 2 |
| 0.9 | 0.9 | 96 | 0 | 1 | 3 |
| 0.9 | 1.0 | 96 | 0 | 1 | 3 |
| 2.0 | 0.0 | 97 | 0 | 0 | 0 |

History support on every outcome-changing choice of the reused ensemble without the history filter (fitting validation only):

| Case / seed | Phase | Outcome change | Distance | 10% cutoff | 75% cutoff | Maximum cutoff |
|---|---|---|---:|---:|---:|---:|
| 35/0 | descend | loss | 2.871023 | 0.186636 | 1.135059 | 2.063864 |
| 39/0 | release | loss | 2.172034 | 1.585990 | 2.095397 | 4.459966 |
| 50/1 | descend | gain | 3.685777 | 0.186636 | 1.135059 | 2.063864 |

In this fixed grid, tighter history support rejects harmful choices but also the only successful rescue. No supported rule gains an episode. Rules through the75% cutoff lose none;90% and maximum-cutoff rules lose one. The fewer-switches tie-break selects full abstention. This does not establish that all history metrics fail, only that this standardized Gram/cross nearest-neighbor rule is not useful under the tested selection criterion.


Selected `{'min_advantage': 2.0, 'max_spread': 0.0, 'quantile': 0.0}`: 97/144 versus E120 97/144, 0 gains, 0 losses, 0 switches. Frozen before evaluation.

## Complete original-episode retest

reference=E118; unbounded=E120 (bounded correction without a gate); previousgate=E124; broadgate=E125; candidate=E127. E126 full abstention exactly matched E120 and is represented by the E120 row.

| Method | Split | DOF | Success | Collision episodes |
|---|---|---|---:|---:|
| reference | all | all | 50/72 | 18 |
| reference | all | 6 | 31/36 | 5 |
| reference | all | 7 | 19/36 | 13 |
| reference | fitting | all | 38/48 | 7 |
| reference | fitting | 6 | 20/24 | 4 |
| reference | fitting | 7 | 18/24 | 3 |
| reference | held | all | 12/24 | 11 |
| reference | held | 6 | 11/12 | 1 |
| reference | held | 7 | 1/12 | 10 |
| unbounded | all | all | 52/72 | 18 |
| unbounded | all | 6 | 33/36 | 3 |
| unbounded | all | 7 | 19/36 | 15 |
| unbounded | fitting | all | 37/48 | 9 |
| unbounded | fitting | 6 | 22/24 | 2 |
| unbounded | fitting | 7 | 15/24 | 7 |
| unbounded | held | all | 15/24 | 9 |
| unbounded | held | 6 | 11/12 | 1 |
| unbounded | held | 7 | 4/12 | 8 |
| previousgate | all | all | 58/72 | 14 |
| previousgate | all | 6 | 34/36 | 2 |
| previousgate | all | 7 | 24/36 | 12 |
| previousgate | fitting | all | 46/48 | 2 |
| previousgate | fitting | 6 | 23/24 | 1 |
| previousgate | fitting | 7 | 23/24 | 1 |
| previousgate | held | all | 12/24 | 12 |
| previousgate | held | 6 | 11/12 | 1 |
| previousgate | held | 7 | 1/12 | 11 |
| broadgate | all | all | 59/72 | 13 |
| broadgate | all | 6 | 34/36 | 2 |
| broadgate | all | 7 | 25/36 | 11 |
| broadgate | fitting | all | 46/48 | 2 |
| broadgate | fitting | 6 | 23/24 | 1 |
| broadgate | fitting | 7 | 23/24 | 1 |
| broadgate | held | all | 13/24 | 11 |
| broadgate | held | 6 | 11/12 | 1 |
| broadgate | held | 7 | 2/12 | 10 |
| candidate | all | all | 52/72 | 18 |
| candidate | all | 6 | 33/36 | 3 |
| candidate | all | 7 | 19/36 | 15 |
| candidate | fitting | all | 37/48 | 9 |
| candidate | fitting | 6 | 22/24 | 2 |
| candidate | fitting | 7 | 15/24 | 7 |
| candidate | held | all | 15/24 | 9 |
| candidate | held | 6 | 11/12 | 1 |
| candidate | held | 7 | 4/12 | 8 |

Retained 6/6 original E120 gains; recovered 0/4 regressions. No E120 success lost: True. Strict preservation of all prior successful episodes and recovery of four regressions: False.

## Every paired outcome change

```json
[
  {
    "baseline": "reference",
    "case": 8,
    "seed": 0,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 20,
    "seed": 0,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 22,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 9,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 13,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 18,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 19,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 23,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 1,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 18,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 8,
    "seed": 0,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 10,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 20,
    "seed": 0,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 22,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 1,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "previousgate",
    "case": 9,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 10,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 11,
    "seed": 1,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 19,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 23,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 10,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 18,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 20,
    "seed": 2,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 22,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 10,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 20,
    "seed": 0,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 22,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 1,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "broadgate",
    "case": 9,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 10,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 19,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 10,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 18,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 20,
    "seed": 2,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "broadgate",
    "case": 22,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  }
]
```

## Interpretation

Read held changes separately from fitting rescues. History support is a diagnostic abstention heuristic, not a calibrated safety guarantee. No held data selected the cutoff, and these are reused development placements on training bodies. No unseen-body claim is made. E120 remains baseline; a higher aggregate alone is insufficient for promotion. All negative outcomes and videos are retained.

The useful descent intervention and a harmful descent intervention share the same validation fold and support group. The useful example has distance3.685777 versus2.871023 for the harmful one. Therefore a single upper-distance cutoff in this metric cannot accept the useful example while rejecting that harmful example. This observed ordering points to the representation/metric, rather than merely a too-tight cutoff. It is a fitting-validation diagnostic, not a new held-tuned rule.

## Verification and media

All fold history banks, leave-family-out distances, OOF support decisions and selected rule independently reconstructed. Fold normalizations and predictions reproduced;1108 labels verified; no held training rows. 648 runtime decisions and21494 actions checked; 48 fitting continuations exact, 72 no-switch E120 continuations exact, two held episodes exactly replayed. Frozen E126 models and E120 inputs unchanged. Saved76 trace-verified success/failure videos.

# E126: Fitting-placement validation of an abstaining gate

E120 remains the immutable baseline and fallback. E126 selects full abstention: every nontrivial rule in the predeclared grid lost two E120 fitting-validation successes while gaining one. The selected fallback is a safety/retention result, not evidence of learned improvement.

## Protocol and split isolation

Four folds, each holding two complete source families from different bodies. Eight ancestral source families total: original placement, prior shifted placement, and all four added descendants stay together, across every phase and estimator seed. Twelve placements per fold. All four bodies remain represented in each training fold. Held placements excluded.

Reuse E125's1108 verified clean fitting pairs from1440 saved trajectories; no new training rollouts. Three MLPs (139-64-32-2, tanh; seeds126100..126102) per fold,1500 full-batch AdamW updates, paired final-success BCE. Each fold computes normalization from its training rows only. All phases and three frozen motion-estimator seeds of a placement family remain in the same fold. Four folds yield144 out-of-fold complete-episode outcomes from the exact saved on/off continuations.

At a phase entry, require the minimum of three predicted off-minus-on advantages to exceed the threshold, and their range to remain within the spread limit. Otherwise E120 continues. At most one up-to8-step correction-off pulse, followed by E120 for the rest of the episode. No physics selector, collision flag, case/robot ID, or future outcome enters the gate. Ensemble agreement is not calibrated confidence.

Twenty-five fixed threshold/spread combinations and full abstention were declared before fitting. Select zero-regression rules by most gains, then fewest switches, then larger advantage threshold and smaller spread. Pooled fitting-validation outcomes select this rule: their score is not an independent test estimate. Three final models are refit on all fitting rows after selection; the rule and checkpoints are hashed before retesting the original72 episodes. No held labels select the rule, threshold, seeds, or checkpoints.

The source-family grouping follows transitive ancestry, including earlier shifted versions. A direct-parent draft was corrected before any E126 training. Validation outcome reconstruction omits starts that already collided, because those episodes cannot recover collision-free success; runtime receives no collision flag. Full retests include all phases.

All bodies are existing training bodies. Held placements are reused development placements, not an untouched final benchmark. This experiment preserves E125's broader fitting support.

## Fitting-fold rule selection

| Minimum advantage | Maximum spread | Success /144 | Gains | Losses | Switches |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0.05 | 96 | 1 | 2 | 4 |
| 0.1 | 0.1 | 96 | 1 | 2 | 6 |
| 0.1 | 0.25 | 96 | 1 | 2 | 7 |
| 0.1 | 0.5 | 96 | 1 | 2 | 13 |
| 0.1 | 2.0 | 96 | 1 | 2 | 23 |
| 0.25 | 0.05 | 96 | 1 | 2 | 4 |
| 0.25 | 0.1 | 96 | 1 | 2 | 6 |
| 0.25 | 0.25 | 96 | 1 | 2 | 7 |
| 0.25 | 0.5 | 96 | 1 | 2 | 13 |
| 0.25 | 2.0 | 96 | 1 | 2 | 19 |
| 0.5 | 0.05 | 96 | 1 | 2 | 4 |
| 0.5 | 0.1 | 96 | 1 | 2 | 6 |
| 0.5 | 0.25 | 96 | 1 | 2 | 7 |
| 0.5 | 0.5 | 96 | 1 | 2 | 13 |
| 0.5 | 2.0 | 96 | 1 | 2 | 13 |
| 0.75 | 0.05 | 96 | 1 | 2 | 4 |
| 0.75 | 0.1 | 96 | 1 | 2 | 6 |
| 0.75 | 0.25 | 96 | 1 | 2 | 7 |
| 0.75 | 0.5 | 96 | 1 | 2 | 7 |
| 0.75 | 2.0 | 96 | 1 | 2 | 7 |
| 0.9 | 0.05 | 96 | 1 | 2 | 4 |
| 0.9 | 0.1 | 96 | 1 | 2 | 6 |
| 0.9 | 0.25 | 96 | 1 | 2 | 6 |
| 0.9 | 0.5 | 96 | 1 | 2 | 6 |
| 0.9 | 2.0 | 96 | 1 | 2 | 6 |
| 2.0 | 0.0 | 97 | 0 | 0 | 0 |

Even the rule requiring every advantage >0.90 and spread <=0.05 makes these outcome-changing decisions on excluded fitting families:

| Case / estimator seed | Fold | Phase | Three predicted advantages | Observed change |
|---|---:|---|---|---|
| 35/0 | 0 | descend | 0.9555, 0.9871, 0.9797 | loss |
| 39/0 | 1 | release | 0.9735, 0.9972, 0.9953 | loss |
| 50/1 | 0 | descend | 0.9937, 0.9462, 0.9747 | gain |

Selected rule: `{'min_advantage': 2.0, 'max_spread': 0.0}`. Validation successes 97/144 versus E120 97/144; 0 gains, 0 losses, 0 switches.

## Complete-episode retest

reference=E118; unbounded=E120 (bounded correction, no gate); previousgate=E124; broadgate=E125; candidate=E126.

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

Retained 6/6 original E120 gains; recovered 0/4 regressions. No E120 success lost: True. Preserve all prior gate successes and recover all regressions: False.

Full abstention deliberately gives up the nine fitting rescues of E125 while restoring its two held losses. Matching E120 does not satisfy the stronger objective of preserving the learned gate's rescues as well. E120 is not replaced.

## Interpretation and next test

Strong agreement did not reject the harmful decisions on held fitting families. Thus three-model agreement and a larger sigmoid margin are insufficient uncertainty controls here. The next candidate should test explicit support in measured-history features and abstain outside that support, chosen with the same ancestry-grouped fitting validation. Do not choose an exception for particular held cases or adjust thresholds using their outcomes.

## Every paired change

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

## Verification and saved media

All four fold predictions and training-only normalizations reproduced; ancestral source families disjoint;1108 fitting labels checked; zero held training rows. 648 gate decisions and21494 action steps verified. 48 exact fitting continuations and72 exact E120 continuations; two held episodes exactly replayed. Selection/checkpoints and E120 inputs unchanged. Saved76 trace-verified success/failure videos. Data collection is referenced by immutable hashes instead of duplicating the large E125 archive.

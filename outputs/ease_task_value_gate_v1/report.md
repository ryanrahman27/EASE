# E124: Fitting-only paired task outcomes and learned gate

reference: 50/72, 18 collisions; unbounded: 52/72, 18 collisions; candidate: 58/72, 14 collisions.

reference = frozen E118; unbounded = E120 per-step correction; candidate = E124 learned task-value gate. The filename unbounded means no task-value gate, not unlimited E120 actions.

## Data and learning protocol

Expanded to all16 fitting placements x3 estimator checkpoints:48 pairs,48 correction-on references and432 independent phase-entry off-pulse continuations. Reused70 fitting E123 trajectories;410 new collection trajectories. All432 intervention snapshots/prefixes checked. Of432 paired labels, 397 start without prior collision and are fitted; contaminated branches remain saved but excluded. Held rows in fitting:0.

One shared MLP (139-64-32-2, tanh) predicts final success for correction-on and a short off-pulse followed by E120. Inputs: phase, joint mask, measured joint position/bias, position-goal error, history Gram/cross matrices, both proposed actions and local-history flag. No robot/case/seed identifier, simulator integration state, collision flag, future data or outcome is an input. Features are normalized using fitting rows only. Fixed seed124000,1500 full-batch AdamW updates, BCE supervision, no held-based threshold/checkpoint selection. Both motion estimators remain frozen.

At each phase entry before an intervention, suppress E120 only if predicted off-success minus on-success exceeds0.10. Suppression lasts up to8 steps within that phase. Then resume E120 for the remainder of the episode. At most one off-pulse; this matches the label continuation and avoids composing untrained interventions. Explicit probe unchanged. No physics branching or geometry selector at inference.

Training labels come from full-task outcomes, not prediction error or immediate position progress. Repeated phase labels within a placement are dependent: this is48 fitting checkpoint/placement trajectories, not hundreds of independent robots. Fitting performance is in-sample. Eight held placements were excluded from optimization; they are reused development cases, not an untouched final test. All bodies are training bodies.

Training diagnostics: head accuracy 1.000; 19 predicted off choices on fitted rows, of which 19 have positive observed advantage and 0 have negative observed advantage. These are in-sample checks, not validation results.

## Complete episodes

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
| candidate | all | all | 58/72 | 14 |
| candidate | all | 6 | 34/36 | 2 |
| candidate | all | 7 | 24/36 | 12 |
| candidate | fitting | all | 46/48 | 2 |
| candidate | fitting | 6 | 23/24 | 1 |
| candidate | fitting | 7 | 23/24 | 1 |
| candidate | held | all | 12/24 | 12 |
| candidate | held | 6 | 11/12 | 1 |
| candidate | held | 7 | 1/12 | 11 |
| analytical | all | all | 18/24 | 5 |
| analytical | all | 6 | 11/12 | 1 |
| analytical | all | 7 | 7/12 | 4 |
| analytical | fitting | all | 12/16 | 3 |
| analytical | fitting | 6 | 7/8 | 1 |
| analytical | fitting | 7 | 5/8 | 2 |
| analytical | held | all | 6/8 | 2 |
| analytical | held | 6 | 4/4 | 0 |
| analytical | held | 7 | 2/4 | 2 |

Retained 3/6 E120 gains; recovered 4/4 E120 regressions. Strict retention gate False.

## Every paired outcome change

```json
[
  {
    "baseline": "reference",
    "case": 10,
    "seed": 0,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 1,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 10,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 11,
    "seed": 1,
    "gain": true,
    "split": "held",
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
    "case": 1,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 10,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 20,
    "seed": 2,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 22,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 8,
    "seed": 0,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 10,
    "seed": 0,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 20,
    "seed": 0,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 22,
    "seed": 0,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 1,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "unbounded",
    "case": 9,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 10,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 11,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 19,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 23,
    "seed": 1,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 10,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 18,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 20,
    "seed": 2,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 22,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  }
]
```

All collection episodes, intervention snapshots, training rows and provenance, gate weights, normalization, evaluation trajectories, decisions and success/failure videos are preserved. E120 remains an immutable baseline. Promote only after retention and held-placement checks; a better fitted score alone is not evidence of generalization.

Verification: all397 clean training rows/features/labels and fitting-only normalization checked; 20752 action decisions verified; all48 fitting evaluations exactly match their corresponding collected on/off continuation; 2 held episodes exactly replayed; 44 trace-verified success/failure videos. E120 weights/inputs unchanged.

## Decision and held-placement errors

Keep E120 as the baseline; do not promote this gate. The learned gate recovers all four targeted fitting regressions and raises fitting success37/48 to46/48 without losing a fitting success. Held success falls15/24 to12/24: one gain and four losses, including all three originally held E120 gains. Overall58/72 versus52/72 therefore does not satisfy retention. These are placement-generalization errors on training bodies, not an unseen-body result.

The sigmoid outputs are uncalibrated learned scores. Several harmful held switches receive large predicted advantages, so the result does not support simply treating a high margin as reliability. Broaden paired fitting-state coverage and test robustness without tuning a threshold to these held outcomes.

| Held case/seed | Change vs E120 | Selected phase | Predicted on | Predicted off |
|---|---|---|---:|---:|
| 8/0 | loss | transport | 0.1822 | 0.9999 |
| 20/0 | loss | release | 0.0607 | 0.9999 |
| 11/1 | gain | descend | 0.0040 | 0.8000 |
| 23/1 | loss | approach | 0.0807 | 0.2688 |
| 20/2 | loss | lower | 0.0326 | 0.9997 |

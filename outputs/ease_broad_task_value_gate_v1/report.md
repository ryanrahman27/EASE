# E125: Broader fitting placements and learned task-value gate

reference: 50/72, 18 collisions; unbounded: 52/72, 18 collisions; previousgate: 58/72, 14 collisions; candidate: 59/72, 13 collisions.

reference = frozen E118; unbounded = E120 per-step correction; previousgate = E124 gate; candidate = E125 broadened-data gate. The filename unbounded means no task-value gate, not unlimited E120 actions.

## Data and learning protocol

Expanded fitting placements16 to48 by adding32 randomized variants. All original24 placements, including8 held, remain unchanged. Across144 fitting checkpoint/placement pairs,144 on references and1296 off-pulse continuations are saved:480 reused E124 episodes and960 new episodes. All1296 intervention snapshots/prefixes checked. 1108 clean paired labels enter fitting. Held rows in fitting:0.

Placement seed125000: two variants per original fitting placement; object XY uniform in[0.275,0.425] x[-0.10,0.10] meters. Destination preserves the source displacement plus independent +/-20mm jitter. Same home and new calibration seeds. Coverage was widened after inspecting placement geometry and before any new outcomes were collected; the unused local-jitter catalog is archived in design_history/. Exact held placements and labels remain excluded, but negative-Y spatial support is now included in fitting. This changes the spatial extrapolation setting. No IK/success screening or held-outcome-driven threshold/model selection.

One shared MLP (139-64-32-2, tanh) predicts final success for correction-on and a short off-pulse followed by E120. Inputs: phase, joint mask, measured joint position/bias, position-goal error, history Gram/cross matrices, both proposed actions and local-history flag. No robot/case/seed identifier, simulator integration state, collision flag, future data or outcome is an input. Features are normalized using fitting rows only. Fixed seed124000,1500 full-batch AdamW updates, BCE supervision, no held-based threshold/checkpoint selection. Both motion estimators remain frozen.

At each phase entry before an intervention, suppress E120 only if predicted off-success minus on-success exceeds0.10. Suppression lasts up to8 steps within that phase. Then resume E120 for the remainder of the episode. At most one off-pulse; this matches the label continuation and avoids composing untrained interventions. Explicit probe unchanged. No physics branching or geometry selector at inference.

Training labels come from full-task outcomes, not prediction error or immediate position progress. Repeated phase labels within a placement are dependent: this is144 fitting checkpoint/placement trajectories, not hundreds of independent robots. Fitting performance is in-sample. Eight held placements were excluded from optimization; they are reused development cases, not an untouched final test. All bodies are training bodies.

Training diagnostics: head accuracy 1.000; 65 predicted off choices on fitted rows, of which 65 have positive observed advantage and 0 have negative observed advantage. These are in-sample checks, not validation results.

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
| previousgate | all | all | 58/72 | 14 |
| previousgate | all | 6 | 34/36 | 2 |
| previousgate | all | 7 | 24/36 | 12 |
| previousgate | fitting | all | 46/48 | 2 |
| previousgate | fitting | 6 | 23/24 | 1 |
| previousgate | fitting | 7 | 23/24 | 1 |
| previousgate | held | all | 12/24 | 12 |
| previousgate | held | 6 | 11/12 | 1 |
| previousgate | held | 7 | 1/12 | 11 |
| candidate | all | all | 59/72 | 13 |
| candidate | all | 6 | 34/36 | 2 |
| candidate | all | 7 | 25/36 | 11 |
| candidate | fitting | all | 46/48 | 2 |
| candidate | fitting | 6 | 23/24 | 1 |
| candidate | fitting | 7 | 23/24 | 1 |
| candidate | held | all | 13/24 | 11 |
| candidate | held | 6 | 11/12 | 1 |
| candidate | held | 7 | 2/12 | 10 |
| analytical | all | all | 18/24 | 5 |
| analytical | all | 6 | 11/12 | 1 |
| analytical | all | 7 | 7/12 | 4 |
| analytical | fitting | all | 12/16 | 3 |
| analytical | fitting | 6 | 7/8 | 1 |
| analytical | fitting | 7 | 5/8 | 2 |
| analytical | held | all | 6/8 | 2 |
| analytical | held | 6 | 4/4 | 0 |
| analytical | held | 7 | 2/4 | 2 |

## Fitting-label coverage

[Placement coverage figure](placement_coverage.png). Original and added fitting outcomes below are descriptive; multiple phase pairs share an episode.

| Source | DOF | Clean pairs | E120 on success | Off helps | Off hurts | Unchanged |
|---|---:|---:|---:|---:|---:|---:|
| original | 6 | 208 | 22/24 | 5 | 14 | 189 |
| original | 7 | 189 | 15/24 | 14 | 7 | 168 |
| added | 6 | 341 | 32/48 | 11 | 8 | 322 |
| added | 7 | 370 | 28/48 | 35 | 25 | 310 |

Retained 5/6 E120 gains; recovered 4/4 E120 regressions. Strict retention gate False.

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
    "case": 19,
    "seed": 1,
    "gain": true,
    "split": "fitting",
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
    "case": 11,
    "seed": 1,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "previousgate",
    "case": 23,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  }
]
```

All collection episodes, intervention snapshots, training rows and provenance, gate weights, normalization, evaluation trajectories, decisions and success/failure videos are preserved. E120 remains an immutable baseline. Promote only after retention and held-placement checks; a better fitted score alone is not evidence of generalization.

Verification: all1108 clean training rows/features/labels and fitting-only normalization checked; 20920 action decisions verified; all48 fitting evaluations exactly match their corresponding collected on/off continuation; 2 held episodes exactly replayed; 44 trace-verified success/failure videos. E120 weights/inputs unchanged.

## Decision and held-placement errors

E120 remains unchanged. Retained 5/6 original E120 gains and recovered 4/4 original regressions. Strict retention check (including E118, E120 and E124 successful episodes): False. See the full paired changes before interpreting aggregate improvement.

E125 preserves all46 E124 fitting successes. Relative to E124 on held placements, it restores8/0 and23/1 but loses11/1: a net gain of one episode. Relative to E120, it adds nine fitting successes but loses held case20 at seeds0 and2, with no held gains. Both held losses are seven-joint episodes. The overall59/72 result therefore does not meet the requested no-loss generalization criterion; E125 is not promoted over E120.

The harmful switches now occur at approach (20/0) and descend (20/2), with predicted off-success near1 despite subsequent failure. Broader fitting coverage recovers two previously lost E120 gains, but perfect fitting classification still does not imply reliable held decisions. This is descriptive evidence, not a demonstrated cause or a calibrated confidence estimate. The next proposed test is placement-grouped validation on fitting data for a conservative abstaining gate; no additional threshold or model was selected on these held outcomes.

Compare the fixed original fitting and held placements against both E120 and the smaller-data E124 gate. The architecture, training seed, update count and threshold are unchanged; normalization is refit using expanded fitting data only. Sigmoid scores remain uncalibrated. No unseen embodiment claims are made.

| Held case/seed | Change vs E120 | Selected phase | Predicted on | Predicted off |
|---|---|---|---:|---:|
| 20/0 | loss | approach | 0.3605 | 0.9995 |
| 20/2 | loss | descend | 0.1009 | 0.9999 |

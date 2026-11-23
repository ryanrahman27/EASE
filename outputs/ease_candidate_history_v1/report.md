# E119: Corrective learning on candidate-reached histories

Task success 50/72 to 47/72; collisions 18 to 21. Strict retention gate FAILED.

All48 E118 fitting trajectories are exactly replayed, including successful and failed episodes, and their pre-update histories are appended to the E118 dataset. Full traces and states match the stored candidates. No candidate histories from the eight held placements are added to fitting. Existing held histories are used only for fixed offline comparisons. Same24 training-body tasks; no unseen morphology evaluation.

Each seed starts its E118 checkpoint and receives a fixed1500-update corrective run. Batches contain own failed-task histories, own successful-task histories, older fitting histories and synthetic replay. Successful cases preserve current E118 predictions/actions. Failed cases target a successful older estimator on that same measured history when available, otherwise analytical ridge. This is target supervision, not rollout reset or an inference-time teacher/selector.

The previous conflicting analytical matrix/response objectives are removed from protected successful contexts. Matrix and action targets agree on which reference is preserved. Measured-response loss applies only to unprotected histories. All fitting rows must be clean up to the current transition; all failed outcomes and their full trajectories remain recorded. Explicit probe actions are excluded from the nominal DLS action objective.

| Method | Group | DOF | Success | Collision episodes |
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
| reference | original_train | all | 19/24 | 4 |
| reference | original_train | 6 | 10/12 | 2 |
| reference | original_train | 7 | 9/12 | 2 |
| reference | original_held | all | 5/12 | 6 |
| reference | original_held | 6 | 5/6 | 1 |
| reference | original_held | 7 | 0/6 | 5 |
| reference | broad_train | all | 19/24 | 3 |
| reference | broad_train | 6 | 10/12 | 2 |
| reference | broad_train | 7 | 9/12 | 1 |
| reference | fresh_held | all | 7/12 | 5 |
| reference | fresh_held | 6 | 6/6 | 0 |
| reference | fresh_held | 7 | 1/6 | 5 |
| candidate | all | all | 47/72 | 21 |
| candidate | all | 6 | 29/36 | 7 |
| candidate | all | 7 | 18/36 | 14 |
| candidate | fitting | all | 33/48 | 11 |
| candidate | fitting | 6 | 18/24 | 6 |
| candidate | fitting | 7 | 15/24 | 5 |
| candidate | held | all | 14/24 | 10 |
| candidate | held | 6 | 11/12 | 1 |
| candidate | held | 7 | 3/12 | 9 |
| candidate | original_train | all | 17/24 | 4 |
| candidate | original_train | 6 | 9/12 | 3 |
| candidate | original_train | 7 | 8/12 | 1 |
| candidate | original_held | all | 8/12 | 4 |
| candidate | original_held | 6 | 6/6 | 0 |
| candidate | original_held | 7 | 2/6 | 4 |
| candidate | broad_train | all | 16/24 | 7 |
| candidate | broad_train | 6 | 9/12 | 3 |
| candidate | broad_train | 7 | 7/12 | 4 |
| candidate | fresh_held | all | 6/12 | 6 |
| candidate | fresh_held | 6 | 5/6 | 1 |
| candidate | fresh_held | 7 | 1/6 | 5 |
| analytical | all | all | 18/24 | 5 |
| analytical | all | 6 | 11/12 | 1 |
| analytical | all | 7 | 7/12 | 4 |
| analytical | fitting | all | 12/16 | 3 |
| analytical | fitting | 6 | 7/8 | 1 |
| analytical | fitting | 7 | 5/8 | 2 |
| analytical | held | all | 6/8 | 2 |
| analytical | held | 6 | 4/4 | 0 |
| analytical | held | 7 | 2/4 | 2 |
| analytical | original_train | all | 7/8 | 0 |
| analytical | original_train | 6 | 4/4 | 0 |
| analytical | original_train | 7 | 3/4 | 0 |
| analytical | original_held | all | 3/4 | 1 |
| analytical | original_held | 6 | 2/2 | 0 |
| analytical | original_held | 7 | 1/2 | 1 |
| analytical | broad_train | all | 5/8 | 3 |
| analytical | broad_train | 6 | 3/4 | 1 |
| analytical | broad_train | 7 | 2/4 | 2 |
| analytical | fresh_held | all | 3/4 | 1 |
| analytical | fresh_held | 6 | 2/2 | 0 |
| analytical | fresh_held | 7 | 1/2 | 1 |

## Paired recovery and retention

[
  {
    "seed": 0,
    "reference": 19,
    "candidate": 16,
    "gains": 0,
    "losses": 3,
    "fitting_gains": 0,
    "fitting_losses": 2,
    "changes": [
      {
        "case": 12,
        "dof": 6,
        "group": "broad_train",
        "fitting": true,
        "gain": false
      },
      {
        "case": 14,
        "dof": 6,
        "group": "fresh_held",
        "fitting": false,
        "gain": false
      },
      {
        "case": 22,
        "dof": 7,
        "group": "broad_train",
        "fitting": true,
        "gain": false
      }
    ]
  },
  {
    "seed": 1,
    "reference": 15,
    "candidate": 14,
    "gains": 1,
    "losses": 2,
    "fitting_gains": 1,
    "fitting_losses": 2,
    "changes": [
      {
        "case": 7,
        "dof": 7,
        "group": "original_train",
        "fitting": true,
        "gain": false
      },
      {
        "case": 9,
        "dof": 7,
        "group": "original_train",
        "fitting": true,
        "gain": false
      },
      {
        "case": 10,
        "dof": 7,
        "group": "original_train",
        "fitting": true,
        "gain": true
      }
    ]
  },
  {
    "seed": 2,
    "reference": 16,
    "candidate": 17,
    "gains": 3,
    "losses": 2,
    "fitting_gains": 0,
    "fitting_losses": 2,
    "changes": [
      {
        "case": 2,
        "dof": 6,
        "group": "original_held",
        "fitting": false,
        "gain": true
      },
      {
        "case": 4,
        "dof": 6,
        "group": "original_train",
        "fitting": true,
        "gain": false
      },
      {
        "case": 8,
        "dof": 7,
        "group": "original_held",
        "fitting": false,
        "gain": true
      },
      {
        "case": 11,
        "dof": 7,
        "group": "original_held",
        "fitting": false,
        "gain": true
      },
      {
        "case": 18,
        "dof": 7,
        "group": "broad_train",
        "fitting": true,
        "gain": false
      }
    ]
  }
]

Unresolved successes from older E117/E99 references:
[
  {
    "case": 8,
    "seed": 0,
    "group": "original_held"
  },
  {
    "case": 12,
    "seed": 0,
    "group": "broad_train"
  },
  {
    "case": 14,
    "seed": 0,
    "group": "fresh_held"
  },
  {
    "case": 1,
    "seed": 1,
    "group": "original_train"
  },
  {
    "case": 7,
    "seed": 1,
    "group": "original_train"
  },
  {
    "case": 8,
    "seed": 1,
    "group": "original_held"
  },
  {
    "case": 9,
    "seed": 1,
    "group": "original_train"
  },
  {
    "case": 18,
    "seed": 1,
    "group": "broad_train"
  },
  {
    "case": 1,
    "seed": 2,
    "group": "original_train"
  },
  {
    "case": 4,
    "seed": 2,
    "group": "original_train"
  },
  {
    "case": 10,
    "seed": 2,
    "group": "original_train"
  },
  {
    "case": 18,
    "seed": 2,
    "group": "broad_train"
  }
]

## Scope and decision

The correction is evaluated on every case and checkpoint, not only the selected fitting failures. Fitting recovery is in-sample; original/fresh held placements are the generalization checks on these same training bodies. Targets from successful prior estimators need not be optimal on off-trajectory histories; their effect must be judged by complete episodes.

No task-phase, bias, controller, collision geometry, physics selector or inference-time estimator architecture changes. No best-checkpoint/seed selection; all three final checkpoints are retained. Promotion requires increased overall success, no lost current success, no collision increase and retention of older successes. Unseen-body testing stays deferred.

All204 inherited trajectories,48 exact fitting-history replays and72 new candidate episodes remain traceable. Checkpoints, contexts, targets/provenance, states, XMLs and outcome-labeled success/failure videos are saved.

## Outcome interpretation

The corrective run recovers four previous failures but loses seven previous successes. On fitting placements it recovers one failure and loses six successes (38/48 to 33/48). On held placements it recovers three failures and loses one success (12/24 to 14/24). These are 24 placements repeated across three learned checkpoints, not 72 independent tasks. All bodies are training bodies.

The result does not support promoting this correction. Matching successful actions on recorded histories is insufficient to guarantee retention in closed-loop execution. The exact source of each regression remains to be isolated; lower aggregate prediction error alone is not evidence of better task control. The next diagnostic should compare the first action/history divergence on these lost-success pairs, then test a bounded learned residual against the frozen reference.

| Seed | Case | Split | Change | Previous first collision | Corrected first collision |
|---|---|---|---|---|---|
| 0 | 12 | broad_train | lost | none | lift |
| 0 | 14 | fresh_held | lost | none | descend |
| 0 | 22 | broad_train | lost | none | descend |
| 1 | 7 | original_train | lost | none | retreat |
| 1 | 9 | original_train | lost | none | none |
| 1 | 10 | original_train | recovered | descend | none |
| 2 | 2 | original_held | recovered | descend | none |
| 2 | 4 | original_train | lost | none | lift |
| 2 | 8 | original_held | recovered | descend | none |
| 2 | 11 | original_held | recovered | none | none |
| 2 | 18 | broad_train | lost | none | lower |

Final verification: 13 independent exact replays, including fresh held placements; 26 trace-verified success/failure videos saved.

# E118: Broader training histories and phase retention

Candidate 50/72 versus E117 reference 49/72; collisions 18 versus 20. Strict retention gate FAILED. No unseen-body evaluation performed.

Four existing training bodies,24 task placements: eight original fitted, four original held, eight shifted fitted and four fresh shifted held. Shifts (+/-12mm x,+/-8mm y) and calibration seeds are fixed before rollout outcomes. No reachability/success filtering. New histories use E117 three-checkpoint and analytical drivers. The original14,906-row context dataset is retained; no held placement enters training.

Three fixed2000-update final checkpoints start from E117. Batches combine original histories, shifted descent/lowering histories, shifted other phases and synthetic replay. Descent/lowering action error has3x weight. Successful reference actions/matrices are preserved on fitting cases; on original fitting cases where E117 failed but E99 succeeded, E99 supplies the preservation target. Other rows use analytical action targets. This is a combined data/objective change, not an isolated loss ablation.

Policy inference remains a learned inverse-free motion estimator with frozen calibrated bias and measured response-history updates. No physics selector, geometry filter, planner or privileged task Jacobian. Original and fresh held tasks are kept separate; all checkpoints and failures remain saved.

| Method | Placement group | DOF | Success | Collision episodes |
|---|---|---|---:|---:|
| reference | all | all | 49/72 | 20 |
| reference | all | 6 | 30/36 | 6 |
| reference | all | 7 | 19/36 | 14 |
| reference | original | all | 26/36 | 9 |
| reference | original | 6 | 15/18 | 3 |
| reference | original | 7 | 11/18 | 6 |
| reference | original_train | all | 20/24 | 3 |
| reference | original_train | 6 | 10/12 | 2 |
| reference | original_train | 7 | 10/12 | 1 |
| reference | original_held | all | 6/12 | 6 |
| reference | original_held | 6 | 5/6 | 1 |
| reference | original_held | 7 | 1/6 | 5 |
| reference | broad_train | all | 17/24 | 5 |
| reference | broad_train | 6 | 9/12 | 3 |
| reference | broad_train | 7 | 8/12 | 2 |
| reference | fresh_held | all | 6/12 | 6 |
| reference | fresh_held | 6 | 6/6 | 0 |
| reference | fresh_held | 7 | 0/6 | 6 |
| candidate | all | all | 50/72 | 18 |
| candidate | all | 6 | 31/36 | 5 |
| candidate | all | 7 | 19/36 | 13 |
| candidate | original | all | 24/36 | 10 |
| candidate | original | 6 | 15/18 | 3 |
| candidate | original | 7 | 9/18 | 7 |
| candidate | original_train | all | 19/24 | 4 |
| candidate | original_train | 6 | 10/12 | 2 |
| candidate | original_train | 7 | 9/12 | 2 |
| candidate | original_held | all | 5/12 | 6 |
| candidate | original_held | 6 | 5/6 | 1 |
| candidate | original_held | 7 | 0/6 | 5 |
| candidate | broad_train | all | 19/24 | 3 |
| candidate | broad_train | 6 | 10/12 | 2 |
| candidate | broad_train | 7 | 9/12 | 1 |
| candidate | fresh_held | all | 7/12 | 5 |
| candidate | fresh_held | 6 | 6/6 | 0 |
| candidate | fresh_held | 7 | 1/6 | 5 |
| analytical | all | all | 18/24 | 5 |
| analytical | all | 6 | 11/12 | 1 |
| analytical | all | 7 | 7/12 | 4 |
| analytical | original | all | 10/12 | 1 |
| analytical | original | 6 | 6/6 | 0 |
| analytical | original | 7 | 4/6 | 1 |
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

## Paired changes

[
  {
    "seed": 0,
    "reference": 14,
    "candidate": 19,
    "gains": 5,
    "losses": 0,
    "changes": [
      {
        "case": 0,
        "dof": 6,
        "group": "original_train",
        "gain": true
      },
      {
        "case": 1,
        "dof": 6,
        "group": "original_train",
        "gain": true
      },
      {
        "case": 13,
        "dof": 6,
        "group": "broad_train",
        "gain": true
      },
      {
        "case": 19,
        "dof": 7,
        "group": "broad_train",
        "gain": true
      },
      {
        "case": 22,
        "dof": 7,
        "group": "broad_train",
        "gain": true
      }
    ]
  },
  {
    "seed": 1,
    "reference": 18,
    "candidate": 15,
    "gains": 0,
    "losses": 3,
    "changes": [
      {
        "case": 1,
        "dof": 6,
        "group": "original_train",
        "gain": false
      },
      {
        "case": 8,
        "dof": 7,
        "group": "original_held",
        "gain": false
      },
      {
        "case": 18,
        "dof": 7,
        "group": "broad_train",
        "gain": false
      }
    ]
  },
  {
    "seed": 2,
    "reference": 17,
    "candidate": 16,
    "gains": 1,
    "losses": 2,
    "changes": [
      {
        "case": 1,
        "dof": 6,
        "group": "original_train",
        "gain": false
      },
      {
        "case": 10,
        "dof": 7,
        "group": "original_train",
        "gain": false
      },
      {
        "case": 20,
        "dof": 7,
        "group": "fresh_held",
        "gain": true
      }
    ]
  }
]

Older E99 successes still lost:
[
  {
    "case": 8,
    "seed": 0,
    "group": "original_held"
  },
  {
    "case": 10,
    "seed": 1,
    "group": "original_train"
  },
  {
    "case": 2,
    "seed": 2,
    "group": "original_held"
  }
]

## Matched held-history diagnostics

| Seed | Group | Action RMSE before/after (mrad) | XYZ error before/after (mm) |
|---|---|---:|---:|
| 0 | original | 9.479 / 9.426 | 0.893 / 0.861 |
| 0 | fresh | 8.986 / 9.039 | 0.846 / 0.812 |
| 1 | original | 9.562 / 9.353 | 0.896 / 0.900 |
| 1 | fresh | 9.430 / 9.377 | 0.787 / 0.797 |
| 2 | original | 9.392 / 9.185 | 0.879 / 0.861 |
| 2 | fresh | 9.127 / 8.866 | 0.817 / 0.794 |

## Descent and lowering on matched held histories

| DOF | Phase | Reference/candidate XYZ error (mm) | Reference/candidate action error (mrad) |
|---|---|---:|---:|
| 6 | descend | 1.307 / 1.276 | 11.995 / 12.068 |
| 6 | lower | 1.661 / 1.595 | 15.502 / 14.988 |
| 7 | descend | 1.255 / 1.193 | 14.245 / 14.316 |
| 7 | lower | 1.765 / 1.744 | 26.041 / 25.986 |

Offline diagnostics use identical clean held histories for reference and candidate, with the analytical same-history action as comparator. The phase table combines original/fresh held histories and averages the three checkpoint means. Probe override actions are excluded. Full-task outcomes remain the primary gate; better offline means do not prove retained successes.

## Decision and scope

The expanded run does not fix retention. Relative to E117, pooled success changes49/72 to50/72 (six gains, five losses), while original-task success decreases26/36 to24/36. Seed0 improves14/24 to19/24 with no E117 success lost; seed1 decreases18/24 to15/24 and seed2 decreases17/24 to16/24. Three older E99 successes also remain unsolved. Do not select the favorable seed as evidence of a replicated repair.

The five new losses are localized: case1 seeds1/2 first collide during lift; case8 seed1 and shifted case18 seed1 first collide during lowering; case10 seed2 grasps but never achieves the lift criterion without a forbidden collision. Thus improved average held-history motion prediction does not guarantee stable contact trajectories. The successful-reference supervision constrains actions on recorded fitting histories, not every history reached by the updated controller. A next diagnostic should collect the failed candidate's own fitting-state histories and test corrections there, retaining held placements for evaluation.

All four targeted phase/joint-count mean XYZ prediction errors decreased, and lowering action error decreased for both joint counts, but descent action error increased slightly. These offline changes are too small and inconsistent to claim that the descent/lowering regressions are resolved.

The strict gate requires increased overall reference success, zero lost reference successes, no collision increase, and retention of the older E99 successes on the original tasks. Zero losses also prevents joint-count or seed success declines. If it fails, retain both experimental candidates and earlier references without promotion. Regardless of the outcome, unseen-body testing remains deferred as requested. The fresh held set tests new placements on familiar training bodies, not new morphology.

All84 inherited original results,48 new reference trajectories and72 new candidate trajectories remain saved. Context provenance, full state sequences, XML scenes, final checkpoints and outcome-labeled success/failure videos are retained.

Final verification: 13 independent exact replays, including fresh held placements; 18 trace-verified success/failure videos saved.

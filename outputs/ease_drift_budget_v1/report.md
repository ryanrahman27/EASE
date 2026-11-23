# E121: Accumulated command residual and episode budget

reference: 50/72, 18 collisions; unbounded: 52/72, 18 collisions; candidate: 46/72, 25 collisions.

reference = frozen E118; unbounded = E120 (per-step bound only); candidate = E121 (per-step plus signed cumulative budget). The unbounded label is inherited tooling terminology: E120 is bounded per step but has no cumulative budget.

Both neural estimators remain frozen. Preserve the E120 25% action residual and 0.001-rad/joint/step cap; additionally keep the signed sum of applied residuals within +/-0.004 rad/joint for the entire task. No resets at phase changes. Opposite residuals can unwind the sum. Calibration and the explicit probe are unchanged and do not spend the budget. This does not bound joint-state drift, path length, contact or physical safety. No physics selector, analytical fallback, privileged Jacobian or additional fitting.

All24 placements on four training bodies, repeated across three checkpoints. Sixteen fitting placements and eight held placements; these are development tasks, not untouched tests or unseen bodies. One fixed budget was tested, without a sweep or seed selection.

## Drift diagnosis

The four successful E118 trajectories were replayed exactly. State/action comparisons align phase and within-phase step. If E118 already left a phase, a paired state is unavailable; we do not substitute a different phase. Cumulative residual includes the action at the first collision; recorded q/XYZ drift is immediately before that action. These observations locate drift but do not establish that command accumulation alone caused the collision.

| Case / seed | First collision | Signed residual max (mrad) | Pre-action XYZ drift (mm) | Recovered |
|---|---|---:|---:|---|
| 22 / 0 | descend step 32 | 23.37 | 29.115530726407968 | False |
| 9 / 1 | lower step 30 | 39.35 | unavailable | True |
| 19 / 1 | lift step 6 | 36.60 | 31.78330703817963 | True |
| 18 / 2 | lift step 2 | 12.08 | 26.03587354896489 | True |

## All episode results

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
| candidate | all | all | 46/72 | 25 |
| candidate | all | 6 | 31/36 | 5 |
| candidate | all | 7 | 15/36 | 20 |
| candidate | fitting | all | 36/48 | 11 |
| candidate | fitting | 6 | 21/24 | 3 |
| candidate | fitting | 7 | 15/24 | 8 |
| candidate | held | all | 10/24 | 14 |
| candidate | held | 6 | 10/12 | 2 |
| candidate | held | 7 | 0/12 | 12 |
| analytical | all | all | 18/24 | 5 |
| analytical | all | 6 | 11/12 | 1 |
| analytical | all | 7 | 7/12 | 4 |
| analytical | fitting | all | 12/16 | 3 |
| analytical | fitting | 6 | 7/8 | 1 |
| analytical | fitting | 7 | 5/8 | 2 |
| analytical | held | all | 6/8 | 2 |
| analytical | held | 6 | 4/4 | 0 |
| analytical | held | 7 | 2/4 | 2 |

Recovered 3/4 regressions; retained 1/6 E120 gains. Strict retention gate FAILED.

## Every paired change

```json
[
  {
    "baseline": "reference",
    "case": 6,
    "seed": 0,
    "gain": false,
    "split": "fitting",
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
    "case": 1,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 2,
    "seed": 1,
    "gain": false,
    "split": "held",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 7,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
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
    "case": 7,
    "seed": 2,
    "gain": false,
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
    "baseline": "unbounded",
    "case": 6,
    "seed": 0,
    "gain": false,
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
    "case": 20,
    "seed": 0,
    "gain": false,
    "split": "held",
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
    "case": 2,
    "seed": 1,
    "gain": false,
    "split": "held",
    "dof": 6
  },
  {
    "baseline": "unbounded",
    "case": 7,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
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
    "case": 13,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 6
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
    "case": 1,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "unbounded",
    "case": 7,
    "seed": 2,
    "gain": false,
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
  }
]
```

All weights, trajectories, command/history logs, signed-budget ledgers and outcomes are saved. Do not promote a higher aggregate score if it loses previously successful cases. A failure of this intervention would show that bounding signed residual alone is insufficient; it would not rule out state/history feedback amplification or phase-timing effects.

Verification: 6 exact candidate replays; 21162 action steps independently checked against both limits; 39 trace-verified videos, including all four regression comparisons.

The episode budget recovered three regressions but retained only one of six E120 gains. Versus E120 it gained four successes and lost ten, reducing52/72 to46/72 and raising collision episodes18 to25. Reject this intervention. Preserve E118 as the frozen reference and E120 as the better correction candidate. The residual ledger remains bounded while substantial physical drift persists; see drift.png. A global signed-command budget does not solve state/history feedback or selectively retain useful corrections.

# E120: First action divergence and bounded correction

Frozen E118: 50/72 successes, 18 collision episodes. Bounded E119 correction: 52/72 successes, 18 collision episodes. Strict retention gate FAILED.

E118 and E119 weights are frozen. Both propose DLS actions on the same currently measured history. Execute E118 action plus 25% of the E119-minus-E118 difference, capped at 0.001 rad per joint per step, with the original 0.04-rad action limit. Calibration and explicit probing are unchanged. This is an action-space residual from two learned estimators, not a bound on the response matrix, accumulated state drift, or physical safety. No training, parameter sweep, checkpoint selection, physics selector, geometry filter or privileged Jacobian.

All 24 placements on four training bodies are evaluated for all three seeds (72 checkpoint/placement episodes, not 72 independent tasks). There are 16 fitting and eight held placements. These placements have been used in prior development; held means excluded from estimator fitting, not an untouched final test. No unseen-body rollout.

## First divergence on the seven previously lost successes

Both full trajectories are exactly replayed for every lost-success pair. At each actually reached state, proposals from both models use identical observations, Gram/cross statistics and calibrated bias. Numerical divergence threshold: 1e-10 rad; material threshold: 1e-4 rad. Step indices are zero-based after calibration. First divergence identifies where behavior starts separating, not the sole cause of the later failure.

| Seed | Case | First differing action | First >0.1 mrad | First collision under E119 | Restored by bound |
|---|---|---|---|---|---|
| 0 | 12 | 0 / approach | 0 / approach | 139 / lift | True |
| 0 | 14 | 0 / approach | 0 / approach | 94 / descend | True |
| 0 | 22 | 0 / approach | 0 / approach | 36 / descend | False |
| 1 | 7 | 0 / approach | 0 / approach | 248 / retreat | True |
| 1 | 9 | 0 / approach | 0 / approach | none | False |
| 2 | 4 | 0 / approach | 0 / approach | 99 / lift | True |
| 2 | 18 | 0 / approach | 0 / approach | 192 / lower | False |

## Full episode comparisons

reference = frozen E118; unbounded = E119; candidate = bounded correction. Analytical results are reused once per placement, not duplicated for neural seeds.

| Method | Split | DOF | Success | Collisions |
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
| unbounded | all | all | 47/72 | 21 |
| unbounded | all | 6 | 29/36 | 7 |
| unbounded | all | 7 | 18/36 | 14 |
| unbounded | fitting | all | 33/48 | 11 |
| unbounded | fitting | 6 | 18/24 | 6 |
| unbounded | fitting | 7 | 15/24 | 5 |
| unbounded | held | all | 14/24 | 10 |
| unbounded | held | 6 | 11/12 | 1 |
| unbounded | held | 7 | 3/12 | 9 |
| unbounded | original_train | all | 17/24 | 4 |
| unbounded | original_train | 6 | 9/12 | 3 |
| unbounded | original_train | 7 | 8/12 | 1 |
| unbounded | original_held | all | 8/12 | 4 |
| unbounded | original_held | 6 | 6/6 | 0 |
| unbounded | original_held | 7 | 2/6 | 4 |
| unbounded | broad_train | all | 16/24 | 7 |
| unbounded | broad_train | 6 | 9/12 | 3 |
| unbounded | broad_train | 7 | 7/12 | 4 |
| unbounded | fresh_held | all | 6/12 | 6 |
| unbounded | fresh_held | 6 | 5/6 | 1 |
| unbounded | fresh_held | 7 | 1/6 | 5 |
| candidate | all | all | 52/72 | 18 |
| candidate | all | 6 | 33/36 | 3 |
| candidate | all | 7 | 19/36 | 15 |
| candidate | fitting | all | 37/48 | 9 |
| candidate | fitting | 6 | 22/24 | 2 |
| candidate | fitting | 7 | 15/24 | 7 |
| candidate | held | all | 15/24 | 9 |
| candidate | held | 6 | 11/12 | 1 |
| candidate | held | 7 | 4/12 | 8 |
| candidate | original_train | all | 19/24 | 3 |
| candidate | original_train | 6 | 11/12 | 1 |
| candidate | original_train | 7 | 8/12 | 2 |
| candidate | original_held | all | 6/12 | 6 |
| candidate | original_held | 6 | 5/6 | 1 |
| candidate | original_held | 7 | 1/6 | 5 |
| candidate | broad_train | all | 18/24 | 6 |
| candidate | broad_train | 6 | 11/12 | 1 |
| candidate | broad_train | 7 | 7/12 | 5 |
| candidate | fresh_held | all | 9/12 | 3 |
| candidate | fresh_held | 6 | 6/6 | 0 |
| candidate | fresh_held | 7 | 3/6 | 3 |
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

## Paired changes from frozen E118

```json
[
  {
    "seed": 0,
    "reference": 19,
    "candidate": 20,
    "gains": 2,
    "losses": 1,
    "changes": [
      {
        "case": 8,
        "group": "original_held",
        "fitting": false,
        "gain": true,
        "collision": false
      },
      {
        "case": 20,
        "group": "fresh_held",
        "fitting": false,
        "gain": true,
        "collision": false
      },
      {
        "case": 22,
        "group": "broad_train",
        "fitting": true,
        "gain": false,
        "collision": true
      }
    ]
  },
  {
    "seed": 1,
    "reference": 15,
    "candidate": 16,
    "gains": 3,
    "losses": 2,
    "changes": [
      {
        "case": 9,
        "group": "original_train",
        "fitting": true,
        "gain": false,
        "collision": true
      },
      {
        "case": 13,
        "group": "broad_train",
        "fitting": true,
        "gain": true,
        "collision": false
      },
      {
        "case": 18,
        "group": "broad_train",
        "fitting": true,
        "gain": true,
        "collision": false
      },
      {
        "case": 19,
        "group": "broad_train",
        "fitting": true,
        "gain": false,
        "collision": true
      },
      {
        "case": 23,
        "group": "fresh_held",
        "fitting": false,
        "gain": true,
        "collision": false
      }
    ]
  },
  {
    "seed": 2,
    "reference": 16,
    "candidate": 16,
    "gains": 1,
    "losses": 1,
    "changes": [
      {
        "case": 1,
        "group": "original_train",
        "fitting": true,
        "gain": true,
        "collision": false
      },
      {
        "case": 18,
        "group": "broad_train",
        "fitting": true,
        "gain": false,
        "collision": true
      }
    ]
  }
]
```

All three seeds and all outcomes are retained. An improved total is insufficient for promotion if any baseline success is lost. The correction bound limits instantaneous action changes; it cannot ensure that later observations or phase timing match the frozen baseline. Full actions, histories, simulator states and labeled success/failure replays are preserved.

Final verification: 6 bounded episodes exactly replayed; zero-correction full episode exactly matches E118; 21494 action steps independently checked against the prescribed bound; 34 trace-verified videos. All168 inherited trials match their source, including original estimator telemetry. Bounded-run telemetry reports the frozen baseline response prediction error; applied actions and both same-history proposals are separately logged in actions/.

Interpretation: all seven lost-success pairs differ on the first executed approach action and first resulting state, despite identical initial observations and history statistics. Maximum per-joint first-action differences span0.667–7.357 mrad. The bound restores four of those seven successes. Relative to E118, it gains six successes but loses four; all four losses are fitting placements with collisions. Held placements improve12/24 to15/24 without losing an E118 held success. This development result is promising but fails the no-regression criterion, so E118 remains the reference. Next isolate accumulated drift and phase transitions in the four remaining regressions before choosing another intervention.

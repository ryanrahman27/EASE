# E107: short physics contact predictions on training bodies

All36 unchanged training-body task/seed trajectories exactly replay E104. All1633 predicted40ms qpos/qvel states and forbidden-contact unions exactly match executed physics. No control intervention, new weights or reserved-test evaluation.

The predictor has privileged access to the exact physical model and complete simulator state. Exact one-step agreement verifies implementation fidelity; it is not evidence of a learned dynamics model or successful collision avoidance. All controllers, commands and collision rules remain unchanged. The held split contains new placements on the same four training bodies, not unseen embodiments.

Static warning means an infeasible filter decision or selected clearance below2mm minus1e-6. Physics warning means any forbidden contact while holding the selected absolute joint/finger target for40/80/160ms. Longer-horizon comparison uses the actual replanning policy as the outcome: false alarms and misses can arise from future command changes. Tails without a complete within-retreat comparison horizon are censored.

## Prediction counts

| Split | Clean start only | Horizon ms | Predictor | N | TP | FP | FN | TN | Censored |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| all | False | 40 | static | 1633 | 43 | 96 | 1 | 1493 | 0 |
| all | False | 40 | physics | 1633 | 44 | 0 | 0 | 1589 | 0 |
| all | False | 80 | static | 1597 | 44 | 93 | 1 | 1459 | 36 |
| all | False | 80 | physics | 1597 | 45 | 1 | 0 | 1551 | 36 |
| all | False | 160 | static | 1525 | 46 | 88 | 1 | 1390 | 108 |
| all | False | 160 | physics | 1525 | 44 | 2 | 3 | 1476 | 108 |
| all | True | 40 | static | 1591 | 2 | 96 | 0 | 1493 | 0 |
| all | True | 40 | physics | 1591 | 2 | 0 | 0 | 1589 | 0 |
| all | True | 80 | static | 1556 | 4 | 93 | 0 | 1459 | 35 |
| all | True | 80 | physics | 1556 | 4 | 1 | 0 | 1551 | 35 |
| all | True | 160 | static | 1486 | 8 | 88 | 0 | 1390 | 105 |
| all | True | 160 | physics | 1486 | 5 | 2 | 3 | 1476 | 105 |
| train | False | 40 | static | 1099 | 10 | 41 | 0 | 1048 | 0 |
| train | False | 40 | physics | 1099 | 10 | 0 | 0 | 1089 | 0 |
| train | False | 80 | static | 1075 | 11 | 40 | 0 | 1024 | 24 |
| train | False | 80 | physics | 1075 | 11 | 0 | 0 | 1064 | 24 |
| train | False | 160 | static | 1027 | 13 | 38 | 0 | 976 | 72 |
| train | False | 160 | physics | 1027 | 11 | 1 | 2 | 1013 | 72 |
| train | True | 40 | static | 1090 | 1 | 41 | 0 | 1048 | 0 |
| train | True | 40 | physics | 1090 | 1 | 0 | 0 | 1089 | 0 |
| train | True | 80 | static | 1066 | 2 | 40 | 0 | 1024 | 24 |
| train | True | 80 | physics | 1066 | 2 | 0 | 0 | 1064 | 24 |
| train | True | 160 | static | 1018 | 4 | 38 | 0 | 976 | 72 |
| train | True | 160 | physics | 1018 | 2 | 1 | 2 | 1013 | 72 |
| held | False | 40 | static | 534 | 33 | 55 | 1 | 445 | 0 |
| held | False | 40 | physics | 534 | 34 | 0 | 0 | 500 | 0 |
| held | False | 80 | static | 522 | 33 | 53 | 1 | 435 | 12 |
| held | False | 80 | physics | 522 | 34 | 1 | 0 | 487 | 12 |
| held | False | 160 | static | 498 | 33 | 50 | 1 | 414 | 36 |
| held | False | 160 | physics | 498 | 33 | 1 | 1 | 463 | 36 |
| held | True | 40 | static | 501 | 1 | 55 | 0 | 445 | 0 |
| held | True | 40 | physics | 501 | 1 | 0 | 0 | 500 | 0 |
| held | True | 80 | static | 490 | 2 | 53 | 0 | 435 | 11 |
| held | True | 80 | physics | 490 | 2 | 1 | 0 | 487 | 11 |
| held | True | 160 | static | 468 | 4 | 50 | 0 | 414 | 33 |
| held | True | 160 | physics | 468 | 3 | 1 | 1 | 463 | 33 |

## Interpretation

At40ms, both methods detect the two contact-onset steps from clean states. Physics has zero false positives versus96 static warnings without subsequent contact. There are no static misses in this training-body baseline sample, so it cannot establish improved recall over static checks. The two positives are case1/seed0 on body84000 (train placement) and case8/seed2 on body84004 (held placement); this is sparse positive coverage. Clean means no forbidden contact at the current boundary, not an entirely collision-free episode prefix. The static2mm warning is a margin criterion, not designed as a calibrated binary physical-contact classifier; its false positives need not be useless safety margins.
At80ms, held-target physics has one false positive and no misses; at160ms it has two false positives and three missed overlapping future windows. These windows cover only two eventual contact events and are not independent failures. Longer hold predictions cannot substitute for replanning-aware prediction. All predictions evaluate the command already selected by the static filter, not alternative candidate actions.
The controller suite passes131 tests (13.51s). Runtime invariants additionally verify all36 full baseline replays and1633 exact one-step states/contact unions. No action-avoidance claim follows from the audit.

## Cost

{
  "40": {
    "median_ms": 2.5141880032606423,
    "p95_ms": 2.8765817995008547
  },
  "80": {
    "median_ms": 4.443752000952372,
    "p95_ms": 4.990254198492039
  },
  "160": {
    "median_ms": 8.308811000460992,
    "p95_ms": 9.163165199424839
  }
}

Wall-clock prediction time includes cloning, step simulation and contact logging; values are machine/load specific and exclude neural inference. Later horizons include earlier work. No claim of hardware real-time readiness.

## Failure examples

[
  {
    "horizon_ms": 40,
    "category": "static_miss_physics_hit",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 40,
    "category": "physics_false_alarm",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 40,
    "category": "physics_miss",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 80,
    "category": "static_miss_physics_hit",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 80,
    "category": "physics_false_alarm",
    "count": 1,
    "first_examples": [
      {
        "case": 2,
        "seed": 0,
        "body": 84000,
        "step": 230,
        "validation": true
      }
    ]
  },
  {
    "horizon_ms": 80,
    "category": "physics_miss",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 160,
    "category": "static_miss_physics_hit",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 160,
    "category": "physics_false_alarm",
    "count": 2,
    "first_examples": [
      {
        "case": 1,
        "seed": 0,
        "body": 84000,
        "step": 326,
        "validation": false
      },
      {
        "case": 2,
        "seed": 0,
        "body": 84000,
        "step": 230,
        "validation": true
      }
    ]
  },
  {
    "horizon_ms": 160,
    "category": "physics_miss",
    "count": 3,
    "first_examples": [
      {
        "case": 1,
        "seed": 0,
        "body": 84000,
        "step": 312,
        "validation": false
      },
      {
        "case": 1,
        "seed": 0,
        "body": 84000,
        "step": 313,
        "validation": false
      },
      {
        "case": 8,
        "seed": 2,
        "body": 84004,
        "step": 344,
        "validation": true
      }
    ]
  }
]

## Next decision

Use40ms physical lookahead as a candidate training-only action check if it catches static misses. A warning is not an avoidance action: the next controlled experiment must evaluate candidate fallback actions in copied physics, account for progress and joint limits, and retain every successful training/held-placement task. Longer held-target horizons are diagnostic and should not be promoted merely for detecting more future contacts. No controller has been changed by E107.

## Artifacts

Full prediction records, source, checkpoint/input hashes and per-trajectory original results are saved. The video gallery reuses verified E104 baseline success/failure executions, explicitly labelled as unchanged trajectories. Counts are correlated control steps, not independent trials; do not interpret them as independent-sample confidence intervals.

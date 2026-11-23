# E105: prospective body and placement replication

15/16 candidate placements passed endpoint IK screening. All accepted cases, including no-route failures, received all three checkpoint seeds and all three controls: 135 physical task rollouts. No replacement. Nine E104 compatibility runs match full traces and outcomes.

This cohort has eight newly perturbed bodies and two new placements per body. It shares the four E87 axis orders and procedural generator: it tests fresh parameterizations/placements, not new axis arrangements or real robots. E99 weights and E104 retreat intervention were frozen before outcomes. Privileged IK, route planning and geometric filtering remain. No reserved final-test evaluation.

| Group | Mode | Success | Collisions |
|---|---|---:|---:|
| all | baseline | 27/45 | 17 |
| all | budget | 27/45 | 17 |
| all | gentle | 27/45 | 17 |
| 6 | baseline | 10/21 | 11 |
| 6 | budget | 10/21 | 11 |
| 6 | gentle | 10/21 | 11 |
| 7 | baseline | 17/24 | 6 |
| 7 | budget | 17/24 | 6 |
| 7 | gentle | 17/24 | 6 |
| seed0 | baseline | 10/15 | 4 |
| seed0 | budget | 10/15 | 4 |
| seed0 | gentle | 10/15 | 4 |
| seed1 | baseline | 7/15 | 8 |
| seed1 | budget | 7/15 | 8 |
| seed1 | gentle | 8/15 | 7 |
| seed2 | baseline | 10/15 | 5 |
| seed2 | budget | 10/15 | 5 |
| seed2 | gentle | 9/15 | 6 |
| route | baseline | 27/42 | 14 |
| route | budget | 27/42 | 14 |
| route | gentle | 27/42 | 14 |
| no_route | baseline | 0/3 | 3 |
| no_route | budget | 0/3 | 3 |
| no_route | gentle | 0/3 | 3 |
| body0 | baseline | 4/6 | 2 |
| body0 | budget | 4/6 | 2 |
| body0 | gentle | 4/6 | 2 |
| body1 | baseline | 5/6 | 1 |
| body1 | budget | 5/6 | 1 |
| body1 | gentle | 6/6 | 0 |
| body2 | baseline | 1/6 | 5 |
| body2 | budget | 1/6 | 5 |
| body2 | gentle | 0/6 | 6 |
| body3 | baseline | 0/3 | 3 |
| body3 | budget | 0/3 | 3 |
| body3 | gentle | 0/3 | 3 |
| body4 | baseline | 4/6 | 1 |
| body4 | budget | 4/6 | 1 |
| body4 | gentle | 4/6 | 1 |
| body5 | baseline | 5/6 | 1 |
| body5 | budget | 5/6 | 1 |
| body5 | gentle | 5/6 | 1 |
| body6 | baseline | 2/6 | 4 |
| body6 | budget | 2/6 | 4 |
| body6 | gentle | 2/6 | 4 |
| body7 | baseline | 6/6 | 0 |
| body7 | budget | 6/6 | 0 |
| body7 | gentle | 6/6 | 0 |

## Paired changes

[
  {
    "control": "baseline",
    "case": 2,
    "seed": 1,
    "body": 1,
    "dof": 6,
    "change": "gain"
  },
  {
    "control": "baseline",
    "case": 5,
    "seed": 2,
    "body": 2,
    "dof": 6,
    "change": "loss"
  },
  {
    "control": "budget",
    "case": 2,
    "seed": 1,
    "body": 1,
    "dof": 6,
    "change": "gain"
  },
  {
    "control": "budget",
    "case": 5,
    "seed": 2,
    "body": 2,
    "dof": 6,
    "change": "loss"
  }
]

## Prespecified operational gate

**FAIL**

{
  "baseline": {
    "success_improves": false,
    "retains_all_successes": false,
    "collisions_nonincreasing": true,
    "per_seed_nondecreasing": false
  },
  "budget": {
    "success_improves": false,
    "retains_all_successes": false,
    "collisions_nonincreasing": true,
    "per_seed_nondecreasing": false
  }
}

## Body-clustered uncertainty

The fixed candidate pipeline denominator is48 (16 placements x3 seeds); an IK rejection counts as zero pipeline success for every mode. Success differences are resampled over eight bodies, keeping both placements and all three seeds together. 10,000 fixed-seed bootstrap samples give descriptive percentile intervals. Eight bodies and four shared axis orders limit precision; these are not 48 independent embodiment samples.

{
  "baseline": {
    "pipeline_difference": 0.0,
    "percentile_95": [
      -0.0625,
      0.0625
    ],
    "body_differences": [
      0.0,
      0.16666666666666666,
      -0.16666666666666666,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "clusters": 8,
    "resamples": 10000
  },
  "budget": {
    "pipeline_difference": 0.0,
    "percentile_95": [
      -0.0625,
      0.0625
    ],
    "body_differences": [
      0.0,
      0.16666666666666666,
      -0.16666666666666666,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "clusters": 8,
    "resamples": 10000
  }
}

## Next decision

Do not promote this as a robust transfer improvement. Keep the frozen outcome and diagnose the paired regressions before proposing another intervention; do not retune this cohort and call it independent confirmation.

Raw traces, protocol, screening records, static planner attempts, checkpoint hashes, source snapshots, and replay-verified success/failure videos are saved alongside this report.

## Changed-pair trace diagnostics

First forbidden contact is sampled at control-step boundaries (40ms); it does not establish the substep contact mechanism. Full pre-retreat prefixes match controls.

[
  {
    "case": 2,
    "seed": 1,
    "mode": "baseline",
    "success": false,
    "first_bad_phase": "retreat",
    "first_bad_pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "final_xy_error": 0.002790932163773178
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "budget",
    "success": false,
    "first_bad_phase": "retreat",
    "first_bad_pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "final_xy_error": 0.0027909321638295665
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "gentle",
    "success": true,
    "first_bad_phase": null,
    "first_bad_pairs": [],
    "final_xy_error": 0.002917784212972583
  },
  {
    "case": 5,
    "seed": 2,
    "mode": "baseline",
    "success": true,
    "first_bad_phase": null,
    "first_bad_pairs": [],
    "final_xy_error": 0.00431118871969557
  },
  {
    "case": 5,
    "seed": 2,
    "mode": "budget",
    "success": true,
    "first_bad_phase": null,
    "first_bad_pairs": [],
    "final_xy_error": 0.004311188720416101
  },
  {
    "case": 5,
    "seed": 2,
    "mode": "gentle",
    "success": false,
    "first_bad_phase": "retreat",
    "first_bad_pairs": [
      [
        "gripper_left_pad",
        "table"
      ]
    ],
    "final_xy_error": 0.008926970904773053
  }
]

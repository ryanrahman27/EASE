# E112: frozen progress-aware selector on prospective unseen bodies

16/16 candidate placements passed endpoint IK screening; 96 complete paired rollouts across all three checkpoint seeds. Every rejection and accepted-task failure is retained. No route planner, extra task budget, weight update or selector tuning.

Eight new parameterizations of four E87 axis orders, two bodies per order and two placements per body. The axis layouts differ from the original training bodies but were inspected in earlier development. This is a prospective body-parameter replication, not an untouched axis-family test or the reserved final test. Procedural simulator arms; exact physics/state/geometry and IK initialization remain privileged.

| Group | Mode | Success | Collisions | Contact-free failures | Changed actions |
|---|---|---:|---:|---:|---:|
| all | baseline | 28/48 | 13 | 7 | 0 |
| all | physics | 31/48 | 10 | 7 | 68 |
| 6 | baseline | 14/24 | 10 | 0 | 0 |
| 6 | physics | 14/24 | 10 | 0 | 65 |
| 7 | baseline | 14/24 | 3 | 7 | 0 |
| 7 | physics | 17/24 | 0 | 7 | 3 |
| seed0 | baseline | 9/16 | 5 | 2 | 0 |
| seed0 | physics | 10/16 | 4 | 2 | 25 |
| seed1 | baseline | 8/16 | 5 | 3 | 0 |
| seed1 | physics | 9/16 | 4 | 3 | 29 |
| seed2 | baseline | 11/16 | 3 | 2 | 0 |
| seed2 | physics | 12/16 | 2 | 2 | 14 |
| body0 | baseline | 4/6 | 2 | 0 | 0 |
| body0 | physics | 4/6 | 2 | 0 | 0 |
| body1 | baseline | 6/6 | 0 | 0 | 0 |
| body1 | physics | 6/6 | 0 | 0 | 0 |
| body2 | baseline | 1/6 | 5 | 0 | 0 |
| body2 | physics | 1/6 | 5 | 0 | 49 |
| body3 | baseline | 3/6 | 3 | 0 | 0 |
| body3 | physics | 3/6 | 3 | 0 | 16 |
| body4 | baseline | 0/6 | 0 | 6 | 0 |
| body4 | physics | 0/6 | 0 | 6 | 0 |
| body5 | baseline | 3/6 | 3 | 0 | 0 |
| body5 | physics | 6/6 | 0 | 0 | 3 |
| body6 | baseline | 5/6 | 0 | 1 | 0 |
| body6 | physics | 5/6 | 0 | 1 | 0 |
| body7 | baseline | 6/6 | 0 | 0 | 0 |
| body7 | physics | 6/6 | 0 | 0 | 0 |

## Paired changes

[
  {
    "case": 10,
    "seed": 0,
    "body": 5,
    "dof": 7,
    "change": "gain"
  },
  {
    "case": 11,
    "seed": 1,
    "body": 5,
    "dof": 7,
    "change": "gain"
  },
  {
    "case": 11,
    "seed": 2,
    "body": 5,
    "dof": 7,
    "change": "gain"
  }
]

## Frozen transfer gate

**PASS**

{
  "success_improves": true,
  "retains_all_successes": true,
  "collisions_nonincreasing": true,
  "per_seed_nondecreasing": true,
  "per_dof_nondecreasing": true
}

## Screening and uncertainty

[
  {
    "candidate": 0,
    "body": 0,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 1,
    "body": 0,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 2,
    "body": 1,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 3,
    "body": 1,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 4,
    "body": 2,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 5,
    "body": 2,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 6,
    "body": 3,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 7,
    "body": 3,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 8,
    "body": 4,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 9,
    "body": 4,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 10,
    "body": 5,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 11,
    "body": 5,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 12,
    "body": 6,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 13,
    "body": 6,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 14,
    "body": 7,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 15,
    "body": 7,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  }
]

Pipeline success uses all48 candidate/seed pairs, with screened-out placements counted as zero for both modes. The bootstrap resamples eight whole bodies, preserving correlated placements and checkpoints; 10,000 fixed-seed resamples provide descriptive intervals, not evidence from48 independent embodiments.

{
  "pipeline_difference": 0.0625,
  "percentile_95": [
    0.0,
    0.1875
  ],
  "body_differences": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0
  ],
  "clusters": 8,
  "resamples": 10000
}

## Compute

{
  "check_only": {
    "n": 8336,
    "median_ms": 4.408034497828339,
    "p95_ms": 5.049674751717248
  },
  "search": {
    "n": 74,
    "median_ms": 519.8373780003749,
    "p95_ms": 600.3635065488197
  }
}

Added prediction/search wall time excludes neural inference, original static filtering and normal execution. Compare search latency with the40ms control interval; this is not a hardware real-time claim.

## Interpretation

The frozen selector passes the prespecified transfer gate on this prospective cohort. This supports transfer of the privileged physics-assisted component; it does not isolate learned history adaptation or demonstrate history-only control.

Before a paper claim, separate the history-conditioned neural component from exact-model search using analytical/specialist/history ablations, assess compute, and broaden independent embodiment evidence. No method changes were made after outcomes.

Recovered episodes occur on 1/8 bodies (body IDs [5]). The body-bootstrap interval includes zero when its lower endpoint is zero; passing the operational retention gate is not statistical confirmation of broad transfer. Each of the three observed recoveries in this run uses one action substitution. The six-joint controller makes65 substitutions without changing any task outcome; safe local progress is insufficient for those failures.

## Remaining failures

{
  "release / unmodified phase": 4,
  "lower / no candidate meets all frozen criteria": 3,
  "descend / no candidate meets all frozen criteria": 3,
  "no collision / task criteria failed": 7
}

In this cohort, all ten remaining collision episodes are six-joint: six first collide in descend/lower when no candidate satisfies every frozen criterion, and four first collide during the unmodified release phase. The seven contact-free failures are seven-joint; six belong to one body that misses descent/close tolerances. These are trace-level diagnostics, not controlled causal ablations. The next diagnostic should examine those six-joint states and release transitions without tuning this frozen evaluation result.

## Verification and artifacts

{
  "exact_compatibility_replays": 12,
  "exact_paired_prefixes": 48,
  "exact_selected_predictions": 17104,
  "hashes_unchanged": true,
  "reserved_test_evaluated": false
}

{
  "frozen_nested_functions": [
    "predict",
    "select",
    "step"
  ],
  "nested_function_asts_identical": true,
  "e111_source_hashes_identical": true,
  "full_serialized_compatibility_replays": 12,
  "distinct_from_original_training_body_axes": true,
  "new_body_seeds": true,
  "component_tests": {
    "passed": 16,
    "seconds": 4.41,
    "files": [
      "test_ease_fm_geometry_filter.py",
      "test_ease_fm_local_history.py",
      "test_ease_fm_local_probe.py",
      "test_ease_fm_task_retention_training.py",
      "test_ease_fm_pick_place.py",
      "test_ease_fm_arm_pick_place.py"
    ]
  }
}

Saved protocol, source snapshots, checkpoint/input hashes, all raw traces and failures, timing measurements, and replay-verified success/failure videos. Twelve prespecified E111 compatibility runs match full serialized results. All selected predicted next states/contact unions match execution; all paired calibration/approach prefixes match. See failure_diagnostics.json for every failed rollout.

Final artifact verification: 12 full-result video replays and SHA256 hashes verified.

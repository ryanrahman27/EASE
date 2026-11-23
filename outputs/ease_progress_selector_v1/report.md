# E111: progress-aware selection with geometric nonpenetration

72 full tasks: all36 original training-body task/seed pairs, baseline and fixed physics selector. Frozen E110 broad bank,40ms prediction, original static admissibility, zero physical contacts, nonnegative minimum geometric clearance and at least1% pose-error reduction (floor1e-8). Rank by smallest predicted pose error, then larger clearance, then name. Preserve original commands unless contact or penetration is predicted; preserve original if no candidate qualifies. Same weights, apertures and budgets. Extend eligibility to descend/lift/transport/lower/retreat, but only before any earlier forbidden contact. Approach/calibration unchanged; no route planner. Descent interventions can alter later probe/history states. No tuning after outcomes.
The24 training and12 held-placement pairs share four original training bodies. No E105 or reserved final test. Exact physical-model/state access is privileged; this is not learned dynamics or history-only control.

| Group | Mode | Success | Collisions | Contact-free failures | Grasped | Lifted | Steps | Changes | Warnings | No safe option |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | baseline | 22/36 | 14 | 0 | 36 | 36 | 13406 | 0 | 14 | 0 |
| all | physics | 31/36 | 3 | 2 | 36 | 36 | 13453 | 54 | 57 | 3 |
| train | baseline | 18/24 | 6 | 0 | 24 | 24 | 8786 | 0 | 6 | 0 |
| train | physics | 24/24 | 0 | 0 | 24 | 24 | 8663 | 21 | 21 | 0 |
| validation | baseline | 4/12 | 8 | 0 | 12 | 12 | 4620 | 0 | 8 | 0 |
| validation | physics | 7/12 | 3 | 2 | 12 | 12 | 4790 | 33 | 36 | 3 |
| 6 | baseline | 14/18 | 4 | 0 | 18 | 18 | 5803 | 0 | 4 | 0 |
| 6 | physics | 17/18 | 1 | 0 | 18 | 18 | 5775 | 4 | 5 | 1 |
| 7 | baseline | 8/18 | 10 | 0 | 18 | 18 | 7603 | 0 | 10 | 0 |
| 7 | physics | 14/18 | 2 | 2 | 18 | 18 | 7678 | 50 | 52 | 2 |
| seed0 | baseline | 7/12 | 5 | 0 | 12 | 12 | 4344 | 0 | 5 | 0 |
| seed0 | physics | 11/12 | 1 | 0 | 12 | 12 | 4319 | 13 | 14 | 1 |
| seed1 | baseline | 7/12 | 5 | 0 | 12 | 12 | 4644 | 0 | 5 | 0 |
| seed1 | physics | 10/12 | 1 | 1 | 12 | 12 | 4535 | 21 | 22 | 1 |
| seed2 | baseline | 8/12 | 4 | 0 | 12 | 12 | 4418 | 0 | 4 | 0 |
| seed2 | physics | 10/12 | 1 | 1 | 12 | 12 | 4599 | 20 | 21 | 1 |

## Paired task changes

[
  {
    "case": 1,
    "seed": 0,
    "dof": 6,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 1,
    "seed": 2,
    "dof": 6,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 2,
    "seed": 1,
    "dof": 6,
    "validation": true,
    "change": "gain",
    "collision": false
  },
  {
    "case": 6,
    "seed": 1,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 8,
    "seed": 0,
    "dof": 7,
    "validation": true,
    "change": "gain",
    "collision": false
  },
  {
    "case": 10,
    "seed": 0,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 10,
    "seed": 1,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 10,
    "seed": 2,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 11,
    "seed": 0,
    "dof": 7,
    "validation": true,
    "change": "gain",
    "collision": false
  }
]

## Retention gate

**PASS**

{
  "train": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true,
    "success_improves": true,
    "per_seed_nondecreasing": true
  },
  "validation": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true
  }
}

## Task progress

Phase reached counts use the original position/tool-axis completion rules, including final-state tolerance on timed-out phases. Phases still advance at the original time limit. Grasped/lifted are original sticky flags, not maintained-grasp guarantees.

| Phase | Mode | Reached /36 | Control steps |
|---|---|---:|---:|
| descend | baseline | 34 | 1024 |
| descend | physics | 35 | 990 |
| lift | baseline | 32 | 1894 |
| lift | physics | 30 | 1954 |
| transport | baseline | 33 | 899 |
| transport | physics | 31 | 981 |
| lower | baseline | 36 | 1134 |
| lower | physics | 35 | 1049 |
| retreat | baseline | 32 | 1633 |
| retreat | physics | 31 | 1657 |

Changed-action progress uses same-state40ms squared position plus .15-scaled tool-axis error. The original action is a simulator counterfactual and may collide; compare it only as a diagnostic. Predicted progress gates and ranks candidate selection in this experiment; this is a selection criterion, not independent evidence of generalization. Actions and seeds are correlated, not independent samples.

[
  {
    "phase": "descend",
    "changed": 15,
    "choices": {
      "directed1.0": 1,
      "random33": 1,
      "random1": 1,
      "axis1_-1_0.04": 1,
      "random21": 1,
      "random43": 1,
      "random53": 1,
      "random35": 1,
      "random15": 1,
      "random11": 1,
      "random45": 1,
      "random37": 1,
      "random55": 1,
      "random27": 1,
      "random57": 1
    },
    "selected_reduces_error": 15,
    "selected_increases_error": 0,
    "original_reduces_error": 1
  },
  {
    "phase": "lift",
    "changed": 15,
    "choices": {
      "random43": 1,
      "random41": 1,
      "random45": 1,
      "random53": 1,
      "random21": 1,
      "random15": 1,
      "random1": 1,
      "random27": 1,
      "random29": 1,
      "random9": 2,
      "random23": 1,
      "random25": 1,
      "random17": 1,
      "random49": 1
    },
    "selected_reduces_error": 15,
    "selected_increases_error": 0,
    "original_reduces_error": 15
  },
  {
    "phase": "transport",
    "changed": 7,
    "choices": {
      "random49": 1,
      "random35": 2,
      "random9": 1,
      "random27": 2,
      "random63": 1
    },
    "selected_reduces_error": 7,
    "selected_increases_error": 0,
    "original_reduces_error": 4
  },
  {
    "phase": "lower",
    "changed": 6,
    "choices": {
      "random47": 1,
      "random40": 1,
      "directed1.0": 3,
      "axis2_1_0.04": 1
    },
    "selected_reduces_error": 6,
    "selected_increases_error": 0,
    "original_reduces_error": 2
  },
  {
    "phase": "retreat",
    "changed": 11,
    "choices": {
      "directed0.5": 1,
      "random21": 1,
      "random41": 1,
      "random25": 1,
      "random6": 1,
      "random63": 1,
      "random53": 1,
      "random11": 1,
      "random37": 1,
      "random57": 1,
      "random61": 1
    },
    "selected_reduces_error": 11,
    "selected_increases_error": 0,
    "original_reduces_error": 10
  }
]

## Failures

[
  {
    "case": 1,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "retreat",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 1,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 2,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 2,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "transport",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "transport"
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "lift",
    "selector_status": "no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 10,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "transport",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "release"
    ]
  },
  {
    "case": 11,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": null,
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport"
    ]
  },
  {
    "case": 11,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 11,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": null,
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "lower",
      "release",
      "retreat",
      "settle"
    ]
  }
]

## Decision

The fixed rule passes the training/held-placement gate; it remains a candidate requiring a separately frozen broader evaluation.
No horizon, candidate or phase tuning after these outcomes. Full traces retain all failures, alternatives and state predictions. Before another control change, identify whether failure is due to no feasible alternative, loss of task progress, or a contact in an unmodified phase.

## Verification

36 full baseline replays, 72 approach prefixes, and 13215 selected next states/contact unions match exactly. All changed actions occur before any prior collision. Inputs/checkpoints unchanged. Videos replay full serialized results; task success/failure videos are executed motion, not lookahead animation.

The broader search uses exact physics and finite-difference simulated response. This is a privileged model-based controller component, not a new learned policy. See protocol.json for frozen candidate counts, random seeds and thresholds.

The inherited held-placement label denotes placement holdout from the original training data, not an untouched evaluation: these placements were inspected in earlier development experiments, including E110. Treat this gate as development retention, not independent generalization confirmation.

## Compute and margin

{
  "physics_all": {
    "n": 6631,
    "median_ms": 4.48717399922316,
    "p95_ms": 5.124834999151062,
    "total_seconds": 59.97994358582582,
    "branches": 12999
  },
  "physics_search": {
    "n": 57,
    "median_ms": 530.0142250016506,
    "p95_ms": 582.4403586018889,
    "total_seconds": 30.281295568998758,
    "branches": 6425
  },
  "physics_check_only": {
    "n": 6574,
    "median_ms": 4.48464999863063,
    "p95_ms": 5.076675300733768,
    "total_seconds": 29.698648016827065,
    "branches": 6574
  }
}

{
  "changed_actions": 54,
  "changed_with_2mm": 19
}
Measured times are CPU wall time for added prediction/search after the original static-filter decision; they exclude neural inference, normal execution and the original static-filter cost. Baseline was instrumented too, so these are not an uninstrumented whole-policy speed comparison. The physical control interval is40ms. A search slower than40ms is not real-time ready. Every changed action has nonnegative predicted clearance; only the separate count maintains2mm throughout prediction.

## Research progress

We have a reproducible state-based simulation pipeline, frozen neural checkpoints, baseline/retention gates, raw trajectories and replay-verified success/failure videos. E104 improved a reused novel-axis development suite from32/48 to35/48, but E105 fresh-body replication tied baseline at27/45 and introduced a paired regression. E108/E109 short physics fallbacks tied the training baseline at22/36; E109 degraded intermediate phase progress. E110 showed broader local action availability in13/14 warning states, which motivated this full-episode trial.
The central unresolved claim is reliable history-conditioned control on unseen embodiments. Current physics-assisted interventions use exact state/model access and are not evidence of history-only adaptation. Before treating this as a conference result, the final method needs reliable full-task improvement, an untouched prospective embodiment evaluation, appropriate analytical/specialist/history ablations, and clear accounting of privileged inputs and realistic-robot scope. This experiment is one controlled step toward that evidence, not publication readiness.
E111 now improves complete-task success from22/36 to31/36 (nine gains, zero losses), including24/24 training-placement successes and7/12 reused held-placement successes. Forbidden-contact episodes fall from14 to3. All54 changed actions reduce immediate pose error, but phase completion is not uniformly improved; five full-task failures remain. This supports the frozen physics-assisted component on development training bodies. The next milestone is a prospectively frozen unseen-body comparison, followed by attribution ablations and reducing search latency.
Regression validation:131 tests passed in12.32 seconds.

## Run integrity

The initial attempt stopped at a tuple-versus-list contact-pair parity assertion before executing its first alternative action. Its source and seven completed trials are preserved in outputs/ease_progress_selector_initial_check. Only representation/check diagnostics were corrected; candidate generation, thresholds and ranking were unchanged. All72 final trials rerun from scratch; no partial outcomes are substituted.

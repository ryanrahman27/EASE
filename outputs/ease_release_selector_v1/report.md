# E113: six-joint failure diagnosis and release-phase extension

48 complete rollouts: all24 six-joint E112 task/seed pairs with original selector and release-extended selector. Here baseline means the E112 physics selector, not the earlier unassisted controller. Only release eligibility changes; exact action bank,40ms horizon, thresholds, progress ranking, apertures and task budgets are unchanged. No retraining or tuning after outcomes. These are reused development bodies, not fresh independent transfer evidence.

| Group | Mode | Success | Collisions | Release changes | Eligible release warnings |
|---|---|---:|---:|---:|---:|
| all | baseline | 14/24 | 10 | 0 | 0 |
| all | physics | 16/24 | 8 | 8 | 10 |
| seed0 | baseline | 4/8 | 4 | 0 | 0 |
| seed0 | physics | 5/8 | 3 | 3 | 4 |
| seed1 | baseline | 4/8 | 4 | 0 | 0 |
| seed1 | physics | 4/8 | 4 | 3 | 4 |
| seed2 | baseline | 6/8 | 2 | 0 | 0 |
| seed2 | physics | 7/8 | 1 | 2 | 2 |
| body0 | baseline | 4/6 | 2 | 0 | 0 |
| body0 | physics | 5/6 | 1 | 2 | 2 |
| body1 | baseline | 6/6 | 0 | 0 | 0 |
| body1 | physics | 6/6 | 0 | 0 | 0 |
| body2 | baseline | 1/6 | 5 | 0 | 0 |
| body2 | physics | 1/6 | 5 | 3 | 4 |
| body3 | baseline | 3/6 | 3 | 0 | 0 |
| body3 | physics | 4/6 | 2 | 3 | 4 |

## Paired changes

[
  {
    "case": 0,
    "seed": 2,
    "change": "gain"
  },
  {
    "case": 6,
    "seed": 0,
    "change": "gain"
  }
]

## Frozen gate

**PASS**

{
  "success_improves": true,
  "retains_all_successes": true,
  "collisions_nonincreasing": true,
  "per_seed_nondecreasing": true
}

## Diagnosis

Constraint counts are sequential intersections: all alternatives, static-admissible, no forbidden physical contact, nonnegative geometric clearance, then required goal-error reduction. A zero at a stage identifies why the tested bank has no eligible action; it does not prove that no safe action exists outside the bank. Physical contact is allowed for object-table and object-pads. Other contacts are failures.

All first-collision states are saved in failure_diagnostics.json; every eligible release warning is in release_diagnostics.json. Earlier failures cannot be repaired by this extension because it starts only at release and disables interventions after any previous forbidden contact.

7/8 remaining failures have static-admissible, contact-free, nonpenetrating40ms alternatives at the first-collision decision, but none meets the required progress gain. The frozen fallback consequently keeps the original unsafe action. This identifies a local conflict between safety and strict immediate goal progress; it does not establish that those alternatives would finish the task. One remaining lower-phase case has no static-admissible contact-free alternative in the tested bank.

[
  {
    "case": 0,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "gripper_left_pad",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 1,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 86,
      "contact_free": 0,
      "nonpenetrating": 0,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.004073995078845598,
      "required_gain": 4.073995078845598e-05,
      "best_safe_gain": null
    },
    "unreached_phases": []
  },
  {
    "case": 1,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 86,
      "contact_free": 0,
      "nonpenetrating": 0,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.004073995078845598,
      "required_gain": 4.073995078845598e-05,
      "best_safe_gain": null
    },
    "unreached_phases": []
  },
  {
    "case": 4,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 91,
      "contact_free": 35,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010981534367269468,
      "required_gain": 1.0981534367269468e-06,
      "best_safe_gain": 3.694962766486782e-07
    },
    "unreached_phases": [
      "descend",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 91,
      "contact_free": 35,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010981534367269468,
      "required_gain": 1.0981534367269468e-06,
      "best_safe_gain": 3.694962766486782e-07
    },
    "unreached_phases": [
      "descend",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 87,
      "contact_free": 14,
      "nonpenetrating": 14,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00017898668859814913,
      "required_gain": 1.7898668859814914e-06,
      "best_safe_gain": -1.838808472145369e-06
    },
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 87,
      "contact_free": 14,
      "nonpenetrating": 14,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00017898668859814913,
      "required_gain": 1.7898668859814914e-06,
      "best_safe_gain": -1.838808472145369e-06
    },
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 89,
      "contact_free": 37,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010202529681817159,
      "required_gain": 1.020252968181716e-06,
      "best_safe_gain": -2.8065011967128484e-06
    },
    "unreached_phases": [
      "close",
      "lift"
    ]
  },
  {
    "case": 4,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 89,
      "contact_free": 37,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010202529681817159,
      "required_gain": 1.020252968181716e-06,
      "best_safe_gain": -2.8065011967128484e-06
    },
    "unreached_phases": [
      "close",
      "lift"
    ]
  },
  {
    "case": 5,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ],
      [
        "5",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 88,
      "contact_free": 39,
      "nonpenetrating": 39,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00022072122249776002,
      "required_gain": 2.2072122249776002e-06,
      "best_safe_gain": 2.1942090262566302e-06
    },
    "unreached_phases": [
      "lower",
      "release"
    ]
  },
  {
    "case": 5,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ],
      [
        "5",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 88,
      "contact_free": 39,
      "nonpenetrating": 39,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00022072122249776002,
      "required_gain": 2.2072122249776002e-06,
      "best_safe_gain": 2.1942090262566302e-06
    },
    "unreached_phases": [
      "lower",
      "release"
    ]
  },
  {
    "case": 5,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": []
  },
  {
    "case": 5,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 94,
      "contact_free": 35,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 5.410090119087839e-06,
      "required_gain": 5.410090119087839e-08,
      "best_safe_gain": -1.7152685337518624e-06
    },
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 84,
      "contact_free": 25,
      "nonpenetrating": 25,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 1.6068680192624934e-05,
      "required_gain": 1.6068680192624936e-07,
      "best_safe_gain": -7.003729006468406e-07
    },
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 84,
      "contact_free": 25,
      "nonpenetrating": 25,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 1.6068680192624934e-05,
      "required_gain": 1.6068680192624936e-07,
      "best_safe_gain": -7.003729006468406e-07
    },
    "unreached_phases": []
  },
  {
    "case": 7,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": []
  },
  {
    "case": 7,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 91,
      "contact_free": 34,
      "nonpenetrating": 34,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 2.4443927407486863e-06,
      "required_gain": 2.4443927407486862e-08,
      "best_safe_gain": -7.579693170879154e-07
    },
    "unreached_phases": [
      "settle"
    ]
  }
]

## Release decisions

[
  {
    "case": 0,
    "seed": 2,
    "step": 249,
    "candidates": 108,
    "static": 100,
    "contact_free": 91,
    "nonpenetrating": 90,
    "progress": 14,
    "choice": "directed1.0",
    "before_error_sq": 0.0023349795841114736,
    "required_gain": 2.3349795841114736e-05,
    "best_safe_gain": 0.0006522615617576136
  },
  {
    "case": 0,
    "seed": 2,
    "step": 250,
    "candidates": 108,
    "static": 100,
    "contact_free": 82,
    "nonpenetrating": 79,
    "progress": 65,
    "choice": "directed0.25",
    "before_error_sq": 0.00168271802235386,
    "required_gain": 1.68271802235386e-05,
    "best_safe_gain": 0.00034061669997597407
  },
  {
    "case": 5,
    "seed": 1,
    "step": 199,
    "candidates": 108,
    "static": 82,
    "contact_free": 32,
    "nonpenetrating": 32,
    "progress": 2,
    "choice": "random2",
    "before_error_sq": 1.1725442236746228e-05,
    "required_gain": 1.1725442236746229e-07,
    "best_safe_gain": 2.3987124906616824e-06
  },
  {
    "case": 5,
    "seed": 1,
    "step": 200,
    "candidates": 108,
    "static": 92,
    "contact_free": 78,
    "nonpenetrating": 78,
    "progress": 13,
    "choice": "directed0.5",
    "before_error_sq": 9.326729746084546e-06,
    "required_gain": 9.326729746084546e-08,
    "best_safe_gain": 3.2041106816363535e-06
  },
  {
    "case": 5,
    "seed": 1,
    "step": 201,
    "candidates": 108,
    "static": 96,
    "contact_free": 27,
    "nonpenetrating": 26,
    "progress": 1,
    "choice": "axis2_1_0.02",
    "before_error_sq": 6.122619064448192e-06,
    "required_gain": 6.122619064448193e-08,
    "best_safe_gain": 7.125289453603539e-07
  },
  {
    "case": 5,
    "seed": 1,
    "step": 202,
    "candidates": 108,
    "static": 94,
    "contact_free": 35,
    "nonpenetrating": 35,
    "progress": 0,
    "choice": "original",
    "before_error_sq": 5.410090119087839e-06,
    "required_gain": 5.410090119087839e-08,
    "best_safe_gain": -1.7152685337518624e-06
  },
  {
    "case": 6,
    "seed": 0,
    "step": 207,
    "candidates": 108,
    "static": 86,
    "contact_free": 58,
    "nonpenetrating": 58,
    "progress": 11,
    "choice": "random61",
    "before_error_sq": 3.291809525911844e-05,
    "required_gain": 3.291809525911844e-07,
    "best_safe_gain": 1.2412041333927836e-05
  },
  {
    "case": 7,
    "seed": 0,
    "step": 190,
    "candidates": 108,
    "static": 88,
    "contact_free": 38,
    "nonpenetrating": 38,
    "progress": 1,
    "choice": "random14",
    "before_error_sq": 6.710609615353138e-06,
    "required_gain": 6.710609615353138e-08,
    "best_safe_gain": 1.70210581467119e-06
  },
  {
    "case": 7,
    "seed": 0,
    "step": 191,
    "candidates": 108,
    "static": 87,
    "contact_free": 76,
    "nonpenetrating": 74,
    "progress": 6,
    "choice": "directed1.0",
    "before_error_sq": 5.008503800681948e-06,
    "required_gain": 5.0085038006819476e-08,
    "best_safe_gain": 2.5641110599332615e-06
  },
  {
    "case": 7,
    "seed": 0,
    "step": 192,
    "candidates": 108,
    "static": 91,
    "contact_free": 34,
    "nonpenetrating": 34,
    "progress": 0,
    "choice": "original",
    "before_error_sq": 2.4443927407486863e-06,
    "required_gain": 2.4443927407486862e-08,
    "best_safe_gain": -7.579693170879154e-07
  }
]

## Interpretation

The release extension passes the prespecified development gate; require a newly frozen prospective replication before claiming general transfer.

The original requirement of at least1% pose-error reduction is unchanged during release. Both remaining release failures have safe alternatives rejected by this requirement. Next test a separately frozen, bounded near-goal safety fallback on training states, measuring full-task progress and preventing indefinite holding. Do not simply remove progress everywhere or relabel this reused cohort as untouched validation.

## Verification

{
  "exact_e112_replays": 24,
  "exact_pre_release_prefixes": 24,
  "exact_predictions": 8339,
  "hashes_unchanged": true
}

All24 original-selector replays match E112 full serialized results; all extended rollouts match before release. Every selected next-state/contact prediction matches execution. Exact simulator access remains privileged; this is not history-only adaptation. Videos are outcome-selected illustrations and must match full serialized replay.

Final verification: 8 full-result video replays and hashes match.

Progress demo: ../ease_progress_demo_v1/progress_demo.mp4 (60 seconds; results through E113; illustrative graphics plus verified footage; final card includes this release experiment).

# E108: selecting safer actions with short physics predictions

All36 original training-body task/seed pairs execute baseline and the fixed physics selector:72 complete tasks. Fixed neural weights, original task budgets, no route planner, no ramp. Third placement per body remains held validation (24 training/12 held pairs). The selector only acts in retreat; it does not repair earlier collision failures.

When the static-filter-selected command predicts a forbidden contact over40ms and current state is contact-free, test five fixed alternatives: half/quarter/bias blends, hold-current-position target, and a velocity-opposing heuristic. Every candidate must pass the original static nonworsening-clearance threshold and predict no contact. Choose the safe candidate closest in joint-target distance to the original command, with fixed ties; if none qualifies keep original. All candidates share the same40ms budget and aperture. No alternative set or horizon tuning after outcomes.

| Group | Mode | Success | Collisions | Steps | Changed actions | Warnings | No safe alternative | Changes after earlier collision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | baseline | 22/36 | 14 | 13406 | 0 | 2 | 0 | 0 |
| all | physics | 22/36 | 14 | 13406 | 5 | 7 | 2 | 2 |
| train | baseline | 18/24 | 6 | 8786 | 0 | 1 | 0 | 0 |
| train | physics | 18/24 | 6 | 8786 | 3 | 4 | 1 | 0 |
| validation | baseline | 4/12 | 8 | 4620 | 0 | 1 | 0 | 0 |
| validation | physics | 4/12 | 8 | 4620 | 2 | 3 | 1 | 2 |
| 6 | baseline | 14/18 | 4 | 5803 | 0 | 1 | 0 | 0 |
| 6 | physics | 14/18 | 4 | 5803 | 3 | 4 | 1 | 0 |
| 7 | baseline | 8/18 | 10 | 7603 | 0 | 1 | 0 | 0 |
| 7 | physics | 8/18 | 10 | 7603 | 2 | 3 | 1 | 2 |
| seed0 | baseline | 7/12 | 5 | 4344 | 0 | 1 | 0 | 0 |
| seed0 | physics | 7/12 | 5 | 4344 | 3 | 4 | 1 | 0 |
| seed1 | baseline | 7/12 | 5 | 4644 | 0 | 0 | 0 | 0 |
| seed1 | physics | 7/12 | 5 | 4644 | 0 | 0 | 0 | 0 |
| seed2 | baseline | 8/12 | 4 | 4418 | 0 | 1 | 0 | 0 |
| seed2 | physics | 8/12 | 4 | 4418 | 2 | 3 | 1 | 2 |

## All paired changes

[]

## Prespecified retention gates

**FAIL**

{
  "train": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true,
    "success_improves": false,
    "per_seed_nondecreasing": true
  },
  "validation": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true
  }
}

## Warning-case audit

[
  {
    "case": 1,
    "seed": 0,
    "success": false,
    "collision": true,
    "changed_actions": 3,
    "no_safe_alternative": 1,
    "choices": [
      {
        "step": 315,
        "choice": "hold",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -1.746878173969693e-05,
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
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -1.746878173969693e-05,
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
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -1.746878173969693e-05,
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
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 0.00014894590896809973,
            "pairs": [],
            "distance": 0.022928897039400816
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": 0.00010012882454130817,
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
            "distance": 0.006874501132785076
          }
        ]
      },
      {
        "step": 316,
        "choice": "hold",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00022134171396499466,
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
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00022134171396499466,
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
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00022134171396499466,
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
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 7.020107215003138e-05,
            "pairs": [],
            "distance": 0.019201659254345072
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -9.429452988622342e-05,
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
            "distance": 0.006221340239492503
          }
        ]
      },
      {
        "step": 317,
        "choice": "hold",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00031698058180011646,
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
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00031698058180011646,
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
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00031698058180011646,
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
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 8.977222225790144e-05,
            "pairs": [],
            "distance": 0.016438246428768347
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00020988646325669977,
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
            "distance": 0.005635365011356991
          }
        ]
      },
      {
        "step": 319,
        "choice": "original",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00045238308001739266,
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
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00045238308001739266,
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
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00045238308001739266,
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
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 1.3739761164192899e-06,
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
            "distance": 0.014054845122771225
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.0002973632030811614,
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
            "distance": 0.0032094070945114872
          }
        ]
      }
    ],
    "phases": [
      {
        "phase": "approach",
        "reached": true,
        "steps": 5
      },
      {
        "phase": "descend",
        "reached": true,
        "steps": 18
      },
      {
        "phase": "probe",
        "reached": true,
        "steps": 32
      },
      {
        "phase": "close",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "lift",
        "reached": true,
        "steps": 88
      },
      {
        "phase": "transport",
        "reached": true,
        "steps": 11
      },
      {
        "phase": "lower",
        "reached": true,
        "steps": 23
      },
      {
        "phase": "release",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "retreat",
        "reached": true,
        "steps": 80
      },
      {
        "phase": "settle",
        "reached": true,
        "steps": 40
      }
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "success": false,
    "collision": true,
    "changed_actions": 2,
    "no_safe_alternative": 1,
    "choices": [
      {
        "step": 347,
        "choice": "hold",
        "prior_episode_collision": true,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.0002296718354263469,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.0002296718354263469,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.0002296718354263469,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 7.666031938170781e-06,
            "pairs": [],
            "distance": 0.027499459895477658
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00014109483460850863,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0026258618321182495
          }
        ]
      },
      {
        "step": 348,
        "choice": "hold",
        "prior_episode_collision": true,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00018422853436433212,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00018422853436433212,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00018422853436433212,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 5.065295702801065e-06,
            "pairs": [],
            "distance": 0.02355210923769029
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00018241891951270572,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.009998881381124236
          }
        ]
      },
      {
        "step": 349,
        "choice": "original",
        "prior_episode_collision": true,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00014864801370761266,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00014864801370761266,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00014864801370761266,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 2.174354508319154e-06,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.020961815191969876
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00014596387151639273,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.009359623581570706
          }
        ]
      }
    ],
    "phases": [
      {
        "phase": "approach",
        "reached": true,
        "steps": 7
      },
      {
        "phase": "descend",
        "reached": true,
        "steps": 17
      },
      {
        "phase": "probe",
        "reached": true,
        "steps": 32
      },
      {
        "phase": "close",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "lift",
        "reached": true,
        "steps": 71
      },
      {
        "phase": "transport",
        "reached": true,
        "steps": 25
      },
      {
        "phase": "lower",
        "reached": true,
        "steps": 36
      },
      {
        "phase": "release",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "retreat",
        "reached": false,
        "steps": 80
      },
      {
        "phase": "settle",
        "reached": false,
        "steps": 40
      }
    ]
  }
]

## Decision and limits

The fixed rule fails the task-success gate. Preserve the baseline and report the full negative result; do not promote collision avoidance alone as task improvement.
The method has exact simulator state and physical-model access. It is a privileged controller component, not a learned dynamics model or history-only policy. A finite candidate set and40ms horizon cannot guarantee progress or safety. The comparison tests this rule, not all possible physics-based controllers. Held placements share training bodies. No fresh E105 state or reserved final test is used.

## Verification

36 full baseline replays and 72 pre-retreat traces match references. All3266 selected-action predicted40ms qpos/qvel/contact unions match execution exactly. Inputs/checkpoints remain unchanged. Full raw results, all action alternatives and outcomes, and exact-replay videos are saved.

## Scope of the retreat intervention

Baseline first-collision phases: descend3, lift4, transport2, lower4, retreat1. Thus13 of14 failed tasks are already invalid before the intervention can act. With the prefix and success rule fixed, a retreat-only change can recover at most one of these14 failures. Case8/seed2 already collided during lift; its two changed retreat actions cannot recover full-task success even if they prevent later contact.
Case1/seed0 is the only baseline failure with first collision in retreat. Three hold actions delay first retreat collision from control step27 to31 (160ms); all five alternatives then predict contact and the rule keeps the original action. Case8/seed2 has two hold actions and first retreat contact shifts from step48 to50 (80ms), but the earlier lift collision remains. These are delays, not recovered tasks. The five action changes occur in two failing pairs; no paired task outcome changes.
Next, audit predictive contact avoidance before the first collision in descend/lift/transport/lower on training bodies. Separate whether the finite candidate set has any contact-free option from whether choosing one preserves task progress. Do not interpret the failure of this retreat-only rule as a general failure of physics lookahead. No new phase-wide control rule or transfer evaluation is performed here.

[
  {
    "case": 1,
    "seed": 0,
    "first_collision_phase": "retreat"
  },
  {
    "case": 1,
    "seed": 2,
    "first_collision_phase": "lower"
  },
  {
    "case": 2,
    "seed": 0,
    "first_collision_phase": "lower"
  },
  {
    "case": 2,
    "seed": 1,
    "first_collision_phase": "lower"
  },
  {
    "case": 6,
    "seed": 1,
    "first_collision_phase": "lower"
  },
  {
    "case": 8,
    "seed": 0,
    "first_collision_phase": "descend"
  },
  {
    "case": 8,
    "seed": 1,
    "first_collision_phase": "transport"
  },
  {
    "case": 8,
    "seed": 2,
    "first_collision_phase": "lift"
  },
  {
    "case": 10,
    "seed": 0,
    "first_collision_phase": "descend"
  },
  {
    "case": 10,
    "seed": 1,
    "first_collision_phase": "transport"
  },
  {
    "case": 10,
    "seed": 2,
    "first_collision_phase": "descend"
  },
  {
    "case": 11,
    "seed": 0,
    "first_collision_phase": "lift"
  },
  {
    "case": 11,
    "seed": 1,
    "first_collision_phase": "lift"
  },
  {
    "case": 11,
    "seed": 2,
    "first_collision_phase": "lift"
  }
]

Regression suite:131 tests pass in12.48s. Prediction and full-replay assertions run on every evaluated task.

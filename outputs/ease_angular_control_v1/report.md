# E51: full angular control with frozen E35 estimators

E50 shows corner-oriented Panda grips and subsequent slip. Test full angular motion identification/control with six independent XYZ/rotation-vector coordinates, unlike E42 nine-component axis tracking. Compare upright initial yaw (hold) with nearest world-cardinal cube-face alignment (align). One configuration for both robots and all seeds; original E35 weights frozen.

Measured angular increment log(R_after R_before.T), scaled .15; same two model calls, shared Gram/cross representation and bias update. Orientation error log(R_target R_current.T). Same gain .35, damping .015, joint clip .04, reference ramp .002. Existing eight-hold .006 m gate now requires full angular error <.08 rad. Native mechanics, phase budgets, object-release rule and final physical success unchanged. No added refresh or target-robot training.

All twelve original E46 homes retained without new IK filtering. Both variants use the same 32 probe actions and seed 41000+case. Targets selected from the common pre-calibration home frame; cubes initially axis-aligned. All failures and total steps retained. Analytical/privileged get the same target. This is a development representation-and-control change, not independent final-test evidence.

| Condition | Robot | Privileged /6 | Analytical /6 | Neural seeds 0 / 1 / 2 (each /6) |
|---|---|---:|---:|---|
| baseline | piper | 6 | 6 | 6 / 6 / 6 |
| baseline | panda | 6 | 6 | 5 / 1 / 5 |
| hold | piper | 6 | 6 | 5 / 2 / 5 |
| hold | panda | 6 | 5 | 3 / 3 / 2 |
| align | piper | 6 | 6 | 5 / 6 / 6 |
| align | panda | 6 | 6 | 3 / 1 / 3 |

## Costs and failures

| Condition | Robot | Method | Collisions | Mean total steps | Failures |
|---|---|---|---:|---:|---|
| hold | piper | privileged | 0 | 960.2 | [] |
| hold | piper | analytical | 0 | 777.8 | [] |
| hold | piper | neural_0 | 1 | 663.7 | [{'case': 1, 'failure': 'forbidden_contact'}] |
| hold | piper | neural_1 | 3 | 362.5 | [{'case': 1, 'failure': 'forbidden_contact'}, {'case': 2, 'failure': 'no_bilateral_grasp'}, {'case': 3, 'failure': 'forbidden_contact'}, {'case': 5, 'failure': 'forbidden_contact'}] |
| hold | piper | neural_2 | 0 | 717.8 | [{'case': 0, 'failure': 'phase_timeout:lift'}] |
| hold | panda | privileged | 0 | 441.8 | [] |
| hold | panda | analytical | 0 | 425.8 | [{'case': 7, 'failure': 'phase_timeout:release'}] |
| hold | panda | neural_0 | 0 | 349.3 | [{'case': 6, 'failure': 'lost_grasp:lift'}, {'case': 7, 'failure': 'phase_timeout:descend'}, {'case': 9, 'failure': 'lost_grasp:lift'}] |
| hold | panda | neural_1 | 1 | 436.0 | [{'case': 7, 'failure': 'lost_grasp:lift'}, {'case': 9, 'failure': 'lost_grasp:lift'}, {'case': 10, 'failure': 'forbidden_contact'}] |
| hold | panda | neural_2 | 1 | 349.5 | [{'case': 6, 'failure': 'forbidden_contact'}, {'case': 7, 'failure': 'phase_timeout:lift'}, {'case': 8, 'failure': 'lost_grasp:lift'}, {'case': 9, 'failure': 'lost_grasp:transport'}] |
| align | piper | privileged | 0 | 995.7 | [] |
| align | piper | analytical | 0 | 806.5 | [] |
| align | piper | neural_0 | 0 | 859.5 | [{'case': 2, 'failure': 'phase_timeout:retreat'}] |
| align | piper | neural_1 | 0 | 797.0 | [] |
| align | piper | neural_2 | 0 | 804.2 | [] |
| align | panda | privileged | 0 | 439.7 | [] |
| align | panda | analytical | 0 | 446.3 | [] |
| align | panda | neural_0 | 1 | 325.7 | [{'case': 6, 'failure': 'lost_grasp:lift'}, {'case': 7, 'failure': 'phase_timeout:close'}, {'case': 9, 'failure': 'forbidden_contact'}] |
| align | panda | neural_1 | 1 | 289.5 | [{'case': 6, 'failure': 'lost_grasp:lift'}, {'case': 7, 'failure': 'lost_grasp:lift'}, {'case': 9, 'failure': 'lost_grasp:lift'}, {'case': 10, 'failure': 'forbidden_contact'}, {'case': 11, 'failure': 'lost_grasp:transport'}] |
| align | panda | neural_2 | 1 | 350.3 | [{'case': 6, 'failure': 'phase_timeout:close'}, {'case': 7, 'failure': 'forbidden_contact'}, {'case': 8, 'failure': 'lost_grasp:lift'}] |

## Interpretation limits

Hold versus align uses the same angular representation and stricter full-orientation gate, changing the yaw target only. Either versus baseline changes both orientation representation and the controlled objective/gate; it is not a pure representation ablation. Final physical success, force limits, friction and object-release rules are unchanged. All homes remain even when the added orientation goal is hard to reach. The same frozen network predicts translational and angular Jacobians separately; no analytical fallback enters neural prediction.

Variants passing the preliminary per-robot/per-seed success and collision retention screen: []. No variant is automatically promoted. Full paired gains/losses are in effects.json. These are repeated development tasks, not independent embodiment validation.

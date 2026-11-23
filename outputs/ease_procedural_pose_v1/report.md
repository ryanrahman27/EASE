# E55: position/tool-direction score validity on procedural histories

Frozen analytical and nine original/E52 models each drive all eight training bodies. Two target poses per body come from seeded +/-0.15-radian home perturbations and forward kinematics, with no outcome/contact filtering. Each driver cold-starts with empty statistics and switches targets every 50 steps, ramping position by at most 2 mm and tool-Z direction by approximately 0.01 radians per step. All ten candidates branch from each measured driver history for five continuing control steps.

Position plus 0.15-scaled unit tool-Z tracking, with free yaw, gain 0.35, damping 0.015, command clip 0.04 and joint limits +/-2.5. This matches the task-coordinate representation of native manipulation but has no object or gripper. Both frozen model calls and measured bias updates are shared. Training-split procedural arms only; no native or held-out geometry and no optimization.

All 80 driver trajectories replay exactly without branches. All 720 sampled first-step driver branches match the actual next step exactly. There are 7200 counterfactual branches. Checkpoint hashes are unchanged.

## History coverage

| Driver group | Trajectories | Driver collisions | Sampled states | Missing planned states |
|---|---:|---:|---:|---:|
| analytical | 8 | 0 | 72 | 0 |
| original | 24 | 0 | 216 | 0 |
| prediction | 24 | 0 | 216 | 0 |
| control | 24 | 0 | 216 | 0 |

Drivers stop on collision; later missing states are retained as missing, not replaced. The 150-step moving-target runs ignore the old static-goal success flag and do not define a new task-success metric.

## Case-macro ordering agreement

At each common state, pairwise proxy order is compared with measured one/five-step error order. Values average within each body then across bodies; ties within 1e-12 are excluded. The eight bodies, not branches or candidate pairs, define clusters. Scores use the privileged instantaneous Jacobian only for diagnostics. All-candidate comparisons include analytical. All results are descriptive, not independent-sample significance claims.

| DoF | Driver group | Horizon | Operator | Local unclipped | Local executed |
|---|---|---:|---:|---:|---:|
| 6dof | all | 1 | 0.486 | 0.720 | 0.804 |
| 6dof | all | 5 | 0.432 | 0.603 | 0.640 |
| 6dof | analytical | 1 | 0.454 | 0.694 | 0.770 |
| 6dof | analytical | 5 | 0.434 | 0.580 | 0.591 |
| 6dof | original | 1 | 0.498 | 0.724 | 0.792 |
| 6dof | original | 5 | 0.445 | 0.630 | 0.664 |
| 6dof | prediction | 1 | 0.531 | 0.745 | 0.813 |
| 6dof | prediction | 5 | 0.450 | 0.610 | 0.658 |
| 6dof | control | 1 | 0.440 | 0.698 | 0.819 |
| 6dof | control | 5 | 0.400 | 0.576 | 0.614 |
| 7dof | all | 1 | 0.445 | 0.745 | 0.786 |
| 7dof | all | 5 | 0.423 | 0.648 | 0.669 |
| 7dof | analytical | 1 | 0.470 | 0.744 | 0.779 |
| 7dof | analytical | 5 | 0.437 | 0.643 | 0.662 |
| 7dof | original | 1 | 0.483 | 0.740 | 0.782 |
| 7dof | original | 5 | 0.445 | 0.652 | 0.677 |
| 7dof | prediction | 1 | 0.455 | 0.764 | 0.786 |
| 7dof | prediction | 5 | 0.414 | 0.647 | 0.666 |
| 7dof | control | 1 | 0.389 | 0.730 | 0.791 |
| 7dof | control | 5 | 0.405 | 0.647 | 0.666 |

## Target-switch sensitivity (all drivers)

| DoF | Samples | Horizon | Operator | Local executed |
|---|---|---:|---:|---:|
| 6dof | switch | 1 | 0.497 | 0.797 |
| 6dof | switch | 5 | 0.421 | 0.618 |
| 6dof | other | 1 | 0.478 | 0.810 |
| 6dof | other | 5 | 0.441 | 0.657 |
| 7dof | switch | 1 | 0.425 | 0.794 |
| 7dof | switch | 5 | 0.388 | 0.670 |
| 7dof | other | 1 | 0.461 | 0.779 |
| 7dof | other | 5 | 0.450 | 0.667 |

## Branch contacts and saturation

| DoF | Candidate group | Branches | Five-step contact branches | First-command clipped branches |
|---|---|---:|---:|---:|
| 6dof | analytical | 360 | 0 | 137 |
| 6dof | original | 1080 | 0 | 499 |
| 6dof | prediction | 1080 | 0 | 491 |
| 6dof | control | 1080 | 0 | 417 |
| 7dof | analytical | 360 | 0 | 82 |
| 7dof | original | 1080 | 0 | 362 |
| 7dof | prediction | 1080 | 0 | 327 |
| 7dof | control | 1080 | 0 | 204 |

## Driver pose tracking (descriptive, not a success gate)

| DoF | Driver group | Mean position error (m) | Mean tool-direction chord error |
|---|---|---:|---:|
| 6 | analytical | 0.01238 | 0.04783 |
| 6 | original | 0.01025 | 0.05492 |
| 6 | prediction | 0.00865 | 0.03116 |
| 6 | control | 0.03066 | 0.14958 |
| 7 | analytical | 0.01385 | 0.05308 |
| 7 | original | 0.00780 | 0.03733 |
| 7 | prediction | 0.00689 | 0.03113 |
| 7 | control | 0.02204 | 0.10097 |

## Interpretation limits

This is a frozen ranking audit, not a new trained controller. It tests position/tool-direction tracking without a gripper or object on eight known training bodies and continuing candidate-generated histories; it does not validate grasp success or unseen embodiments. The local executed score assumes measured bias approximates actuator response; it does not differentiate through the simulator. Five-step rankings also include subsequent feedback updates and target changes that a one-step proxy does not forecast. Scores selected from these development results require a separate protocol before optimization. Keep original E35 checkpoints; no objective/model is automatically promoted. Per-body ranking rates, collision-free and neural-only comparisons are retained in summary.json.

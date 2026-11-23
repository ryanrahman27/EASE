# E54: action-score validity on procedural training histories

Frozen analytical and nine original/E52 models each drive all eight training bodies (four six-joint, four seven-joint). Each driver cold-starts with empty statistics and tracks alternating first/last training goals, switching every 50 steps with a 2 mm reference ramp. Driver histories are separate. The same ten candidates receive each sampled driver history and branch for five steps, continuing the moving-target schedule and online estimator/bias updates.

Position-only reaching with the existing procedural actuators, gain 0.5, damping 0.015, command clip 0.04 and joint limits +/-2.5. This is not a pose/contact manipulation benchmark. Native data, validation bodies and final-test bodies are excluded; no training is performed.

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
| 6dof | all | 1 | 0.413 | 0.797 | 0.812 |
| 6dof | all | 5 | 0.398 | 0.747 | 0.757 |
| 6dof | analytical | 1 | 0.425 | 0.745 | 0.856 |
| 6dof | analytical | 5 | 0.386 | 0.673 | 0.732 |
| 6dof | original | 1 | 0.414 | 0.819 | 0.820 |
| 6dof | original | 5 | 0.405 | 0.777 | 0.778 |
| 6dof | prediction | 1 | 0.485 | 0.867 | 0.867 |
| 6dof | prediction | 5 | 0.460 | 0.829 | 0.829 |
| 6dof | control | 1 | 0.336 | 0.723 | 0.735 |
| 6dof | control | 5 | 0.333 | 0.660 | 0.671 |
| 7dof | all | 1 | 0.404 | 0.787 | 0.810 |
| 7dof | all | 5 | 0.390 | 0.711 | 0.725 |
| 7dof | analytical | 1 | 0.553 | 0.689 | 0.868 |
| 7dof | analytical | 5 | 0.386 | 0.491 | 0.601 |
| 7dof | original | 1 | 0.425 | 0.774 | 0.775 |
| 7dof | original | 5 | 0.415 | 0.737 | 0.735 |
| 7dof | prediction | 1 | 0.410 | 0.863 | 0.862 |
| 7dof | prediction | 5 | 0.412 | 0.790 | 0.789 |
| 7dof | control | 1 | 0.326 | 0.756 | 0.774 |
| 7dof | control | 5 | 0.344 | 0.680 | 0.693 |

## Target-switch sensitivity (all drivers)

| DoF | Samples | Horizon | Operator | Local executed |
|---|---|---:|---:|---:|
| 6dof | switch | 1 | 0.371 | 0.788 |
| 6dof | switch | 5 | 0.367 | 0.749 |
| 6dof | other | 1 | 0.447 | 0.831 |
| 6dof | other | 5 | 0.423 | 0.763 |
| 7dof | switch | 1 | 0.371 | 0.805 |
| 7dof | switch | 5 | 0.346 | 0.697 |
| 7dof | other | 1 | 0.430 | 0.814 |
| 7dof | other | 5 | 0.426 | 0.748 |

## Branch contacts and saturation

| DoF | Candidate group | Branches | Five-step contact branches | First-command clipped branches |
|---|---|---:|---:|---:|
| 6dof | analytical | 360 | 0 | 10 |
| 6dof | original | 1080 | 0 | 120 |
| 6dof | prediction | 1080 | 0 | 107 |
| 6dof | control | 1080 | 0 | 24 |
| 7dof | analytical | 360 | 0 | 12 |
| 7dof | original | 1080 | 0 | 167 |
| 7dof | prediction | 1080 | 0 | 158 |
| 7dof | control | 1080 | 0 | 58 |

## Interpretation limits

This is a frozen ranking audit, not a new trained controller. It tests position tracking on eight known training bodies and continuing candidate-generated histories; it does not validate tool orientation, gripper contacts, grasp success or unseen embodiments. The local executed score assumes measured bias approximates actuator response; it does not differentiate through the simulator. Five-step rankings also include subsequent feedback updates and target changes that a one-step proxy does not forecast. Scores selected from these development results require a separate protocol before optimization. Keep original E35 checkpoints; no objective/model is automatically promoted. Per-body ranking rates, collision-free and neural-only comparisons are retained in summary.json.

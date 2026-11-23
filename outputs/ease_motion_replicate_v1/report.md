# E30–E32: explicit motion Transformer and shared control

The redesigned model is a 235,011-parameter, three-layer joint-token Transformer (width 96, four attention heads). Measured displacement Gram/cross statistics become joint tokens; the network predicts one 3D Jacobian column per joint. A shared damped Cartesian controller and measured command-tracking bias produce actions. This replaces direct action-chunk prediction; it is not a fine-tuned E27 checkpoint. Model weights stay frozen during evaluation while motion statistics update from task transitions.

Each model trains for 15,000 updates × 128 synthetic linear systems. Supervision distills ridge estimates from random Jacobians and excitation profiles drawn only from 128 training-robot probes, mixed with Gaussian excitation. No validation geometry or expert actions are used. Data, objective, representation, controller and online adaptation all change from E27; this is not an isolated architecture ablation.

Seed 0 was observed before the unchanged seeds 1/2 replication. Every trained model is evaluated with three probes. Empty mode has no probe and its three identical repetitions are explicitly deduplicated in interpretation. Hybrid uses the network only on step one, then analytical identification; it is not neural-only estimation.

## Validation success

Each entry lists three probe results out of 384 for a training seed. Empty and random controls have one meaningful result per seed.

| Mode | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| neural_frozen | 346 / 365 / 358 | 370 / 340 / 366 | 349 / 364 / 370 |
| neural_online | 383 / 383 / 383 | 384 / 383 / 384 | 384 / 384 / 384 |
| hybrid_online | 384 / 384 / 384 | 384 / 384 / 384 | 384 / 384 / 384 |
| empty_neural_online | 384 | 384 | 384 |
| twin_neural_online | 384 / 384 / 383 | 384 / 384 / 384 | 381 / 384 / 384 |
| random_neural_online | 45 | 6 | 54 |

## Unseen six-joint cases

Each entry is successes out of 48, across probe seeds. Empty has no probe.

| Mode | Training seed | Familiar posture | Shifted posture |
|---|---:|---|---|
| neural_frozen | 0 | 48 / 47 / 48 | 48 / 47 / 46 |
| neural_frozen | 1 | 48 / 48 / 48 | 48 / 48 / 48 |
| neural_frozen | 2 | 48 / 48 / 48 | 48 / 48 / 46 |
| neural_online | 0 | 48 / 48 / 48 | 48 / 47 / 47 |
| neural_online | 1 | 48 / 48 / 48 | 48 / 48 / 48 |
| neural_online | 2 | 48 / 48 / 48 | 48 / 48 / 48 |
| empty_neural_online | 0 | 48 | 48 |
| empty_neural_online | 1 | 48 | 48 |
| empty_neural_online | 2 | 48 | 48 |

## Interaction cost

Means average all validation cases, including full-horizon failures. Calibrated modes pay 32 steps per task; empty mode pays zero separate calibration. Reset and computation time are excluded.

| Mode | Mean task steps | Mean including probe | Mean successes by 50 total steps / 384 |
|---|---:|---:|---:|
| neural_frozen | 32.69 | 64.69 | 206.00 |
| neural_online | 17.27 | 49.27 | 238.44 |
| hybrid_online | 19.52 | 51.52 | 204.78 |
| empty_neural_online | 19.08 | 19.08 | 380.67 |
| twin_neural_online | 20.02 | 52.02 | 200.67 |
| random_neural_online | 139.91 | 171.91 | 8.67 |

## Training control

| Seed | Mode | Success / 256 | Gate |
|---:|---|---:|---|
| 0 | neural_frozen | 237 | False |
| 0 | neural_online | 256 | True |
| 0 | hybrid_online | 256 | True |
| 1 | neural_frozen | 255 | True |
| 1 | neural_online | 256 | True |
| 1 | hybrid_online | 256 | True |
| 2 | neural_frozen | 248 | False |
| 2 | neural_online | 256 | True |
| 2 | hybrid_online | 256 | True |

## Cold-start attribution controls

E32 was added after seeing seed-0 empty-history success. All rows have no separate calibration. Neural bootstrap/constant use the seed-0 model; the fixed first action is +0.01 rad for every joint, counted as task step one.

| Controller | Success / 384 | Collisions | Mean steps |
|---|---:|---:|---:|
| analytical_cold | 336 | 0 | 38.32 |
| fixed_probe_analytical | 384 | 0 | 18.45 |
| neural_bootstrap_analytical | 384 | 0 | 18.28 |
| neural_constant | 18 | 278 | 147.93 |

The trained network must be distinguished from the analytical controller around it. Random-network and constant-output controls test whether useful learned predictions and task-time updates matter. A successful fixed-probe analytical controller prevents claiming learned superiority merely from cold-start success. Matching this benchmark is a functional milestone, not evidence of novelty or broad morphology generalization.

All 2304 primary neural/empty replay cases match exactly while simulator Jacobian access, all torch.linalg.solve calls, and NumPy solves larger than the 3×3 Cartesian control system are forbidden. Thus these neural modes do not secretly substitute an analytical joint-model inverse. Other analytical operations, measured tracking-bias updates, and the 3×3 control solve remain intentional components.

The same fixed procedural validation families and expert-filtered reaching goals are used throughout. No final-test policy evaluation, hardware, vision, orientation control, or manipulation claim is made. Further claims require broader morphology families and tasks plus a demonstrated benefit over analytical identification.

The reusable single-robot MotionController interface independently achieves 384/384 without calibration using the seed-0 checkpoint, with the same inverse/API guards. One model instance is shared across six/seven-joint controller states. This separately checks the deployment interface rather than relying only on the batched evaluator.

# E28–E29: observation-only adaptive kinematic control

This is an analytical controller, not an improved Transformer checkpoint. It estimates a local Cartesian Jacobian from measured joint/end-effector displacements and a joint-command tracking bias from executed actions. Only measured observations enter the controller; no axis sequence, links, masses, gravity forces, simulator Jacobian, or target-arm expert actions are used.

The 32-step calibration, goals, resets, action limit, 150-step task horizon, 3 cm / five-step success criterion, and collision rejection match E27. Online variants additionally update their estimates from task transitions. E28 compares frozen, online, and wrong-twin initialization. E29 is a post-E28 mechanism check with fixed gains, isolating Jacobian and bias updates and blocking mj_jacSite during every rollout.

## Unseen six-joint validation

Entries are successes for three calibration seeds, each out of 48 cases. These are probe variations on fixed bodies, not independent learned-policy training seeds.

| Controller | Familiar posture | Shifted posture |
|---|---|---|
| identified_frozen | 48 / 48 / 48 | 48 / 47 / 48 |
| identified_online | 48 / 48 / 48 | 48 / 48 / 48 |
| twin_online | 48 / 48 / 48 | 48 / 48 / 48 |
| jacobian_online | 48 / 48 / 48 | 48 / 48 / 48 |
| bias_online | 48 / 47 / 48 | 48 / 47 / 48 |

## All validation and cost

| Controller | Successes / 1152 | Collisions | Mean task steps | Including one probe per task | Success by 50 total steps |
|---|---:|---:|---:|---:|---:|
| identified_frozen | 1044 | 9 | 41.10 | 73.10 | 518 |
| identified_online | 1152 | 0 | 19.66 | 51.66 | 604 |
| twin_online | 1151 | 0 | 23.14 | 55.14 | 447 |
| jacobian_online | 1117 | 0 | 23.47 | 55.47 | 607 |
| bias_online | 1110 | 6 | 32.65 | 64.65 | 574 |

The initial Jacobian estimate has median relative Frobenius error 0.372 and maximum 4.172 across 96 distinct validation probes. These privileged diagnostics are computed outside the controller. The complete online replay (1408 training/validation cases) is exactly unchanged with simulator Jacobian access blocked.

Probe cost is 32 control interactions, including for wrong-twin diagnostics. All failures cost 150 task steps. Reset and computation costs are excluded. Reusing calibration is possible only for the same robot/reset; the table charges a fresh probe for every task. The bias estimate is a measured command-minus-motion residual, not an identified gravity model.

The result establishes working measured-feedback control for this fixed simulated reaching benchmark, including previously failing six-joint arrangements. It does not establish neural cross-embodiment transfer, novelty, real-world control, pose/orientation manipulation, or unseen final-test performance. Wrong-twin recovery suggests online feedback can correct much of an initially incorrect model; initial-history necessity is not established. Online task adaptation also changes the information protocol relative to E27 frozen-context Transformers.

The next learned architecture should expose a kinematic/motion predictor and update it from observed transitions, retaining this controller as a strong baseline. A neural version must earn its value over this analytical solution; adding a Transformer around it is not evidence of a contribution.

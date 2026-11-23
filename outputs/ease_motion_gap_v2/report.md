# E58: predicted versus actual procedural motion

32 analytical/E35 trajectories exactly replay E55; 288 fixed states, 2,880 five-step branches and 288 exact matching-driver first steps. Eight training bodies; no independent validation, native data or new training. All candidate checkpoints remain frozen.

## Pairwise ranking (body-macro agreement)

| DoF | Horizon | Estimated J | Geometric J + predicted motion | Geometric J + realized joint motion |
|---|---:|---:|---:|---:|
| 6dof | 1 | 58.516% | 78.236% | 99.413% |
| 6dof | 5 | 56.861% | 63.446% | 63.847% |
| 7dof | 1 | 51.846% | 76.382% | 99.475% |
| 7dof | 5 | 50.989% | 66.184% | 68.792% |

Ties at 1e-12 are excluded. Bodies, states and branches are dependent. Realized joint motion is a post-hoc diagnostic unavailable before executing the action. Five-step reference and online estimates continue changing.

## First-step displacement error decomposition (RMS scaled displacement)

| DoF | Group | Estimator | Actuator/bias | Nonlinear geometry | Total |
|---|---|---:|---:|---:|---:|
| 6dof | analytical | 0.003437 | 0.006124 | 0.000107 | 0.006055 |
| 6dof | original | 0.012942 | 0.009802 | 0.000160 | 0.005395 |
| 6dof | prediction | 0.008194 | 0.008325 | 0.000140 | 0.004070 |
| 6dof | control | 0.007873 | 0.007428 | 0.000133 | 0.004041 |
| 7dof | analytical | 0.001312 | 0.004229 | 0.000044 | 0.003920 |
| 7dof | original | 0.011516 | 0.010754 | 0.000061 | 0.004314 |
| 7dof | prediction | 0.004074 | 0.006237 | 0.000039 | 0.004592 |
| 7dof | control | 0.005461 | 0.006779 | 0.000047 | 0.004017 |

Vectors sum exactly; magnitudes can cancel and squared terms are not additive. Units combine XYZ metres and 0.15 times tool-Z direction. These are free-space procedural arms with no object/gripper.

## Matched control versus prediction-only candidates

| DoF | Seed | Geometric score favors control / states | Actual first step favors control / states | Actual fifth step favors control / states |
|---|---:|---:|---:|---:|
| 6dof | 0 | 88/144 | 79/144 | 78/144 |
| 6dof | 1 | 97/144 | 88/144 | 90/144 |
| 6dof | 2 | 96/144 | 79/144 | 90/144 |
| 7dof | 0 | 88/144 | 95/144 | 99/144 |
| 7dof | 1 | 97/144 | 111/144 | 104/144 |
| 7dof | 2 | 107/144 | 107/144 | 103/144 |

Five-step branches with contact: 0/2880. All outcomes retained. This audit cannot explain native grasp/contact failures directly; it tests local mechanisms on fixed histories overlapping E57 training data. Keep E35; no promotion or final-test evaluation.

## Interpretation

Replacing predicted joint displacement with realized joint displacement raises one-step geometric-score agreement from 78.2% to 99.4% for six joints and 76.4% to 99.5% for seven. This is a diagnostic oracle, not an implementable improvement. Actuator/bias mismatch is much larger than geometric nonlinearity in the vector decomposition; estimated-J error is also substantial and can cancel actuator error.

Control wins over prediction-only in more than half the sampled states for every seed at both measured horizons, despite worse E57 native manipulation. Better short-horizon performance on these overlapping procedural histories therefore does not establish native retention. No contact occurred here, so contact and grasp mechanisms remain untested. The five-step ordering gap persists even with the first-step realized-motion oracle.

Next bounded test: fit a joint-response predictor using procedural measured commands, velocities and histories, evaluate it on whole held-out bodies/trajectories before integrating it into a training objective, and compare against the existing command-minus-bias approximation. Keep native data diagnostic-only and E35 as the retained controller. No model training was performed by this audit.

Validation: 81 regression tests pass. Initial v1 stopped on a float64-versus-float32 command-log assertion (maximum observed difference 7.4e-10); its source and failure status are preserved. v2 compares commands exactly at logged float32 precision while retaining full precision in the physical-motion decomposition.

# E79: geometry attribution of successful-run warnings

All 110 E78 response-predictor warnings on successful learned runs arise in two six-joint trajectories. Both are replayed, together with the single missed learned collision. All three full traces, probes, geometry-filter logs and prediction records reproduce exactly. This is an outcome-selected development diagnostic, not an independent performance test.

| Case / estimator | Outcome | Diagnosed steps | Phase counts | Actual substep margin warnings | Margin already violated at step start |
|---|---|---:|---|---:|---:|
| 1 / candidate_2 | success | 87 | {'lift': 82, 'transport': 5} | 86 | 85 |
| 2 / parent_0 | success | 23 | {'release': 23} | 0 | 0 |
| 10 / candidate_0 | failure | 1 | {'descend': 1} | 1 | 0 |

**Lift/transport: real near-table motion.** Case 1 / candidate_2 has 87 warnings (82 lift, five transport), all involving link1 or link2 against the table. The actual sampled motion enters the fixed 2 mm margin on 86 steps, and 85 start inside it. The smallest measured clearance is about 0.298 mm; none of these steps penetrates. These warnings mostly describe genuinely close but successful motion, rather than a fictional carried-object collision. They remain false alarms under the original next-step physical-contact metric.

**Release: predicted proximity not observed.** Case 2 / parent_0 has 23 warnings involving link4 and the left gripper pad. Predicted minimum clearance ranges from 1.669 to 1.989 mm; measured substep clearance stays at the 2 mm reporting cap throughout. This identifies a prediction/path error for this pair, but does not isolate whether arm or finger response, or path interpolation, causes it.

**Missed collision: velocity matters in this example.** Case 10 / candidate_0 collides during descent at step 53. Command-only and no-velocity response paths stay at the 2 mm reporting cap, while the measured right-pad/table clearance reaches -0.369 mm. Both existing velocity-aware paths predict -0.085 mm and detect it. This single selected event does not establish a velocity-aware controller advantage or reverse its false-alarm results.

The object pose is included in retrospective actual-state clearance reconstruction. None of the closest predicted warning pairs involves the object, so frozen carried-object pose does not explain these particular warning minima. All clearances use the same expanded-margin shadow model; the physical model is unchanged. Twenty substeps plus the starting state are sampled, not a mathematical continuous-time clearance proof.

Decision: preserve E78 as a failed gate and keep actions unchanged. Before deploying a warning-triggered filter, distinguish an existing small clearance from a newly predicted approach to contact. A useful next controlled diagnostic is same-state, one-step action comparison at actual collisions and successful near-margin states, with no threshold retuning. Report avoidance, new collisions and task-direction error; one-step avoidance would still require full-task retention validation. Do not simply stop every time the 2 mm margin is entered.

Verification: three exact full replays, 111 attributed steps, 21 samples per predicted path and 21 measured states per diagnosed step. Prediction records match E78 exactly; input, checkpoint and source hashes verified. No training, action intervention, reserved-test evaluation or task-success improvement.

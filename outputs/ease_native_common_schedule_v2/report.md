# E63: complete common native pose-tracking schedule

All twelve observed E46 cases, ten frozen methods. One 32-step calibration per case is verified against E46, then full simulator state and measured controller statistics are cloned for every method. All geometry contype/conaffinity disabled. Fixed complete 1700-step schedule, no early stopping or object feedback. Native nominal nine pose waypoints and original phase budgets (220,260,60,260,260,260,60,260,60). One schedule per case starts at calibrated analytical tool position; ramp .002 m/step continuously across phases, independent of model observations. Native aperture sequence retained. Tool-Z target +Z. Same gains/damping/clips/limits/online updates.

Phase passes only if all final eight samples have goal XYZ error <.006 m and tool-Z direction norm <.08. Complete schedule passes all nine phases. Also retain tracking RMS, joint speeds and action clipping. No velocity threshold, task-success termination or outcome filtering.

Artificial unconstrained tracking, not manipulation, safety or deployable success. Fixed nominal tool poses omit object-offset correction and object-height early release. Observed development cases and two native robot designs, not independent novel-embodiment validation. No training or parameter tuning.

| Group | Robot | Complete schedules pass | Per-seed passes /6 | Phase holds pass | Tracking RMS (m) |
|---|---|---:|---|---:|---:|
| analytical | piper | 6/6 | [6] | 54/54 | 0.014367 |
| analytical | panda | 6/6 | [6] | 54/54 | 0.003223 |
| original | piper | 18/18 | [6, 6, 6] | 162/162 | 0.013869 |
| original | panda | 15/18 | [6, 3, 6] | 155/162 | 0.009391 |
| uncorrected | piper | 18/18 | [6, 6, 6] | 162/162 | 0.010706 |
| uncorrected | panda | 14/18 | [6, 5, 3] | 153/162 | 0.011625 |
| corrected | piper | 16/18 | [4, 6, 6] | 160/162 | 0.016731 |
| corrected | panda | 12/18 | [4, 5, 3] | 143/162 | 0.019461 |

## Missed phase holds

analytical / piper: {}
analytical / panda: {}
original / piper: {}
original / panda: {'retreat': 2, 'lower': 2, 'release': 2, 'settle': 1}
uncorrected / piper: {}
uncorrected / panda: {'lower': 2, 'release': 2, 'descend': 1, 'close': 1, 'lift': 2, 'retreat': 1}
corrected / piper: {'lift': 2}
corrected / panda: {'retreat': 5, 'settle': 5, 'descend': 2, 'close': 1, 'lower': 3, 'release': 3}

All 120 complete schedules run without early termination: 204,000 control steps plus 384 physical calibration steps reused as 120 identical starts (32-step interaction cost charged per trial). Nine checkpoint hashes and archived traces verified. This is a fair common-schedule tracking comparison within this diagnostic, not a native manipulation result. No checkpoint promotion or final-test evaluation.

## Interpretation and decision

E35 passes 33/36 complete schedules, uncorrected E61 32/36 and corrected E61 28/36, versus analytical 12/12. Thirty of the 37 missed learned phase holds never achieve eight consecutive in-tolerance samples; seven acquire a hold but lose the terminal criterion. See [phase acquisition and stability](dynamics.md). Most misses occur on Panda, but response-corrected training also regresses on six-joint PiPER. Keep E35. This diagnostic establishes residual unconstrained tracking deficits; neither a uniform tracking improvement nor an analytical-baseline advantage is supported.

The schedule changes contact forces, timing and nominal goals versus manipulation. It does not isolate every original task failure's cause. Tool-Z control leaves yaw unconstrained. Native data remains diagnostic-only; investigate learned motion estimates and bias during failing pose phases before another training objective.

All 86 regression tests pass. Initial v1 stopped after twenty completed schedules at a repeated-calibration equality assertion; the actual mismatched probe was not persisted, so its cause is unresolved. v2 verifies one calibration per case and clones its complete state/history for every model, guaranteeing shared starts. The failed version is preserved and excluded. The phase figure was visually inspected.

![Phase holds passed out of six cases](phase_holds.png)

# E123: Correction-on/off continuations from identical states

Across90 phase-entry pulse comparisons: correction-on helps final success in16, hurts in9, and leaves final success unchanged in65. 70/90 starts have no prior forbidden collision. E120 remains unchanged.

Scope: six E120 gains and four E120 regressions relative to E118. These ten selected checkpoint/placement pairs are development diagnostics, including previously exposed held placements. They are not an unbiased success-rate benchmark. There is no new training, selector or unseen-body evaluation.

Intervention: at each of all nine task-phase entries, compare E120 continuously on with E120 residual off for up to eight control actions in that phase (0.32 seconds), then resume E120. The base action is still computed from the branch's current measured history; bias remains frozen after calibration. Explicit probing is unchanged. Episodes run to completion. Phases shorter than eight actions have shorter pulses and no matched eighth-step metric.

Identical-state guarantee: replay the deterministic E120 prefix, asserting exact per-step fingerprints through the intervention. Fingerprints include full MuJoCo integration state (warmstart, controls and time included), Gram/cross statistics, frozen bias, recent history, local-history flag, task flags, phase step, probe counter, goal and probe anchor. Saved integration snapshots are compared independently. This is not a qpos-only reset. All ten correction-on traces and state sequences exactly match E120.

Interpretation: help means on succeeds and off fails; hurt means off succeeds and on fails. Unchanged final success does not imply identical motion or no effect. These are effects of short phase-entry pulses, not labels for every individual action or proof of a deployable gate. Long-term differences include feedback/history and phase timing induced by the intervention. Contaminated starts cannot erase an earlier forbidden collision and are separately marked.

## Clean-start comparisons by phase

| Phase | Pairs | Helps | Hurts | Unchanged |
|---|---:|---:|---:|---:|
| approach | 10 | 4 | 3 | 3 |
| descend | 10 | 3 | 2 | 5 |
| close | 9 | 3 | 2 | 4 |
| lift | 9 | 2 | 2 | 5 |
| transport | 7 | 2 | 0 | 5 |
| lower | 7 | 1 | 0 | 6 |
| release | 6 | 1 | 0 | 5 |
| retreat | 6 | 0 | 0 | 6 |
| settle | 6 | 0 | 0 | 6 |

## Every comparison

Position advantage = off minus on goal error; positive favors correction-on. One-step and eighth-step metrics are kept separate from whole-task outcomes. No unmatched phase duration is silently substituted.

| Case/seed | Phase | Clean start | Off actions | On success | Off success | Position advantage at step1 (mm) | Step8 (mm) |
|---|---|---|---:|---|---|---:|---:|
| 8/0 | approach | True | 7 | True | False | -0.0013 | n/a |
| 8/0 | descend | True | 8 | True | False | -0.0935 | -1.2732 |
| 8/0 | close | True | 8 | True | False | -0.0065 | 0.1047 |
| 8/0 | lift | True | 8 | True | False | 0.0000 | -4.8586 |
| 8/0 | transport | True | 8 | True | False | 0.0564 | 0.3255 |
| 8/0 | lower | True | 8 | True | True | 0.0618 | 0.6778 |
| 8/0 | release | True | 8 | True | True | -0.0173 | 0.0308 |
| 8/0 | retreat | True | 8 | True | True | 0.0478 | 3.9809 |
| 8/0 | settle | True | 8 | True | True | 0.0059 | 0.0635 |
| 20/0 | approach | True | 8 | True | False | 0.0466 | 0.0851 |
| 20/0 | descend | True | 8 | True | True | 0.0862 | 0.1146 |
| 20/0 | close | True | 8 | True | True | 0.0068 | -0.0051 |
| 20/0 | lift | True | 8 | True | False | -0.0357 | 5.9073 |
| 20/0 | transport | True | 8 | True | False | 0.0415 | 1.5708 |
| 20/0 | lower | True | 8 | True | True | 0.0984 | 0.1467 |
| 20/0 | release | True | 8 | True | False | 0.0573 | -1.9823 |
| 20/0 | retreat | True | 8 | True | True | 0.0000 | 0.5767 |
| 20/0 | settle | True | 8 | True | True | 0.0120 | 0.0160 |
| 22/0 | approach | True | 8 | False | True | 0.0431 | 0.0493 |
| 22/0 | descend | True | 8 | False | False | -0.0124 | 1.0700 |
| 22/0 | close | False | 8 | False | False | 0.0020 | -0.0152 |
| 22/0 | lift | False | 8 | False | False | -0.0060 | 10.8748 |
| 22/0 | transport | False | 8 | False | False | 0.0928 | -0.0683 |
| 22/0 | lower | False | 8 | False | False | 0.0645 | 0.1345 |
| 22/0 | release | False | 8 | False | False | 0.0361 | -0.1563 |
| 22/0 | retreat | False | 8 | False | False | -0.0445 | -2.9410 |
| 22/0 | settle | False | 8 | False | False | 0.0641 | -0.3023 |
| 9/1 | approach | True | 5 | False | True | 0.0187 | n/a |
| 9/1 | descend | True | 8 | False | True | -0.0401 | -0.0726 |
| 9/1 | close | True | 8 | False | True | -0.0156 | 0.0069 |
| 9/1 | lift | True | 8 | False | True | -0.0192 | -0.2921 |
| 9/1 | transport | True | 8 | False | False | 0.0507 | 0.9811 |
| 9/1 | lower | True | 8 | False | False | 0.0352 | -0.7447 |
| 9/1 | release | False | 8 | False | False | -0.0012 | -0.0812 |
| 9/1 | retreat | False | 8 | False | False | 0.0000 | -0.3698 |
| 9/1 | settle | False | 8 | False | False | 0.0358 | 0.4539 |
| 13/1 | approach | True | 7 | True | True | 0.0738 | n/a |
| 13/1 | descend | True | 8 | True | False | -0.0746 | 3.9964 |
| 13/1 | close | True | 8 | True | False | -0.0102 | 0.1933 |
| 13/1 | lift | True | 8 | True | True | -0.0723 | 8.2107 |
| 13/1 | transport | True | 8 | True | True | 0.0556 | 0.3207 |
| 13/1 | lower | True | 8 | True | False | -0.1188 | -0.3852 |
| 13/1 | release | True | 8 | True | True | -0.0124 | 0.0131 |
| 13/1 | retreat | True | 8 | True | True | -0.1391 | -4.1671 |
| 13/1 | settle | True | 8 | True | True | 0.0069 | -0.0054 |
| 18/1 | approach | True | 8 | True | True | -0.0214 | -0.0384 |
| 18/1 | descend | True | 8 | True | True | -0.0867 | 0.5380 |
| 18/1 | close | True | 8 | True | True | 0.0256 | -0.4990 |
| 18/1 | lift | True | 8 | True | True | -0.0359 | 7.1097 |
| 18/1 | transport | True | 8 | True | True | -0.0027 | -1.2512 |
| 18/1 | lower | True | 8 | True | True | -0.0032 | 0.3374 |
| 18/1 | release | True | 8 | True | True | -0.0291 | -0.2623 |
| 18/1 | retreat | True | 8 | True | True | -0.0815 | -1.4250 |
| 18/1 | settle | True | 8 | True | True | -0.0102 | -0.0238 |
| 19/1 | approach | True | 8 | False | False | -0.0445 | 0.0820 |
| 19/1 | descend | True | 8 | False | True | 0.0145 | 0.4674 |
| 19/1 | close | True | 8 | False | True | 0.0189 | 0.0444 |
| 19/1 | lift | True | 8 | False | True | -0.0027 | -10.8610 |
| 19/1 | transport | False | 8 | False | False | -0.0779 | -0.2515 |
| 19/1 | lower | False | 8 | False | False | -0.0156 | 0.0516 |
| 19/1 | release | False | 8 | False | False | 0.0839 | -0.0788 |
| 19/1 | retreat | False | 8 | False | False | 0.0669 | 0.0836 |
| 19/1 | settle | False | 8 | False | False | -0.0737 | -0.5630 |
| 23/1 | approach | True | 8 | True | False | 0.0612 | 0.6523 |
| 23/1 | descend | True | 8 | True | True | -0.1201 | 1.3301 |
| 23/1 | close | True | 8 | True | True | 0.0043 | 0.0136 |
| 23/1 | lift | True | 8 | True | True | 0.0000 | -1.0796 |
| 23/1 | transport | True | 8 | True | True | -0.0318 | -0.3224 |
| 23/1 | lower | True | 8 | True | True | 0.0000 | 0.0022 |
| 23/1 | release | True | 8 | True | True | -0.0087 | -0.0456 |
| 23/1 | retreat | True | 8 | True | True | 0.0069 | 0.1666 |
| 23/1 | settle | True | 8 | True | True | -0.0159 | 0.0644 |
| 1/2 | approach | True | 5 | True | False | 0.0004 | n/a |
| 1/2 | descend | True | 8 | True | False | -0.0160 | 0.7456 |
| 1/2 | close | True | 8 | True | False | -0.0027 | -0.1773 |
| 1/2 | lift | True | 8 | True | True | 0.0000 | -0.0078 |
| 1/2 | transport | True | 8 | True | True | 0.0390 | -0.0075 |
| 1/2 | lower | True | 8 | True | True | 0.0724 | -0.5342 |
| 1/2 | release | True | 8 | True | True | 0.0006 | 0.0031 |
| 1/2 | retreat | True | 8 | True | True | -0.1332 | 0.3313 |
| 1/2 | settle | True | 8 | True | True | 0.0097 | 0.0347 |
| 18/2 | approach | True | 8 | False | True | -0.0134 | 0.1088 |
| 18/2 | descend | True | 8 | False | False | 0.0909 | 0.2823 |
| 18/2 | close | True | 8 | False | False | 0.0078 | 0.0218 |
| 18/2 | lift | True | 8 | False | False | 0.0934 | -3.3258 |
| 18/2 | transport | False | 8 | False | False | -0.1852 | -0.6340 |
| 18/2 | lower | False | 8 | False | False | -0.0023 | -0.3598 |
| 18/2 | release | False | 8 | False | False | 0.0222 | -0.2343 |
| 18/2 | retreat | False | 8 | False | False | 0.0225 | -9.1095 |
| 18/2 | settle | False | 8 | False | False | 0.0006 | 0.0045 |

## Local progress versus final outcome

13 clean-start outcome-changing comparisons have first-step position-error advantage opposite to the final-success effect (excluding magnitudes <=0.001 mm). This compares one-step position progress with an up-to-eight-step intervention, not a one-action causal label. Local progress is not automatically a task-value signal.

```json
[
  {
    "case": 8,
    "seed": 0,
    "phase": "approach",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.0012695838156381173
  },
  {
    "case": 8,
    "seed": 0,
    "phase": "descend",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.09354479686569939
  },
  {
    "case": 8,
    "seed": 0,
    "phase": "close",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.0064597135590680616
  },
  {
    "case": 20,
    "seed": 0,
    "phase": "lift",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.03572479610505774
  },
  {
    "case": 22,
    "seed": 0,
    "phase": "approach",
    "success_effect": -1,
    "first_step_position_advantage_mm": 0.04306185293643551
  },
  {
    "case": 9,
    "seed": 1,
    "phase": "approach",
    "success_effect": -1,
    "first_step_position_advantage_mm": 0.018704509228580773
  },
  {
    "case": 13,
    "seed": 1,
    "phase": "descend",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.07461417116839952
  },
  {
    "case": 13,
    "seed": 1,
    "phase": "close",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.010243004665342823
  },
  {
    "case": 13,
    "seed": 1,
    "phase": "lower",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.11876724748463574
  },
  {
    "case": 19,
    "seed": 1,
    "phase": "descend",
    "success_effect": -1,
    "first_step_position_advantage_mm": 0.014463481917928456
  },
  {
    "case": 19,
    "seed": 1,
    "phase": "close",
    "success_effect": -1,
    "first_step_position_advantage_mm": 0.01888138220889745
  },
  {
    "case": 1,
    "seed": 2,
    "phase": "descend",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.016008368947315055
  },
  {
    "case": 1,
    "seed": 2,
    "phase": "close",
    "success_effect": 1,
    "first_step_position_advantage_mm": -0.0027370166747267555
  }
]
```

All100 full episodes, every state/command trace, intervention snapshot, history fingerprint and outcome are saved. Videos include successes and failures and paired examples of helpful/harmful pulses. Offline simulator intervention outcomes are diagnostic labels only; no simulator lookahead is added to E120.

Final audit: all90 snapshot/prefix pairs independently verified; 31978 action choices independently checked; 4 full episodes independently reexecuted exactly; 38 trace-verified videos. Original E120 input/model hashes remain unchanged.

All four selected E120 regressions have at least one successful off-pulse continuation. Rescuing phases by case/seed:

```json
{
  "22/0": [
    "approach"
  ],
  "9/1": [
    "approach",
    "descend",
    "close",
    "lift"
  ],
  "19/1": [
    "descend",
    "close",
    "lift"
  ],
  "18/2": [
    "approach"
  ]
}
```

This does not constitute a repaired policy: choosing these phases by known outcome would be an oracle. The next useful step is broader matched-state intervention data on fitting placements and a task-value gate evaluated separately on held placements. The same phase contains both beneficial and harmful pulses, and local position progress can disagree with final success. Keep E120 unchanged.

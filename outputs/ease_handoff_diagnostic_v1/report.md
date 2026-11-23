# E49: probe-to-descent command diagnostic

192 exact prefix replays: all E48 conditions/controllers/cases. Stop after 16 descent steps (0.64 s), or earlier task termination. Episode outcomes are inherited from E48, not new full-episode evaluations. No controller intervention, tuning or training.

Log the actual damped-Jacobian motion term, pre-update bias, raw sum, clipped action, executed command, measured dq, singular values and pose response. Command_jump compares each descent action with the last probe command; only the first value is the handoff discontinuity. Diagnostic only.

| Condition | Robot | Method | E48 successes /6 | First motion norm | First bias norm | First command jump | First clipped cases /6 | Clipped descent steps / observed |
|---|---|---|---:|---:|---:|---:|---:|---:|
| both | piper | analytical | 6 | 0.01047 | 0.00379 | 0.01059 | 0 | 60/96 |
| both | piper | neural_0 | 6 | 0.01448 | 0.00378 | 0.01371 | 0 | 68/96 |
| both | piper | neural_1 | 6 | 0.01154 | 0.00377 | 0.01124 | 0 | 67/96 |
| both | piper | neural_2 | 6 | 0.01215 | 0.00378 | 0.01221 | 0 | 62/96 |
| both | panda | analytical | 6 | 0.03014 | 0.00659 | 0.02489 | 2 | 46/96 |
| both | panda | neural_0 | 4 | 0.01829 | 0.00660 | 0.01864 | 0 | 26/96 |
| both | panda | neural_1 | 1 | 0.04535 | 0.00647 | 0.04269 | 3 | 57/96 |
| both | panda | neural_2 | 2 | 0.01394 | 0.00658 | 0.01415 | 0 | 26/96 |
| motion_only | piper | analytical | 6 | 0.01047 | 0.04452 | 0.04778 | 6 | 96/96 |
| motion_only | piper | neural_0 | 5 | 0.01448 | 0.05271 | 0.05684 | 6 | 96/96 |
| motion_only | piper | neural_1 | 6 | 0.01154 | 0.05341 | 0.05614 | 6 | 96/96 |
| motion_only | piper | neural_2 | 6 | 0.01215 | 0.03915 | 0.04162 | 6 | 96/96 |
| motion_only | panda | analytical | 6 | 0.03014 | 0.02270 | 0.03796 | 2 | 33/96 |
| motion_only | panda | neural_0 | 3 | 0.01829 | 0.02082 | 0.03213 | 2 | 32/96 |
| motion_only | panda | neural_1 | 2 | 0.04535 | 0.03145 | 0.05539 | 5 | 74/96 |
| motion_only | panda | neural_2 | 5 | 0.01394 | 0.01912 | 0.03119 | 0 | 27/96 |
| bias_only | piper | analytical | 6 | 0.01046 | 0.00379 | 0.01056 | 0 | 58/96 |
| bias_only | piper | neural_0 | 6 | 0.01450 | 0.00378 | 0.01373 | 0 | 68/96 |
| bias_only | piper | neural_1 | 5 | 0.01150 | 0.00377 | 0.01123 | 0 | 67/96 |
| bias_only | piper | neural_2 | 6 | 0.01048 | 0.00378 | 0.01069 | 0 | 58/96 |
| bias_only | panda | analytical | 6 | 0.02300 | 0.00659 | 0.02130 | 2 | 42/96 |
| bias_only | panda | neural_0 | 4 | 0.01893 | 0.00660 | 0.01913 | 0 | 28/96 |
| bias_only | panda | neural_1 | 2 | 0.04176 | 0.00647 | 0.04065 | 1 | 54/96 |
| bias_only | panda | neural_2 | 5 | 0.01283 | 0.00658 | 0.01285 | 0 | 32/96 |
| neither | piper | analytical | 6 | 0.01046 | 0.04452 | 0.04774 | 6 | 96/96 |
| neither | piper | neural_0 | 6 | 0.01450 | 0.05271 | 0.05676 | 6 | 96/96 |
| neither | piper | neural_1 | 6 | 0.01150 | 0.05341 | 0.05614 | 6 | 96/96 |
| neither | piper | neural_2 | 6 | 0.01048 | 0.03915 | 0.04107 | 6 | 96/96 |
| neither | panda | analytical | 4 | 0.02300 | 0.02270 | 0.03589 | 2 | 37/96 |
| neither | panda | neural_0 | 3 | 0.01893 | 0.02082 | 0.03278 | 2 | 32/96 |
| neither | panda | neural_1 | 1 | 0.04176 | 0.03145 | 0.05302 | 5 | 73/96 |
| neither | panda | neural_2 | 4 | 0.01283 | 0.01912 | 0.02884 | 0 | 36/96 |

Command norms are in radians; the first jump compares bounded descent action with the last executed probe command. Later descent trajectories diverge across conditions, so aggregate differences are descriptive. Early termination is retained with actual denominators. All 192 prefixes and refresh states reproduce E48 exactly; this is an observational diagnostic, not a successful intervention or 192 new complete task evaluations. No model or controller promotion.

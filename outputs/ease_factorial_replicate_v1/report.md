# E27: fixed-data replication and interaction cost

Three training seeds; seed 0 is the E26 pilot. Eight new 15,000-update policies use the unchanged training data and fixed 3,211-chunk recovery union. No final-test evaluation or validation tuning.

Each cell below lists successes for seeds 0 / 1 / 2, each out of 96 matched cases. Training gates are retained even when failed.

| Data | Context | Seen axes, seen posture | New axes, seen posture | Seen axes, shifted posture | New axes, shifted posture |
|---|---|---|---|---|---|
| original | reactive | 54 / 30 / 39 | 39 / 24 / 40 | 51 / 26 / 60 | 31 / 13 / 36 |
| original | correct | 96 / 96 / 96 | 59 / 43 / 56 | 95 / 50 / 67 | 35 / 24 / 39 |
| original | empty | 49 / 52 / 40 | 28 / 18 / 20 | 54 / 35 / 39 | 21 / 13 / 25 |
| original | twin | 12 / 9 / 12 | 21 / 7 / 12 | 41 / 12 / 11 | 12 / 7 / 14 |
| shared_recovery | reactive | 93 / 96 / 96 | 74 / 78 / 48 | 96 / 92 / 85 | 48 / 45 / 46 |
| shared_recovery | correct | 96 / 96 / 96 | 70 / 72 / 73 | 95 / 96 / 96 | 51 / 48 / 48 |
| shared_recovery | empty | 96 / 96 / 96 | 75 / 79 / 66 | 88 / 96 / 90 | 51 / 49 / 48 |
| shared_recovery | twin | 49 / 61 / 77 | 57 / 61 / 65 | 94 / 93 / 91 | 47 / 48 / 48 |

## Overall validation and cost

Success columns are out of 384 per seed. Costs average all cases and seeds, including full-horizon failures.

| Data | Context | Seed 0 | Seed 1 | Seed 2 | Mean task steps | Mean single-use steps | Mean reused steps |
|---|---|---:|---:|---:|---:|---:|---:|
| original | reactive | 175 | 93 | 175 | 103.2 | 103.2 | 103.2 |
| original | correct | 285 | 213 | 258 | 62.5 | 94.5 | 65.1 |
| original | empty | 152 | 118 | 124 | 106.5 | 106.5 | 106.5 |
| shared_recovery | reactive | 311 | 311 | 275 | 50.1 | 50.1 | 50.1 |
| shared_recovery | correct | 312 | 312 | 313 | 42.3 | 74.3 | 45.0 |
| shared_recovery | empty | 310 | 320 | 300 | 45.1 | 45.1 | 45.1 |

## Recovery-trained transfer by joint count

Each entry lists seed 0 / 1 / 2 successes out of 48.

| Context | Joint count | New axes, seen posture | New axes, shifted posture |
|---|---:|---|---|
| reactive | 6 | 26 / 31 / 0 | 0 / 0 / 0 |
| reactive | 7 | 48 / 47 / 48 | 48 / 45 / 46 |
| correct | 6 | 22 / 24 / 25 | 3 / 0 / 0 |
| correct | 7 | 48 / 48 / 48 | 48 / 48 / 48 |
| empty | 6 | 27 / 31 / 18 | 7 / 1 / 1 |
| empty | 7 | 48 / 48 / 48 | 44 / 48 / 47 |

## Training control

| Data | Context | Seed | Success / 256 | Gate |
|---|---|---:|---:|---|
| original | reactive | 0 | 154 | False |
| original | correct | 0 | 256 | True |
| original | empty | 0 | 143 | False |
| original | twin | 0 | 44 | False |
| shared_recovery | reactive | 0 | 246 | False |
| shared_recovery | correct | 0 | 256 | True |
| shared_recovery | empty | 0 | 256 | True |
| shared_recovery | twin | 0 | 135 | False |
| original | reactive | 1 | 95 | False |
| original | correct | 1 | 256 | True |
| original | empty | 1 | 141 | False |
| original | twin | 1 | 41 | False |
| shared_recovery | reactive | 1 | 255 | True |
| shared_recovery | correct | 1 | 256 | True |
| shared_recovery | empty | 1 | 256 | True |
| shared_recovery | twin | 1 | 161 | False |
| original | reactive | 2 | 123 | False |
| original | correct | 2 | 256 | True |
| original | empty | 2 | 118 | False |
| original | twin | 2 | 43 | False |
| shared_recovery | reactive | 2 | 256 | True |
| shared_recovery | correct | 2 | 256 | True |
| shared_recovery | empty | 2 | 256 | True |
| shared_recovery | twin | 2 | 197 | False |

## Interaction accounting

Costs are control steps (25 Hz), excluding resets and computation. All unsuccessful episodes consume the full 150-step horizon. Correct/twin history requires a 32-step probe. Reactive/empty needs no probe: the evaluator may generate an ignored probe for empty, so zero cost describes the minimal deployment input, not literal evaluator runtime.

Single-use cost adds 32 steps per task. Reuse cost adds one probe for each body/posture group of 12 goals (8 probes per 96-case cell); reuse is aggregate suite accounting, not per-task latency. Frozen context is goal-independent and identical within each group. Report success jointly with cost; lower capped cost alone does not establish better control.

The table averages over three seeds; success is mean successes / 96. Budget success includes calibration and counts only successful tasks finished by that budget. Full seed-level costs, six/seven-joint results, collisions, and paired wins are in comparison.json.

| Data | Context | Condition | Success / 96 | Mean task steps | Single use | Reused | Success by 50 | Success by 100 | Success by 150 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| original | reactive | seen_axes_seen_posture | 41.0 | 97.3 | 97.3 | 97.3 | 34.7 | 40.0 | 41.0 |
| original | reactive | new_axes_seen_posture | 34.3 | 107.6 | 107.6 | 107.6 | 28.7 | 33.3 | 34.3 |
| original | reactive | seen_axes_shifted_posture | 45.7 | 92.4 | 92.4 | 92.4 | 38.0 | 44.3 | 45.7 |
| original | reactive | new_axes_shifted_posture | 26.7 | 115.6 | 115.6 | 115.6 | 23.7 | 26.3 | 26.7 |
| original | correct | seen_axes_seen_posture | 96.0 | 13.5 | 45.5 | 16.2 | 90.0 | 96.0 | 96.0 |
| original | correct | new_axes_seen_posture | 52.7 | 81.2 | 113.2 | 83.9 | 20.3 | 51.7 | 52.0 |
| original | correct | seen_axes_shifted_posture | 70.7 | 50.5 | 82.5 | 53.2 | 64.0 | 70.3 | 70.3 |
| original | correct | new_axes_shifted_posture | 32.7 | 104.6 | 136.6 | 107.2 | 25.7 | 32.7 | 32.7 |
| original | empty | seen_axes_seen_posture | 47.0 | 87.2 | 87.2 | 87.2 | 43.0 | 46.7 | 47.0 |
| original | empty | new_axes_seen_posture | 22.0 | 122.1 | 122.1 | 122.1 | 18.7 | 22.0 | 22.0 |
| original | empty | seen_axes_shifted_posture | 42.7 | 93.5 | 93.5 | 93.5 | 40.7 | 41.0 | 42.7 |
| original | empty | new_axes_shifted_posture | 19.7 | 123.0 | 123.0 | 123.0 | 19.7 | 19.7 | 19.7 |
| shared_recovery | reactive | seen_axes_seen_posture | 95.0 | 18.1 | 18.1 | 18.1 | 95.0 | 95.0 | 95.0 |
| shared_recovery | reactive | new_axes_seen_posture | 66.7 | 70.6 | 70.6 | 70.6 | 54.0 | 62.7 | 66.7 |
| shared_recovery | reactive | seen_axes_shifted_posture | 91.0 | 24.8 | 24.8 | 24.8 | 90.0 | 91.0 | 91.0 |
| shared_recovery | reactive | new_axes_shifted_posture | 46.3 | 87.0 | 87.0 | 87.0 | 45.3 | 46.3 | 46.3 |
| shared_recovery | correct | seen_axes_seen_posture | 96.0 | 13.6 | 45.6 | 16.2 | 89.7 | 96.0 | 96.0 |
| shared_recovery | correct | new_axes_seen_posture | 71.7 | 61.9 | 93.9 | 64.6 | 23.7 | 64.7 | 70.0 |
| shared_recovery | correct | seen_axes_shifted_posture | 95.7 | 12.7 | 44.7 | 15.4 | 95.3 | 95.7 | 95.7 |
| shared_recovery | correct | new_axes_shifted_posture | 49.0 | 80.9 | 112.9 | 83.6 | 38.3 | 49.0 | 49.0 |
| shared_recovery | empty | seen_axes_seen_posture | 96.0 | 18.7 | 18.7 | 18.7 | 96.0 | 96.0 | 96.0 |
| shared_recovery | empty | new_axes_seen_posture | 73.3 | 58.7 | 58.7 | 58.7 | 63.0 | 71.7 | 73.3 |
| shared_recovery | empty | seen_axes_shifted_posture | 91.3 | 21.6 | 21.6 | 21.6 | 91.3 | 91.3 | 91.3 |
| shared_recovery | empty | new_axes_shifted_posture | 49.3 | 81.5 | 81.5 | 81.5 | 49.0 | 49.3 | 49.3 |

Three training seeds do not represent three independent robot-family samples. Seed 0 was already observed when replication was proposed. The fixed recovery union tests training randomness, not data-collection randomness. Six-joint and shifted-posture failures remain relevant; no equivalence or broad embodiment-adaptation claim follows from close aggregate success.

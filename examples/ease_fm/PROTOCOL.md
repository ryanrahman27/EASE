# EASE-FM v2: benchmark and matched history experiments

This is a standalone MuJoCo/PyTorch experiment inside OpenPI. It uses neither
pretrained OpenPI weights nor hardware. The arm is a procedural seven-joint
capsule model, **not a validated Franka model**.

## Reproduce

Run from the repository root in an independent environment with NumPy, MuJoCo
and PyTorch. Recorded versions: Python 3.10.11, NumPy 2.2.6, MuJoCo 3.8.0,
PyTorch 2.11.0+cu128. The main OpenPI project has different requirements.
Use `--device cpu` if CUDA is unavailable. All output directories must be new.

```powershell
python -m unittest discover -s tests -p 'test_ease_fm*.py' -v
python -m examples.ease_fm.benchmark --output outputs/ease_new_data
python -m examples.ease_fm.run_suite --data outputs/ease_new_data --output outputs/ease_new_comparison --steps 5000 --device cuda
python -m examples.ease_fm.report outputs/ease_new_comparison
```

A suite writes its protocol before training both methods with seeds 0, 1 and 2.
It evaluates **validation only**, using the final prescribed checkpoint without
early stopping or checkpoint selection. Evaluation is batched but the MuJoCo
environments are independent. For individual runs:

```powershell
python -m examples.ease_fm.experiment train --data outputs/ease_new_data --output outputs/ease_one_model --method history --steps 5000 --seed 0 --device cuda
python -m examples.ease_fm.experiment evaluate --data outputs/ease_new_data --checkpoint outputs/ease_one_model/policy.pt --output outputs/ease_one_model/validation.json --device cuda
```

The explicit `--split test` option evaluates reserved interpolation/extrapolation
bodies. Keep it unused until the method and protocol are fixed.

## Benchmark and information boundaries

- Ten training bodies, three validation bodies, three reserved interpolation
  bodies and three reserved extrapolation bodies. Extrapolation uses all links
  25–35% longer than nominal; other splits use independent ±20% link changes.
  Masses, servo strength and damping vary; joint topology is fixed.
- Sixty training goals, twelve validation goals and twelve test goals. Each XYZ
  goal and starting-state seed is shared across bodies. A privileged expert must
  succeed on **every** body before a candidate is accepted. Rejected goals and
  failing bodies are logged. This evaluates a common expert-feasible workspace,
  not the entirety of each robot's workspace. Test morphology is used for
  benchmark feasibility screening, never as a student input or training example.
- Only training-body demonstrations are saved. Training uses no validation/test
  trajectories or physical parameters. Normalization uses training data only.
- Control is 25 Hz, physics is 500 Hz, and joint-position increments are bounded
  to ±0.04 radians. Success requires five consecutive steps within 3 cm and no
  contact anywhere in the episode. Contacts are checked at every physics substep.
  Episodes last at most 150 steps.
- Student observation is joint position/velocity, end-effector XYZ and goal XYZ.
  End-effector position is an assumed external measurement supplied by simulation;
  this sensing assumption must be explicit in any eventual paper.
- Exact XML, body parameters, seeds, candidate rejection audit, calibration data
  and file SHA-256 hashes are saved. Checkpoints include normalization buffers.
  Training records include source snapshots, parameter counts, runtime, seeds,
  losses and checkpoint/manifest hashes.

## Models and interventions

Both methods share a two-layer width-128, four-head flow Transformer, 16-action
chunks, ten Euler steps and one executed action before replanning. Matching seeds
produce identical initial shared weights, minibatches, interpolation times and
flow noise. Both use AdamW at 3e-4 and batch size 64. The history model adds a
three-layer width-128, four-head causal Transformer and latent projection. It has
more parameters, so this is not a parameter-count-matched baseline; same-model
history interventions supply the more direct causal check.

Each of up to 32 tokens contains measured `[q, dq, ee_xyz]`, the **executed clipped
command**, and measured state difference with wrapped angle deltas. Goals, IDs,
Jacobians and true morphology parameters are excluded. A learned empty token
handles zero history. Training samples variable-length contexts from calibration
or a **different task episode on the same training body**. There is no future
target-episode context or auxiliary dynamics loss.

Every evaluation case receives 32 separate-episode bounded excitation actions
(1.28 seconds), then resets to the same task starting state. This tests context
transfer across a reset; it is not continuous within-episode self-exploration.

| Condition | Context during the task |
| --- | --- |
| `baseline/none` | No encoder; calibration budget still counted |
| `history/empty` | Empty context throughout |
| `history/correct` | Frozen calibration from the target body |
| `history/wrong` | Frozen calibration from a different training body with the same excitation seed |
| `history/shuffled` | Target-body transitions in shuffled temporal order |
| `history/online` | Correct calibration initially; rolling context updated after each task transition |

Calibration uses no expert. Donor IDs appear only in evaluation logs. Inference
noise is paired by task and step across methods/interventions; early termination
cannot shift another trial's stream. All weights remain frozen. Compare `correct`
with `wrong` to test history identity, and `online` with `correct` to test updating.

## Recorded experiments, October 15, 2026

`outputs/ease_fm_v2` contains 600 successful training demonstrations (17,953
transitions), 80 calibration episodes and 84 accepted goals. Screening rejected
107 candidates. Six test bodies remain unused for learned-policy evaluation.

`outputs/ease_fm_v2_comparison` is the fixed 1,500-step development suite. Across
three training seeds and 36 validation cases per seed:

| Condition | Successes | Rate |
| --- | ---: | ---: |
| Baseline | 86/108 | 79.6% |
| Correct history | 79/108 | 73.1% |
| Empty history | 79/108 | 73.1% |
| Wrong-body history | 80/108 | 74.1% |
| Shuffled history | 80/108 | 74.1% |
| Online history | 81/108 | 75.0% |

This does not support the adaptation claim. Baseline seed rates range from 55.6%
to 94.4%, motivating a second fixed 5,000-step comparison for **both** methods.
That protocol and its outputs are stored separately in
`outputs/ease_fm_v2_comparison_5000`. This is validation-driven development, not an
independent confirmation experiment.

The completed 5,000-step suite produced:

| Condition | Successes | Rate |
| --- | ---: | ---: |
| Baseline | 75/108 | 69.4% |
| Correct history | 73/108 | 67.6% |
| Empty history | 71/108 | 65.7% |
| Wrong-body history | 72/108 | 66.7% |
| Shuffled history | 74/108 | 68.5% |
| Online history | 72/108 | 66.7% |

Correct minus wrong history is +0.9 percentage points with an exploratory 95%
interval of [−11.1, +11.1]. Correct minus baseline is −1.9 points [−16.7, +11.1].
Neither budget supports a reliable history benefit. Longer training reduced loss
but did not stabilize closed-loop success; seed 0 deteriorated while seeds 1 and
2 stayed substantially stronger. Baseline and history parameter counts are
288,263 and 711,943, respectively.

A post-hoc diagnostic of the weak 5,000-step baseline seed 0 tested three training
bodies and six goals per condition. With ten integration steps, successes were
1/18 on seen training goals, 3/18 on new validation goals for training bodies,
and 4/18 on validation bodies. Increasing integration to fifty steps yielded
3/18, 4/18 and 3/18. Thus this seed's failure is not confined to unseen bodies,
and solver resolution alone did not fix it. This small probe does not identify
the root cause: optimization instability and state-distribution drift still need
to be separated. Its records are in
`outputs/ease_fm_v2_comparison_5000/baseline_seed0/diagnostic.json`.

```powershell
python -m examples.ease_fm.diagnose_baseline --data outputs/ease_fm_v2 --checkpoint outputs/ease_fm_v2_comparison_5000/baseline_seed0/policy.pt --output outputs/ease_diagnostic_new.json
```

The optional `plot_results` module requires matplotlib and writes a static PNG:

```powershell
python -m examples.ease_fm.plot_results outputs/ease_fm_v2_comparison_5000
```

Each suite's `summary.txt`, `summary.json` and `trials.csv` report aggregate,
per-seed and per-body results. Paired differences use 5,000 crossed bootstrap
draws over training seeds, bodies and goals. With only three validation bodies and
three training seeds, intervals are exploratory, not evidence for broad
embodiment generalization. Repeated trials do not count as independent robots.

Do not mix v2 results with `outputs/ease_fm_pilot` or
`outputs/ease_fm_baseline_pilot`, which used the old instantaneous-success and
contact-at-control-boundaries contract. The old README records that initial pilot.

## Remaining scientific work

If correct history fails to beat wrong/empty history, investigate identifiability,
excitation coverage and whether the task requires body-specific adaptation before
scaling the architecture. Loss reduction alone is not evidence of adaptation.

The immediate gate remains a stable history-free controller. Next investigate
teacher-forced action error versus rollout divergence, EMA/checkpoint behavior,
and recovery-state demonstration coverage using training and validation bodies.
If collecting recovery data, regenerate one shared dataset for both methods and
retain the original runs. Do not use the current results as a positive paper claim.

### Follow-up: control failure diagnosis and recovery data

The training-state action audit found similar first-action RMSE across seeds
(approximately 0.0063–0.0065 radians). Swapping goals increased flow loss markedly,
so the policy was using goal information. Train/eval forward parity was verified.
The closed-loop audit localized a failure mechanism: the final joint of this
position-only fixture does not change end-effector XYZ, and the expert's command
for it is zero. Learned command errors accumulated in that joint, producing states
outside its very narrow demonstration distribution. On failed raw rollouts, the
median maximum standardized final-joint position and velocity were about 5.7 and
36.5, respectively.

Setting only that final joint command to zero during a **diagnostic intervention**
changed the three baseline seed success counts from 6/36, 35/36 and 34/36 to
36/36 each. This fixture-specific command override is not used in the learned
policy or the subsequent main comparison.

Instead, `recovery_data.py` added one physically simulated recovery episode per
original training episode, starting from a recorded training state with **all
seven** joint positions perturbed by up to 0.35 radians and velocities by up to
0.15 rad/s. Of 600 recoveries, 593 succeeded. Failures remain logged and excluded
from imitation training. All original files, body splits and goals are preserved;
the new manifest points to the parent's SHA-256 hash. The accepted combined
dataset has 32,415 transitions.

At the same 5,000-step budget and seeds 0–2, the ordinary history-free policy
reached 108/108 validation trials (36/36 per seed). Correct-history success was
101/108, empty-history 102/108, wrong-history 99/108 and online-history 100/108.
The data expansion substantially improved this limited control test, but did not
establish a history benefit. Raw results live in
`outputs/ease_fm_v2_recovery_comparison/summary.json`. Both methods used the same
expanded dataset; their actions were not overridden.

```powershell
python -m examples.ease_fm.recovery_data --data outputs/ease_fm_v2 --output outputs/ease_recovery_new
python -m examples.ease_fm.run_suite --data outputs/ease_recovery_new --output outputs/ease_recovery_comparison_new --steps 5000 --device cuda
python -m examples.ease_fm.report outputs/ease_recovery_comparison_new
```

The user's clarified goal is transfer across kinematically different arms,
including six and seven DoF, rather than only this common-topology benchmark.
See [the mixed-topology design](MIXED_DESIGN.md) for the current direction.

Still absent: a validated Franka-like model, pushing, grippers, dynamics-change
recovery, exploration-budget curves, a parameter-count-matched control, privileged
references, fine-tuning comparisons, hardware validation and a fresh related-work
audit. No novelty or conference-readiness claim has been established. Keep the
final test split reserved while resolving development questions.

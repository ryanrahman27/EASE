# E122: Select corrections using preceding measured prediction errors

reference: 50/72, 18 collision episodes; unbounded: 52/72, 18 collision episodes; candidate: 51/72, 21 collision episodes.

reference = E118 frozen estimator; unbounded = E120 fixed per-step correction; candidate = E122 selective correction. E120 still has a per-step bound; unbounded means no additional selection in these inherited filenames.

Keep both estimator weights frozen. Estimate both motion-response matrices before each action, then score them against realized measured joint and task displacement after that action. This is conditional response-model error using realized joint motion, not a command-conditioned forward rollout. Update squared-error EWMA with decay0.9. Use unchanged E120 correction for the first8 transitions, then permit it only if E119 preceding EWMA error is <=0.95 times E118 error; otherwise use the frozen E118 action on the current history. No phase reset. The explicit probe remains fixed but contributes evidence. Current-transition or future errors cannot affect the action that generated them. No contact, outcome, task identity, geometry, analytical fallback or physics rollout enters the gate.

The threshold was fixed before inspecting the ten shadow-gate replays. All24 training-body placements and three seeds are evaluated;16 fitting and8 held placements. These are reused development placements, not an untouched test or unseen bodies. No fitting, threshold sweep or checkpoint selection. Lower motion prediction error is a proposed signal, not a proven measure of action value.

## Help-versus-hurt diagnostic

All six E120 gains and four E120 losses relative to E118 are exactly replayed with original actions and a shadow gate. Statistics below use active steps before any prior forbidden collision. The transition causing the first collision is included; subsequent transitions are excluded. Held gains are inspected diagnostically and are not used to fit thresholds.

| Case / seed | E120 outcome change | Shadow gate-on fraction | Mean E119/E118 prediction MSE | Selective success |
|---|---|---:|---:|---|
| 8 / 0 | gain | 0.148 | 1.007 | True |
| 20 / 0 | gain | 0.399 | 0.970 | False |
| 22 / 0 | loss | 0.952 | 0.714 | False |
| 9 / 1 | loss | 0.122 | 1.030 | True |
| 13 / 1 | gain | 0.249 | 1.025 | False |
| 18 / 1 | gain | 0.539 | 0.939 | True |
| 19 / 1 | loss | 0.132 | 1.109 | False |
| 23 / 1 | gain | 0.158 | 1.006 | False |
| 1 / 2 | gain | 0.262 | 1.016 | True |
| 18 / 2 | loss | 0.141 | 1.023 | False |

## All task outcomes

| Method | Split | DOF | Success | Collision episodes |
|---|---|---|---:|---:|
| reference | all | all | 50/72 | 18 |
| reference | all | 6 | 31/36 | 5 |
| reference | all | 7 | 19/36 | 13 |
| reference | fitting | all | 38/48 | 7 |
| reference | fitting | 6 | 20/24 | 4 |
| reference | fitting | 7 | 18/24 | 3 |
| reference | held | all | 12/24 | 11 |
| reference | held | 6 | 11/12 | 1 |
| reference | held | 7 | 1/12 | 10 |
| unbounded | all | all | 52/72 | 18 |
| unbounded | all | 6 | 33/36 | 3 |
| unbounded | all | 7 | 19/36 | 15 |
| unbounded | fitting | all | 37/48 | 9 |
| unbounded | fitting | 6 | 22/24 | 2 |
| unbounded | fitting | 7 | 15/24 | 7 |
| unbounded | held | all | 15/24 | 9 |
| unbounded | held | 6 | 11/12 | 1 |
| unbounded | held | 7 | 4/12 | 8 |
| candidate | all | all | 51/72 | 21 |
| candidate | all | 6 | 32/36 | 4 |
| candidate | all | 7 | 19/36 | 17 |
| candidate | fitting | all | 36/48 | 12 |
| candidate | fitting | 6 | 21/24 | 3 |
| candidate | fitting | 7 | 15/24 | 9 |
| candidate | held | all | 15/24 | 9 |
| candidate | held | 6 | 11/12 | 1 |
| candidate | held | 7 | 4/12 | 8 |
| analytical | all | all | 18/24 | 5 |
| analytical | all | 6 | 11/12 | 1 |
| analytical | all | 7 | 7/12 | 4 |
| analytical | fitting | all | 12/16 | 3 |
| analytical | fitting | 6 | 7/8 | 1 |
| analytical | fitting | 7 | 5/8 | 2 |
| analytical | held | all | 6/8 | 2 |
| analytical | held | 6 | 4/4 | 0 |
| analytical | held | 7 | 2/4 | 2 |

Retained 3/6 gains; recovered 1/4 regressions. Strict retention gate False.

## Every paired change

```json
[
  {
    "baseline": "reference",
    "case": 6,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 8,
    "seed": 0,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 18,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 19,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 22,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 1,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 8,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 10,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 18,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 19,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 20,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 1,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 3,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "reference",
    "case": 10,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "reference",
    "case": 18,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 6,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 18,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 19,
    "seed": 0,
    "gain": false,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 20,
    "seed": 0,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 1,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "unbounded",
    "case": 8,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 9,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 10,
    "seed": 1,
    "gain": true,
    "split": "fitting",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 13,
    "seed": 1,
    "gain": false,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "unbounded",
    "case": 20,
    "seed": 1,
    "gain": true,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 23,
    "seed": 1,
    "gain": false,
    "split": "held",
    "dof": 7
  },
  {
    "baseline": "unbounded",
    "case": 3,
    "seed": 2,
    "gain": false,
    "split": "fitting",
    "dof": 6
  },
  {
    "baseline": "unbounded",
    "case": 10,
    "seed": 2,
    "gain": true,
    "split": "fitting",
    "dof": 7
  }
]
```

All outcomes, models, simulator trajectories, causal error ledgers and success/failure videos are preserved. E120 is not replaced unless selective correction retains its successes and repairs the regressions without introducing new losses. Shadow gate association and full closed-loop intervention answer different questions; changing actions also changes later observations, evidence and phase timing.

Verified 20826 causal gate/action decisions, 6 exact candidate replays, all168 inherited trial contents, unchanged input/model hashes, and 47 trace-verified videos.

Decision: keep E120. This gate produced51/72 versus52/72, with21 versus18 collision episodes. It retained3/6 prior gains and recovered1/4 regressions. Across all pairs versus E120 it gained six successes and lost seven. In the shadow diagnosis, the harmful case22/seed0 has lower conditional response error for E119 and95% gate admission, while gain cases admit corrections on only15–54% of clean active steps. The particular recent-error criterion therefore does not reliably identify beneficial corrections. These episode-level associations do not label each individual correction as causal help or harm. Next isolate correction value with matched-state action/phase interventions on fitting cases, preserving held placements for checks, rather than tuning this threshold against known outcomes.

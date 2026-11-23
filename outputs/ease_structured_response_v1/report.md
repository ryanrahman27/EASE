# E133: structured one-step action response

Measured-history inputs only; same four fitting-family folds and 1108 clean starts. E120 unchanged. No unseen-body or held-placement evaluation.

Predict paired off-minus-on Cartesian displacement as estimated history Jacobian times command difference, followed by a fitted scalar, phase-specific scalar, or 3x3 linear response. Models have no intercept and use no final-success targets. Scalar fits use an additive 1e-18 denominator floor; linear fit uses 1e-6 times mean Gram diagonal ridge. This clarifies the protocol’s general ridge description.

| Model | Progress RMSE (micrometers) | Sign accuracy |
|---|---:|---:|
| analytical | 64.41 | 90.0% |
| scalar | 17.69 | 90.9% |
| phase_scalar | 17.97 | 91.2% |
| linear | 17.88 | 90.7% |
| zero | 50.67 | — |

Previous E132 MLP one-step RMSE 98.61 micrometers, zero reference 50.67. Scalar achieves 17.69 on the same family splits. Input representation and model class both change, so this is a structured-method comparison rather than an architecture-only ablation. Sign is evaluated on 974 rows with absolute truth above 1 micrometer.

Primary scalar gate at 10 micrometers: 101/144 complete-task successes versus E120 97/144,12 rescues and 8 losses,131 switches. These outcomes are reconstructed from existing 8-step off-pulse continuations. One-step prediction improvements do not establish safe 8-step control. All prespecified no-loss grid candidates abstain; E120 remains baseline.

Next controlled experiment: one-step intervention matching the prediction horizon, keeping scalar/10-micrometer rule fixed. Videos are outcome-selected counterfactual replays, not fresh E133 rollouts. All predictions, pre-action features, fitted weights and task reconstructions reproduced; source inputs unchanged.

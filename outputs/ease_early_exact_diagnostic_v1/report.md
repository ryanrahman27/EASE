# E167: privileged exact-motion route diagnostic

**PRIVILEGED DIAGNOSTIC ONLY.** The controller reads the exact current simulator tool-motion Jacobian. It is not E120, a learned estimator result, an unseen-body result, or a mathematical task-success upper bound. No diagnostic labels are supplied to training.

| Exact-map controller | Six-joint successes | Collision episodes | First-contact phases |
|---|---:|---:|---|
| release_only | 37/48 | 11 | {'lift': 1, 'approach': 4, 'descend': 6} |
| conditional_high | 37/48 | 11 | {'descend': 11} |

Conditional higher route versus matched exact-map release-only reference: **0 rescues, 0 regressions**.

Both variants retain the265mm release fix, original measured32-step calibration, frozen measured bias, close probe, DLS gain/damping, phase budgets and strict contact/success scoring. The route rule raises approach330→400mm only if initial measured lateral travel exceeds8mm. No geometry-distance query, collision-aware route planner, future physics or success filtering is used. The original E120 learned correction is absent in both exact-map conditions, so only their within-diagnostic comparison cleanly isolates the route effect.

All48 six-joint fitting placements were included for each condition. Release-only trajectories before lowering exactly reproduce the prior E149 exact-map reference. Inactive conditional routes exactly match the new release-only full states/traces. Calibration and frozen bias are identical, and E120 checkpoint hashes remain unchanged. This diagnostic was registered after fitting E164/E165 outcomes; it is adaptive diagnosis rather than independent validation.

Residual contacts with exact current motion maps indicate that accurate instantaneous kinematics alone does not solve this fixed controller/route problem. This does not prove collision-free paths are impossible or that every learned controller must fail. A clearance-aware feasibility study would be needed to separate path feasibility from controller limitations.

Videos are explicitly labeled privileged executed-state replays, including successes, failures and matched changed cases. Every saved state is checked against the task trace and decoded frame counts.

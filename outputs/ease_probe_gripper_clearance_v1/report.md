# E169: privileged probe and gripper clearance

Fitting-only diagnostic with frozen E120 and the 265mm release fix. No held-body evaluation, retraining, or learned-only transfer claim.

| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E168 |
|---|---:|---:|---:|---|
| E120 + release | 37 | 11 | 0 | — |
| E168 task projection | 39 | 7 | 2 | — |
| probe_only | 41 | 4 | 3 | [27, 63] / [] |
| gripper_only | 39 | 7 | 2 | [] / [] |
| both | 41 | 4 | 3 | [27, 63] / [] |

All 144 intervention episodes are retained. Four corrected exact reference replays passed; four earlier exact replays are archived separately. Initial calibration, E120 weights, frozen bias, phase budgets, jaw commands and strict success/contact scoring are unchanged. The scheduled probe remains 32 steps, but protected variants modify its executed arm commands; per-case measured-motion Gram spectra and changed-command counts are saved in summary.json.

Arm constraints retain E168 bounded-table capsule distances. New palm/pad constraints use the lowest box vertex above the infinite tabletop plane: conservative outside the table footprint. Before any intervention outcome, native box/table queries failed a directional-gradient preflight. The failed implementation and protocol are archived, and the geometric replacement passed independent derivative checks. This is a disclosed preflight correction, not outcome-driven tuning.

The same task-metric linearized QP, 3mm margin and 30mm activation apply. Probe protection acts after the scheduled probe command is formed. Gripper gradients hold jaw positions fixed; jaw actions are unchanged. Ground, self and object contact are not constrained. Linearized feasibility does not guarantee nonlinear clearance. All fallbacks and actual contacts remain scored.

summary.json includes paired rescues/losses against both references, first arm-link contact phases, collision types, phase progress, solver fallbacks and probe excitation. Verification reconstructs every saved post-state distance and actual command, accepted constraints, independent gradients and matching pre-probe prefixes. Success/failure videos replay saved states; 80ms video sampling can miss contacts scored at 2ms. E120 remains the baseline regardless of this diagnostic result.

## Interpretation and execution-error follow-through

Probe-only and combined coverage rescue cases27 and63, retain both E168 gains13 and66 and lose no E168 or E120+release success. Arm-link/table collision episodes fall from3 to0; four gripper/table collision episodes remain, all beginning in descent. Three episodes are collision-free failures rather than successes. Gripper coverage alone changes neither task successes nor collision counts. The simpler probe-only extension therefore accounts for the measured task gain; added gripper coverage has no demonstrated aggregate benefit here.

| Coverage | Episodes with modified probe | Modified probe commands | Median smallest Gram-eigenvalue ratio to E168 (modified episodes) |
|---|---:|---:|---:|
| probe_only | 7 | 220 | 0.08928441770940906 |
| gripper_only | 0 | 0 | None |
| both | 14 | 325 | 0.17374379703313603 |

These are measured arm-motion excitation statistics, not estimator accuracy or safety guarantees. Per-case spectra remain in summary.json. Reduced probe excitation can accompany a successful contact-avoidance intervention; task success and motion information must be judged separately.

In combined-coverage case26, the first pad/table contact is step113 during descent. The left-pad row predicted 1.717mm clearance, but the saved actual clearance was -0.421mm. The QP passed and did not fall back. This directly identifies a command-to-motion prediction discrepancy; it does not identify its complete dynamical cause. A useful next fitting diagnostic is a conservative execution-error allowance or short-horizon dynamics treatment, holding this probe protection fixed.

Audit implementation note: independent gripper-gradient reconstruction uses an eligible step with a saved preceding full state. The initial post-calibration jaw state was not stored separately, so substituting the first post-action jaw state produced a small audit-only mismatch. The checker was corrected to use the next eligible saved state; tolerances and all controller outcomes were unchanged. Every post-action geometry state was still checked.

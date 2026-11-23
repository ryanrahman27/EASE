# E94: measured excitation retained by recent history

This read-only audit covers all96 saved E93 handoffs, including78 admitted-route pairs where history changes. Temporal diagnostics exactly replay all21 changed-outcome pairs and the earliest retained-success control per DOF, in both modes (46 tasks). Per-step statistics are recorded after the measured update. Long shadow and recent32 statistics are computed on the same executed trajectory; the shadow does not affect actions. No controller modification or new policy-success claim is made.

Subtract the1e-6 isotropic ridge before computing measured Gram spectra. Directions above ridge count measured eigenvalues greater than1e-6, tied to the existing regularizer rather than fitted to outcomes. Effective rank is squared total eigenvalue mass divided by sum of squared eigenvalues. These metrics describe excitation; they are not validated success/failure thresholds.

| DOF | History | Handoffs | Median measured energy | Median effective rank | Median directions above ridge | Median largest-direction fraction |
|---|---|---:|---:|---:|---:|---:|
| 6 | long | 30 | 0.00146304 | 1.2053 | 6.0 | 0.909526 |
| 6 | recent | 30 | 0.00072916 | 1.0000 | 1.0 | 1.000000 |
| 7 | long | 48 | 0.00122307 | 1.2509 | 6.0 | 0.892543 |
| 7 | recent | 48 | 0.000596632 | 1.0000 | 1.0 | 1.000000 |

## Findings and decision

Recent32 contains only one direction above the ridge at66/78 admitted-route handoffs (24/30 six-joint and42/48 seven-joint). Median effective rank is approximately1 in both groups; the older weighted history has median effective rank1.205/1.251 and six directions above ridge in both groups. The weak directions are small relative to the dominant one, so longer history is not uniformly well-conditioned either. Recent history retains median48.4%/47.9% of long-history measured energy for six/seven joints.

This pattern is consistent with a straight joint-space descent exciting predominantly one joint combination. Sample count alone does not imply coverage of all joint directions. Windowing removes weaker independent directions alongside older potentially nonlocal data. Since this pattern occurs in both DOFs while E93 effects differ, it cannot alone explain every regression or justify a DOF-specific rule.

Next investigate an estimator-side diagnostic that preserves directional coverage while addressing locality. A controlled option is a small, fixed, separately budgeted local excitation probe at the handoff, with every contact and task cost retained, followed by frozen estimation. Predeclare its bounds and retention criteria before rollout; compare probe/no-probe under matched budgets and include old successes. No probe or replacement controller has been implemented or promoted in E94. Merely changing window length would not address the observed directional deficiency.

Verification: all46 full replays match E93 result dictionaries exactly, with96 saved handoff pairs audited and unchanged inputs/checkpoints. All119 tests pass, including zero-motion/ridge exclusion and known rank-one/isotropic cases. Existing six E93 executed videos are referenced with verified hashes rather than relabeled as new successes. No training or reserved-test evaluation occurred.

## Artifacts

- Figure: `outputs/ease_history_excitation_v1/excitation.png` and `.pdf`.
- Per-step spectra: `profiles/`; all-handoff statistics: `handoffs.json`; aggregate phase summaries: `summary.json`.
- Gallery with reused success/failure videos: `outputs/ease_history_excitation_v1/videos/index.html`.

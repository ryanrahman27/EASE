# EASE-FM: History-conditioned control across kinematically different arms

**Technical report TR-001 · Revision 129 · 15 November 2026**  
**Status:** Development experiments; simulated benchmark success, not a paper submission or broad generalization result.

## Abstract

We study a shared policy for six- and seven-degree-of-freedom serial arms that uses a short measured interaction history without inference-time weight updates. The earlier state-based Transformer has joint tokens, a shared scalar action head, and a causal history encoder. After control diagnostics favor direct action-chunk regression over matched flow, a privileged physical-parameter controller is trained across twelve bodies spanning six axis families. A measured-history encoder then learns its conditioning latents while the controller is frozen. Across three teacher/student seeds, correct32 history gives 28/64, 20/64, and 17/64 successes on eight validation bodies with withheld axis arrangements; empty gives 0/64, 4/64, 4/64 and wrong-arrangement history gives 4/64, 4/64, 5/64. The history advantage repeats, but transfer remains modest and variable. Two seeds pass the complete training-control gate; one fails and is retained as diagnostic. A withheld seven-joint arrangement fails for every student, while true physical parameters only partly help its teachers. These results support limited history-conditioned transfer on this fixed procedural reaching benchmark, not robust general embodiment adaptation. Subsequent frozen-policy interventions show that repeating the initial pose nearly matches intact-history success across all seeds, so these results do not establish action-response dynamics identification. No target-arm training occurs, final-test evaluation is reserved, and all negative experiments are retained in the cumulative record.

A separately versioned benchmark pairs kinematically distinct arms with exactly identical initial policy observations. On this benchmark, three teacher/history seeds all reach 256/256 training successes. Across three calibration seeds per model, correct-history validation ranges are 57–62/96, 52–62/96, and 44–48/96, exceeding empty, twin-history, pose-only, and directly supervised reactive controls in each aggregate comparison. All pose/reactive training gates fail, and transfer is concentrated in the held-out seven-joint subset. This extends the evidence to repeatable interaction-derived conditioning on the paired benchmark, while leaving strong reactive baselines, six-joint transfer, action-response identification, and broader generality unresolved.

The subsequent crossed axis/posture benchmark with a fixed shared recovery union narrows that conclusion. Across three directly action-supervised training seeds, recovery-trained correct history achieves 312/384, 312/384, and 313/384 validation successes; reactive control achieves 311/384, 311/384, and 275/384; empty history achieves 310/384, 320/384, and 300/384. Both correct and empty history pass training at 256/256 in every seed. On unseen axes alone, correct history totals 362/576 successes versus empty history's 368/576. Mean interaction cost across all validation cases is 74.3 steps with per-task calibration, versus 50.1 reactive and 45.1 empty; reusing each calibration across 12 goals reduces correct-history cost to 45.0 steps. These descriptive results show improved stability against one separately trained reactive seed, but do not establish a consistent inference-time calibration benefit. Six-joint transfer remains weak, and final-test evaluation is reserved.

An observation-only analytical identification/controller baseline subsequently solves 288/288 unseen six-joint cases across three calibration seeds and both postures, and 1,152/1,152 total validation cases with online updates. Even freezing the probe-derived model solves 287/288 unseen six-joint cases. A replay with simulator Jacobian access blocked is exact. These results motivate explicit motion modeling.

The redesigned motion Transformer predicts Jacobian columns from measured displacement statistics and uses a shared Cartesian controller. Across three model seeds and three probes, online neural estimation solves 3,452/3,456 validation cases without collisions. Without a separate probe, it solves 1,152/1,152 across three model seeds, including 288/288 unseen six-joint cases. Inverse-guarded replays confirm neural motion estimation, and a reusable single-robot interface reproduces 384/384. This resolves the immediate learned-control failure on the fixed benchmark. However, a simple analytical controller with one fixed probing action also achieves 384/384 with comparable cost. Broader transfer, learned superiority, and novelty remain unresolved.

## 1. Research question and claim criteria

**E113 completed:** Release-only extension on all24 six-joint E112 pairs: original selector 14/24 (10 collisions), release extension 16/24 (8 collisions). Gate PASS; 8 release substitutions. All24 original-selector replays and24 pre-release prefixes exact. Reused development cohort; frozen40ms bank/progress/nonpenetration criteria, no tuning. 8 verified videos. Separate60-second concept/progress reel covers results through E113. See Section4.103.

**E112 completed:** Frozen E111 selector on eight prospectively perturbed bodies from four previously studied novel-axis families: baseline 28/48 (13 collisions), physics 31/48 (10 collisions); 3 gains and 0 losses. 16/16 placements pass prespecified endpoint screening. Transfer gate PASS. No route planner, weight update, budget extension or selector tuning. Exact model/state remain privileged; not history-only adaptation or untouched axis-family evidence. Saved 12 verified success/failure videos. See Section4.102.

**E111 completed:** Frozen broad progress-aware selector with nonnegative predicted geometric clearance, tested in72 paired training-body episodes: baseline 22/36 (14 collisions; 0 contact-free failures), physics 31/36 (3 collisions; 2 contact-free failures). Training/held-placement gate PASS. Same weights and task budgets; no route planner. Reports phase progress,2mm-margin fraction and search compute cost. All36 baseline replays,72 approach prefixes and 13215 selected next states/contact unions verify exactly. Saves 24 verified executed videos. Privileged exact model/state; no unseen-body or final-test evaluation. See Section4.101.

**E110 completed:** Broader copied-physics action audit on 44 training-body states. Warning-state contact-free progress availability at40ms: narrow 1/14, broad 13/14; broad candidates maintaining2mm margin 5/14. The40ms-selected broad action retains contact-free progress at160ms in 12/14. 20 source tasks replay exactly; 116 snapshot branches reconstruct exactly. No new actions executed in policy, no task-success improvement claim, no final test. See Section4.100.

**E109 completed:** Before-first-collision physics selection across descend/lift/transport/lower/retreat, same fixed E108 alternatives and40ms horizon, tested in72 paired training-body tasks: baseline 22/36 (14 collisions; 0 contact-free failures), physics 22/36 (14 collisions; 0 contact-free failures). Training/held-placement gate FAIL. Records phase completion and same-state action progress. Frozen weights and task budgets; no route planner. All36 baselines,72 approach prefixes and 13343 selected next-state/contact predictions verify exactly. Saves 12 verified executed videos. Privileged physics; no unseen-body or final-test evaluation. See Section4.99.

**E108 completed:** Fixed 40ms exact-physics safer-action selection tested in 72 paired training-body tasks: baseline 22/36 (14 collisions), physics 22/36 (14 collisions). Training/held-placement gate FAIL. Frozen weights and unchanged task budgets; no route planner. All36 baselines,72 pre-retreat prefixes and 3266 selected next-state/contact predictions verify exactly. Saves 4 new verified full-task videos. Privileged physics access; no unseen-body or final-test evaluation. See Section 4.98.

**E107 completed:** Unchanged training-body baseline audit: 36 exact task replays and 1633 exact 40ms predicted states/contact unions. Clean-start 40ms counts: static: TP 2, FP 96, FN 0, TN 1493; physics: TP 2, FP 0, FN 0, TN 1589. Reports held-placement split, longer held-target horizons, latency and all errors. Exact physical-model/state access is privileged; this verifies prediction, not collision avoidance or learned-dynamics transfer. No controller change or final test. See Section 4.97.

**E106 completed:** Six exact E105 diagnostic replays save 20580 contact samples. The fresh-body regression contacts the table 3.150s into retreat, after the ramp ends, despite a preceding static filter decision marked feasible at its 2mm clearance cap. The successful baseline reaches a lower tool height, so tool height alone does not explain the failure. No causal mechanism isolated and no controller change. Six original verified videos retained. See Section 4.96.

**E105 completed:** Prospective replication on eight fresh body parameterizations and two new placements per body, using the same four axis orders. 15/16 placements pass endpoint IK screening; no replacement. baseline 27/45 (17 collisions), budget 27/45 (17 collisions), gentle 27/45 (17 collisions). Strict retention gate FAIL. Reports all 48 candidate/seed opportunities separately from the 45 executable pairs, with body-clustered uncertainty. Saves 131 passing tests, nine exact E104 compatibility replays and 9 verified success/failure videos. Privileged planning/geometry remain; no reserved final test. See Section 4.95.

**E104 completed:** All four isolated ramps screened on training bodies (324 tasks). Training-gate passing phases ['retreat'] receive the complete novel-axis development comparison: retreat: baseline 32/48 (16 collisions), budget 32/48 (16 collisions), gentle 35/48 (13 collisions); transfer gate PASS. Every selected phase is reported; no seed or DOF switching. Fixed weights and unchanged privileged planning/geometry; no final test. Saves 131 passing tests and 30 verified videos across training and transfer galleries. See Section 4.94.

**E103 completed:** Fixed training-arm phase-transition screen: baseline 22/36 successes, 14 collisions; budget 23/36 successes, 13 collisions; gentle 22/36 successes, 12 collisions. Gentle ramps the task-command correction over20 steps at lift/transport/lower/retreat onset; budget-only receives the same extra phase allowance. Strict training/held-placement gate FAILS; no transfer evaluation or policy adoption. All108 full tasks,36 exact baseline references and108 prefixes are saved, with131 passing tests and 9 verified videos. No contact feedback or weight changes. Next isolate the four affected phases on training bodies before choosing any further controller change. See Section 4.93.

**E102 completed:** Ten exact diagnostic replays of five placement regressions and matched successful references save 55020 physics-substep contact samples. Failures differ: transport grasp loss, early-lowering loss of bilateral contact, off-target setdown, and post-release pad recontact during retreat. Two seven-joint failures show only 12-20% bilateral contact before table contact and 67-70 mm relative object displacement. All successful references maintain bilateral contact over their pre-table lowering intervals. Corrects E101 prose: acceptable transport XY error does not establish a maintained grasp, and one miss occurs during retreat rather than lowering/release. No controller change or new benchmark result. Saves ten verified videos, figures and129 passing tests. Next test gentle phase transitions on training bodies with explicit contact monitoring and fixed checkpoints. See Section 4.92.

**E101 completed:** parent 29/48 successes, 18 collisions; start 32/48 successes, 16 collisions; replay 33/48 successes, 13 collisions; onpolicy 30/48 successes, 15 collisions. Strict retention gates against original parent and E99 starting student: replay FAIL; onpolicy FAIL. One fixed collection round on original training arms adds 6366 training /3514 held-placement eligible contexts, retaining failures. Both conditions start E99 retained checkpoints and receive 2000 updates; only 16 extra task contexts per batch change source. Saves six checkpoints,144 new evaluations,48 reference copies,129 passing tests and 15 verified new success/failure videos. Privileged-planner development evaluation; no final test. Do not expand this fixed student-history mixture or select a favorable seed. Next audit gripper contact and object-relative motion during lowering/release, including the five non-collision regressions and matched successes, then define any task-level correction on training bodies before repeating the full retention evaluation. Prediction loss and pre-filter action drift alone have not established reliable task retention. See Section 4.91.

**E100 completed:** Read-only same-state command audit covers all three E99 retained regressions and the one available same-task unchanged-success control. Eight full replays remain exact, with unchanged physical/controller state during queries. Case13 seed1 has identical filtered parent/student commands for all four pre-contact transport steps despite raw disagreement; earlier lift-state divergence remains relevant. Other failures and the successful control both show disagreement, so there is no validated failure threshold. No policy change or new benchmark score. Saves per-step vectors/filter decisions, scientific figures, 129 passing tests and eight verified new videos. Next test training-body on-policy histories against a matched replay-only training control. See Section 4.90.

**E99 completed:** Matched task-data training and action retention reach 31/48 and 32/48 successes, respectively, versus parent 29/48; collisions are 17 and 16 versus 18. Retained gains six cases but loses three seven-joint parent successes, and seed2 declines 12/16 to 11/16. Both replacement gates FAIL. Six-joint retained reaches 14/24 without parent-success losses; data_only reaches 15/24. Training adds 36 full trajectories on four original training bodies, with 24/12 disjoint training/validation trajectories and matched batches across conditions. No evaluation-arm data enters training. All 144 evaluations, six checkpoints, 129 passing tests, scientific figures and 13 verified success/failure videos are saved. Evaluation retains privileged planning and frozen parent prefixes. Next audit command divergence in the three retained regressions before choosing a training-only remedy. See Section 4.89.

**E98 completed:** Three fixed-budget Transformer checkpoints trained on16384 local-history contexts from eight training bodies (12288 train/4096 trajectory validation). Post-probe neural substitution improves29/48 to31/48 full-task successes and collisions18 to17, but the retention gate FAILS: six gains/four seven-joint losses, all losses in seed2. Seed results are8->12,9->11,12->8 of16. Six-joint success improves13/24 to15/24 with no losses (15/15 admitted-route pairs); seven-joint stays16/24. Held-trajectory MSE improves29.7-31.3 percent in every seed but does not prevent task regression. No checkpoint is adopted. All48 parent replays and96 handoffs verify;127 tests and eight new video replays pass. Training uses no evaluation-arm histories; evaluation retains privileged planning and frozen parent prefixes. Next add training-arm task trajectories and explicit behavior retention. See Section4.88.

**E97 completed:** With an identical learned/planner/probe prefix, post-probe analytical continuation reaches68/96 successes with27 collisions versus learned57/96 with37; exact continuation reaches78/96 with18. Analytical gains15 and loses4, all losses seven-joint; both E95 six-joint regressions recover and no learned six-joint success is lost. Six-joint success is26/48 versus30/48; seven-joint31/48 versus38/48 (exact48/48). Exact solves all78 route-assisted pairs without collision;18 no-route pairs remain unchanged failures. The analytical diagnostic gate passes, but this is not a new neural-policy result or an unqualified controller replacement. All96 references,36 fallbacks and288 handoffs verify;127 tests and12 new video replays pass. Next improve the neural estimator with training-only measured-history targets while preserving these regressions. See Section4.87.

**E96 completed:** All 288 E95 rollouts replay exactly during a read-only same-state estimator audit. In all 78 route-assisted post-probe states, the analytical fit has lower Jacobian error and lower linearized task residual than the frozen learned estimator. Median relative Jacobian error is 0.419 learned versus 0.070 analytical for six joints, and 0.410 versus 0.083 for seven joints. Both estimates improve after probing, including in the two lost six-joint cases, so instantaneous accuracy does not establish task recovery. No policy change is adopted. All 125 tests pass; scientific figures, snapshots and ten hash-verified reused success/failure videos are saved. Next compare learned, analytical and exact continuation after an identical probe prefix. See Section 4.86.

**E95 completed:** A fixed 32-step local probe restores median above-ridge directional coverage from one to six/seven, but achieves 57/96 successes with 37 collisions, versus matched hold 59/96 with 35 and no probe 58/96 with 34. Two unique six-joint successes are lost against both controls; the adoption gate fails. Neither intervention contacts forbidden geometry during its 32 steps; regressions occur later. All 96 references and 36 no-route fallbacks match, and 123 tests pass. This remains a frozen learned estimator with privileged planning on reused development tasks. Next audit estimation at identical post-intervention states. See Section 4.85.

**E94 completed:** Read-only excitation audit finds only one measured joint-motion direction above the existing ridge scale in66/78 admitted-route recent histories. Longer history retains a median six directions, though motion remains dominated by one. This supports a locality-versus-directional-coverage limitation, not a proven failure rule. All46 diagnostic rollouts replay exactly,96 handoffs are audited and119 tests pass. Spectral figures and a gallery of unchanged, hash-verified E93 success/failure videos are saved. See Section4.84.

**E93 completed:** Fixed32-transition history after planned descent yields58/96 successes versus57/96, but collisions increase33 to34. Six-joint success improves22/48 to28/48; seven-joint falls35/48 to30/48. There are11 gains and10 losses, including one lost successful six-joint hybrid case, so the retention gate fails and window32 is not adopted. Five of ten earlier regressions recover. All96 references and18 fallbacks match;117 tests and six video replay checks pass. See Section4.83.

**E92 completed:** Exact-Jacobian continuation from identical post-descent states recovers all 10 E91 longer-route regressions and preserves all six selected retained-success controls. This is a privileged, outcome-selected diagnostic, not a new learned success rate. Jacobian-error ranges overlap, so no threshold is justified. All 16 learned replays match; 115 tests and four executed video replay checks pass. Next investigate local-history estimation with frozen weights. See Section 4.82.

**E91 completed:** Frozen learned estimators with disclosed privileged route assistance reach 49/96 successes at the original descent budget and 57/96 with the longer budget, versus 43/96 baseline. Route100 has 17 gains/11 losses; route300 has 24 gains/10 losses. Both lose five baseline-successful six-joint cases and fail retention, so neither is promoted. Five of six checkpoints recover difficult case 4 with longer descent. All 96 baselines and 36 fallbacks replay exactly; 114 tests and four video replay checks pass. See Section 4.81.

**E90 completed:** Physical tracking of frozen E89 descent routes improves privileged full-task success from 8/16 to 11/16 under the original descent budget and 13/16 with a 300-step descent allowance. No baseline successes are lost. Difficult six-joint case 4 succeeds with 213 descent steps; cases 5-7 have no clear selected route and remain failures. Learned-policy scores are unchanged. All 16 baselines and six no-route runs match, 112 tests pass, and six executed success/failure video replays are saved. See Section 4.80.

**E89 completed:** A bounded static descent-path search finds sampled-clear routes in 13/16 cases. In the difficult six-joint ZYXYZY family, case 4 has one alternative clear route, while cases 5-7 have reachable endpoints but no clear straight joint-space route among the tested candidates. Case 4 requires 105 nominal command increments versus the 100-step descent budget. This is path-geometry evidence, not recovered task success. All 16 E88 replays match and 110 tests pass. Clear/blocked kinematic videos are saved with explicit non-execution labels. See Section 4.79.

**E88 completed:** Exact-Jacobian substitution improves the analytical reference from 7/16 to 8/16 successes, with collisions falling from seven to six. Only case 2 recovers; all four difficult ZYXYZY six-joint cases still fail. This privileged diagnostic points to remaining shared path/contact-control limits, not estimator error alone. All 16 baselines replay exactly and 109 tests pass. Four labeled success/failure videos reproduce their saved trajectories exactly. See Section 4.78.

**E87 completed:** Four exact joint-axis sequences excluded from training and checked catalogs were evaluated on eight new procedural bodies and 16 fixed placements. Response correction yields 46/96 learned successes versus 43/96 baseline and 44/96 command correction; collisions are 27, 40 and 35 respectively. Six-joint success is 17/48 versus 15/48 for both controls. No baseline successes are lost, but the two response-specific gains share one body and checkpoint. All 112 independent baselines match; 108 tests pass. Low absolute success and 13.22x geometry-query cost remain major limitations. See Section 4.77.

**E86 completed:** Expanded frozen comparison samples12 new parameterizations, accepts21/24 placements on11 bodies, and yields116/126 learned successes for response correction versus114/126 for baseline and command-only correction. Collisions fall11 to9 with no losses. Two response-specific recoveries occur on different seven-joint bodies/families/checkpoints; six-joint remains53/54. All147 baseline checks and108 tests pass. Small effect and unchanged joint-axis arrangements limit the claim. See Section4.76.

**E85 completed:** Prospective new-morphology pilot accepts11/12 fixed placements on six parameter-perturbed bodies (same axis families). Response correction reaches59/66 learned successes versus58/66 for both baseline and matched command-only correction, with seven versus eight collisions and no losses. The single gain is seven-joint; six-joint remains28/30. Both response gates pass, but one favorable paired outcome is preliminary mechanism evidence. All77 independent baselines match exactly;108 tests pass. See Section4.75.

**E84 completed:** Frozen sustained direction correction replicates on 17 separate historical placements:95/102 learned successes versus93/102, seven versus nine collisions, and47/48 six-joint successes versus46/48. Two recoveries, zero casewise losses; analytical remains16/17. All119 baselines replay exactly;107 tests pass. Added geometry-query cost is605,010. Same robot bodies and previously used development placements limit the claim. See Section4.74.

**E83 completed:** Full-task sustained correction yields 71/96 learned successes versus baseline 68/96, with 18 versus 26 collisions. Pulse yields 69/96. Sustained has 3 casewise gains and 0 losses; six-joint success is 37/42. The primary retention gate passes. All 112 baselines reproduce exactly; 107 tests pass. These are historical development tasks, not independent generalization. See Section 4.73.

**E82 completed:** Bounded response-based direction corrections reduce colliding 20-step local branches from 24/27 to 17/27 (five-step pulse) or 15/27 (sustained), with seven/nine paired avoidances and no new contacts. Both variants avoid three of five six-joint contacts; two successful controls remain clear. All 27 full replays and 20-step matched baselines reproduce exactly; 107 tests pass. These are local avoidance results, not recovered pick-and-place tasks. See Section 4.72.

**E81 completed:** First-warning five-step slowing/bias pulses provide no retained avoidance over 20 steps: every strategy and matched baseline has 24/27 colliding local branches, including all five six-joint cases. Bias-only initially avoids four contacts, but that gain disappears by step five. Advance warning exists for 14/26 original collisions; two successful controls remain clear. All 27 full replays and selected first-step checks pass. Reject this pulse mechanism; task success is unchanged. See Section 4.71.

**E80 completed:** Across 26 imminent-contact states, half/quarter/bias-only actions avoid 0/1/4 contacts and none of five six-joint contacts; they cause zero contacts in 110 successful controls. Analytical same-history actions avoid 10/26 but introduce 68/110 contacts in successful controls. Existing filtering already selects bias-only in 13 collision states and 108 controls. All 28 full trajectories and 136 selected-action branches match exactly; 680 one-step comparisons are diagnostic, not task recoveries. See Section 4.70.

**E79 completed:** Geometry attribution separates real near-table motion from prediction error. Of 87 lift/transport warnings in one successful six-joint run, 86 correspond to measured clearance inside 2 mm without penetration. Another successful six-joint run has 23 release warnings for predicted link4/gripper proximity that is absent in sampled actual motion. The missed descent collision is detected by the existing velocity-aware predictors. Three full traces and prediction logs replay exactly. E78 remains a failed gate; actions and success rates are unchanged. See Section 4.69.

**E78 completed:** All-phase response prediction detects 25/26 learned first contacts versus command-only 15/26, with 288 versus 280 one-step false alarms. Six-joint detection is 5/5; successful learned runs have 110 warnings across 17,409 noncollision steps. The fixed advancement rule fails. All 112 trajectories and prior E76 predictions match exactly; 105 tests pass. No action changes or task recoveries. See Section 4.68.

**E77 completed:** Seven of eight response-predictor one-step false alarms precede contact by 40-80 ms on unchanged trajectories; four response-only warnings minus one command-only warning explain the net increase of three. All four response-only warnings share placement 15. Three of four unscored six-joint contacts involve arm links hitting the table during lift/transport. E76 remains a failed gate; no actions or success rates changed. See Section 4.67.

**E76 completed:** Frozen no-velocity response prediction detects 13/14 in-scope learned first contacts, versus command-only 6/14, with 8 versus 5 false alarms across 4,556 noncollision steps. The prespecified prediction gate fails. Only 14/26 first contacts fall in the evaluated descent/lowering phases. All 112 separate historical trajectories replay exactly; 105 tests pass. Task outcomes are unchanged. See Section 4.66.

**E75 completed:** On unchanged learned-controller trajectories, measured response without velocity detects 8/9 first contacts at the fixed 2 mm threshold with zero false alarms in 4,920 noncollision steps, versus command-only 1/9. Velocity models detect 8/9 with one false alarm, failing their stricter prespecified gate. The simpler comparator is an exploratory candidate requiring separate validation, not a recovered-task result. All 119 traces replay exactly; 105 tests pass. See Section 4.65.

**E74 completed:** All ten E73 geometry-filter failures first hit the table with a finger. Nine commanded joint paths remain 2 mm clear under 21-point sampling, yet all ten executed dynamic paths penetrate forbidden geometry. From identical pre-contact states, analytical same-history actions avoid 5/10 immediate contacts, velocity reset 2/10, and half-motion/bias-only 0/10. These are one-step failure-conditioned diagnostics, not recovered tasks. Ten full replays and 50 branches are audited; 104 tests pass. See Section 4.64.

**E73 completed:** On 17 E71 placements, pooled learned success is original 93/102, geometry 93/102 and fixed slowing 87/102. Collision counts are 9, 9 and 13. The frozen geometry filter fails the prespecified replication rule. Retain the original controller; no promotion of the filter. All 357 trials are retained, 119 baseline traces reproduce exactly and 103 tests pass. See Section 4.63.

**E72 completed:** A shared filter with privileged geometry changes learned success from 65/96 to 68/96 and collision cases from 29 to 26. There are 3 recoveries and 0 regressions. The prespecified development gate passes; freeze the filter for fresh-placement validation, retaining all casewise regressions. This tests controller assistance with extra robot/scene information, not improved history-only adaptation. All 224 v2 trials are retained; 112 baseline replays are exact and 102 tests pass. See Section 4.62.

**E71 completed:** Fresh-placement validation yields baseline 93/102 and extended 92/102 learned successes, with 2 recoveries and 3 regressions. Collisions change from 9 to 10. 17/18 placements pass structural screening on the same six bodies; this is not new-embodiment transfer. The replication rule fails; retain the original phase budgets. Do not promote extended timing. All 238 trials and screening rejections are retained; 99 tests pass. See Section 4.61.

**E70 completed:** Learned procedural success is baseline 65/96, extended budget 67/96 and gated transitions 62/96. Collision counts are 29, 26 and 26; gated control aborts 34 trials. Retain these as development candidates for broader validation: extended. Keep the baseline until that validation passes. All 336 trials are retained; 112 baseline and 112 gate-prefix checks pass, with 99 tests passing. See Section 4.60.

**E69 completed:** Exact telemetry replays show collisions in 29/31 learned procedural failures. All nine failures with both collision and sustained carry-contact loss collide first; one other failure loses contact without collision. The remaining failure retains carry contact but misses final placement. Successful trials also show relative object motion, so a 5 mm drift threshold cannot label slip reliably. Prioritize collision clearance and phase control before active regrasp. All 112 baseline traces reproduce exactly; 97 tests pass. See Section 4.59.

**E68 completed:** Bounded procedural grasp settling yields baseline 65/96, fixed waiting 64/96 and contact-ended waiting 65/96 pooled learned successes. Six-joint scores are 34, 35, 35 /42; collisions are 29, 28, 27. The prespecified advancement gate fails. Retain the original sequencer and E35; do not expand this settling rule to native evaluation. All 336 trials are retained, all 64 historical baseline traces reproduce exactly, and 95 tests pass. See Section 4.58.

**E67 completed:** Frozen E66 transfer is mixed. On 64 new procedural bodies, success changes from E35 2871/3072 to E66 2884/3072, with two fewer six-joint successes and one additional collision. On twelve new native placements, both achieve 34/36 tracking and 28/36 manipulation; E66 has three manipulation gains and three losses, including a new PiPER regression. Analytical native control remains 12/12. Audited 7,168 procedural and 168 native trials; 92 tests pass. Retain E35, preserve E66 experimentally, and investigate grasp stability with frozen estimators. See Section 4.57.

**E66 completed:** Matched geometric-versus-analytical-target training yields a promising analytical-target Transformer: 36/36 native tracking (E35 33/36) and 32/36 manipulation (E35 29/36), with all 18/18 six-joint successes retained. Manipulation has four paired recoveries and one regression; the net gain is concentrated in seed 1. Geometric-target scores are 32/36 and 27/36. All six original reaching gates pass 256/256; 92 tests pass. Preserve E35 and freeze E66 candidates for broader validation; no uniform-retention or analytical-baseline superiority claim. See Section 4.56.

**E65 completed:** All 240 full-schedule trials confirm that blanket bias removal regresses tracking: analytical 12 to 8/12, E35 33 to 25/36, E61 uncorrected 32 to 22/36 and corrected 28 to 24/36. Learned models have five paired gains and 27 losses; PiPER retention also regresses in two checkpoints despite two corrected-seed-0 recoveries. All 120 baseline traces replay exactly; 90 EASE-FM tests pass. Keep E35 and measured bias; focus subsequent procedural training on the learned motion estimator. See Section 4.55.

**E64 completed:** Matched-state interventions on all 37 failed E63 phase holds recover 24 with analytical motion estimation, 15 with bias disabled and 20 with both; baseline remains 0. Both PiPER failures recover under each intervention. All 25,500 original replay steps and 37 baseline branches match exactly; 148 branches (26,480 steps) audited and 90 EASE-FM tests pass. This implicates motion estimation but does not establish whole-task improvement. Retain E35. See Section 4.54.

**E63 completed:** All 120 complete common schedules ran for 1,700 steps from shared calibrated states, with geometry contacts disabled. Analytical passes 12/12; E35 33/36; E61 uncorrected 32/36; response-corrected 28/36. PiPER scores are 18/18, 18/18 and 16/18 for the learned groups; most misses are on Panda. Of 37 missed phase holds, 30 never acquire tolerance and seven acquire then lose it. This demonstrates residual unconstrained tracking deficits under a common full schedule, not solely contact/grasp failure. All 86 tests pass; keep E35. See Section 4.53.

**E62 completed:** Across 120 native cases, exact originals and intact reference replays are paired with cube-disabled and all-geometry-contact-disabled replays (480 executions). Of 38 original failures, 17 meet their recorded endpoint tolerances unconstrained and 21 do not; remaining misses are mainly descend/lift. Cube-only removal confounds PiPER with table/gripper contacts. One corrected Panda success misses its unconstrained endpoint. Prefix horizons differ and may be truncated, so these are not full-task or causal failure counts. All 84 tests pass; keep E35. See Section 4.52.

**E61 completed:** Response-corrected local-action training scores 19/36 native successes versus matched uncorrected 22/36 and E35 29/36. Both new conditions have seven forbidden-contact failures. All six original reaching gates pass 256/256, but native retention fails. Corrected trajectory-validation loss is worse in two of three seeds. Audited 1,536 fresh reaching episodes, 72 fresh native trials, paired batches and frozen response fits; 84 tests pass. Keep E35. See Section 4.51.

**E60 completed:** On 2,880 exactly replayed counterfactual branches, held-out response fits improve first-step true-J ranking from 78.2% to 88.6% (six joints) and 76.4% to 88.5% (seven), improving every body. Estimated-J composition remains weak (57.8%/55.7%) and five-step results are mixed. Verified 576 independent response predictions and unchanged weights; all 83 tests pass. This validates a better local response surrogate, not a better deployed controller. Keep E35. See Section 4.50.

**E59 completed:** A shared linear joint-response predictor reduces held-out-body motion RMSE by 79.2% for six joints and 83.8% for seven versus command-minus-bias. It improves all eight omitted bodies and all 32 body/driver strata. Audited 32 exact source replays, 4,800 transitions, 31,200 joint rows and 16 reproduced fits; 83 tests pass. This is response prediction on procedural driver trajectories, not improved control or novel-family validation. Keep E35. See Section 4.49.

**E58 completed:** Frozen procedural motion audit: 32 exact driver replays, 288 states and 2,880 five-step branches. Replacing predicted with realized joint displacement raises geometric-score one-step ranking agreement from 78.2% to 99.4% (six joints) and 76.4% to 99.5% (seven). Actuator/bias mismatch exceeds local geometric nonlinearity, but this post-hoc oracle is not deployable. Local-action candidates win most short procedural comparisons despite failing native retention; no contact occurs here, so grasp/contact causes remain unresolved. All 81 tests pass. Keep E35; see Section 4.48.

**E57 completed:** Training on 12,000 coherent procedural states improves held-out trajectory losses but fails native retention. Prediction-only scores 25/36 and local-action 18/36 versus original E35 29/36; forbidden-contact failures are one and five respectively. All six models retain 256/256 original reaching successes. All 1,536 gate episodes and 72 candidate native trials are fresh. Paired batches, unchanged parents, exact dataset replays and calibration are audited; 79 tests pass. Keep E35. See Section 4.47.

**Latest local-action training result (E56):** All three local-action candidates retain 256/256 original reaching successes. Native PiPER scores are 5/5/5 and Panda 4/1/5 (each /6), totaling 25/36 versus matched prediction-only 26/36 and original 29/36. Five forbidden-contact failures versus prediction-only one; no candidate passes retention. Synthetic local loss improves in all seeds, but no reliable native advantage follows. Six 2,000-update training runs completed; baseline checkpoints/batches/loss traces exactly reproduce E52, allowing audited reuse of 768 gate episodes and 36 native trials. The local-action arm adds 768 new gate episodes and 36 new native trials. All 77 tests pass; keep E35. See Section 4.46.

**Latest procedural pose validation (E55):** With XYZ plus scaled tool-Z control, all 80 driver trajectories and 720 matched first steps replay exactly. Across 7,200 branches, local executed-command agreement is 80.4%/64.0% (one/five steps) for six joints and 78.6%/66.9% for seven, versus operator 48.6%/43.2% and 44.5%/42.3%. The advantage holds across every driver group. No contacts occur; grasping remains unvalidated. No training or model promotion; all weights unchanged and 75 tests pass. See Section 4.45.

**Latest procedural score validation (E54):** All eight training bodies and ten separate driver histories complete 80 exact-replay trajectories. At 720 sampled states, 7,200 continuing moving-target branches give local executed-command agreement of 81.2%/75.7% (one/five steps) for six joints and 81.0%/72.5% for seven, versus operator agreement 41.3%/39.8% and 40.4%/39.0%. The local-score advantage holds across all driver groups, but there are no contacts and this is position-only tracking. No training or model promotion; all weights unchanged and 73 tests pass. See Section 4.44.

**Latest objective/action audit (E53):** All twelve analytical driver trajectories replay exactly. Across 202 fixed-history states and 2,020 five-step counterfactual branches, the E52 operator loss agrees with measured one-step rankings only 45.0% on PiPER and 47.3% on Panda. A local proxy including clipping, joint limits and measured bias agrees 73.4% and 74.0%; five-step agreement falls to 67.9% and 58.6%. No branch has forbidden contact, so this audit does not explain E52's full-episode collisions. Weights remain unchanged; no training or promotion. All 71 tests pass. See Section 4.43.

**Latest matched objective result (E52):** Six fixed-budget fine-tuning runs completed with exactly shared paired synthetic batches. Prediction-only passes 256/256 training tasks in all seeds; control-aware scores 255/256, 256/256 and 256/256. Native PiPER scores are original 6/6/6, prediction-only 5/6/5, control-aware 0/0/1; Panda scores are 5/1/5, 3/3/4, 1/0/1 (each /6). Control-aware improves its synthetic response loss but produces 28 contact failures across 36 native trials. Neither candidate passes retention; keep E35. All 1,536 training-gate episodes and 72 native trials are retained, with matched calibration and unchanged parent hashes. All 68 tests pass. See Section 4.42.

**Latest contact/control result (E50–E51):** All 48 original full-episode contact replays are exact. Actual grasp loss occurs during lift/transport, including initially centered, aligned grasps. Across 120 trials, full angular control with initial yaw gives PiPER neural 5/2/5 and Panda 3/3/2; face alignment gives 5/6/6 and 3/1/3 (each /6), versus original 6/6/6 and 5/1/5. Neither passes retention; keep E35. All 120 initial calibrations match the reference, weights are unchanged, and 66 tests pass. See Section 4.41.

**Latest handoff diagnostic (E49):** All 192 E48 prefixes reproduce exactly through the first 16 descent steps. Panda seed 2's both-update condition has no first-step clipping and a smaller mean command jump than motion only, despite its worse inherited task score. Clipping and jump size alone do not explain that regression; no smoothing fix is established. All 64 tests pass; no new full-episode evaluations or policy changes. See Section 4.40. An initial related-work screen and publication-readiness assessment are now recorded in Section 5.1; the central learned advantage and independent validation remain open.

**Latest update-isolation result (E48):** Separating motion-statistic and actuator-bias updates reveals a closed-loop interaction, not a universally faulty component. Panda neural scores are both 4/1/2, motion only 3/2/5, bias only 4/2/5, and neither 3/1/4 of six. Each isolated choice introduces a PiPER regression in one seed. Both updates improve their respective matched-probe prediction residuals, yet combined task success can worsen. All 192 trials, 96 exact E47 replays and 48 matched groups are audited; 64 tests pass. Keep the original reference and inspect the transition from probing to task control. See Section 4.39.

**Latest local-refresh result (E47):** A matched 32-step pre-grasp probe does not improve the original neural controller consistently. PiPER remains 6/6 in every seed; Panda changes from 5/1/5 to 4/1/2 of six. Waiting scores Panda 3/0/4 and identical probing with estimator updates frozen scores 3/1/4. Analytical probing retains 6/6 on both robots. All 240 trials and failures are retained; 60 baseline replays and matched probe traces are exact, weights are unchanged, and 63 tests pass. Keep the original reference. See Section 4.38.

**Latest retention/fresh-placement result (E45–E46):** Retention-regularized candidates pass the original training-control gate at 256/256 in all three seeds. On six fresh placements per robot, original versus candidate neural scores are PiPER 6/6/6 versus 6/6/5 and Panda 5/1/5 versus 6/3/5; analytical and privileged both solve all twelve tasks. However, the candidate regresses on native reaching and one previously observed PiPER manipulation seed. It is not promoted. All 1,038 rollout records, failed retention checks and unchanged reference checkpoints are retained. See Section 4.37.

**Latest estimator result (E43–E44):** Matched one-step diagnostics show greater neural prediction error than the analytical shadow estimate. Frozen rescaling/smoothing are mixed; synthetic long-history fine tuning improves Panda manipulation from 7/2/4 to 7/5/6 of nine but changes PiPER from 9/7/8 to 8/9/6. Native reaching improves to PiPER 10/11/11 of twelve, retaining Panda 12 each. These are experimental candidates, not a uniform replacement: two PiPER manipulation seeds regress. Completed 264 rollout records and 2,000 training updates per seed with no native robot data in training. See Section 4.36.

**Latest Panda result (E42):** A measured cube-height release trigger improves neural successes from 4/0/2 to 7/2/4 of nine observed tasks; analytical improves 3 to 8 and privileged 3 to 9. All calibration and pre-lowering trajectories match exactly. The intervention fixes later placement failures, not earlier grasp/control failures. Yaw alignment, contact-phase joint holds and stricter settling were also evaluated and retained as mixed/negative results. Completed 225 trials with unchanged weights/mechanics, 61 passing tests and an exact rendered recovered neural trial. See Section 4.35.

**Latest reliability result (E41):** Completed 270 native-gripper development rollouts. Ramped targets and phase gates give PiPER neural scores 9/8/8 of nine expanded tasks versus analytical 8; object-offset-aware placement changes these to 9/7/8 versus analytical 9. On Panda, the same placement intervention improves neural 1/0/1 to 4/0/2 of nine, versus analytical 2 to 3, but reliability remains poor. One PiPER neural trial regresses. Frozen estimator weights, native mechanics and all negative revisions are preserved. No consistent neural superiority or independent confirmation is established. See Section 4.34.

**Latest native-model result (E39–E40):** Imported Menagerie PiPER (six arm joints) and Panda (seven), retaining native mechanics and grippers. Frozen neural seeds solve 9/9/11 of 12 PiPER reaching tasks and 12/12 each on Panda; analytical solves 10 and 12. Compact PiPER pick-and-place succeeds on 1/2/3 of three tasks versus analytical 3. Panda manipulation remains unreliable. These are simulations of existing robot designs, with no new weight training or hardware validation. All layout revisions and failures are retained separately. See Section 4.33.

**Latest manipulation result (E37–E38):** Physical arm-driven pick-and-place now works with the frozen learned motion estimator and a shared scripted task sequencer. On 12 familiar-family tasks, neural seeds solve 7/7/9 versus analytical 8 and privileged 10. On four endpoint-screened unseen-family tasks, neural solves 3/2/3 versus analytical and privileged 3; every model solves the sole retained unseen six-joint case. Eight of twelve proposed unseen placements failed IK screening, so this is a small feasibility pilot, not broad manipulation transfer. Task phases and gripper timing remain scripted. See Section 4.32.

**Latest development result (E35–E36):** Noisy/short-history fine tuning improves higher-noise reaching to 988/983/989 successes out of 1,024, versus the fixed analytical estimator's 979, with lower mean task steps in every seed. This is a small advantage on an already-observed development suite, not independent confirmation or superiority to noise-aware analytical estimation. Short-history improvements are inconsistent across seeds. Shared delay-aware gain recovers performance for both estimator types; analytical remains strongest at 120 ms. All three updated models retain 256/256 training-task success. Separately, a physical gripper fixture passes three scripted Cartesian pick-and-place trials; arm scenes compile, but no learned arm pick-and-place result exists yet. See Sections 4.30–4.31.

**Latest robustness result (E34):** All three frozen neural models remain close to analytical control under modest sensor noise and 40 ms delay, but no tested condition improves final success over the analytical baseline. At 120 ms delay, neural success is 678/723/683 versus analytical 725 of 1,024; retaining only four transitions gives 877/949/861 versus 971. This reveals delay and short-memory weaknesses, particularly for six-joint arms. At higher noise, neural mean interaction cost is lower despite slightly worse final success. See Section 4.29; final test remains reserved.

**Latest broader transfer result (E33):** With unchanged checkpoints and no separate calibration, three neural seeds solve 985, 1,000, and 995 of 1,024 tasks across 16 new axis arrangements and 64 physical bodies. Six-joint success is 477, 489, and 486 of 512. The fixed-probe analytical baseline solves 1,003/1,024, including 491/512 six-joint tasks, with lower mean interaction cost. This supports broader procedural-arm transfer, but no learned advantage. Goals were not filtered by controller success; contact failures are retained. See Section 4.28. Final test remains reserved.

**Latest learned six-joint result (E30–E32):** A redesigned motion Transformer predicts joint-wise Jacobian columns from measured displacement statistics and uses a shared Cartesian controller. Calibrated online inference achieves 3,452/3,456 validation successes across three model seeds and three probes, with no collisions; unseen six-joint cases are 862/864. Starting without a separate calibration probe achieves 384/384 for every model seed, including 48/48 unseen six-joint cases at each posture. Neural weights remain frozen while measured-motion statistics update. Guarded replays exclude an analytical joint-model inverse. This fixes the learned controller's six-joint benchmark performance, but a simple analytical controller with one fixed initial probing action also solves 384/384, so learned superiority and novelty remain unestablished. See Section 4.27.

**Latest six-joint control result (E28–E29):** An analytical, observation-only controller identifies local kinematics and command-tracking bias from the existing 32-step probe, then updates them from task transitions. It solves 48/48 unseen six-joint cases at each posture for all three probe seeds, and 1,152/1,152 validation cases across both joint counts without collisions. Even the frozen identified controller solves 287/288 unseen six-joint cases. Thus the available measurements are sufficient for control on this benchmark; the learned Transformer still has a representation/control generalization failure. This is a working adaptive-control baseline, not a successful neural-policy transfer result or a novelty claim. A complete online replay matches exactly with simulator Jacobian access blocked. See Section 4.26.

**Latest replication and efficiency result (E27):** Eight new models complete seeds 1/2 for all four E26 policies, with seed 0 retained as the observed pilot. Recovery-trained correct-history validation is stable at 312, 312, 313 out of 384, versus reactive 311, 311, 275 and empty history 310, 320, 300. However, on unseen-axis cases alone, correct history scores 121, 120, 121 out of 192, versus empty 126, 128, 114. There is no consistent calibration advantage over the same policy with empty context. Single-use calibration costs more than either control; amortization across 12 goals brings its overall cost close to empty history. New six-joint shifted-posture success remains only 3/48, 0/48, 0/48. See Section 4.25 for costs, gates, and limitations.

**Latest controlled comparison (E26):** The new crossed axis/posture benchmark and fixed shared recovery data largely remove the final-success gap. After recovery, correct history succeeds 312/384, a directly trained reactive policy 311/384, and the history policy with empty context 310/384. On new axes at matched posture, the respective scores are 70/96, 74/96, and 75/96. This single-seed result does not establish equivalence, but it does not support a consistent success-rate advantage from history. The empty-context variant passes training control at 256/256; the separate reactive baseline reaches 246/256 but fails one body's gate. Earlier results remain valid under their protocols, with stronger limits on claims of history necessity.

**Latest diagnosis (E25):** Privileged teachers also fail on most six-joint validation tasks (6/48, 12/48, 2/48), while solving all seven-joint tasks. A post-hoc coverage audit found a reset-posture confound: six-joint validation joints 4/5 start outside training demonstration ranges; seven-joint validation uses the training home vector. Thus these splits do not isolate kinematic transfer. Independent action shuffling reduces history-policy success in 8/9 comparisons, whereas whole-transition shuffling stays near intact performance. Reactive first-action prediction is close to the lower bound imposed by identical twin observations with different expert targets. Reset coverage and recovery training now take priority over simply increasing optimization budgets.

**Latest replication (E24):** All three teachers and full-history students achieve 256/256 training successes. Across three calibration seeds per model, correct-history validation ranges are 57–62/96, 52–62/96, and 44–48/96 for model seeds 0/1/2. Correct history beats empty, twin, pose-only, and directly supervised reactive baselines in every aggregate seed/probe comparison. However, all pose/reactive training gates fail, and six-joint transfer remains weak (2–17/48 versus seven-joint 41–48/48). These results support repeatable history-derived conditioning on this fixed benchmark, not robust general adaptation or superiority to an optimally trained reactive policy. E21's earlier static-cue finding remains valid for its different benchmark.

**Latest diagnostic (E20, completed in WSL on 18 October):** Repeated initial pose achieves 27/64, 21/64, and 17/64 successes on withheld arrangements versus intact history's 28/64, 20/64, and 17/64. Static cues retain nearly all aggregate control performance in these frozen interventions. E19's correct-versus-wrong history advantage therefore does not establish action-response dynamics identification. WSL restores simulation without changing Windows security policy; intact-history outcomes reproduce E19, and all 31 tests pass.

**Latest evidence (E19):** Across teacher/student seeds 0, 1, and 2, correct32 succeeds on 28/64, 20/64, and 17/64 withheld-arrangement tasks, versus empty 0/64, 4/64, 4/64 and wrong32 4/64, 4/64, 5/64. The history advantage repeats, but absolute success is modest and variable. Seeds 0 and 2 pass both training gates; seed 1 fails, so its validation is retained as diagnostic. The student fails `seven_e` in every seed, and true-parameter teachers achieve only 6/16, 0/16, and 4/16 there. Both controller transfer and history-derived conditioning remain relevant limitations. No target-arm training or weight updates occur; final-test evaluation remains reserved.

Can a single learned policy infer useful embodiment behavior from a brief sequence of executed actions and measured responses, then control a previously unseen robot without gradient updates? The immediate scope is Cartesian reaching with genuine six- and seven-joint chains. The more demanding test changes joint-axis sequence while keeping six available actuators. Extending the framework to locomotion is a separate future research problem.

The adaptation claim requires correct history to improve control over both empty history and history from a different body with the same joint count, across multiple held-out bodies and training seeds. Better prediction loss, variable output dimensionality, and success from an analytical controller do not independently satisfy that claim. We distinguish the number of calibration interactions from task interactions and disclose whether context is updated during execution.

## 2. Implemented method

### 2.0 Current motion-model architecture (E30 onward)

The current candidate is `MotionTransformer` plus `MotionController`, trained from scratch; the older action-chunk checkpoints and methods below remain historical baselines. The new 235,011-parameter model has three Transformer layers, width 96, four attention heads, feed-forward width 192, and one shared three-value output head per active joint. Six- and seven-joint arms share weights and use a padding mask. There is no vision/language encoder or robot-geometry input.

Measured joint displacement D and end-effector displacement X produce G = DᵀD + 10⁻⁶I and C = DᵀX. Each joint token contains its row of the padded 7×7 G and its three C values, normalized by the mean active diagonal of G. The network predicts the joint's three-dimensional Jacobian column. It does not invert G during inference. A shared damped 3×3 Cartesian control solve converts the predicted Jacobian, goal error, and measured command-tracking bias to clipped joint commands. This is a neural estimator with an analytical controller, not an end-to-end learned action head.

During task execution, the online variant updates G and C with forgetting 0.98 and updates the tracking bias with weight 0.2, then recomputes the neural prediction. Network weights remain frozen. Cold start initializes G to 10⁻⁶I, C and tracking bias to zero, and obtains identification information from task interactions without a separate probe. The history representation is now explicit sufficient statistics, replacing the older raw-history latent encoder. The reusable interface supports `action(observation)` followed by `update(before, after, executed_command)` and can share one model across separate robot-specific state containers.

### 2.1 Observations, actions, and information boundaries

At each control step the student observes joint positions and velocities, measured end-effector XYZ, target XYZ, and a validity mask for available joints. It outputs bounded joint-position increments, one per actuator, at 25 Hz. Actions are limited to 0.04 radians per step. The interface supplies a consistent base-to-tip channel order; discovering unknown actuator wiring is outside the current problem.

History contains up to 32 transitions. Each joint token has 11 features: position, velocity, normalized executed command, wrapped position change, velocity change, current end-effector XYZ, and measured end-effector displacement. Neither goal nor robot identity, axis sequence, link lengths, masses, or simulator Jacobians enters the history encoder. Privileged kinematics are used by the demonstration expert and feasibility screening, not the student controller. All normalization statistics are fitted on training data.

### 2.2 Variable-joint flow policy

The network uses width 64, four attention heads, and feed-forward width 128. A shared projection embeds the noisy scalar command and current position/velocity for each joint. Task and flow-time embeddings, joint-order encodings, and action-chunk position encodings are added. Two alternating spatial and temporal attention stages predict flow velocity for a 16-step action chunk. A shared scalar output projection handles each valid joint; there are no body-specific action heads. Ten Euler integration steps generate a chunk; only its first command is executed before replanning.

The history encoder applies spatial attention across joints within each transition, then two causal temporal layers per joint, yielding one 64-dimensional latent per joint. A learned empty token represents absent context. Padding is excluded from spatial attention, statistics, and action loss. Training samples history from a different episode of the same training body, preventing future transitions from the supervised action episode from leaking into context.

All weights are trained from scratch. There is currently no vision encoder, language encoder, pretrained foundation model, or frozen perception module. The original mixed baseline has 138,881 parameters; its history-conditioned counterpart has 244,289, so that earlier comparison is not capacity matched.

### 2.3 Auxiliary measured-response experiment (E07)

The new training-only head predicts measured end-effector displacement from current joint positions/velocities, the executed normalized command, and history latents. It uses a shared projection, one spatial Transformer layer, masked mean pooling across joints, and a small output MLP. It receives no task goal, future observation, or privileged physical target. Future measured displacement appears only as a supervised label. The head does not generate control actions.

The loss is `L = L_flow + lambda * mean(((predicted_delta_ee - measured_delta_ee) / scale)^2)`, with per-axis scales equal to training displacement RMS, floored at 0.0001 m. The implementation predicts normalized displacement directly. We prespecify lambda = 0 versus 0.1, training seed 0, 10,000 updates, batch 32, AdamW learning rate 0.0003, and gradient clipping at norm 1. Both variants instantiate the same architecture and have identical initial weights and random streams for data, flow noise, flow time, and context sampling. The lambda-zero head is untrained and is not a meaningful prediction baseline; zero displacement is reported as a diagnostic baseline.

The query transitions for auxiliary supervision come from the same accepted training demonstrations as the flow objective. Independent random-excitation episodes on validation bodies probe displacement prediction under empty, correct, and wrong history. These query transitions never enter the calibration context or training. This probe intentionally differs from expert-action training, so poor performance may also reflect distribution shift.

## 3. Benchmark and evaluation protocol

### 3.1 Mixed-topology dataset

The local-frame axis sequences are six-joint Z/Y/Y/X/Y/X, seven-joint Z/Y/X/Y/X/Y/Z, and held-out six-joint Z/Y/X/Y/Y/X. Lengths, masses, damping, and actuator strength also vary. These are procedural capsule arms, not validated commercial robots.

Training uses three bodies in each of the first two families. Validation contains one new body from each training family and one body with the novel six-joint axis sequence. One further six-joint body remains reserved for final testing. There are 40 shared training goals, eight validation goals, and eight test goals. Expert screening across all bodies establishes a common feasible workspace; this uses test-body information for benchmark construction, though no test rollouts enter student training or model selection.

Of 480 nominal/broad-start training episodes, 470 collision-free successful episodes are accepted, comprising 22,259 transitions. There are also 48 separate training calibration episodes. Broad starts perturb joint positions by up to 0.35 radians. Uniform transition sampling yields 15,079 six-joint and 7,180 seven-joint samples; it is not body balanced. Dataset manifests, XML models, and asset hashes are stored under `outputs/ease_mixed_v1`.

### 3.2 Success and interventions

Each task allows 150 control steps. Success requires end-effector error below 0.03 m for five consecutive steps with no contact during the episode; contacts are checked at each physics substep. Evaluation records success, final error, contacts, and elapsed control steps.

Learned-policy conditions are empty history, eight correct calibration transitions (0.32 s), 32 correct transitions (1.28 s), and 32 transitions from a different training body with the same joint count. Calibration is a separate bounded-excitation episode followed by task reset. Weights and history are frozen throughout each task. Per-case flow noise and task resets are paired across conditions. The analytical controller instead updates a rolling history while acting; its interaction use differs and its success rate is not a matched estimate of improvement over the neural policy.

## 4. Experiment registry and results

Results from different datasets and success contracts are not pooled. Historical fixed-topology results use three training seeds and 108 validation trials; mixed pilots use one training seed and 24 tasks. The latter comprise only one validation body per family, not 24 independent robots.

| ID | Experiment | Principal result | Artifact directory under `outputs/` |
|---|---|---|---|
| E00 | Legacy fixed-seven-joint smoke/pilot | Historical only; older instantaneous-success/contact contract | `ease_fm_baseline_pilot` |
| E01 | Fixed topology, 1,500 updates, seeds 0-2 | Baseline 86/108; correct history 79/108; wrong 80/108 | `ease_fm_v2_comparison` |
| E02 | Fixed topology, 5,000 updates, seeds 0-2 | Baseline 75/108; correct 73/108; wrong 72/108 | `ease_fm_v2_comparison_5000` |
| E03 | Recovery data, 5,000 updates, seeds 0-2 | Baseline 108/108; correct 101/108; wrong 99/108 | `ease_fm_v2_recovery_comparison` |
| E04 | Mixed topology, 2,000 updates, seed 0 | Zero successes for both models and all history conditions | `ease_mixed_v1_pilot` |
| E05 | Mixed topology, 10,000 updates, seed 0 | History-free 1/24; correct32 6/24; wrong32 6/24 | `ease_mixed_v1_pilot_10000` |
| E06 | Analytical online identification | Six 7/8; seven 8/8; novel axes 8/8 | `ease_mixed_v1/history_control.json` |
| E07 | Matched auxiliary response loss, 10,000 updates, seed 0 | Correct32: control 2/24, auxiliary 1/24; no six-joint successes | `ease_mixed_aux_v1` |
| E08 | Exact training-episode audit of E07 checkpoints | Expert replay 470/470; flow-only 74/470; auxiliary 73/470 | `ease_mixed_training_audit_v1` |
| E09 | Single-body four-goal overfit | Flow 10/21, direct chunk 5/7, MLP 1/7; gate not passed | `ease_overfit_v1` |
| E10 | Shared recovery-state intervention | Flow 16/21; direct chunk 7/7; MLP 6/7 | `ease_overfit_retrained_v1` |
| E11 | Frozen flow sampling diagnostics | Euler50 16/21; mean8 17/21; zero noise 6/7 | `ease_overfit_sampler_v1` |
| E12 | Corrected full training-body comparison | Recovery direct 450/470; recovery flow 297/470; direct passes exploratory gate | `ease_full_control_v1` |
| E13 | Post-hoc training-body history intervention | Correct 450/470; empty 427/470; wrong 449/470 | `ease_full_control_v1/history_probe_summary.json` |
| E14 | Frozen unseen validation arms | Correct32: direct 16/16 familiar-family and 3/8 novel axes; empty 4/8 novel axes | `ease_frozen_validation_v1` |
| E15 | Three training axis arrangements per joint count | Recovery direct 870/939; flow 79/939; training gate not met | `ease_diverse_control_v1` |
| E15-oracle | Supply true physical parameters at the same capacity and budget | Privileged direct 816/939; training gate not met | `ease_diverse_oracle_v1` |
| E16 | Increase update budget on the fixed E15 recovery union | At 30k: history 880/939 fails gate; privileged 915/939 passes | `ease_budget_v1` |
| E17 | Distill privileged conditioning into measured-history encoder with frozen controller | Correct 914/939; empty 50/939; wrong-arrangement 79/939 | `ease_distill_v1` |
| E18 | Frozen distilled policy on unseen arms | Correct32 familiar 46/48; novel axes 28/64 versus empty 0/64 and wrong 4/64 | `ease_distill_validation_v1` |
| E19 | Seeds 1/2 replication and privileged unseen-arm diagnostic | Novel correct32: 28/64, 20/64, 17/64; advantage repeats; seed 1 training gate fails | `ease_replicate_v1` |
| E20 | Frozen history-feature interventions and teacher state-fitting audit | Completed in WSL: repeated initial pose 27/64, 21/64, 17/64 nearly matches full history | `ease_history_audit_wsl_v1` |
| E21 | Matched trained initial-pose encoder, three seeds | Novel pose 30/64, 17/64, 17/64 versus history 28/64, 20/64, 17/64 | `ease_pose_baseline_v1` |
| E22 | Initially indistinguishable kinematic twin fixtures | Twelve pairs pass observation equality, distinct responses, and expert feasibility | `ease_ambiguous_fixture_v1` |
| E23 | Paired dataset and gated matched-encoder pilot | Teacher/history training 256/256; validation history 62/96, pose 47/96, twin 29/96 | `ease_paired_v1`, `ease_paired_pilot_v1` |
| E24 | Three-seed replication, probe variation, directly supervised reactive baseline | History beats controls in all 9 aggregate seed/probe cases; reactive training gates all fail | `ease_paired_replicate_v1` |
| E25 | Frozen privileged transfer, history interventions, reactive fitting and reset coverage | Six-joint failure also affects teachers; reset confound found; action shuffle hurts 8/9 | `ease_paired_diagnostics_v1` |
| E26 | Crossed axis/posture dataset and shared recovery comparison | After recovery: history 312/384, reactive 311/384, empty-history 310/384; no consistent success advantage | `ease_paired_factorial_v2`, `ease_factorial_recovery_v2` |
| E27 | Fixed-data three-seed replication and calibration-inclusive efficiency | Correct history 312/312/313, reactive 311/311/275, empty 310/320/300 out of 384; calibration does not consistently improve unseen-axis success over empty | `ease_factorial_replicate_v1` |
| E28 | Observation-only kinematic identification and control | Online 1,152/1,152 validation; unseen six-joint 288/288; frozen identified six-joint 287/288 | `ease_identified_control_v1` |
| E29 | Isolated online updates and privileged-API-blocked replay | Full online replay exact; Jacobian-only 1,117/1,152 and bias-only 1,110/1,152 validation | `ease_identified_ablation_v1` |
| E30 | Motion-Transformer pilot with shared controller | Neural online 383/384, training 256/256; unseen six-joint 96/96 on the first probe | `ease_motion_transformer_v1` |
| E31 | Three training seeds, three probes, empty/twin/random controls | Neural online 3,452/3,456; cold start 1,152/1,152 after deduplication; guarded replay exact | `ease_motion_replicate_v1` |
| E32 | Cold-start attribution and reusable interface | Fixed-probe analytical also 384/384; constant neural model 18/384; standalone neural interface 384/384 | `ease_motion_coldstart_v1`, `ease_motion_coldstart_v2`, `ease_motion_api_check_v1` |
| E33 | Frozen-checkpoint transfer to 16 new axis families, crossed geometry/dynamics/posture | Neural 985/1000/995 out of 1024; analytical 1003; no learned advantage | `ease_broad_development_v1` (generation failure), `ease_broad_development_v2` |
| E34 | Frozen robustness: noise, delay, restricted history, equal budgets | Analytical final success exceeds all neural seeds in every condition; delay and four-transition history expose weaknesses | `ease_robustness_v1` |
| E35 | Noisy/short-history fine tuning and matched delay-aware control | Higher-noise neural 988/983/989 versus analytical 979; mixed short-memory results; analytical stronger under delay | `ease_robust_training_v1`, `ease_robust_training_gate_v1` |
| E36 | Physical pick-and-place mechanics and arm capability audit | Scripted Cartesian fixture 3/3; 64 arm scene assets compile; no arm-policy manipulation result | `ease_pick_place_fixture_v1/v2/v3`, `ease_pick_place_arm_audit_v1` |
| E37 | Arm-driven pose/grasp feasibility and mounting correction | Free-yaw reference initially 8/12; revised matched reference 10/12; unseen 4/12 endpoint-IK feasible | `ease_arm_pick_place_v1/v2`, `ease_arm_pick_place_unseen_v1` |
| E38 | Frozen measured position/direction control plus shared task sequencer | Neural familiar 7/7/9 of 12, unseen 3/2/3 of 4; analytical 8/12 and 3/4 | `ease_pick_place_control_v1/v2`, `ease_pick_place_unseen_control_v1`, `ease_pick_place_report_v1` |
| E39 | Pinned native PiPER/Panda assets, adapters and unfiltered reaching | PiPER neural 9/9/11 of 12 versus analytical 10; Panda all 12/12 | `ease_real_arm_assets_v1`, `ease_real_arm_smoke_v1`, `ease_real_arm_reach_v1` |
| E40 | Native-gripper manipulation development and workspace audit | Compact PiPER neural 1/2/3 of 3 versus analytical 3; Panda unreliable; versions separate | `ease_real_arm_pick_place_v1/v2/v3`, `ease_real_arm_workspace_v1`, `ease_real_arm_report_v1` |
| E41 | Ramped motion, phase gates and measured object-offset placement | Expanded final PiPER neural 9/7/8 of 9 versus analytical 9; Panda 4/0/2 versus analytical 3; one PiPER offset regression | `ease_native_reliability_v1/v2/v3`, `ease_native_reliability_expanded_v1/v2`, `ease_native_reliability_report_v1` |
| E42 | Matched Panda yaw, joint-hold, settling and object-height release interventions | Release gives neural 7/2/4 of 9 versus analytical 8 and privileged 9; earlier failures persist | `ease_panda_grasp_baseline_v1`, `ease_panda_grasp_yaw_v1`, `ease_panda_grasp_hold_v1`, `ease_panda_grasp_settle_v1`, `ease_panda_grasp_supported_v1`, `ease_panda_grasp_report_v1` |
| E43 | One-step measured-motion diagnostics and frozen rescaling/smoothing | Baseline Panda 7/2/4; rescaling 5/3/4; smoothing 8/3/3 of nine | `ease_panda_motion_baseline_v1`, `ease_panda_motion_scale_v1`, `ease_panda_motion_smooth_v1` |
| E44 | Synthetic long-history fine tuning and native retention gates | Panda manipulation 7/5/6; PiPER 8/9/6 of nine; reaching PiPER 10/11/11 and Panda 12 each of twelve | `ease_motion_lowrank_v1`, `ease_motion_lowrank_manipulation_v1`, `ease_motion_lowrank_reach_v1`, `ease_motion_adaptation_report_v1` |
| E45 | Teacher-anchored synthetic fine tuning and original/native retention gates | Original train 256/256 each; observed PiPER manipulation 8/9/9 and Panda 6/3/5 of nine; native reaching regresses | `ease_motion_retention_v1`, `ease_motion_retention_training_gate_v1`, `ease_motion_retention_manipulation_v1`, `ease_motion_retention_reach_v1` |
| E46 | Frozen old/new checkpoints on prespecified fresh placements | PiPER old 6/6/6, new 6/6/5; Panda old 5/1/5, new 6/3/5 of six; candidate not promoted | `ease_retention_fresh_v1`, `ease_retention_report_v1`, `ease_retention_fresh_preview_v1` |
| E47 | Local pre-grasp excitation versus matched waiting and frozen updates | PiPER probe 6/6/6; Panda baseline 5/1/5, wait 3/0/4, probe 4/1/2, frozen probe 3/1/4 of six; no promotion | `ease_local_refresh_v1`, `ease_local_refresh_preview_v1` |
| E48 | Factorial isolation of motion-statistic and actuator-bias updates during refresh | Panda both 4/1/2, motion only 3/2/5, bias only 4/2/5, neither 3/1/4 of six; isolated choices regress PiPER; no promotion | `ease_split_refresh_v1` |
| E49 | Exact early-descent replays separating motion, bias and clipping | 192 exact prefixes; first-step clipping absent in all Panda seed-2 conditions; command jump alone does not explain regression; observational only | `ease_handoff_diagnostic_v1` |
| E50 | Full contact-state telemetry with original control | 48 exact replays; genuine grasp loss, including aligned starts; yaw alone insufficient | `ease_contact_diagnostic_v1` |
| E51 | Full angular control: initial yaw versus face alignment | 120 trials; neither variant retains all robot/seed scores; no promotion | `ease_angular_control_v1` |
| E52 | Matched synthetic control-aware versus prediction-only fine tuning | Complete: 26/36 prediction-only vs 3/36 control-aware native successes; neither retains reference; 1,536 training-gate episodes | `ease_control_training_v1` |
| E53 | Fixed-history objective/action ranking audit | 202 states, 2,020 branches; local executed-command proxy ranks one-step outcomes better, but longer-horizon/contact validity unresolved | `ease_action_objective_v1` |
| E54 | Procedural training-arm ranking with model-generated histories | 80 exact drivers, 720 states, 7,200 branches; local score ranks position-tracking outcomes better across all driver groups | `ease_procedural_action_v1` |
| E55 | Procedural XYZ/tool-Z action-score validation | 80 exact drivers, 720 states, 7,200 branches; local pose-score advantage with weaker five-step agreement; no contacts | `ease_procedural_pose_v1` |
| E56 | Matched clipped, bias-aware local-action training | Complete: local 25/36 vs paired prediction 26/36 native successes; all reaching gates pass, native retention fails | `ease_local_action_training_v1` |
| E57 | Matched coherent procedural-context training | Complete: prediction 25/36, local 18/36; all reaching gates pass, native retention fails | `ease_coherent_pose_data_v1`, `ease_coherent_training_v1` |
| E58 | Frozen predicted-versus-actual procedural motion audit | Complete: actuator-response mismatch identified locally; no native causal or promotion claim | `ease_motion_gap_v2` |
| E59 | Held-out-body joint-response regression | Complete: 79.2%/83.8% lower six/seven-joint RMSE; no policy integration | `ease_joint_response_v1` |
| E60 | Held-out response counterfactual action ranking | Complete: true-J first-step ranking improves on all bodies; estimated-J and five-step results mixed | `ease_response_ranking_v1` |
| E61 | Matched response-corrected local-action training | Complete: corrected 19/36 vs uncorrected 22/36; original reaching passes, native retention fails | `ease_response_corrected_training_v1` |
| E62 | Matched native contact/pose-tracking isolation | Complete: 17/38 failed originals track unconstrained endpoints; 21 miss, with truncated-prefix limits | `ease_native_contact_isolation_v2` |
| E63 | Complete common native pose schedule | Complete: analytical 12/12, E35 33/36, uncorrected 32/36, corrected 28/36 | `ease_native_common_schedule_v2` |
| E64 | Matched-state phase interventions | Complete: analytical motion recovers 24/37, zero bias 15/37, both 20/37 | `ease_native_phase_interventions_v1` |
| E65 | Full-schedule estimator/bias interventions | Complete: removing bias regresses every group; five learned gains, 27 losses | `ease_native_full_interventions_v1` |
| E66 | Geometric versus analytical-target estimator training | Complete: analytical-target 36/36 tracking, 32/36 manipulation; one paired manipulation regression | `ease_estimator_distillation_v1` |
| E67 | Frozen broader embodiment and placement validation | Complete: procedural +13/3072; native tracking/manipulation tie E35, with six-joint regression | `ease_frozen_transfer_v1` |
| E68 | Frozen-estimator procedural grasp settling | Complete: bounded contact and fixed-duration comparison; advancement gate fails | `ease_grasp_settling_v1` |
| E69 | Grasp/tracking telemetry and collision chronology | Complete: 29/31 learned failures include collision; prioritize clearance | `ease_grasp_diagnostics_v1` |
| E70 | Bounded phase budgets and transition retention | Complete: baseline/extended/gated matched comparison | `ease_phase_retention_v1` |
| E71 | Frozen extended timing on fresh procedural placements | Complete: prospective placement retention; all screening outcomes retained | `ease_phase_fresh_v1` |
| E72 | Shared privileged-geometry action filter | Complete: original budgets, paired frozen estimators; extra information disclosed | `ease_geometry_filter_v2` |
| E73 | Frozen geometry transfer and geometry-free slowing | Complete: 357 paired trials on E71 placements | `ease_geometry_transfer_v1` |
| E74 | Exact pre-contact counterfactual audit | Complete: 10 failed trajectories, 50 one-step branches; dynamic path mismatch | `ease_collision_counterfactual_v1` |
| E75 | Causal joint-response prediction and false-alarm controls | Complete: 119 exact trajectories; prediction-only audit | `ease_response_prediction_v1` |
| E76 | Frozen response predictor on separate historical placements | Complete: 112 exact trajectories; prediction-only validation | `ease_response_validation_v1` |
| E77 | Warning timing and six-joint contact coverage | Complete: read-only audit of 96 learned trajectories | `ease_warning_timing_v2` |
| E78 | Frozen response prediction across every task phase | Complete: 112 exact trajectories, all first contacts covered | `ease_response_allphase_v1` |
| E79 | Geometry attribution of successful-run warnings and missed contact | Complete: three exact replays, 111 attributed steps | `ease_warning_geometry_v1` |
| E80 | Bounded actions at collisions and successful near-margin states | Complete: 680 one-step branches, 28 exact full replays | `ease_action_retention_audit_v1` |
| E81 | First-warning pulses with matched multi-step continuations | Complete: 108 branches / 2,160 steps, 27 exact full replays | `ease_early_warning_v1` |
| E82 | Measured-response bounded direction correction | Complete: 81 local branches, 27 exact replays; full-task validation pending | `ease_direction_correction_v1` |
| E83 | Frozen direction correction, full-task retention | Complete: 336 tasks, 112 exact baseline replays | `ease_direction_fulltask_v1` |
| E84 | Frozen direction correction on separate placements | Complete: 238 tasks, 119 exact baselines, retention gate passes | `ease_direction_transfer_v1` |
| E85 | New parameterizations and matched command-response ablation | Complete: 231 tasks, 77 verified baselines, one response-specific recovery | `ease_direction_newbody_v1` |
| E86 | Expanded parameterization replication and response ablation | Complete: 441 tasks, 147 verified baselines, two response-specific recoveries | `ease_direction_expanded_v1` |
| E87 | Unseen axis arrangements and matched response ablation | Complete: 336 tasks, 112 verified baselines; 46/96 learned successes, two response-specific six-joint recoveries | `ease_direction_novelaxis_v1` |
| E88 | Exact-Jacobian diagnostic on unseen arrangements | Complete: 32 tasks, 16 exact baselines, 8/16 privileged successes; four verified demos | `ease_novelaxis_oracle_v1` |
| E89 | Static descent-path feasibility and posture search | Complete: 320 IK attempts, 182 endpoints, clear sampled routes in 13/16 cases; two labeled path videos | `ease_novelaxis_paths_v1` |
| E90 | Physical execution of frozen descent routes | Complete: 48 tasks; privileged success 8/16 baseline, 11/16 route100, 13/16 route300; six verified videos | `ease_novelaxis_path_execution_v1` |
| E91 | Frozen learned estimators with privileged route assistance | Complete: 288 tasks; 43/96 baseline, 49/96 route100, 57/96 route300; both retention gates fail | `ease_learned_path_transfer_v1` |
| E92 | Same-state post-descent Jacobian intervention | Complete: 10/10 selected regressions recover; 6/6 controls retained under privileged continuation; 32 tasks | `ease_learned_handoff_v1` |
| E93 | Fixed recent-history intervention with frozen weights | Complete: 192 tasks; 58/96 vs57/96 but more collisions and one six-joint loss; gate fails | `ease_local_history_v1` |
| E94 | Read-only history excitation audit | Complete: 96 handoffs, 46 exact temporal replays; recent history has one above-ridge direction in66/78 route-assisted handoffs | `ease_history_excitation_v1` |
| E95 | Fixed local probe versus matched hold | Complete: 288 rollouts; probe57/96, hold59/96, no-probe58/96; gate FAIL despite restored directional coverage | `ease_local_probe_v1` |
| E96 | Same-state post-probe estimator audit | Complete: 288 exact replays; analytical estimate beats learned on both diagnostics in78/78 post-probe states; no task intervention | `ease_probe_estimation_v1` |
| E97 | Paired post-probe continuation | Complete: learned57/96, analytical68/96, exact78/96; diagnostic gate PASS; four seven-joint analytical regressions retained | `ease_probe_continuation_v1` |
| E98 | Training-only local-history distillation | Complete: three trained seeds;29/48 to31/48; six-joint gains retained but four seven-joint seed2 losses; gate FAIL | `ease_probe_distillation_v1` |
| E99 | Training-task histories and matched action retention | Complete: 29/48 parent, 31/48 data_only, 32/48 retained; both gates FAIL; retained loses three seven-joint successes | `ease_task_retention_training_v1` |
| E100 | Same-state command divergence audit | Complete: eight exact diagnostic replays; three regressions and one control; no new policy or success claim | `ease_command_divergence_v1` |
| E101 | Student-collected histories versus matched replay | Complete: parent 29/48 successes, 18 collisions; start 32/48 successes, 16 collisions; replay 33/48 successes, 13 collisions; onpolicy 30/48 successes, 15 collisions; replay FAIL; onpolicy FAIL | `ease_onpolicy_history_v1` |
| E102 | Substep grasp/contact audit | Complete: ten exact replays, five distinct placement regressions,55020 samples; no policy change | `ease_grasp_contact_audit_v1` |
| E103 | Gentle phase-transition training screen | Complete: baseline 22/36 successes, 14 collisions; budget 23/36 successes, 13 collisions; gentle 22/36 successes, 12 collisions; gate FAIL, transfer withheld | `ease_gentle_transition_v1` |
| E104 | Single-phase ramps and conditional development transfer | Complete: retreat: baseline 32/48 (16 collisions), budget 32/48 (16 collisions), gentle 35/48 (13 collisions); transfer gate PASS | `ease_single_phase_v1` |
| E105 | Prospective fresh body/placement replication | baseline 27/45 (17 collisions), budget 27/45 (17 collisions), gentle 27/45 (17 collisions); gate False | `ease_fresh_retreat_v1` |
| E106 | Opposite retreat contact diagnosis | Six exact replays; static feasibility fails to guarantee dynamic clearance; no policy change | `ease_retreat_contact_audit_v1` |
| E107 | Training-body short physics prediction | static: TP 2, FP 96, FN 0, TN 1493; physics: TP 2, FP 0, FN 0, TN 1589; no control intervention | `ease_physics_prediction_v1` |
| E108 | Physics-based safer action selection | baseline 22/36 (14 collisions), physics 22/36 (14 collisions); gate False | `ease_physics_action_v1` |
| E109 | Earlier-phase physics selection and progress | baseline 22/36 (14 collisions; 0 contact-free failures), physics 22/36 (14 collisions; 0 contact-free failures); gate False | `ease_early_physics_v1` |
| E110 | Broad goal-directed action availability | Narrow 1/14, broad 13/14 warning states; audit only | `ease_goal_action_audit_v1` |
| E111 | Broad progress-aware physics selection | baseline 22/36 (14 collisions; 0 contact-free failures), physics 31/36 (3 collisions; 2 contact-free failures); gate True | `ease_progress_selector_v1` |
| E112 | Frozen progress selector on prospective unseen bodies | baseline 28/48 (13 collisions), physics 31/48 (10 collisions); 3 gains, 0 losses; gate True | `ease_unseen_progress_v1` |
| E113 | Six-joint release extension | original selector 14/24 (10 collisions), release extension 16/24 (8 collisions); gate True | `ease_release_selector_v1` |

### 4.1 Fixed-topology failure and recovery

At 5,000 updates, baseline seed success counts were 6/36, 35/36, and 34/36 despite similar teacher-state action RMSE (about 0.0063-0.0065 radians). Increasing flow integration from ten to fifty steps did not resolve the weak seed. A diagnostic located drift in the final joint, which does not affect end-effector XYZ in that fixture and receives zero expert commands. Small prediction errors pushed this joint outside its narrow training distribution. A diagnostic command override gave 36/36 for all seeds, but is not part of the learned controller.

Instead, 600 recovery episodes perturbed all joint positions and velocities from training states; 593 succeeded and were added to the shared dataset. With unchanged training budget, unmodified baseline actions then achieved 108/108. Correct history remained below baseline. Thus data coverage addressed a limited control failure without establishing embodiment adaptation.

For E01, correct-minus-wrong success was -0.9 percentage points, exploratory 95% interval [-9.3, +5.6]. For E02 it was +0.9 points [-11.1, +11.1]; for E03 +1.9 points [-7.4, +13.9]. Intervals use 5,000 paired crossed-factor bootstrap draws over seeds, bodies, and goals. Three bodies and three training seeds are too few for strong generalization claims.

### 4.2 Mixed topology before auxiliary supervision

| E05 policy/context | New six-joint body | New seven-joint body | Novel six-joint axes |
|---|---:|---:|---:|
| History-free | 0/8 | 1/8 | 0/8 |
| History model, empty | 2/8 | 3/8 | 0/8 |
| Correct8 | 2/8 | 4/8 | 0/8 |
| Correct32 | 2/8 | 4/8 | 0/8 |
| Wrong32 | 2/8 | 4/8 | 0/8 |

Correct and wrong histories produce identical success counts. This does not establish useful body identification. E06 estimates a local Jacobian by ridge regression on measured joint/end-effector displacements, with measured servo-bias compensation and online updates. Its 23/24 success is an information-interface sanity check and an explicit classical baseline, not a learned EASE-FM result or a novelty claim.

### 4.3 Auxiliary experiment results

The prespecified protocol is [protocol.json](../../outputs/ease_mixed_aux_v1/protocol.json). Both runs completed 10,000 updates. Each has 282,372 parameters and used the same dataset hash, source snapshot, and initial flow loss (1.552526474). Training took 248.3 s for flow-only and 278.7 s for flow plus response supervision. The dataset SHA-256 is `eaee66531407c973185931fa2c2b74630dc236bca99bcbc6a9615a9f05727118`.

| E07 variant/context | Six | Seven | Novel six | Total |
|---|---:|---:|---:|---:|
| flow_only / empty | 0/8 | 1/8 | 0/8 | 1/24 |
| flow_only / correct8 | 0/8 | 1/8 | 0/8 | 1/24 |
| flow_only / correct32 | 0/8 | 2/8 | 0/8 | 2/24 |
| flow_only / wrong32 | 0/8 | 3/8 | 0/8 | 3/24 |
| flow_response / empty | 0/8 | 1/8 | 0/8 | 1/24 |
| flow_response / correct8 | 0/8 | 1/8 | 0/8 | 1/24 |
| flow_response / correct32 | 0/8 | 1/8 | 0/8 | 1/24 |
| flow_response / wrong32 | 0/8 | 1/8 | 0/8 | 1/24 |


**Control outcome:** The auxiliary objective did not improve success in this pilot. Correct32 gives 1/24 versus 2/24 for the matched control; both fail all six-joint and novel-axis goals. The prespecified primary contrast, correct32 minus wrong32 on the familiar-family held-out six-joint arm, is zero for both variants. This is a floor result, not evidence of equivalence. There are no multi-seed uncertainty estimates.

For the trained auxiliary head, independent response-probe RMSE is shown below in millimeters per coordinate, aggregated over eight query episodes of 32 transitions. Zero predicts no motion. These are short random-excitation episodes, not task-rollout accuracy measurements.

| Family | Empty | Correct32 | Wrong32 | Zero-motion baseline |
|---|---:|---:|---:|---:|
| six | 1.234 | 1.293 | 1.261 | 2.481 |
| seven | 1.366 | 1.061 | 1.028 | 2.786 |
| six_novel | 2.792 | 2.635 | 2.800 | 4.881 |


**Representation outcome:** The trained response head beats zero motion on these queries, but correct history is slightly worse than wrong history for the familiar six- and seven-joint families. On the novel-axis body, correct history reduces RMSE from 2.800 to 2.635 mm, a small descriptive difference on one body with no task success. It does not establish useful adaptation. The flow-only head is unsupervised and its errors must not be interpreted as a trained representation probe.

The last-500-update mean flow losses are 0.10288 (control) and 0.10233 (auxiliary), while the auxiliary model's normalized response MSE is 0.02651. Similar flow losses and improved response prediction coexist with poor closed-loop control. Possible explanations include insufficient within-count body diversity, limited excitation coverage, and the flow policy not making effective use of the learned representation; this experiment does not separate them.

Full per-condition task errors and response metrics are in [summary.md](../../outputs/ease_mixed_aux_v1/summary.md) and [summary.json](../../outputs/ease_mixed_aux_v1/summary.json). Per-episode records, separate loss traces, source snapshots, and checkpoint hashes remain in each variant directory. No run was overwritten and no final-test body was evaluated.

### 4.4 Training-episode control audit (E08)

This post-hoc diagnostic tests whether the E07 checkpoints can execute the episodes used for imitation. It uses all 470 accepted training episodes across all six training bodies and 40 training goals, including nominal and broad starts. Resets are asserted equal to the stored initial observations. Replaying recorded expert commands succeeds on 470/470 episodes. Policy rollouts use fresh, separate correct-body 32-transition calibration, frozen context, ten flow integration steps, and paired per-case noise. Weights are unchanged. This is performance on seen bodies/goals/starts with fresh calibration, not action replay or a held-out generalization estimate.

| Policy | Six nominal | Six broad | Seven nominal | Seven broad | Total |
|---|---:|---:|---:|---:|---:|
| flow_only | 21/120 | 11/111 | 24/120 | 18/119 | 74/470 |
| flow_response | 23/120 | 14/111 | 22/120 | 14/119 | 73/470 |

Flow-only succeeds on 15.7% of training episodes; auxiliary supervision succeeds on 15.5%. Even nominal starts are poor: both achieve 45/240. This makes basic imitation and rollout stability the immediate research priority, ahead of increasing embodiment diversity. Low validation success cannot currently be attributed mainly to transfer failure.

Teacher-state diagnostics sample every fifth recorded transition, generate a first action with one fixed noise draw, and compare it to the stored executed expert command. RMSE is pooled over valid joint scalars; all values below are radians per command. Zero-action error provides an action-scale reference, not a control baseline.

| Policy | Family | First-action RMSE | Zero-action RMSE |
|---|---|---:|---:|
| flow_only | six | 0.00728 | 0.03491 |
| flow_only | seven | 0.01128 | 0.02875 |
| flow_response | six | 0.00735 | 0.03491 |
| flow_response | seven | 0.01126 | 0.02875 |

The policies predict expert commands better than zero on recorded states, but imperfect predictions coexist with poor control. This audit does not isolate function approximation, optimization, stochastic action sampling, or compounding state-distribution shift. It does not establish that the history latent identifies the robot.

**Revised next-step order:** (1) deliberately overfit one training arm and a few goals, with closed-loop success as the gate; (2) compare the flow controller with a deterministic one-step action-regression baseline on exactly the same data and inspect per-joint errors/rollout drift; (3) address the observed failure with targeted recovery data or a justified training/inference change, then restore all six training bodies; (4) only after reliable training control, broaden within-count kinematic diversity and resume paired history interventions on multiple unseen bodies and seeds. These are planned experiments, not completed results.

Artifacts: [protocol](../../outputs/ease_mixed_training_audit_v1/protocol.json), [summary](../../outputs/ease_mixed_training_audit_v1/summary.json), and full per-episode/teacher-state records in that directory. No final-test evaluation was performed. Reproduce in a new directory:

```powershell
python -m examples.ease_fm.training_audit --data outputs/ease_mixed_v1 --run outputs/ease_mixed_aux_v1 --output outputs/ease_mixed_training_audit_new --device cuda
```

### 4.5 Tiny-subset overfit gate (E09)

One training six-joint body (seed 20261018), the first four training goals, and all seven accepted nominal/broad-start episodes yield 522 training transitions. History is disabled to remove embodiment ambiguity in this single-body diagnostic. All models train from scratch, seed 0, batch 32, AdamW learning rate 0.0003, for 5,000 updates. Checkpoints at 1,000 and 5,000 updates are evaluated without choosing a winner after the fact. The flow and deterministic chunk-regression models have the same architecture and initialization; direct chunk regression sets noisy-action/time inputs to zero and regresses masked normalized action chunks. A smaller two-hidden-layer MLP directly regresses the next action and is not capacity matched.

| Model | Success at 1,000 updates | Success at 5,000 | Final recorded-state action RMSE (rad) |
|---|---:|---:|---:|
| Flow | 2/21 | 10/21 | 0.00266 |
| Direct chunk regression | 4/7 | 5/7 | 0.00115 |
| Direct one-step MLP | 2/7 | 1/7 | 0.00150 |

Flow counts include three sampling seeds per original episode; deterministic models use one rollout per episode. These are seven seen tasks on one body, not 21 independent tasks or bodies. No model passes the prespecified gate of success on every selected episode and, for flow, every noise repeat. Teacher-state errors are computed over all original transitions, using first actions; action RMSE alone fails to predict rollout reliability. For example, the MLP's recorded-state error improves while its task success drops.

Saved traces compare executed policy commands with a diagnostic expert command on the actual visited state and flag joint positions outside the original training ranges. Failed trajectories show growing command disagreement and state drift; the traces suggest a coverage issue but do not independently establish causality. Expert diagnostic commands never replace learned commands. Raw protocols, checkpoints, losses, and traces are under `outputs/ease_overfit_v1`; source snapshots preserve the initial implementation.

### 4.6 Shared visited-state recovery intervention (E10)

Following E09, collect expert-labeled 16-step recovery chunks from every fifth non-colliding state visited by each of the three frozen final policies, using the same seven training starts and flow noise repeat zero. Branches copy MuJoCo's full integration state plus task, horizon, success, and contact bookkeeping, and verify joint-observation equality. A branch stops at success or the remaining episode horizon. Branches with collisions are logged and excluded. This uses privileged expert labels during training only, as in the original imitation setup.

All variants share the union of 472 additional recovery queries plus the original 522 transitions (994 examples). This is a targeted dataset-aggregation intervention, not new held-out evidence. Retraining uses fresh initialization, the same seed, optimizer, and 5,000-update budget. Evaluation stays on the original seven starts and probes recorded-state action errors on the original 522 transitions, so added labels do not change that denominator. The source checkpoints and recovery-file hash are recorded and checked. Artifacts are in `outputs/ease_overfit_recovery_v1` and `outputs/ease_overfit_retrained_v1`. The completed results are:

| Model | Success at 1,000 updates | Success at 5,000 | Final original-state action RMSE (rad) |
|---|---:|---:|---:|
| Flow | 7/21 | 16/21 | 0.00512 |
| Direct chunk regression | 1/7 | **7/7** | 0.00215 |
| Direct one-step MLP | 1/7 | 6/7 | 0.00262 |

The deterministic chunk model passes the seven-episode gate at 5,000 updates. Flow does not pass its all-noise-repeat gate. All models improve closed-loop success while original-state action RMSE becomes worse than E09. This intervention supports recovery-state coverage as a major contributor on this subset; lower demonstration error alone is an inadequate selection criterion. Data size and distribution both change, so it does not isolate which aspect of the additional supervision matters. It is one training seed, one body, and four seen goals; it is not a robustness, transfer, or adaptation result. The flow and direct-chunk models each have 138,881 parameters; the MLP has 19,718.

![Tiny-subset control and action-error comparison](../../outputs/ease_overfit_retrained_v1/comparison.png)

The [machine-readable comparison](../../outputs/ease_overfit_retrained_v1/comparison.json) preserves exact counts. Reproduce these steps in fresh output directories:

```powershell
python -m examples.ease_fm.overfit --data outputs/ease_mixed_v1 --output outputs/ease_overfit_new --steps 5000 --device cuda
python -m examples.ease_fm.overfit_recovery --data outputs/ease_mixed_v1 --run outputs/ease_overfit_new --output outputs/ease_overfit_recovery_new --device cuda
python -m examples.ease_fm.overfit --data outputs/ease_mixed_v1 --output outputs/ease_overfit_retrained_new --steps 5000 --device cuda --recovery outputs/ease_overfit_recovery_new/recovery.npz
```

### 4.7 Frozen-checkpoint flow sampling diagnostic (E11)

Using E10's final flow checkpoint without any weight updates, the post-hoc sampler probes give:

| Inference mode | Success | Interpretation |
|---|---:|---|
| Original ten Euler steps | 16/21 | E10 reference |
| Fifty Euler steps | 16/21 | Finer integration does not improve the aggregate count |
| Mean of eight sampled bounded first actions | 17/21 | Modest descriptive gain at eightfold sampling compute |
| Zero initial flow noise, ten steps | 6/7 | Deterministic heuristic; one unique trial per episode |

Zero-noise repeats generated by the shared evaluator are identical and are counted only once per episode. These probes are not matched-compute comparisons or new learned methods; none passes the full flow gate. Sampling variability contributes to some failures, but these interventions do not resolve all of them. The deterministic regression baseline remains the successful tiny-set reference. Protocol, checkpoint hash, source, and full traces are preserved in `outputs/ease_overfit_sampler_v1`.

**Current next milestone:** restore the successful direct-chunk baseline across the full training set, with explicitly shared recovery data and appropriate body conditioning, while retaining flow as a comparison. Do not infer cross-body adaptation from the single-body gate. Once training control is reliable across bodies and seeds, resume held-out-body and paired-history evaluations.

### 4.8 Full training-body comparison and synchronized observations (E12)

A new recovery-clone test found a temporal inconsistency in the legacy mixed simulator: MuJoCo integrates joint positions after computing derived Cartesian fields, so reading end-effector position immediately after `mj_step` can pair current joint positions with a Cartesian position from the preceding physics state. A clone followed by `mj_forward` therefore had matching joints but a different end-effector observation (approximately 0.39 mm in the reproducing test). The discrepancy depends on motion; this single magnitude is not a global bound. Earlier expert queries and success checks also used these cached fields. E04-E11 are retained as legacy-timing results and should not be pooled with the corrected experiment. In particular, the previous recovery comparisons do not establish the size or effect of this timing confound.

The new explicit `synchronize=True` path calls `mj_forward` after the control step's final integration, before observation, error/success checks, or subsequent expert queries; it also checks contacts at that refreshed state. The legacy default is retained to preserve earlier run behavior. The dataset was regenerated in `outputs/ease_mixed_sync_v2` using `mixed_data --synchronize`, with the observation timing recorded in its manifest. Existing files were not edited. E12 requires the synchronized flag and refuses the legacy dataset. Recovery cloning now verifies both joint and task observation equality. Tests check source-state preservation, remaining-horizon masks, and deterministic regression's independence from sampling noise.

E12 compares two identically initialized 244,289-parameter history-conditioned policies: flow matching and deterministic masked action-chunk regression. There is no auxiliary response head. All six training bodies, all 40 training goals, and all accepted nominal/broad-start episodes are included. Training uses seed 0, 10,000 updates per model, batch 32, AdamW at 0.0003, uniform transition sampling, and paired batch/context streams within each data condition. History comes from original same-body, different-episode trajectories or calibration; recovery examples exclude their source episode from context. No morphology parameters or body IDs enter policy inputs.

First train both methods on the corrected demonstrations. On every accepted training episode, evaluate each model from its exact initial configuration with separate correct-body 32-transition calibration, frozen history, and paired flow noise. Every tenth non-colliding visited state supplies an expert 16-step recovery branch; colliding branches are logged and excluded. Both retrained methods use the same union of both methods' recovery queries and all original demonstrations, with fresh initialization and the same 10,000-update budget. Recovery-target future states are labels only, never policy/context inputs. Recovery files are hashed and checked before reuse.

The prespecified exploratory control gate is at least 95% success overall and at least 90% for every training body. This is one training seed and is not a confirmatory generalization test. Final-test policy evaluation remains unused. The four-run protocol, source snapshot, training traces, query logs, raw outcomes, and dataset/checkpoint hashes are stored in `outputs/ease_full_control_v1`. The completed corrected dataset contains 470 accepted episodes and 22,239 transitions. Recorded expert commands replay successfully on all 470 episodes. Flow collection produced 5,907 accepted queries out of 5,968 attempted branches; direct collection produced 4,668 out of 4,670. The shared expanded dataset therefore has 32,814 examples. Colliding branches remain in the query logs but are excluded from training.

| Training data | Objective | Six-joint arms | Seven-joint arms | Overall |
|---|---|---:|---:|---:|
| Demonstrations | Flow | 20/231 | 50/239 | 70/470 (14.9%) |
| Demonstrations | Direct regression | 110/231 | 133/239 | 243/470 (51.7%) |
| Shared recovery + demonstrations | Flow | 126/231 | 171/239 | 297/470 (63.2%) |
| Shared recovery + demonstrations | Direct regression | 218/231 | 232/239 | **450/470 (95.7%)** |

The recovery-trained direct policy passes the prespecified exploratory gate. Its per-body counts are 70/75, 78/80, 70/76, 79/80, 77/80, and 76/79 (92.1%-98.8%). Nominal-start success is 237/240 and broad-start success is 213/230. The remaining 20 failures have no recorded collisions; their median final error is 0.0541 m. Flow does not pass the gate. All four models have 244,289 parameters and the same update budget; direct regression retains its advantage on both data conditions. Data volume, state distribution, and fitted normalization change between original and recovery conditions, so this is not an isolated test of any one of those factors.

Training runtimes were 354.0 s (original flow), 203.1 s (original direct), 188.5 s (recovery flow), and 220.1 s (recovery direct). These are observed wall times, not controlled throughput measurements. One training seed and six seen bodies support a development milestone only; repeated training seeds and genuinely unseen-body tests remain necessary.

![Full training-body comparison](../../outputs/ease_full_control_v1/control_comparison.png)

Full tables: [control report](../../outputs/ease_full_control_v1/control_report.md). Machine-readable results and source hashes: [control_report.json](../../outputs/ease_full_control_v1/control_report.json). Reproduction requires the synchronized dataset and fresh directories:

```powershell
python -m examples.ease_fm.mixed_data --output outputs/ease_mixed_sync_new --synchronize
python -m examples.ease_fm.full_control --data outputs/ease_mixed_sync_new --output outputs/ease_full_control_new --steps 10000 --device cuda
python -m examples.ease_fm.full_control_report outputs/ease_full_control_new
```

### 4.9 Post-hoc history intervention on training bodies (E13)

After E12, freeze the recovery-trained direct checkpoint and repeat all 470 training starts with empty context or a 32-transition calibration from the first different training body with the same joint count. Tasks, resets, weights, and calibration seed rules are unchanged. This is explicitly post-hoc and is not a new training experiment, an unseen-body evaluation, or an independently replicated test.

| Context | Success |
|---|---:|
| Correct32 | 450/470 (95.7%) |
| Empty | 427/470 (90.9%) |
| Wrong32, same joint count | 449/470 (95.5%) |

Correct history improves the aggregate count over empty history by 23 episodes, but over wrong-body history by only one episode. Thus the control milestone does not establish body-specific identification: a generic nonempty history may account for much of the gain. The identity intervention is more informative than empty-history improvement alone. Paired discordant outcomes are retained in [history_probe_summary.json](../../outputs/ease_full_control_v1/history_probe_summary.json); no significance or equivalence claim is made.

**Next scientific gate:** evaluate the frozen direct baseline on validation bodies with correct, empty, and same-DoF wrong histories, separately for familiar-family parameter transfer and novel-axis transfer. Preserve the final-test split. If correct and wrong histories remain similar, broaden within-count kinematic diversity and test representation mechanisms; do not present the 95.7% seen-task success as rapid adaptation. Replicate promising findings over multiple training seeds before paper claims.

### 4.10 Frozen unseen-validation-arm evaluation (E14, 17 October 2026)

Evaluate the two final E12 recovery-trained checkpoints without changing weights, normalization, or architecture. The synchronized benchmark supplies one unseen six-joint body and one unseen seven-joint body from familiar axis families, plus one unseen six-joint body with a novel axis arrangement. Each receives eight validation goals, for 24 cases per context condition. The reserved final-test body is not evaluated. Checkpoint hashes are verified before loading and again after evaluation.

History conditions are empty, correct8 (0.32 s), correct32 (1.28 s), and wrong32 from the first training body with the same joint count. Calibration is a separate bounded-excitation episode followed by task reset; context stays frozen during control. Calibration uses goal seed + 500000; flow noise uses goal seed + 600000 and is paired across conditions. The primary diagnostic is correct32 versus wrong32 on the novel-axis body, while the adaptation claim also requires improvement over empty history. All settings were written before evaluation. This is one training seed, one body per family, and one flow-noise realization per case.

| Model | Context | New six-joint | New seven-joint | Novel six-joint axes |
|---|---|---:|---:|---:|
| direct | empty | 6/8 | 8/8 | 4/8 |
| direct | correct8 | 8/8 | 8/8 | 3/8 |
| direct | correct32 | 8/8 | 8/8 | 3/8 |
| direct | wrong32 | 8/8 | 8/8 | 0/8 |
| flow | empty | 4/8 | 5/8 | 2/8 |
| flow | correct8 | 5/8 | 4/8 | 0/8 |
| flow | correct32 | 5/8 | 5/8 | 0/8 |
| flow | wrong32 | 6/8 | 4/8 | 0/8 |

**Findings:** With correct32, direct regression succeeds on 16/16 familiar-family validation goals and 3/8 novel-axis goals; flow achieves 10/16 and 0/8 respectively. Direct regression is therefore retained as the working controller for the next embodiment study, while flow remains a matched comparison. This is an empirical choice for these experiments, not a claim that direct regression is universally superior.

For direct regression, wrong32 also gives 16/16 on familiar-family arms. On the novel-axis arm, correct32 gives 3/8 versus wrong32's 0/8, but empty history gives 4/8. Thus correct-versus-wrong alone would overstate the evidence: correct history does not beat having no history. Neither model establishes the desired rapid adaptation to a new joint-axis arrangement. Correct8 and correct32 have identical direct-policy success counts; this pilot does not establish a benefit from the larger calibration budget. Differences between the old and corrected benchmarks must not be pooled.

**Method direction:** The joint-token architecture, shared scalar action head, and causal history encoder remain unchanged. The direct objective predicts the action chunk in one forward pass; flow predicts a vector field integrated for ten steps. The research question is history-based embodiment adaptation. Flow matching is now a comparison rather than an assumption that the eventual method must retain; the EASE-FM name remains a working project label.

**Next experiment:** broaden axis configurations within each joint count so training cannot identify the kinematic family from joint count alone, then evaluate correct/empty/same-DoF-wrong histories on multiple withheld configurations. Keep the current direct baseline and frozen E14 results. Replicate with multiple model seeds and body instances before a generalization claim; retain the final-test split until the protocol is settled. This broader study has not yet been run.

Full artifacts: [protocol](../../outputs/ease_frozen_validation_v1/protocol.json), [summary](../../outputs/ease_frozen_validation_v1/summary.md), per-case outcomes, source snapshot, and hashes in the same directory. Both models produce 96 case records, with no final-test records; all 24 regression tests pass. Reproduce into a fresh directory:

```powershell
python -m examples.ease_fm.frozen_validation --data outputs/ease_mixed_sync_v2 --run outputs/ease_full_control_v1 --output outputs/ease_frozen_validation_new --device cuda
```

### 4.11 Expanded within-count kinematic diversity (E15, 17 October 2026)

This experiment removes the association between joint count and a single training axis family while holding the policy architecture, optimizer, and per-model update budget fixed. The benchmark contains three training axis arrangements per joint count, two physical bodies per arrangement (12 training bodies), one additional familiar-family validation body per arrangement (six bodies), two unseen arrangements per joint count with two bodies each (eight novel-family validation bodies), and one separate final-test arrangement per joint count. All arrangements are distinct; the final-test pair is never used for policy evaluation. The earlier `six_novel` family is explicitly included in this new training split, so E14 and E15 are different benchmarks and must not be pooled.

The family axis sequences, expressed in local link frames, are:

| Split | Six joints | Seven joints |
|---|---|---|
| Training | ZYYXYX; ZYXYYX; ZYYYXX | ZYXYXYZ; ZYYXYXY; ZYXYYXX |
| Novel validation | ZYXXYY; ZYYXYZ | ZYYYXXY; ZYXXYYX |
| Final test | ZYXYXX | ZYYXXYZ |

Physical parameters vary within each family. All six-joint bodies share home joint vector [0, 0.5, -0.9, 0, 0.6, 0]; all seven-joint bodies share [0, 0.5, 0, -0.9, 0, 0.6, 0]. Reset perturbation seeds are shared for each task. This removes an arrangement-specific reset-joint shortcut, but current measured Cartesian position can still contain embodiment information; the design does not mathematically guarantee that history is necessary.

A preliminary dataset with family-dependent homes is preserved in `outputs/ease_axis_diverse_v1` as a feasibility check and is not used for training. The standardized dataset is `outputs/ease_axis_diverse_v2`. It has 40 training goals, eight validation goals, and eight final-test goals. Expert screening across all 28 bodies accepted these after 63, 13, and 16 candidate attempts respectively; the audit retains all rejections. As in earlier benchmarks, this shared-workspace construction uses final-test geometry for feasibility screening, but no final-test episode enters policy training or model selection. Of 960 nominal/broad-start training episodes, 939 are accepted (41,184 transitions), plus 96 calibration episodes. Accepted counts by training family are 160, 158, 143 for the six-joint families and 160, 158, 160 for the seven-joint families. Uniform transition sampling remains unbalanced; failures stay logged and excluded from imitation.

E15 uses the same 244,289-parameter history-conditioned architecture, seed 0, batch 32, AdamW at 0.0003, and 10,000 updates per model as E12. Both direct regression and flow first train on demonstrations, then each is evaluated on every accepted training start while collecting an expert recovery branch from every tenth non-colliding visited state. Both models retrain from scratch on the same union of recovery queries and original demonstrations. The larger dataset means fewer average passes through the data at this fixed budget; a training failure must not be interpreted as a transfer failure. The gate remains 95% overall training success and at least 90% for every training body. No architecture or body-ID input is added.

The expanded validation runner compares empty history and paired correct/wrong histories at 8, 16, and 32 transitions. Wrong history is from a training body with the same joint count and a different axis arrangement. The runner checks the direct policy's training gate before normal validation; an explicitly labeled diagnostic override is available for a confounded probe. It reports familiar-family parameter transfer separately from unseen-arrangement transfer. Multi-seed replication is deferred until the training-control gate and a meaningful history benefit are established.

**Completed control results:** Expert replay succeeds on all 939 accepted training episodes. Original flow contributes 10,983 accepted recovery queries from 11,150 branches, and original direct contributes 11,455 from 11,498. The common retraining set contains 63,622 examples, including 22,438 recovery queries. Collision branches are excluded, with every attempt retained in the audit.

| Supervision | Objective | Six-joint | Seven-joint | All training episodes |
|---|---|---:|---:|---:|
| Original | Flow | 17/461 | 26/478 | 43/939 (4.6%) |
| Original | Direct | 74/461 | 155/478 | 229/939 (24.4%) |
| Shared recovery | Flow | 28/461 | 51/478 | 79/939 (8.4%) |
| Shared recovery | Direct | 425/461 | 445/478 | 870/939 (92.7%) |

Direct regression improves substantially with recovery data but misses both the 95% overall gate and the requirement that every body exceed or equal 90%. Four of twelve bodies fall below 90%: 69/79 for one `six_novel`, 60/71 and 64/72 for the two `six_c`, and 70/78 for one `seven_b`. None of its 69 failed rollouts involves contact; their median final distance is 0.04375 m. This is a training-control limitation, not a measurement of unseen-arrangement adaptation. Expanded validation and model-seed replication are therefore deferred. The privileged physical-parameter diagnostic below follows the prespecified conditional branch.

![Per-body E15 training results](../../outputs/ease_diverse_control_v1/control_comparison.png)

Artifacts: dataset designs, source snapshots, manifests and feasibility logs in the two dataset directories; training protocol, checkpoints, recovery queries and outcome records in `outputs/ease_diverse_control_v1`. The [full control summary](../../outputs/ease_diverse_control_v1/control_report.md) retains every body's outcome, and `expert_replay.json` records the expert check. All 26 tests pass, including disjoint axis arrangements, identical reset joint states within each count, synchronized observations, recovery-state cloning, and identity-free privileged token construction.

Reproduce into fresh directories:

```powershell
python -m examples.ease_fm.diverse_data --output outputs/ease_axis_diverse_new
python -m examples.ease_fm.full_control --data outputs/ease_axis_diverse_new --output outputs/ease_diverse_control_new --steps 10000 --device cuda --experiment E15
python -m examples.ease_fm.full_control_report outputs/ease_diverse_control_new
```

### 4.12 Privileged physical-parameter reference (E15-oracle)

The failed E15 gate triggers a reference that supplies each joint's true local axis XYZ, link length, link mass, actuator strength, and damping. These seven values and four zero columns occupy the existing 11-feature context interface, repeated over 32 token positions. They replace measured transition history. There are zero physical calibration interactions; these synthetic token positions must not be described as 32 calibration actions. Context contains no explicit body seed, split label, goal, or home configuration. Training-only normalization is recomputed for these physical inputs.

The reference retains the same 244,289-parameter network, direct objective, seed 0 initialization, 63,622-example recovery union, paired example sampling, batch size, learning rate, and 10,000 updates. Context information and its distribution change, including constant length rather than variable-length history, so this is not a clean causal isolation of the history encoder. It is a privileged diagnostic, not a deployable policy or a formal achievable upper bound.

It succeeds on **816/939 (86.9%)**, with 373/461 six-joint and 443/478 seven-joint episodes. By family: `six` 144/160, `six_novel` 108/158, `six_c` 121/143, `seven` 147/160, `seven_b` 148/158, and `seven_c` 148/160. There are no collisions; median final error among the 123 failures is 0.04530 m. Both aggregate and per-body gates fail. Supplying true physical parameters therefore does not resolve training control under this representation and budget. The result does not establish that physical parameters are unhelpful in general, nor identify history extraction as the sole bottleneck.

Artifacts, per-body gates, verified checkpoint/data hashes, and source snapshots are in `outputs/ease_diverse_oracle_v1`; see [summary](../../outputs/ease_diverse_oracle_v1/summary.json). No E15 held-out policy evaluation has been run. Reproduce after a failed full-control gate:

```powershell
python -m examples.ease_fm.diverse_oracle --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --output outputs/ease_diverse_oracle_new --steps 10000 --device cuda
```

### 4.13 Fixed-data training-budget diagnostic (E16)

E16 tests whether the E15 control failures persist with more optimization. Both direct-policy context variants train for 30,000 updates, saving checkpoints at 10,000, 20,000, and 30,000. Architecture, seed 0, batch 32, learning rate 0.0003, uniform sampling, and the 63,622-example shared recovery union remain fixed. No recovery queries are added. All budgets are reported; there is no held-out tuning or early stopping. Every checkpoint is evaluated on the same 939 accepted training starts with the previous overall/per-body gate.

E15 saved model weights but not AdamW state. E16 therefore replays each variant from the original initialization, preserving its optimizer and random streams through 30,000 updates. A tensor-level comparison at 10,000 checks reproduction of the earlier checkpoint. Intermediate saving does not run evaluation or alter model mode during fitting. New checkpoints save optimizer state and sampling/context/PyTorch random states for future continuation. A regression test verifies that serialization leaves a short training trajectory exactly unchanged. All 28 tests pass.

The privileged variant still receives true physical parameters instead of measured history and has a different context distribution; longer training does not remove that comparison limitation. Both 10,000-update replays match every earlier checkpoint tensor exactly and reproduce the earlier rollout totals. The complete results are:

| Context | Updates | Overall success | Lowest body success | Gate |
|---|---:|---:|---:|---|
| Measured history | 10,000 | 870/939 (92.7%) | 60/71 (84.5%) | Fail |
| Measured history | 20,000 | 834/939 (88.8%) | 54/80 (67.5%) | Fail |
| Measured history | 30,000 | 880/939 (93.7%) | 46/71 (64.8%) | Fail |
| True parameters | 10,000 | 816/939 (86.9%) | 51/79 (64.6%) | Fail |
| True parameters | 20,000 | 907/939 (96.6%) | 64/71 (90.1%) | Pass |
| True parameters | 30,000 | 915/939 (97.4%) | 67/71 (94.4%) | Pass |

The history policy's aggregate improvement at 30k does not imply uniformly better control: one `six_c` body falls to 46/71, while the other reaches 62/72. Together that family declines from 124/143 at 10k to 108/143 at 30k, even as seven-joint performance rises from 445/478 to 462/478. There are no collisions at any history-policy budget. The privileged reference has one collision at 20k and none at 10k or 30k; that collision counts as failure. Its 30k failures have median final distance 0.04396 m.

**Interpretation:** The same-sized architecture can pass the diverse training-control gate when supplied true physical parameters and enough optimization. Longer fitting alone does not achieve that result for measured history. The E15 conclusion that training capacity/fitting remained unresolved is narrowed by this new evidence: extracting or using useful embodiment information from history is now a stronger candidate bottleneck. This is not proof that the history encoder alone is defective, because privileged and measured inputs differ in information, distribution, and context-length sampling. No unseen-body inference or adaptation claim follows from this training-only, single-seed result.

![E16 overall and weakest-body learning curves](../../outputs/ease_budget_v1/learning_curve.png)

All six checkpoint and continuation hashes, dataset linkage, training-example counts, and 939-case outcome files were verified. Both exact replay checks are retained in `verification.json`. Raw artifacts and source snapshots are in `outputs/ease_budget_v1`; see [summary](../../outputs/ease_budget_v1/summary.md). Reproduce into a fresh directory:

```powershell
python -m examples.ease_fm.budget_experiment --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --oracle-source outputs/ease_diverse_oracle_v1 --output outputs/ease_budget_new --device cuda
```

### 4.14 Frozen-controller history distillation (E17)

This experiment uses the E16 30k privileged checkpoint (915/939 training successes) as a fixed reference. The complete action-producing network and its joint/task normalization remain bitwise frozen. Only fresh seed-0 history modules train, including their empty-context token. Measured-history normalization is fitted on training contexts. Teacher per-joint 64-dimensional latent targets are computed from true physical parameters for the twelve training bodies; those parameters and body identifiers do not enter the student's history encoder or inference inputs.

The objective is masked raw latent MSE, with no action imitation or task loss. Training uses 10,000 AdamW updates, batch 32, learning rate 0.0003, gradient clipping at 1, and the earlier variable 0..32-transition, same-body/different-episode history sampler. Sampling rows from the fixed 63,622-example union preserves the earlier body/context distribution; action labels in those rows are not used by this objective. This diagnostic includes the reference's previous 30k training and is not a matched total-compute comparison against the end-to-end history policy.

After training, an independent calibration probe uses four fixed excitation seeds per training arm to measure latent MSE under correct32, empty, and wrong-arrangement32 contexts. Control evaluation uses every one of the same 939 training starts, with history collected in a separate calibration episode and frozen throughout the task. Wrong history comes from the first training arm with the same joint count and a different axis family. Both latent error and full control success must be reported: matching a representation numerically does not establish useful control. No validation or final-test policy evaluation is part of E17.

**Comparison clarification:** E15–E17 compare shared models on known training embodiments, changing the conditioning information or training procedure. They do not compare unseen-arm adaptation against a separately trained target-arm specialist. That eventual comparison requires disjoint training/held-out arm configurations; the intended shared policy would receive only a brief measured calibration history on the new arm, without gradient updates. A target-arm-trained specialist would be a separately labeled reference with access to target-arm training data, not evidence that the shared model adapted.

**Results:** The model retains 244,289 total parameters, with only 105,408 history parameters trained. Correct32 reaches **914/939 (97.3%)**, with every body above 90% and no collisions. Empty history gives 50/939 with 332 collisions; wrong-arrangement32 gives 79/939 with 64 collisions. Collisions count as failures. Independent calibration-probe mean latent MSE is 0.003961 for correct32, 0.396664 empty, and 0.512983 wrong-arrangement32. This demonstrates useful history-dependent conditioning on training arms, not yet unseen-embodiment adaptation. The poor empty-context performance is also a limitation: the distilled policy cannot reliably control these arms without its calibration input.

The frozen controller tensors match the teacher exactly after training and evaluation, and the original teacher hash is unchanged. The source snapshot, prespecified protocol, per-case results, and [summary](../../outputs/ease_distill_v1/summary.md) are in `outputs/ease_distill_v1`. All 29 tests pass, including an optimizer-step check that encoder weights change while the entire controller remains exactly unchanged. Reproduce into a fresh directory:

```powershell
python -m examples.ease_fm.history_distill --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --teacher outputs/ease_budget_v1/privileged/30000/policy.pt --output outputs/ease_distill_new --device cuda
```

### 4.15 Frozen unseen-arm validation after distillation (E18)

After E17 passes the training gate, its checkpoint is frozen and evaluated using the previously prepared expanded validation protocol. Six new bodies retain training axis families; eight new bodies use four completely withheld axis arrangements, two bodies per arrangement. Each body receives the eight validation goals. No target-arm policy training or weight updates occur. The student receives only measured joint/task observations and action-response calibration histories, not physical parameters. Calibration lasts 8, 16, or 32 transitions (0.32, 0.64, or 1.28 seconds at 25 Hz), followed by a task reset; history then remains frozen. These times exclude reset overhead and are not hardware measurements.

| Context | Familiar-family new bodies | Withheld axis arrangements |
|---|---:|---:|
| Empty | 4/48 | 0/64 |
| Correct8 | 46/48 | 29/64 |
| Wrong8 | 0/48 | 5/64 |
| Correct16 | 46/48 | 29/64 |
| Wrong16 | 0/48 | 6/64 |
| Correct32 | 46/48 | 28/64 |
| Wrong32 | 1/48 | 4/64 |

For novel families, correct32 outcomes are `six_d` 5/16, `six_e` 15/16, `seven_d` 8/16, and `seven_e` 0/16. Correct8 and correct16 differ only in `six_e`, reaching 16/16 there. Thus correct context beats both controls in aggregate, but performance is uneven and no longer-history benefit is established. Correct histories cause no collisions on either validation group at any tested length. Empty history causes 16 familiar-family and 42 novel-family collisions; wrong history causes zero and four respectively at each length.

**Interpretation:** This is the first expanded-benchmark result here showing a correct-history advantage over both controls on withheld axis arrangements. It supports preliminary history-conditioned transfer, not reliable general adaptation across all kinematics. It is one seed, a small set of procedural bodies and reaching goals, and uses privileged supervision during training. The frozen context is not online adaptation during task execution. We have not compared against a target-arm-trained specialist or run final-test evaluation. Matched multi-seed replication and broader bodies/tasks are needed before a paper-level claim.

![Frozen unseen-arm history interventions](../../outputs/ease_distill_validation_v1/history_comparison.png)

Artifacts: [summary](../../outputs/ease_distill_validation_v1/summary.md), 784 case records, frozen checkpoint hash, protocol and source snapshot in `outputs/ease_distill_validation_v1`. The policy hash is unchanged; no final-test case appears. Shared-workspace feasibility screening still uses final-test geometry as disclosed in E15; this is distinct from policy evaluation. Reproduce with:

```powershell
python -m examples.ease_fm.distill_validation --data outputs/ease_axis_diverse_v2 --run outputs/ease_distill_v1 --output outputs/ease_distill_validation_new --device cuda
python -m examples.ease_fm.distill_report --training outputs/ease_distill_v1 --validation outputs/ease_distill_validation_new
```

### 4.16 Seed replication and privileged transfer diagnostic (E19)

E19 adds prespecified teacher/student seeds 1 and 2 to the previously observed seed-0 pilot. Each teacher receives 30,000 direct-regression updates on the identical E15 recovery union; each fresh history encoder receives 10,000 distillation updates against its frozen teacher. Initialization and training batch/context random streams vary with seed; data, normalization inputs, optimizer settings, architecture, task resets, and evaluation histories remain fixed. This measures optimization variability conditional on the seed-0-collected dataset, not full data-collection variability. Teacher and student seeds vary together, so their variance contributions are not isolated.

Every seed is retained. There is no checkpoint selection, early stopping, or validation-driven adjustment. If a training gate fails, subsequent validation is explicitly labeled diagnostic instead of dropping that seed. The primary comparison is correct32 versus empty and wrong32 on novel families, with every 8/16/32 condition and family also reported. The same frozen teacher is evaluated on each validation body with true physical parameters and zero calibration interactions; it receives no target-arm training. Final-test policy evaluation remains reserved.

**Training results:** Teacher/student counts are 915/939 and 914/939 for seed 0; 902/939 and 891/939 for seed 1; and 916/939 and 911/939 for seed 2. Seeds 0 and 2 pass both gates. Seed 1's teacher exceeds 95% overall but gets only 56/71 and 60/72 on its two `six_c` training bodies; its student gives 55/71 and 59/72 there and also misses the aggregate gate. This seed is retained, not excluded. Student correct32 training collisions are 0, 1, and 4 for seeds 0, 1, and 2; all count as failures.

| Seed | New familiar-family bodies: correct32 | Novel axes: correct32 | Novel axes: empty | Novel axes: wrong32 | Novel axes: true parameters |
|---|---:|---:|---:|---:|---:|
| 0, earlier pilot | 46/48 | 28/64 | 0/64 | 4/64 | 31/64 |
| 1, training gates failed | 38/48 | 20/64 | 4/64 | 4/64 | 15/64 |
| 2 | 39/48 | 17/64 | 4/64 | 5/64 | 19/64 |

Correct8 novel outcomes are 29/64, 18/64, and 18/64; correct16 outcomes are 29/64, 21/64, and 18/64. There is no consistent monotonic gain from more history. Mean per-seed novel success at correct32 is 33.9% (range 26.6–43.8%), versus 4.2% empty and 6.8% wrong32. These are descriptive averages over three optimization seeds sharing the same bodies and goals, not estimates from 192 independent novel tasks. The full summary reports every mode and family, including failed-gate seed 1. Its student correct32 validation has nine novel-family collisions; seeds 0 and 2 have none. Familiar-family correct32 has no collisions in any seed.

**Failure diagnosis:** For `seven_e`, the student gives 0/16 in all three seeds; true-parameter teachers give 6/16, 0/16, and 4/16. Explicit physical inputs therefore do not reliably solve this family either. Teacher/student differences vary by family: for `seven_d`, true parameters give 13/16, 13/16, 5/16 versus student 8/16, 7/16, 2/16; for `six_e`, teachers give 6/16, 2/16, 8/16 versus students 15/16, 6/16, 10/16. Thus the privileged reference is not an upper bound. The student can supply conditioning that works better than the teacher's physical-input encoder on some new bodies. These results cannot isolate whether this reflects latent interpolation, information extraction, or another representation effect.

**Conclusion:** A correct-history advantage on this fixed validation suite repeats across the two additional seeds, but reliable training control and broad kinematic transfer do not. In particular, the failed teacher/student training gate in seed 1 and persistent `seven_e` failure rule out describing the current method as robust general adaptation. No successful seed was selected for reporting and no protocol was adjusted after the new results.

![Three-seed validation and privileged reference](../../outputs/ease_replicate_v1/replication.png)

All teacher/student hashes, all frozen controller tensors, and 2,688 validation records (three times 784 student plus 112 teacher cases) were verified. Privileged evaluations use zero calibration actions and no target-arm training. All 30 tests pass, including seed reproducibility and encoder-only initialization changes. The source of the final report/plot is hashed in `verification.json`.

Protocol, per-seed checkpoints, training gates, frozen validation outcomes, source snapshots, and the [full summary](../../outputs/ease_replicate_v1/summary.md) are in `outputs/ease_replicate_v1`. Reproduce into a fresh directory:

```powershell
python -m examples.ease_fm.replicate --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --output outputs/ease_replicate_new --device cuda
python -m examples.ease_fm.replication_report outputs/ease_replicate_new
```

### 4.17 History-feature and teacher fitting audit (E20, 18 October 2026)

The intended frozen rollout diagnostic compares intact correct32 history, permuted action columns, permuted whole transitions, pose-sequence-only inputs, repeated initial pose, and velocity/action/delta-only inputs. A fixed permutation is shared across seeds/cases. Pose means joint angle plus end-effector XYZ; velocity/action/delta columns are zeroed in static conditions. Conversely, pose columns are zeroed for dynamics-only. Every altered sequence keeps 32 token positions. All conditions are feature interventions on collected histories, not separately trained baselines or demonstrations of reduced calibration cost. Shuffling transitions preserves within-transition action/response pairing; shuffling only actions breaks that pairing while leaving state/response sequences intact.

**Execution limitation:** Windows now rejects the installed MuJoCo native module with `ImportError: DLL load failed while importing _functions: An Application Control policy has blocked this file.` A minimal import reproduced the failure. No security policy was modified or bypassed. Simulator intervention rollouts and the complete regression suite therefore could not run. The new standalone feature-transformation test passes after correcting NumPy advanced-index ordering; syntax checks pass. The 30-test pass reported for E19 is historical, not a claim about the blocked current suite.

The unaffected offline audit uses stored, hash-verified tensors and ordinary PyTorch inference without importing the simulator. Across each of the three frozen teachers, it covers all 41,184 original and 22,438 recovery states. Targets are the recorded first expert action, and predictions use the bounded first command of the teacher's chunk. Errors are averaged over valid joints and states, not balanced over episodes. Near means current end-effector distance below 0.06 m.

| `six_c` data | Distance band | States | Seed 0 MAE (rad) | Seed 1 MAE (rad) | Seed 2 MAE (rad) |
|---|---|---:|---:|---:|---:|
| Original | Near | 1,099 | 0.005384 | 0.006608 | 0.004872 |
| Original | Far | 8,591 | 0.002045 | 0.002404 | 0.001661 |
| Recovery | Near | 128 | 0.009165 | 0.009215 | 0.008428 |
| Recovery | Far | 3,853 | 0.005577 | 0.006254 | 0.005521 |

Seed 1 has the largest `six_c` errors in each band, consistent with its poor training control but not a causal explanation. Its original near-goal MAE is also the largest of the six training families (others range 0.003257–0.004883 rad). Only 128/3,981 `six_c` recovery states are near-goal (3.2%), compared with 1,099/9,690 original states (11.3%). This identifies a candidate coverage issue; it does not prove that more near-goal queries will solve control.

An additional offline representation probe applies the history interventions to all 96 saved training calibration episodes per seed, comparing predicted latents to the corresponding teacher's physical-parameter latents. These episodes were available during training, so this is not independent validation. Mean raw latent MSE is:

| Intervention | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Intact correct32 | 0.003932 | 0.004074 | 0.004959 |
| Shuffle actions | 0.009990 | 0.020031 | 0.007049 |
| Shuffle complete transitions | 0.005060 | 0.004703 | 0.005368 |
| Pose sequence only | 0.006228 | 0.006941 | 0.008090 |
| Repeat initial pose only | 0.005652 | 0.005726 | 0.007197 |
| Velocity/action/deltas only | 0.422260 | 0.301348 | 0.311502 |

**Interpretation:** Static pose cues can approximately recover the training-body latent representation, and removing them strongly degrades it in this frozen probe. Action shuffling also increases error, particularly in seed 1, so the result does not establish complete action independence. The low repeated-pose error makes a claim of necessary online dynamics identification premature. Feature zeroing/shuffling can be out of distribution, and latent error is not task success. No conclusion about altered-history rollout control or unseen-body inference follows until the blocked simulation experiment runs.

Artifacts and [summary](../../outputs/ease_history_audit_offline_v2/summary.md) are in `outputs/ease_history_audit_offline_v2`; the initial teacher-only audit is retained as v1. Verified counts are 63,622 states per teacher and 1,728 latent-probe records. All six teacher/student hashes are unchanged. No validation or final-test rollout was added. Offline reproduction and the pending simulator experiment are:

```powershell
python -m examples.ease_fm.history_audit --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --replication outputs/ease_replicate_v1 --output outputs/ease_history_audit_offline_new --device cuda --offline-only
# Requires an environment where Windows permits MuJoCo to load:
python -m examples.ease_fm.history_audit --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --replication outputs/ease_replicate_v1 --output outputs/ease_history_audit_new --device cuda
```

### 4.18 WSL completion of E20 rollout interventions

At the user's request, E20 ran in the existing Ubuntu WSL2 environment using `/home/faisa/miniconda3/bin/python`. MuJoCo 3.8.0 loads successfully there; the Windows native-DLL restriction was not modified. Runtime is Python 3.13.11, PyTorch 2.11.0+cu130, CUDA 13.0, and NumPy 2.4.4. Saved relative Windows paths are normalized for Linux without modifying earlier artifact manifests. Cross-platform replay comparisons are recorded rather than assuming floating-point identity. Each seed's 112 intact-history validation cases matches E19 on every success and collision outcome; maximum final-distance difference is below 7.5e-15 m.

**Withheld axis arrangements:**

| Frozen input intervention | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Intact correct32 | 28/64 | 20/64 | 17/64 |
| Shuffle actions only | 26/64 | 18/64 | 15/64 |
| Shuffle complete transitions | 27/64 | 20/64 | 16/64 |
| Pose sequence only | 28/64 | 19/64 | 16/64 |
| Repeat initial pose only | 27/64 | 21/64 | 17/64 |
| Remove pose; keep velocity/action/deltas | 5/64 | 2/64 | 3/64 |

**New bodies in familiar axis families:**

| Frozen input intervention | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Intact correct32 | 46/48 | 38/48 | 39/48 |
| Shuffle actions only | 45/48 | 40/48 | 39/48 |
| Shuffle complete transitions | 46/48 | 40/48 | 38/48 |
| Pose sequence only | 45/48 | 39/48 | 37/48 |
| Repeat initial pose only | 46/48 | 40/48 | 38/48 |
| Remove pose; keep velocity/action/deltas | 4/48 | 11/48 | 3/48 |

Repeated initial pose and intact history have the same aggregate novel success count over the three seeds, but not identical per-case outcomes: full-only versus static-only successes are 3 versus 2, 2 versus 3, and 2 versus 2 for seeds 0, 1, and 2. These small differences do not establish statistical equivalence. The two interventions also differ slightly on familiar-family cases. All results retain seed 1's failed-training-gate limitation.

**Revised interpretation:** The current benchmark permits effective conditioning from initial joint angles and end-effector position; the complete calibration action-response sequence adds little to aggregate success in this frozen-policy test. Shuffling actions has a small negative effect on novel success in each seed, so exact action independence is not established. Removing pose features degrades performance strongly, but this is an out-of-distribution intervention and does not prove that a separately trained dynamics-only model would fail. The intended claim of rapid dynamics identification is not supported by these results. A matched static-only trained baseline and a benchmark that reduces initial-pose identification cues are required before pursuing that claim. No task interactions are removed by this analysis: these are altered context inputs, not a newly trained zero-calibration policy.

![Frozen history intervention outcomes](../../outputs/ease_history_audit_wsl_v1/interventions.png)

All 2,016 rollout records and original model hashes were checked. The repeated teacher audit agrees with the Windows offline audit to less than 9.4e-10 rad in MAE. All 31 regression tests pass in WSL. Per-case results, all collisions, runtime metadata, source snapshots, replay checks, [summary](../../outputs/ease_history_audit_wsl_v1/summary.md), and verification are in `outputs/ease_history_audit_wsl_v1`. No final-test evaluation or weight updates occurred. Reproduction from the repository root in WSL:

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.history_audit --data outputs/ease_axis_diverse_v2 --source outputs/ease_diverse_control_v1 --replication outputs/ease_replicate_v1 --output outputs/ease_history_audit_wsl_new --device cuda
```

### 4.19 Matched trained pose-only baseline (E21)

The prespecified three-seed comparison completed in WSL at `outputs/ease_pose_baseline_v1`. Each fresh history encoder sees only the first joint positions and end-effector XYZ of the sampled context, repeated across token positions. Velocities, actions, deltas, and later poses are removed during both training and inference. Architecture, frozen privileged teacher, seed, sampling, 10,000-update budget, optimizer, context lengths, and full-history normalization match E17/E19. The action network still observes current joint state and task position online. The first pose in a sampled training window need not be the first pose of its episode; evaluation uses the calibration's initial pose. Cross-runtime training differences remain a confound: earlier full-history models trained on Windows, whereas E21 trains in WSL. E20 verified cross-runtime inference replay, not identical training trajectories.

All seeds and training-gate failures are retained. The paired correct32 comparison is:

| Seed | Training full / pose (of 939) | Familiar full / pose (of 48) | Novel full / pose (of 64) | Both student gates pass |
|---|---:|---:|---:|---|
| 0 | 914 / 917 | 46 / 45 | 28 / 30 | Yes |
| 1 | 891 / 900 | 38 / 43 | 20 / 17 | No |
| 2 | 911 / 915 | 39 / 36 | 17 / 17 | Yes |

On novel arrangements, paired full-only versus pose-only successes are 2/4, 10/7, and 3/3. Summed counts are 65/192 versus 64/192, but these reuse the same bodies and goals across seeds and must not be treated as 192 independent embodiment samples. Pose-only improves training success in all three seeds; seed 1 still fails the per-body gate and inherits a failed teacher, so its transfer remains diagnostic. There is no consistent full-history benefit across seeds, and no statistical equivalence claim. Static conditioning plus online current-state feedback is sufficient to reproduce much of the observed performance on this benchmark.

`comparison.json` and `comparison.md` retain all seven context conditions for both validation splits and every seed. The report generator verified checkpoint hashes, bitwise frozen controller tensors, and matching identities for all 2,352 paired validation records. Final-test policy evaluation remains untouched. Calibration interaction budgets are unchanged for matching; this study does not measure the savings from skipping the unused probe actions.

### 4.20 Ambiguous initial-pose fixture audit (E22)

`outputs/ease_ambiguous_fixture_v1/audit.json` records twelve procedural arm pairs (24 bodies), with two physical instances per joint count and split. Within each pair, link lengths, masses, actuator strength, damping, home joint positions, and target are identical; only joint-axis arrangements differ. Nonzero home angles occur only at shared axes. At reset with zero random spread, every float32 policy observation is exactly identical within each pair. Both Cartesian Jacobians have rank three. This removes the initial-pose cue within each pair, not across all bodies.

Identical open-loop commands of 0.01 rad, four steps per joint, yield maximum paired end-effector separation of 0.04973–0.11739 m without contact. Every pair has at least one common candidate goal reached by both analytical experts under the normal success criterion. All four candidate goals and both outcomes, including failures, are retained. Train, validation, and test axis arrangements are disjoint within this fixture version; neither legacy reserved-test arrangement is reused. New test geometry is used for this disclosed expert feasibility audit, with no learned-policy training or evaluation.

This is a fixture feasibility result, not an adaptation result or a full training dataset. Current-state feedback during a task may still allow reactive control without an explicit history encoder. The next benchmark must compare matched policies and calibration conditions rather than assuming that indistinguishable initial observations prove history is necessary. Source snapshots and generated XMLs are retained. All 34 regression tests pass in WSL, including observation equality, axis-split isolation, and distinct responses.

### 4.21 Paired dataset and gated pilot (E23)

The separately versioned `outputs/ease_paired_v1` contains eight training arms in four pairs, eight validation arms in four withheld-axis pairs, and eight reserved test geometries. Each body has a unique bookkeeping identifier; IDs never enter policy tensors. There are 32 accepted goals per training pair (256 demonstrations) and 12 per validation pair (96 cases). Goals are offsets of radius 0.06–0.12 m from the pair's shared starting end-effector position, with negative vertical offset. Goals are shared within each pair, not across all bodies. Only candidates reached by both analytical experts are accepted, and all evaluated candidate outcomes are retained. This local reaching pilot is narrower than broad-workspace manipulation.

Every task resets with zero joint spread; paired initial q, dq, end-effector position, target, and joint mask are identical. Each training body has 16 separate 32-step calibration episodes. Commands are piecewise constant for four steps, uniformly sampled in [-0.018, 0.018] rad, and identical within a pair. Training histories use calibration prefixes only, never later task-window poses. This prevents the trained pose-only encoder from exploiting a pose acquired after movement. Validation calibration files are separate from training; policy evaluation uses a fresh deterministic paired probe. Context and weights remain frozen during task execution, while current joint/task observations remain available to the controller.

The independent `paired_audit` verifies 256/256 expert action replays, all 68 stored paired calibrations, unique IDs, disjoint axis splits, and asset/source hashes. This dataset performs no test-goal screening or test rollout. E22's earlier expert test-geometry feasibility audit remains disclosed. All 36 regression tests pass in WSL.

The prespecified single-seed pilot trained a privileged direct-regression Transformer for 10,000 updates, batch 32, AdamW learning rate 0.0003, on original demonstrations without recovery data. It passed the 95%-overall/90%-per-body gate with 256/256 successes, 32/32 on every body. Matched fresh full-history and initial-pose encoders then each trained for 5,000 latent-distillation updates against the same frozen teacher with identical uniform-body/context sampling, seed, architecture, and training-only normalization. Both models run in the same WSL environment.

| Encoder | Context | Training success | Withheld-axis validation success |
|---|---|---:|---:|
| Full history | Correct | 256/256 | 62/96 |
| Full history | Empty | 29/256 | 13/96 |
| Full history | Twin | 38/256 | 29/96 |
| Initial pose | Correct | 214/256 | 47/96 |
| Initial pose | Empty | 30/256 | 13/96 |
| Initial pose | Twin | 214/256 | 47/96 |

Full-history correct-context training passes every body gate. Pose-only fails, so its validation remains diagnostic. Correct-history transfer is 14/48 on six-joint bodies and 48/48 on seven-joint bodies; pose-only is 8/48 and 39/48, respectively. Paired validation wins are 16 history-only versus 1 pose-only, and 34 correct-only versus 1 twin-only. Pose-only correct/twin outcomes match exactly on all training and validation cases, including final error and steps. The report generator verified both frozen controllers bitwise, checkpoint hashes, and all 2,112 student rollout records. No final-test policy evaluation occurred.

Step-10 validation error is 0.06279 m for correct history (88 available cases), 0.09072 m empty (91), 0.08369 m twin (90), and 0.07937 m pose-only (92). Episodes ending earlier have null values; unequal retained subsets prevent interpreting these means as an unbiased paired early-error comparison. Raw records retain the exclusions.

**Interpretation and limits:** The pilot demonstrates a benefit from interaction-derived conditioning beyond identical initial pose inputs on this particular dataset. It does not distinguish action-conditioned identification from recognizing the post-probe pose trajectory; action-shuffle and pose-trajectory controls remain necessary. Latent distillation can impose different targets on identical pose-only inputs, so the pose baseline is constrained by an inherently ambiguous objective. An end-to-end reactive baseline is needed before claiming an advantage over the best history-free controller. Only one teacher/student seed, two training arrangements per joint count, local reaching goals, and one new probe seed per pair are tested. Replication and broader calibration seeds must precede a general adaptation claim.

### 4.22 Replication and directly trained reactive baseline (E24)

The prespecified run `outputs/ease_paired_replicate_v1` reuses immutable E23 seed 0 and adds teacher/student seeds 1 and 2 with the same 10,000/5,000 update budgets and fixed dataset. Every failed teacher or student gate is retained; a failed teacher prevents student fitting for that seed. Validation adds probe seed bases 940000 and 950000 to the existing 930000, with the pair ID added to each base. These reuse the same tasks and bodies and are not independent embodiment samples. No final-test evaluation is performed.

For each seed, a fresh reactive direct-regression Transformer trains on the same demonstration action chunks for 15,000 updates, batch 32, AdamW learning rate 0.0003. This equals teacher-plus-one-encoder update count, not FLOPs. It receives only current joint/task observations and a learned constant context, with no calibration, body metadata, or initial-pose conditioning. All action-producing weights train directly from action supervision. The allocated network has the same 244,289 parameters, but effective capacity differs: history input/spatial modules have no effect at context length zero, while temporal/output modules transform the empty token into a learned constant. A separate clarification preserves the original protocol wording and corrects this distinction without changing training.

The reactive baseline addresses E23's contradictory pose-only latent-target limitation. Completed training results are:

| Model seed | Privileged teacher | Full-history student | Pose-only student | Reactive policy |
|---|---:|---:|---:|---:|
| 0 | 256/256 | 256/256 | 214/256 | 96/256 |
| 1 | 256/256 | 256/256 | 107/256 | 83/256 |
| 2 | 256/256 | 256/256 | 209/256 | 131/256 |

All teachers and full-history students pass every body gate. Every pose-only and reactive policy fails; their validation comparisons remain diagnostic.

| Model seed | Probe seed base | Correct history | Empty | Twin history | Pose only | Reactive |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 930000 | 62/96 | 13/96 | 29/96 | 47/96 | 20/96 |
| 0 | 940000 | 57/96 | 13/96 | 35/96 | 47/96 | 20/96 |
| 0 | 950000 | 61/96 | 13/96 | 38/96 | 47/96 | 20/96 |
| 1 | 930000 | 52/96 | 26/96 | 15/96 | 16/96 | 15/96 |
| 1 | 940000 | 61/96 | 26/96 | 19/96 | 16/96 | 15/96 |
| 1 | 950000 | 62/96 | 26/96 | 22/96 | 16/96 | 15/96 |
| 2 | 930000 | 44/96 | 6/96 | 10/96 | 34/96 | 32/96 |
| 2 | 940000 | 45/96 | 6/96 | 10/96 | 34/96 | 32/96 |
| 2 | 950000 | 48/96 | 6/96 | 12/96 | 34/96 | 32/96 |

Reactive values repeat for display; reactive inference performs no probe. Correct history beats every listed control in all nine aggregate model/probe combinations. This does not imply superiority on every body or joint-count subset. Six-joint correct-history success is 14/48, 9/48, 13/48 for seed 0; 10/48, 17/48, 14/48 for seed 1; and 2/48, 4/48, 3/48 for seed 2. Corresponding seven-joint counts are 48/48, 48/48, 48/48; 42/48, 44/48, 48/48; and 42/48, 41/48, 45/48. The seven-joint subset accounts for most aggregate success. Only two held-out axis arrangements per joint count are represented, and probe repetitions are dependent samples on the same tasks/bodies.

`paired_replication_report` verified all 12 checkpoint hashes, the six frozen student controllers, 5,472 validation records, paired case identities, and exact pose-only correct/twin outcomes. The split-by-joint-count plot was rendered and visually inspected; its bars denote probe ranges, not confidence intervals. All 37 regression tests pass, including reactive input independence and zero gradients through excluded history input/spatial modules. No final-test policy evaluation occurred.

**Interpretation:** History-conditioned control repeats across model and probe seeds, beyond initial-pose cues. The reactive action-supervision comparison removes one objective confound but leaves basic baseline fitting unresolved: poor imitation/control on training tasks is not proof that reactive control is impossible. Shared recovery-data comparisons and privileged validation diagnostics are appropriate next tests before attributing the six-joint failure to the encoder. Action-shuffle and pose-trajectory controls are still needed to isolate action-response correspondence from recognizing a post-probe trajectory.

### 4.23 Frozen diagnostics and reset-coverage confound (E25)

All checkpoints remain frozen. The prespecified diagnostic evaluates the three privileged teachers on validation tasks and five history conditions across all three model seeds and three probe seeds. No final-test policy evaluation occurs. All 4,608 new rollout records are retained; the nine intact conditions exactly reproduce prior success, collision, final error, and step count. Nine checkpoint hashes are unchanged, and all 37 regression tests pass.

| Teacher seed | Six-joint validation | Seven-joint validation |
|---|---:|---:|
| 0 | 6/48 | 48/48 |
| 1 | 12/48 | 48/48 |
| 2 | 2/48 | 48/48 |

Providing true physical parameters does not resolve six-joint transfer. This places a limitation upstream of history extraction, though privileged teacher performance is not a formal upper bound: a student can outperform its teacher on a particular held-out family.

**Post-hoc reset audit:** Six-joint training homes have joint vector `[0, .45, 0, 0, -.65, 0]`, whereas validation homes are `[0, .45, 0, -.65, 0, 0]`. Across all training demonstration observations, joint 4 spans [-0.33063, 0.19338] rad and joint 5 spans [-1.08952, -0.61670] rad. Validation starts those joints at -0.65 and 0 respectively, outside both ranges. Seven-joint validation instead uses the same home vector as training, with no joint outside demonstration ranges at tolerance 1e-6 rad. Pairwise observation equality remains valid, but the six-joint split changes reset posture as well as kinematics. This is an identified confound, not causal proof that posture alone explains every failure. The reproducible audit is saved in `reset_coverage.json`; no earlier dataset was modified.

| Seed | Probe | Intact | Shuffled actions | Shuffled transitions | Pose trajectory only | Initial pose only |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 930000 | 62/96 | 13/96 | 62/96 | 13/96 | 13/96 |
| 0 | 940000 | 57/96 | 13/96 | 57/96 | 14/96 | 13/96 |
| 0 | 950000 | 61/96 | 23/96 | 60/96 | 11/96 | 13/96 |
| 1 | 930000 | 52/96 | 40/96 | 54/96 | 29/96 | 28/96 |
| 1 | 940000 | 61/96 | 46/96 | 63/96 | 30/96 | 28/96 |
| 1 | 950000 | 62/96 | 67/96 | 60/96 | 28/96 | 28/96 |
| 2 | 930000 | 44/96 | 23/96 | 44/96 | 9/96 | 8/96 |
| 2 | 940000 | 45/96 | 25/96 | 45/96 | 6/96 | 8/96 |
| 2 | 950000 | 48/96 | 27/96 | 52/96 | 9/96 | 8/96 |

Independent action shuffling changes only the executed-action column, breaking its alignment with measured responses. It lowers success in eight comparisons but improves one; dependence is not universal. Shuffling complete transitions preserves within-transition action/response tuples and changes aggregate success by at most four tasks. Some collision outcomes change despite equal success counts, so this is not exact trajectory equivalence. Pose-only interventions zero velocities/actions/deltas, retaining q/EE across time or repeating their first values. These perform poorly, but are out-of-distribution interventions on frozen history encoders, not retrained baselines. The pattern supports useful action/response alignment more than long-range temporal order; it does not establish a particular learned identification algorithm.

**Reactive fitting audit:** All 256 demonstrations supply 1,664 initial and 21,303 later valid-joint samples per model. Twin first observations are verified identical while expert actions differ. Under equal twin weighting, any deterministic reactive first-action prediction has per-joint MSE at least 0.000210500496 rad², or RMSE 0.0145086 rad. Reactive initial RMSE is 0.0147450, 0.0146802, and 0.0147013 rad for seeds 0/1/2, close to that bound. Later expert-state RMSE is 0.001059, 0.001127, and 0.001037 rad; privileged teacher initial RMSE is 0.000960, 0.001309, and 0.001090 rad. Thus low reactive task success should not be read simply as failure to optimize the supervised objective. A compromise first action may leave expert-state coverage; that is a recovery-training hypothesis, not established causality. This first-action imitation bound does not bound achievable task success: a reactive controller may choose an informative or recoverable action and use subsequent current-state feedback.

### 4.24 Crossed kinematics/posture benchmark and shared recovery (E26)

`outputs/ease_paired_factorial_v2` preserves earlier artifacts and supplies 256 new training demonstrations plus 384 validation cases. Every training and seen-posture validation arm uses the same joint home vector within DoF: zero except zero-based q1=.45 and q4=-.65. Shifted posture uses q1=.75 and q4=-1.05. Eight training bodies and eight held-out-axis bodies share physical parameters across matched family instances. Validation crosses seen/new axes with seen/shifted posture; each cell contains 96 cases, half six-joint and half seven-joint. Seen axes means the actual training bodies with new goals, not new-body transfer. All four cells share relative Cartesian target offsets within each physical group. Offsets are accepted only when both twins in every cell succeed with the analytical expert; all attempted outcomes are retained. Cartesian states and absolute goals still vary with geometry/posture, so this is a controlled factor comparison rather than identical observation distributions across all cells.

An independent audit verifies 256/256 expert replays, 48 complete four-condition offset groups, matched masses/lengths/actuators, identical twin observations, and rank-three Cartesian Jacobians at both postures. Reserved test geometries have no task screening or rollout. All 38 tests pass.

The fixed recovery union uses only training tasks and two unchanged sources: the E24 seed-0 reactive model and E23 seed-0 history model. Expert branches of up to 16 actions are queried at steps 0,1,2,4,8,16,32,64,100 on noncolliding source rollouts; branches with contact are rejected. Collection yields 1,897 reactive-source and 1,314 history-source chunks, 3,211 total. Source checkpoint hashes, per-query provenance, rejected branches, episode mappings, and recovery hashes are retained. No new model contributes further data after training begins.

Four fresh policies compare reactive versus history conditioning on original demonstrations versus the exact shared recovery union. Each uses seed 0, 15,000 updates, batch 32, AdamW learning rate 0.0003, and the same direct action-chunk architecture and learned-weight initialization. Normalization is fitted separately on each training-data condition. Unlike E23/E24 students, these history policies train end-to-end on actions, with calibration-prefix contexts only; no privileged latent teacher is used. The objective change is deliberate to make the recovery-data comparison matched. Original-data runs contain 3,424 supervised rows; both recovery runs contain the identical 6,635-row union. History training includes variable prefix lengths, including empty context; correct/empty/twin evaluation is retained. Three frozen old privileged teachers are supplementary references trained on the previous dataset.

| Training data | Policy | Training success | Full training gate |
|---|---|---:|---|
| Original | Reactive | 154/256 | Fail |
| Original | History, correct | 256/256 | Pass |
| Shared recovery | Reactive | 246/256 | Fail: body 84000 reaches 25/32 |
| Shared recovery | History, correct | 256/256 | Pass |
| Shared recovery | History, empty | 256/256 | Pass |

Recovery substantially improves reactive control. Its aggregate rate exceeds 95%, but the per-body criterion still fails; that limitation is retained. The history policy's empty-context variant passes every body, making its near-equal validation performance particularly relevant.

| Data | Policy/context | Seen axes / seen posture | New axes / seen posture | Seen axes / shifted posture | New axes / shifted posture |
|---|---|---:|---:|---:|---:|
| Original | Reactive | 54/96 | 39/96 | 51/96 | 31/96 |
| Original | History / correct | 96/96 | 59/96 | 95/96 | 35/96 |
| Original | History / empty | 49/96 | 28/96 | 54/96 | 21/96 |
| Original | History / twin | 12/96 | 21/96 | 41/96 | 12/96 |
| Shared recovery | Reactive | 93/96 | 74/96 | 96/96 | 48/96 |
| Shared recovery | History / correct | 96/96 | 70/96 | 95/96 | 51/96 |
| Shared recovery | History / empty | 96/96 | 75/96 | 88/96 | 51/96 |
| Shared recovery | History / twin | 49/96 | 57/96 | 94/96 | 47/96 |

Across the 384 factorial cases, recovery-trained correct history, reactive, and empty history succeed 312, 311, and 310 times respectively. These totals mix distinct conditions and are not equivalence tests. For new axes/seen posture, paired history-only versus reactive-only wins are 7 versus 11; for new axes/shifted posture, 3 versus 0. Correct history is not consistently better than empty context or the separate reactive model. Twin history remains harmful in some conditions, but correct-versus-wrong context alone does not show a benefit over omitting history.

Joint-count breakdown reveals persistent six-joint limits. In the two new-axis cells (seen/shifted posture), recovery reactive succeeds 26/48 and 0/48 on six-joint arms; recovery history succeeds 22/48 and 3/48. Both achieve 48/48 on seven-joint arms in both cells. Original history handles shifted posture on seen axes (95/96) much better than new axes even with matched posture (59/96). Removing the identified joint-reset mismatch therefore does not eliminate kinematic transfer problems. Cartesian-state distributions still differ across axes, as disclosed in the design.

The frozen old privileged teachers score, in the table's column order: seed 0 = 96,52,96,48; seed 1 = 96,82,76,50; seed 2 = 96,51,92,50, each out of 96. They are supplementary transfer diagnostics, not retrained upper bounds.

The final report verifies the shared recovery hashes, unchanged collector checkpoints, accepted/rejected query accounting, training-only episode provenance, valid-joint masks, four trained checkpoint hashes, and exact paired evaluation case identities. The factorial heatmap was rendered and visually inspected. All 38 tests pass. No final-test policy evaluation occurred.

**Interpretation:** The E24 history advantage is sensitive to baseline training-state coverage and protocol. In this controlled single-seed action-supervision experiment, shared recovery closes most of the final-success gap. This does not establish that history is useless, nor that all methods are equivalent. It motivates replication before claiming a stable difference, and interaction-efficiency measurements that include calibration cost rather than simply another terminal-success comparison. History policies here collect 32 pre-task actions; the reactive policy does not require calibration. Additional task or metric changes must be prespecified and motivated by the intended deployment, not chosen because they favor history.

The first collector attempt, `ease_factorial_recovery_v1`, stopped before training because a NumPy query index could not be serialized to JSON. It is preserved with a failure status. Casting logging scalars to native types fixed the issue; the unchanged experiment restarted in `ease_factorial_recovery_v2`.

### 4.25 Fixed-data replication and interaction efficiency (E27)

**Protocol.** Reuse the completed E26 seed-0 pilot and train seeds 1 and 2 for each original/recovery × reactive/history combination. All eight new policies receive 15,000 direct-regression updates, batch size 32, AdamW learning rate 0.0003, and 244,289 parameters. The dataset, training-only recovery union (3,211 chunks), and evaluation calibration seed remain fixed. Original/recovery data contain 3,424/6,635 training rows. There is no recollection, validation-dependent tuning, or final-test evaluation. The additional seeds and interaction budgets were specified before new training; seed 0 was already observed. Results are stored in `outputs/ease_factorial_replicate_v1`.

**Training control.** Original-data reactive training scores are 154, 95, 123 out of 256; all fail the gate. Original-data correct history scores 256/256 in every seed. With recovery, reactive scores 246, 255, 256: seeds 1/2 pass, while seed 0 fails the per-body gate. Recovery-trained correct and empty history both score 256/256 in every seed. Failed gates are retained rather than used to filter comparisons.

**Validation.** Counts below are for seeds 0 / 1 / 2. All validation includes seen training bodies with new goals as well as unseen-axis bodies; these are distinguished explicitly.

| Training data | Context | All validation / 384 | Unseen axes, familiar posture / 96 | Unseen axes, shifted posture / 96 |
|---|---|---|---|---|
| Original | Reactive | 175 / 93 / 175 | 39 / 24 / 40 | 31 / 13 / 36 |
| Original | Correct | 285 / 213 / 258 | 59 / 43 / 56 | 35 / 24 / 39 |
| Original | Empty | 152 / 118 / 124 | 28 / 18 / 20 | 21 / 13 / 25 |
| Recovery | Reactive | 311 / 311 / 275 | 74 / 78 / 48 | 48 / 45 / 46 |
| Recovery | Correct | 312 / 312 / 313 | 70 / 72 / 73 | 51 / 48 / 48 |
| Recovery | Empty | 310 / 320 / 300 | 75 / 79 / 66 | 51 / 49 / 48 |
| Recovery | Twin | 247 / 263 / 281 | 57 / 61 / 65 | 47 / 48 / 48 |

Correct history is more stable than the separately trained reactive policy across these three seeds, but differs from empty history by +2, -8, +13 successes over all 384 cases. On unseen axes alone it differs by -5, -8, +7 out of 192. Pooling these repeated benchmark evaluations gives 362/576 correct, 368/576 empty, and 339/576 reactive; these are descriptive totals, not independent robot-family samples. The original-data history advantage repeats, but improved recovery coverage greatly narrows it.

**Joint-count failure analysis.** Recovery-trained correct history solves 22/48, 24/48, 25/48 new six-joint cases at familiar posture, and only 3/48, 0/48, 0/48 after the posture shift. Reactive scores 26/48, 31/48, 0/48 at familiar posture and 0/48 in every shifted-posture seed. Empty history scores 27/48, 31/48, 18/48 and 7/48, 1/48, 1/48 respectively. Correct history solves every new seven-joint case in both posture conditions. The stability of aggregate correct-history success does not establish robust six-joint adaptation.

**Interaction cost.** The horizon is 150 control steps; all failures consume the full horizon. A successful case counts within a budget only if its task steps plus required calibration fit the budget. Correct/twin history uses a 32-step probe. Reactive and empty require no calibration input; the evaluator generates an ignored probe for empty, so zero represents minimal deployment cost rather than literal evaluator runtime. Costs exclude reset and computation time. Control rate is 25 Hz.

| Recovery-trained policy | Mean task steps | Calibration per task | Calibration reused across 12 goals |
|---|---:|---:|---:|
| Reactive | 50.1 | 50.1 | 50.1 |
| Correct history | 42.3 | 74.3 | 45.0 |
| Empty history | 45.1 | 45.1 | 45.1 |

These averages include all validation cases and all three seeds; success rates must be considered alongside capped costs. Reuse charges one probe per robot/posture group, verified to contain 12 goals with identical home and probe seed. This is aggregate suite accounting, not per-task latency. On new axes at familiar posture, mean successes within 50 total interactions are 54.0/96 reactive, 23.7/96 correct, and 63.0/96 empty. At 100 interactions they are 62.7, 64.7, and 71.7. Single-use calibration therefore has no general efficiency advantage. Amortization helps, but overall correct-history cost is nearly the same as empty history and unseen-axis mean success is slightly lower.

**Verification and artifacts.** The report audits all 12 policy hashes, fixed data/recovery/source hashes, 6,144 training records, and 9,216 validation records. Every policy uses the same expected case keys; per-condition counts, robot/reset reuse groups, full-horizon failures, model capacity, and training-row counts are checked. All 40 tests pass, including calibration budget and failure-cost tests. `comparison.json` retains seed/condition/joint-count successes, collisions, budget curves, training gates, and paired wins. `audit.json` hashes all rollout evidence. `report.md` contains full tables, and the rendered figure was inspected.

![E27 calibration-inclusive efficiency](../../outputs/ease_factorial_replicate_v1/interaction_efficiency.png)

**Interpretation.** This fixed-data replication tests training randomness, not recovery-collection randomness or broad morphology diversity. Correct history is useful relative to weak original-data controls and more stable than one recovery-trained reactive seed, but inference-time calibration does not consistently beat empty context. Wrong-history damage establishes sensitivity to the conditioning input, not necessity or accurate identification. Frozen history is not continuous online adaptation. Before a paper claim about rapid unseen-embodiment identification, diagnose the persistent six-joint failures and test whether the existing probe supports accurate local kinematic prediction. Do not change the task merely to produce a history advantage.

### 4.26 Working six-joint control through measured kinematics (E28–E29)

**Question and change.** E27's learned policy fails many unseen six-joint cases despite receiving calibration. To determine whether these observations support control, E28 replaces neural action prediction with an explicit local kinematic estimator and damped inverse controller. This is a new analytical baseline. The same implementation handles six or seven joints without an axis catalog or robot-specific controller parameters.

For calibration transitions, let D contain measured joint displacements and X measured end-effector displacements. Estimate J = [solve(DᵀD + 10⁻⁶I, DᵀX)]ᵀ. The command-tracking bias is the mean executed joint command minus measured joint displacement. Given Cartesian error e, command clip(Jᵀ solve(JJᵀ + 0.015²I, 0.5e) + bias, -0.04, 0.04). The bias is an empirical tracking correction, not a gravity model. The online variant exponentially updates the displacement sufficient statistics with forgetting 0.98 and the tracking bias with weight 0.2. It receives only joint state, end-effector position, goal, and executed actions. No target-arm expert queries, axis sequence, link lengths, physical parameters, simulator Jacobian, or simulator forces enter the controller.

**Protocol.** Preserve E27's benchmark manifest, goals, homes, action limit, 150-step horizon, 3 cm/five-step criterion, and collision rejection. Use the existing 32-step random probe with seeds 930000/940000/950000 plus pair ID. Training cases use seed 930000; validation uses all three. E28 compares a frozen identified model, full online updates, and full updates initialized from the paired wrong-arm probe. No gains are selected against validation outcomes. E29 is a declared post-E28 ablation with the same constants, updating only J or only tracking bias, plus a complete full-online replay. During every E29 rollout, `mujoco.mj_jacSite` raises if called. The final test remains untouched.

**Unseen six-joint results.** Entries list successes for the three probe seeds. These are repeated probes on fixed robots, not independent learned-model seeds.

| Controller | Familiar posture / 48 | Shifted posture / 48 |
|---|---|---|
| Frozen identified model | 48 / 48 / 48 | 48 / 47 / 48 |
| Full online identification | 48 / 48 / 48 | 48 / 48 / 48 |
| Wrong-twin initialization + online | 48 / 48 / 48 | 48 / 48 / 48 |
| Jacobian updates only | 48 / 48 / 48 | 48 / 48 / 48 |
| Tracking-bias updates only | 48 / 47 / 48 | 48 / 47 / 48 |

The nearly perfect frozen six-joint result is important: online adaptation is not required to explain most of this improvement over E27. A simple explicit model extracted from the existing history already controls these cases. This does not isolate whether the Transformer failure lies in extracting kinematics, representing them, or converting them into actions, but it rules out insufficient measured information as a blanket explanation on these cases.

**All validation and costs.** Across 384 cases and three probes:

| Controller | Success / 1152 | Collisions | Mean task steps | Mean steps with per-task probe |
|---|---:|---:|---:|---:|
| Frozen identified model | 1044 | 9 | 41.10 | 73.10 |
| Full online identification | 1152 | 0 | 19.66 | 51.66 |
| Wrong-twin + online | 1151 | 0 | 23.14 | 55.14 |
| Jacobian updates only | 1117 | 0 | 23.47 | 55.47 |
| Tracking-bias updates only | 1110 | 6 | 32.65 | 64.65 |

Full online identification also solves 256/256 training cases. Its 32-step probe is 1.28 seconds of simulated interaction at 25 Hz; mean task execution is another 0.79 seconds. These are control-time equivalents, excluding resets and computation. All failures retain their full 150-step cost. E27 and E28 differ in estimator/controller and task-time adaptation, so this is a working baseline comparison, not a controlled neural-architecture ablation. For new six-joint arms specifically, the frozen identified baseline is also strong.

**Identification and verification.** The initial J estimate has median relative Frobenius error 0.372 and maximum 4.172 across 96 distinct validation probes. It need not be an exact global kinematic model to support local reaching. These simulator-Jacobian comparisons are diagnostics outside the controller; disabling that API yields exactly identical full-online outcomes, errors, and step counts on all 1,408 training/validation replay cases. Both experiment files contain 4,224 audited rollout records. Expected cases, success/error/collision consistency, failure horizons, manifest hashes, and exact replay are checked. All 43 tests pass, including known-linear-system recovery, unexcited-input stability, and isolated-update tests. The six-joint figure was rendered and inspected.

![E28–E29 six-joint control](../../outputs/ease_identified_ablation_v1/six_joint_control.png)

**Research consequence.** We now have functioning cross-kinematic six-joint control on this fixed simulated reaching benchmark. We do not yet have a learned Transformer that achieves it. Wrong-twin recovery also prevents claiming that a correct initial probe is necessary for the online controller. The concrete learned-policy direction is an explicit joint-wise motion model (Jacobian columns and tracking corrections, with uncertainty), updated from measured transitions and coupled to a shared controller. Any learned component must demonstrate an advantage over the analytical estimator, rather than merely wrapping this working controller. Broad generalization, novelty, manipulation, hardware, and publication readiness remain unestablished.

Artifacts: `outputs/ease_identified_control_v1` preserves E28's source, protocol, rollouts and identification diagnostics; `outputs/ease_identified_ablation_v1` preserves E29 and contains `report.md`, `summary.json`, `audit.json`, and the verified figure.

### 4.27 Learned motion-model Transformer reaches six-joint control (E30–E32)

**Redesign and training.** Following E28's explicit-model success, E30 replaces direct action prediction with a Transformer that predicts local Jacobian columns from measured Gram/cross statistics. The shared damped controller and measured tracking-bias correction remain analytical components. The predictor's forward path has no least-squares solve or simulator geometry. Three fresh models each train for 15,000 AdamW updates at learning rate 0.0003 and batch size 128. Synthetic linear systems use random Jacobians, randomly mixed/scaled excitation profiles from the 128 training-robot probes, and Gaussian excitation. Supervision distills the ridge solution on these synthetic systems. Each model sees 1.92 million generated systems. No validation-arm geometry, validation response data, or target-arm expert actions enter training.

This changes training distribution, objective, history representation, controller, and online adaptation together; it is a functional redesign rather than an isolated architectural ablation or fine-tuning of E27. Model capacity is 235,011 parameters versus E27's 244,289, but data and update workloads are not matched. The same physical benchmark, goals, resets, success criterion, action limits, and reserved final test are preserved.

**Replication protocol.** E30 seed 0 is an observed pilot. E31 adds seeds 1/2 without changing the training recipe, then evaluates all models with probe seeds 930000/940000/950000. Modes are frozen neural estimation, online neural estimation, an explicitly labelled neural-first-step/analytical-update hybrid, cold-start neural estimation, wrong-twin initialization, and an untrained-network control. In cold start the probe seed has no effect; all three repetitions match exactly and are counted only once per training seed. Wrong-twin and hybrid outcomes do not independently establish correct-history necessity or neural-only performance.

**Validation results.** Entries below are successes /384; three values denote the three calibration seeds. All primary neural-online and cold-start trials are collision-free.

| Mode | Model seed 0 | Model seed 1 | Model seed 2 |
|---|---|---|---|
| Frozen neural | 346 / 365 / 358 | 370 / 340 / 366 | 349 / 364 / 370 |
| Online neural | 383 / 383 / 383 | 384 / 383 / 384 | 384 / 384 / 384 |
| Hybrid analytical updates | 384 / 384 / 384 | 384 / 384 / 384 | 384 / 384 / 384 |
| Cold-start neural (no probe) | 384 | 384 | 384 |
| Twin-initialized online neural | 384 / 384 / 383 | 384 / 384 / 384 | 381 / 384 / 384 |
| Untrained neural (one probe) | 45 | 6 | 54 |

Calibrated online neural estimation totals 3,452/3,456 successes (99.88%). Seed 0 retains one shifted seven-joint failure at probe 930000 and one shifted six-joint failure at each other probe; seed 1 retains one seen-axis shifted seven-joint failure at probe 940000. Seed 2 has none. Cold start totals 1,152/1,152 after removing redundant probe-labelled repetitions. These are repeated tasks on fixed validation families, not thousands of independently sampled embodiments.

**Unseen six-joint cases.** Each entry is successes /48 at the three probe seeds; cold start has no separate calibration.

| Mode | Model seed | Familiar posture | Shifted posture |
|---|---:|---|---|
| Online neural | 0 | 48 / 48 / 48 | 48 / 47 / 47 |
| Online neural | 1 | 48 / 48 / 48 | 48 / 48 / 48 |
| Online neural | 2 | 48 / 48 / 48 | 48 / 48 / 48 |
| Cold-start neural | 0 | 48 | 48 |
| Cold-start neural | 1 | 48 | 48 |
| Cold-start neural | 2 | 48 | 48 |

The calibrated variant reaches 862/864 unseen six-joint cases; cold start reaches 288/288 across model seeds. This is a substantial functional improvement over E27's calibrated six-joint scores of 22/48, 24/48, 25/48 at familiar posture and 3/48, 0/48, 0/48 at shifted posture. It must not be attributed solely to a better Transformer layer: the explicit system-identification objective, synthetic training, and shared controller also change.

**Training control and costs.** Online neural estimation solves 256/256 training tasks in every seed. Frozen neural estimation scores 237/256, 255/256, 248/256; only seed 1 passes the complete gate. Cold-start validation consumes mean 19.08 control steps per task, including identification interactions. Calibrated online neural inference uses 17.27 task steps plus 32 probe steps = 49.27 total. Hybrid uses 51.52 total and frozen neural 64.69. These averages include 150-step failures and exclude resets and compute. Cold start achieves mean 380.67/384 successes within 50 total interactions; calibrated online reaches 238.44/384. A separate probe is not beneficial on this particular protocol; this does not establish that probes are unnecessary for other morphologies or tasks.

**Attribution controls (E32).** Added after observing seed-0 cold-start success, with unchanged validation cases and no gain tuning. Zero-initialized analytical identification solves 336/384 (mean 38.32 steps). A seed-0 neural first-step bootstrap followed by analytical updates solves 384/384 (18.28 steps). Holding the neural empty-history Jacobian fixed while only updating tracking bias solves 18/384 and incurs 278 collisions. A fixed +0.01 rad command on every joint at task step one, followed by analytical identification, also solves 384/384 (18.45 steps). The initial three controls are preserved in `ease_motion_coldstart_v1`; the added fixed-action control and repeated originals are in `v2`. Thus training and task-time prediction updates matter within the neural design, but the simple analytical probing controller remains equally successful and slightly cheaper on average. No learned-superiority claim follows.

**Verification and reusable implementation.** All 2,304 primary calibrated/cold-start neural replay cases match exactly while simulator Jacobian access, `torch.linalg.solve`, and NumPy solves other than the 3×3 Cartesian control system are blocked. The joint-wise motion model is therefore predicted by the neural network rather than silently replaced by a ridge inverse. Shared control and measured bias updates remain intentional analytical components. The report verifies all three checkpoint hashes, expected case keys, training gates, record hashes, per-condition/joint-count assignments, and redundant empty-mode repeats. There are 18,432 stored replication validation records, of which 2,304 are redundant empty-mode repetitions, plus separately preserved training and attribution runs. All 46 tests pass, including padding invariance, joint reordering of measurements, no neural inverse, and unchanged network weights during online updates.

`MotionController` provides a reusable inference API with checkpoint-hash verification and independent state per robot. The seed-0 checkpoint, through this single-robot interface, solves all 384 validation cases without calibration while the same inverse/API guards are active. It shares one model between six- and seven-joint instances. This independently tests the interface; it is not an additional robot-family sample. Source and record hashes are in `ease_motion_api_check_v1/audit.json`.

![E30–E31 six-joint Transformer results](../../outputs/ease_motion_replicate_v1/six_joint_transformer.png)

**Interpretation and next gate.** The redesigned Transformer-plus-controller now matches the analytical solution on the six-joint benchmark in cold-start mode, and nearly matches it with a calibration probe. It is a learned motion estimator with a structured controller, not a wholly neural action policy. The result resolves the immediate functional failure, but not novelty or broad embodiment transfer. Next lock these checkpoints and evaluate broader withheld kinematics and task distributions against the fixed-probe analytical baseline, with interaction cost and failure rates retained. Any paper claim must identify where learning improves robustness, sample efficiency, or capabilities beyond that baseline. Final-test policy evaluation remains reserved.

Artifacts: `ease_motion_transformer_v1` contains the pilot; `ease_motion_replicate_v1` contains seeds 1/2, all probe/context controls, `summary.json`, `audit.json`, `report.md`, and the inspected figure. `motion_transformer.py` contains the trained architecture and reusable controller.

### 4.28 Broader frozen-checkpoint embodiment transfer (E33)

**Protocol.** All three E31 checkpoints remain unchanged. A separate development suite samples eight new axis sequences per joint count, excluding every registered prior sequence, including reserved axes. Base axes can now be X, Y, or Z. Four crossed geometry/dynamics variants per family, two shared joint postures, and eight target joint offsets per posture produce 64 physical bodies and 1,024 tasks. Sampling seed 330000, checkpoint hashes, and evaluation rules were recorded before generation and evaluation. No new training or gain tuning occurred.

Nominal link multipliers are 0.8–1.2 and masses 0.7–1.3; shifted geometry uses 0.55–1.55 and 0.5–1.8 respectively. Nominal actuator strength/damping are 1/2; shifted values are sampled from 0.3–2.0 and 0.3–5.0. Near/wide homes use joint ranges ±0.45/±0.85 radians with joint 2 offset by 0.4. All variants share each joint-space home and target offset, but their Cartesian goals differ with geometry. Thus the crossing does not isolate identical Cartesian observations.

Homes must be contact-free with rank-3 positional Jacobians. Target offsets have norm 0.2–0.55 radians; target configurations must be contact-free, within ±2.2 radians, and place the end effector 6–18 cm from its start for all four variants. No expert or policy success filtering is applied. Reachable endpoints do not guarantee a collision-free path. Version 1 stopped during generation on an intrinsically rank-deficient family, before any controller evaluation. It is preserved; version 2 adds a generic-pose rank check before accepting axis sequences. The completed generation logs three rejected axis candidates, zero rejected homes, and 333 rejected goal candidates.

**Results.** Every controller starts without separate calibration. The fixed analytical probe commands +0.01 radians on each joint at task step one, counted in its budget. The unchanged horizon is 150 steps, action limit 0.04 radians, and success requires five consecutive steps within 3 cm with no contact during the episode. Means include failures at 150 steps.

| Controller | Success /1,024 | Six-joint /512 | Seven-joint /512 | Contact episodes | Mean task steps |
|---|---:|---:|---:|---:|---:|
| Privileged reference | 1,012 | 500 | 512 | 12 | 13.46 |
| Fixed-probe analytical | 1,003 | 491 | 512 | 19 | 19.76 |
| Zero-start analytical | 1,003 | 491 | 512 | 19 | 19.48 |
| Neural seed 0 | 985 | 477 | 508 | 19 | 27.27 |
| Neural seed 1 | 1,000 | 489 | 511 | 18 | 21.29 |
| Neural seed 2 | 995 | 486 | 509 | 16 | 24.90 |

Neural six-joint success is 93.2–95.5%, versus analytical 95.9%; overall neural success is 96.2–97.7%, versus analytical 97.9%. Paired neural-only/analytical-only successes are 4/22, 5/8, and 4/12 for seeds 0/1/2. The fixed-probe analytical controller solves 992 tasks within 50 interactions, versus 930, 977, and 950 for the neural models. Learning does not improve aggregate success or interaction cost here. Tasks share families, bodies and postures; the 1,024 cases are not independent embodiment samples.

**Post-hoc failure diagnosis.** The six-joint `XXXZYZ` family accounts for all 21 analytical failures (19 contacts, two timeouts), all 12 privileged-reference failures (contacts), and 22/16/22 neural failures for seeds 0/1/2. On this family, successes are analytical 43/64 and neural 42/48/42 out of 64. Its shifted dynamics use strength 0.519 and damping 4.686. All three neural models and the fixed-probe analytical controller incur contact on all 16 shifted-dynamics/wide-posture tasks; the privileged reference fails on 12 of those tasks. This identifies a shared difficult condition, not proof that the tasks are impossible or that actuator strength alone causes failure. Neural failures also occur elsewhere: 17, eight, and seven cases respectively. Retain this family and all failures in subsequent comparisons.

**Verification.** The audit checks all 64 body assets, 128 homes, 1,024 forward-kinematic target endpoints, crossed parameter ties, axis nonoverlap, expected record keys, and all six methods' 6,144 records. Training-manifest and checkpoint hashes remain unchanged. Nonprivileged evaluation blocks simulator Jacobian access; neural evaluation additionally blocks analytical joint-model solves while allowing the intentional 3×3 control solve. All 48 tests pass. The per-family figure was visually inspected. There are no final-test rollouts.

![E33 broader embodiment transfer](../../outputs/ease_broad_development_v2/broad_transfer.png)

**Interpretation.** Frozen learned motion estimation transfers across substantially more procedural six- and seven-joint arrangements, geometry, dynamics and postures. This strengthens functional generalization evidence while retaining clean observations, position-only reaching, known joint count/order, and synthetic linear-system training. It does not establish a conference contribution beyond analytical identification. The next experiment should prespecify noise, delay and limited-excitation conditions, compare both estimators at equal interaction budgets, and diagnose contact trajectories without dropping difficult cases.

Artifacts: `outputs/ease_broad_development_v2` contains the frozen protocol, source snapshot, dataset, structural exclusions, six result files, `summary.json`, `audit.json`, `report.md`, and figure. Reproduction entry points are `broad_embodiment.py` and `broad_report.py`.

### 4.29 Sensor, delay and restricted-history robustness (E34)

**Prespecified protocol.** Evaluate every E33 task with all three unchanged neural checkpoints and the fixed-probe analytical controller, retaining the 150-step task budget, no separate calibration, and unchanged true-state success/contact criteria. Conditions were written before evaluation: clean; Gaussian position noise at sigma 0.001/0.005 in joint radians and end-effector meters; synchronized joint/EE observation delays of one/three control steps (40/120 ms); and statistics restricted to the most recent four/eight transitions. No gains or weights were fitted. Noise realizations are deterministic per task and shared across methods at equal elapsed time; targets remain exact. Delayed transitions use timestamp-aligned executed commands, and repeated startup observations do not create fictitious updates. Restricted-history conditions reconstruct both motion statistics and tracking bias from the retained window using the original prior and forgetting constants. They test memory restriction, not independently controlled physical excitation.

**Complete results.** Successes out of 1,024; the parenthesized values are six-joint successes out of 512. Seed order is fixed, not selected by outcome.

| Condition | Analytical | Neural 0 | Neural 1 | Neural 2 |
|---|---:|---:|---:|---:|
| Clean | 1003 (491) | 985 (477) | 1000 (489) | 995 (486) |
| Noise 1 mm / 0.001 rad | 1002 (490) | 987 (475) | 997 (486) | 999 (488) |
| Noise 5 mm / 0.005 rad | 979 (473) | 972 (464) | 974 (467) | 974 (466) |
| Delay 40 ms | 1004 (492) | 989 (478) | 993 (482) | 986 (482) |
| Delay 120 ms | 725 (360) | 678 (341) | 723 (346) | 683 (335) |
| Last 4 transitions | 971 (467) | 877 (401) | 949 (445) | 861 (376) |
| Last 8 transitions | 1008 (496) | 980 (471) | 1003 (491) | 982 (470) |

**Costs and failure modes.** At 5 mm noise, neural mean task steps are 34.93/32.60/35.87 versus analytical 36.89, including 150-step failures. Success by step 50 is 880/885/853 versus analytical 858. Thus lower mean interaction cost does not establish uniformly better control: neural final success is lower, and one seed has fewer early successes. At 120 ms, analytical failure comprises 17 contact episodes and 282 failures without contact; neural counts are 30/29/19 contact episodes and 316/272/322 failures without contact. The main aggregate degradation is therefore timeout without contact. With four-transition memory, neural contacts rise to 91/57/87 versus analytical 39; six-joint neural success falls to 401/445/376 versus analytical 467. Eight transitions recover much of the loss. These are observations, not a causal identification of a particular network defect.

**Audit and limits.** Clean success, collision, and step counts reproduce E33 exactly for all four methods. All 28,672 records pass case-key, finite-error, success/contact and horizon checks; per-family results and paired task comparisons are preserved. Checkpoint hashes are unchanged. Simulator Jacobian access is blocked for all rollouts, and neural runs additionally prohibit analytical joint-model solves while permitting the shared 3×3 control solve. All 50 tests pass, including nonmutating reproducible sensor noise and truncated-statistics agreement with the existing online estimator. The figure was visually inspected. There is one disturbance realization per task and correlated cases within 16 families; these counts do not establish statistical significance or broad real-world robustness. The analytical estimator is not noise-tuned. There are no final-test rollouts.

![E34 robustness comparison](../../outputs/ease_robustness_v1/robustness.png)

**Decision.** These frozen tests do not demonstrate a learned final-success advantage. The next development step should investigate training-distribution coverage for noisy and rank-deficient short histories, and explicit delay handling, with equally equipped analytical controls. Any new training must use training-only systems and a separate development protocol; retain E34 as an observed diagnostic rather than repeatedly tuning to it. Physically restricted excitation remains a separate experiment. Artifacts and full cost tables are in `outputs/ease_robustness_v1/report.md`, `summary.json`, `audit.json` and the immutable protocol/rollout files.

### 4.30 Noisy/short-history training and matched delay handling (E35)

**Fixed development protocol.** Each E31 model receives 5,000 additional AdamW updates at learning rate 0.0001, batch 128, using seeds 35000/35001/35002. The 235,011-parameter architecture is unchanged. Training-only excitation profiles are mixed with Gaussian synthetic motions and random linear Jacobians. History length is uniform from one to 32 transitions; position-noise sigma is sampled from {0, 0.0005, 0.001, 0.002, 0.005}, in radians for joints and meters for the end effector. Noise is applied to positions before differencing, retaining adjacent-transition error correlation. Inputs use noisy exponentially weighted statistics, while supervision is the clean weighted ridge solution. This simultaneously changes training distribution, history length and target statistics, and adds training compute; it is not an isolated ablation or a compute-matched comparison.

All 1,024 E33/E34 development tasks and seven conditions are retained. No post-result tuning or seed selection occurs. These task families and disturbance outcomes were already observed, so this is development evidence. A shared delay-aware gain, 0.5/(1 + delay steps), is applied to analytical, old neural and new neural estimators. It is a conservative gain rule, not prediction of the delayed state. Non-delay old-model and analytical results are inherited from E34. There are 29,696 new robustness rollouts and 20,480 inherited comparison rows, plus 768 separate training-task checks.

**Results.** All entries are successes out of 1,024. Delay rows use the same new gain for every column; old neural results there are newly evaluated controls.

| Condition | Analytical | Old neural 0/1/2 | Updated neural 0/1/2 |
|---|---:|---|---|
| Clean | 1003 | 985 / 1000 / 995 | 1004 / 1004 / 997 |
| Noise 1 mm / 0.001 rad | 1002 | 987 / 997 / 999 | 1003 / 1000 / 1001 |
| Noise 5 mm / 0.005 rad | 979 | 972 / 974 / 974 | 988 / 983 / 989 |
| Delay 40 ms, shared gain | 998 | 991 / 986 / 993 | 999 / 1002 / 993 |
| Delay 120 ms, shared gain | 986 | 956 / 962 / 955 | 954 / 940 / 955 |
| Last 4 transitions | 971 | 877 / 949 / 861 | 986 / 937 / 976 |
| Last 8 transitions | 1008 | 980 / 1003 / 982 | 1006 / 1006 / 1006 |

**Higher-noise lead.** Updated six-joint success is 481/483/478 of 512 versus analytical 473 and old neural 464/467/466. Mean task steps including failures are 30.68/30.99/28.88 versus analytical 36.89. Thus all three updated seeds improve both final success and mean interaction cost against this fixed analytical baseline in this condition. The total gains are only four to ten tasks out of 1,024, with one noise realization per correlated task; do not treat this as a statistically established or general learned advantage. A noise-aware analytical estimator and fresh morphology/noise repetitions are necessary controls before a paper claim. The model has no explicit noise-scale input and no inference-time gradient update.

**Delay attribution and short-history limits.** At 120 ms, analytical success rises from E34's 725 to 986 solely with the shared gain change. Old neural scores similarly rise from 678/723/683 to 956/962/955. Updated models do not exceed their corresponding old-model controls, so this recovery cannot be attributed to noisy-history training. Four-transition fine tuning improves seeds 0/2 substantially but worsens seed 1 overall (949 to 937); updated six-joint scores are 482/450/468 versus analytical 467. Eight-transition updated scores agree at 1,006 overall and 494/512 six-joint, slightly below analytical 1,008 and 496. Preserve the seed-dependent failure and the stronger analytical delay result.

**Verification.** All updated models solve 256/256 original training tasks with simulator Jacobians and analytical joint-model inverses blocked. Training/evaluation dataset hashes, original and new checkpoint hashes, case keys, success/contact consistency and failure horizons are verified. The 3×3 Cartesian control solve remains intentional. Original checkpoints are unchanged. There are 53 passing tests across the research code, including synthetic masked statistics, training gradients and physical fixture checks. The comparison figure was inspected. Final test remains reserved.

![E35 training and matched-controller comparison](../../outputs/ease_robust_training_v1/comparison.png)

Artifacts: `outputs/ease_robust_training_v1` holds the frozen protocol, source snapshot, three new checkpoints, training traces, full rollout records, summary, audit and report. `ease_robust_training_gate_v1` preserves the separate training checks. Next independently replicate the higher-noise effect against an analytical estimator equipped for noisy measurements; do not repeatedly select conditions from this observed development suite.

### 4.31 Pick-and-place task foundation (E36)

**Implemented mechanics.** `pick_place_fixture.py` adds a physical parallel-jaw gripper, a free 3 cm cube of mass 0.04 kg, a tabletop and contact-based grasping. A scripted Cartesian carriage executes approach, close, lift, transport, lower, release, retreat and settle. All three predefined object/target cases pass in 370 control steps each. The cube is neither welded to the gripper nor teleported. Success requires bilateral contact during grasp, object height above 7.5 cm while held, final XY error below 2.5 cm, settled tabletop height within 5 mm, final generalized speed norm below 0.02, zero final finger contact and no gripper/table contact. The generalized speed norm combines translational and angular components and is only a fixture settling check. Final positions are within a few micrometers of their target XY in this deterministic rig; this precision is not a robot-policy result.

Version 1 is retained; version 2 makes the final zero-contact requirement explicit and stores source; version 3 adds readable lighting/colors and source snapshots. All physical outcomes are identical across versions. Lift and release frames were rendered and inspected. This is a mechanics test driven by scripted Cartesian actuators, not learned manipulation or cross-embodiment pick-and-place.

**Arm assets and capability audit.** `arm_fixture_xml` attaches the same two-finger gripper to a serial arm, with n arm actuators plus two jaw actuators, a grasp-center site, free cube and target marker. All 64 six/seven-joint scene assets compile. An audit of the original 128 body/posture configurations finds 80 local rank-six pose Jacobians and 48 rank-five Jacobians. The six-joint split is 32 rank-six/32 rank-five; seven-joint is 48/16. All have rank-three position Jacobians. These are local kinematic ranks, not proof of downward-grasp reachability or collision-free task execution. The exported scenes are not yet validated manipulation resets or a task dataset.

**Next manipulation gate.** Use a known cube pose and place target, a downward-facing grasp with free yaw where feasible, and one shared gripper aperture command. First validate full arm-driven sequences with a privileged expert and record structural exclusions. Extend the measured motion model from position to pose (or a justified five-dimensional task) before evaluating a learned arm controller. Initially share a scripted task sequencer between analytical and neural arm controllers; label it as a hybrid and report grasp, lift, transport, release and complete-task success separately. The existing three-row Jacobian model cannot regulate orientation, and no arm-driven pick-and-place result has been produced. Detailed task/interface gates are in `MIXED_DESIGN.md`.

Artifacts: `outputs/ease_pick_place_fixture_v3` contains the definitive physical trials, XML, source and rendered stages; `outputs/ease_pick_place_arm_audit_v1` contains exported arm scenes and the capability audit. Earlier fixture versions remain available.

### 4.32 Physical arm-driven pick-and-place with frozen motion estimation (E37–E38)

**Scene and task.** Corrected the gripper mounting by rotating it 180 degrees about its local X axis so the fingers extend away from the last link; the earlier scene assets compiled but their fingers extended back along the terminal capsule. A free 3 cm cube rests on a tabletop at height 0.2 m. A shared sequencer commands approach, descend, close, lift, 6 cm lateral transport, lower, release, retreat and settle. All controllers regulate the same tool-position and downward-tool-direction goals; yaw is free. There is no object attachment, weld or teleportation. The arm starts at an IK-derived approach configuration; arbitrary-start motion planning is not tested.

**Pose extension without new weights.** The existing E35 Transformer is applied twice with shared frozen weights: once to measured XYZ displacement statistics and once to measured tool-Z-direction displacement statistics, scaled by 0.15. Both predictions share the joint Gram matrix and form a six-row task Jacobian. Direction derivatives have only two independent components, so the task constrains five dimensions. A shared damped 6×6 solve generates arm commands; this is an intentional analytical controller. The analytical comparator instead fits the same six measured outputs directly from the same statistics. Both update motion statistics and tracking bias online. The networks receive no arm axes, link geometry, simulator Jacobians or expert actions. This reuses the learned identification operation rather than training a new manipulation policy. Phase transitions, aperture commands and grasp timing remain scripted.

**Development revisions and retained failures.** Initial full-orientation endpoint IK found four feasible seven-joint tasks and no six-joint tasks in twelve proposed training-family placements; all four reference rollouts succeeded. Switching to free yaw made all twelve endpoint sets feasible; the privileged reference solved 8/12, with four contact failures. Initial observation-only trials with free-running random calibration and 5 mm nominal finger/table clearance yielded analytical 0/12 and neural 0/1/1 out of 12. Many trials lifted and placed the object but touched the table, which remains a failure. These initial results are retained.

The corrected comparison holds probe joint targets within ±0.012 rad of the supplied approach home, changing random offsets every four steps for 32 steps. This bounds calibration drift using measured joint feedback. Grasp/lower tool-center heights rise from 0.230/0.235 m to 0.240/0.245 m, increasing finger/table clearance by 1 cm. The identical corrections apply to every method. All trials continue from calibration without resetting; its 32 interactions and any contacts count toward total task outcomes. The privileged reference also receives the same physical calibration sequence. These are deliberate development corrections, not an independent validation of a frozen manipulation design.

**Complete corrected comparison.** Familiar-family tasks use one body from each of four training axis families, two per joint count, and three fixed cube locations. Neural models did not train on manipulation demonstrations. Each table entry shows complete-task successes; all phase-level outcomes and contacts are retained.

| Controller | Familiar total /12 | Familiar six /6 | Familiar seven /6 | Unseen total /4 | Unseen six /1 | Unseen seven /3 |
|---|---:|---:|---:|---:|---:|---:|
| Privileged reference | 10 | 6 | 4 | 3 | 1 | 2 |
| Analytical identified | 8 | 6 | 2 | 3 | 1 | 2 |
| Neural seed 0 | 7 | 3 | 4 | 3 | 1 | 2 |
| Neural seed 1 | 7 | 5 | 2 | 2 | 1 | 1 |
| Neural seed 2 | 9 | 5 | 4 | 3 | 1 | 2 |

Familiar-family mean total interaction counts are privileged 281.2, analytical 301.3, and neural 322.6/337.3/286.2. Forbidden-contact episodes are respectively 2, 4, and 4/4/3. Every observation-only familiar trial achieves bilateral grasp; all analytical and neural seeds 0/2 trials lift, while seed 1 lifts 9/12. Therefore reaching the object or lifting it alone is insufficient: transport, safe placement and release matter. Unseen mean interactions are privileged 333.3, analytical 328.0, and neural 338.8/390.8/310.0. There is no consistent learned advantage across seeds.

**Unseen screening denominator.** The first two distinct E33 development axis families per joint count were selected by manifest order, without learned-success selection, using the same three cube locations. IK found every required endpoint for only four of twelve candidates: one six-joint and three seven-joint tasks. All four were evaluated even when the reference failed. Eight IK failures are retained; failure of this finite-restart solver does not prove physical impossibility. Only two of the four requested families contribute evaluated tasks. The successful unseen six-joint result is one case repeated across three model seeds, not three independent embodiments. These small, screened results demonstrate feasibility and do not establish broad cross-embodiment manipulation.

**Success and verification.** Success requires bilateral grasp, a held cube above 0.275 m, final placement within 2.5 cm of the target, object center within 5 mm of tabletop-rest height 0.215 m, final generalized speed norm below 0.02, zero final finger contact and no forbidden contact during calibration or execution. Finger/cube and cube/table contacts are permitted; other contacts fail the episode. The generalized speed norm mixes translation and rotation and is only a settling check. All 80 corrected/reference/unseen rollouts have audited case keys, probe counts, step counts, finite errors and success/contact consistency. Original checkpoint hashes remain unchanged. Privileged Jacobian APIs are blocked during observation-only evaluation; neural prediction is separately tested with both NumPy and Torch solves blocked. The shared 6×6 task solve is retained intentionally.

All 56 tests pass, including finite-difference verification of the direction Jacobian, mounting direction, and unchanged neural weights. A rendered seed-0 six-joint success reproduces the stored trace, final object position and all 320 interactions exactly; lift and release frames were inspected. No final-test family is evaluated. The next gate is broader feasible task coverage and contact-reliable sequencing on more unseen six-joint families, retaining analytical and privileged controls. Learned task sequencing, visual perception and arbitrary-start approach planning remain outside this pilot.

![Frozen neural estimator controlling a six-joint arm during physical lift](../../outputs/ease_pick_place_preview_v1/lift.png)

Artifacts: `outputs/ease_pick_place_report_v1` contains the combined report, summary and audit; `ease_pick_place_control_v2` and `ease_pick_place_unseen_control_v1` preserve corrected rollout records and protocols; `ease_pick_place_preview_v1` contains the exact rendered replay. Initial geometry, IK and contact-failure versions remain preserved.

### 4.33 Native PiPER and Panda simulation transfer (E39–E40)

**Assets and interface.** Downloaded 168 XML, mesh, license and documentation files (67,552,223 bytes) from [MuJoCo Menagerie commit cda5236](https://github.com/google-deepmind/mujoco_menagerie/tree/cda5236f98383f6fbaa4820c6b18dba892cd087f). Each file passed upstream Git-blob verification and local SHA-256 verification. PiPER is MIT licensed; Panda is Apache-2.0 licensed. Original asset directories remain unchanged. Derived scenes add task sites, floor/table and a free cube. Adapters retain native inertias, collisions, joint limits, actuators and finger coupling. PiPER has six arm joints, two finger joints and seven actuators; Panda has seven, two and eight respectively. Normalized openness maps to native PiPER 0–0.035 m and Panda 0–255 control ranges. PiPER retains nine gravity-compensated bodies; Panda has none. Consequently the plants are not dynamically harmonized. Control runs at 25 Hz with twenty native 2 ms substeps. Geometry-based tool sites are not hardware-calibrated frames.

**Frozen reaching gate.** The three E35 checkpoints are unchanged. Each robot receives twelve unfiltered home-relative targets: positive/negative XYZ offsets of 4 and 8 cm, retaining initial tool direction. Every controller receives 32 measured probe steps and at most 200 task steps. Success requires five holds within 1.5 cm and direction difference 0.08, with no contact. Nonprivileged evaluations block simulator Jacobian access; neural prediction blocks numerical inverse solves while the shared task controller retains its intentional task-space solve. All 120 trials had zero contacts.

| Native model | Privileged /12 | Analytical /12 | Neural seed 0 /12 | Seed 1 /12 | Seed 2 /12 |
|---|---:|---:|---:|---:|---:|
| PiPER | 11 | 10 | 9 | 9 | 11 |
| Panda | 12 | 12 | 12 | 12 | 12 |

**Manipulation development.** Native physical fingers grasp a 3 cm free cube through frictional contact; no attachment or teleportation is used. Task phases remain scripted and reuse the frozen position/direction estimator. Nine endpoint IK checks screen candidates before any policy result; all accepted tasks remain even when the privileged reference fails. Calibration costs and forbidden contacts count in outcomes. Robot-specific grasp heights accommodate the native finger geometry.

| Version and accepted robot | Accepted/proposed candidates across robots | Privileged | Analytical | Neural 0 | Neural 1 | Neural 2 |
|---|---:|---:|---:|---:|---:|---:|
| v1: Panda, table above base | 3/6 | 2/3 | 0/3 | 1/3 | 0/3 | 0/3 |
| v2: Panda, base at tabletop | 3/6 | 0/3 | 1/3 | 0/3 | 0/3 | 0/3 |
| v3: PiPER, compact tabletop fixture | 3/3 | 3/3 | 3/3 | 1/3 | 2/3 | 3/3 |

Both initial versions reject all three PiPER candidates at endpoint IK. A subsequent structural grid at four X positions and four heights accepts 0/16 poses with the base below the table and 12/16 with the base at tabletop height; none at the highest 0.36 m level. Finite-restart IK failure does not establish impossibility. This motivated the separately versioned compact PiPER fixture: cube X positions 0.20/0.225/0.25 m, 4 cm Y transfer, approach height 0.30 m and lift height 0.32 m instead of 0.33/0.35 m. Versions are different observed development layouts and must not be pooled or interpreted as learning gains. Compact PiPER forbidden-contact counts are 0/0/2/0/0 in table controller order; every controller grasps and lifts all three cubes. Panda still has contact and placement failures. Full stage outcomes, interaction costs and all negative trials are in `outputs/ease_real_arm_report_v1/report.md`.

**Verification and interpretation.** Audited 165 evaluation records (120 reaching, 45 manipulation), pinned assets and unchanged checkpoint hashes. Successful Panda neural-0 v1 case 2 and PiPER neural-2 v3 case 0 render replays reproduce saved traces, outcomes, final object positions and 364/659 total steps exactly. Images and replay audits are in `ease_native_panda_preview_v1` and `ease_native_piper_preview_v1`. All 58 regression tests pass, including joint/actuator mapping, clipping, native gripper ranges and finite-difference task Jacobians. This establishes initial simulated transfer to two real robot designs and successful native-gripper trials, not reliable manipulation, learned sequencing, broad embodiment coverage or analytical superiority. No new training or reserved final-test evaluation occurred.

### 4.34 Native manipulation reliability and expanded placements (E41)

**Failure diagnosis and changes.** E40 traces show finger/table contacts, grasp loss, and continued sequencing after motion timeouts. The first E41 revision limits Cartesian tracking error to 4 mm; this suppresses necessary corrections and produces substantial PiPER drift. Retained pilot results are PiPER privileged/analytical 0/3 each and neural 2/3/3 of three; Panda all zero. Revision 2 instead ramps a Cartesian reference by at most 2 mm per 40 ms control step, retaining full tracking error, task gain 0.35, damping 0.015 and joint-command clip 0.04 rad. Eight holds within 6 mm and tool-direction difference 0.08 gate motion phases; failures stop on forbidden contacts, phase timeout, absent bilateral grasp after closing, or lost grasp after lift/transport. These stricter completion rules differ from E40's final-state gate.

Both bases remain at tabletop height 0.2 m. Approach/lift are 0.30/0.32 m; PiPER/Panda grasp heights are 0.24/0.222 m. Initial lowering uses grasp height plus 8 mm. Transfers are 4 cm in Y. The shared frozen E35 estimators continue updating measured motion statistics; no weight training occurs. Native gripper forces, coupling, inertias, limits and contact mechanics are unchanged. All methods receive 32 calibration steps, counted in total steps. Comparison with E40 changes multiple controller/fixture settings, IK and calibration seeds, and gating, so it does not isolate a causal improvement from smoothing alone.

**Pilot and expansion.** All six nominal pilot tasks pass nine endpoint IK checks. Revision 2 gives PiPER 3/3 for every controller; Panda privileged/analytical 1/3 and all neural zero. The expanded candidates were fixed before screening: PiPER X = 0.205/0.230/0.255 m, Panda X = 0.31/0.36/0.41 m, crossed with Y = -0.03/0.03/0.06 m. All eighteen candidates pass nominal endpoint screening; no controller-success filter is applied. These are only two robot models and one transfer direction, not eighteen independent embodiments.

**Placement-offset intervention.** Panda's repeated lowering failures show the cube hanging below the assumed tool-relative position, then reaching the table before the commanded lowering endpoint. Revision 3 estimates the tool-minus-object offset from allowed simulated object-position feedback at transport/lower phase entry. It offsets the XY target and lowers to an object-center target of 0.222 m, 7 mm above resting height, then holds that tool target during release. Actual offset-adjusted targets are not covered by the nominal endpoint IK screen. The pilot gives PiPER 3/3 for every controller; Panda privileged 2/3, analytical 1/3, and neural 1/0/0. We then reuse the observed expanded set for a matched comparison rather than claiming fresh confirmation.

| Robot / revision | Privileged /9 | Analytical /9 | Neural 0 /9 | Neural 1 /9 | Neural 2 /9 |
|---|---:|---:|---:|---:|---:|
| PiPER: ramped reference, fixed placement | 9 | 8 | 9 | 8 | 8 |
| PiPER: object-offset placement | 9 | 9 | 9 | 7 | 8 |
| Panda: ramped reference, fixed placement | 1 | 2 | 1 | 0 | 1 |
| Panda: object-offset placement | 3 | 3 | 4 | 0 | 2 |

The paired expanded audit verifies exactly identical task definitions, calibration records and every pre-transport trajectory across revisions. Offset placement gains one analytical PiPER success but loses one neural-seed-1 success to contact. On Panda it gains 2/1/3/0/1 successes in table controller order, with no lost successes. Final forbidden-contact counts are PiPER 0/0/0/1/0 and Panda 6/5/1/3/2. Remaining failures include missing/lost grasps, motion timeouts and inaccurate final placements. The change is not uniformly beneficial, and neither controller revision supports consistent neural superiority. Both remain available as immutable source snapshots; the current runner uses revision 3.

**Cost and verification.** Final expanded mean total steps are PiPER privileged 942, analytical 738, neural 747.9/692.0/662.7; Panda 373.1/374.6/424.1/367.4/339.1. Early stopping makes faster failures look cheaper, so these means are not efficiency gains. Audited 270 rollout records across three 30-trial pilots and two 90-trial expansions, asset/checkpoint hashes, complete method/case coverage and step accounting. Successful Panda pilot neural-0 case 5 (536 steps) and expanded PiPER neural-0 case 8 (728 steps) reproduce saved traces, phases, final states and success exactly in rendered replays. All 59 regression tests pass. The inspected comparison figure, complete failure breakdown and audit are in `outputs/ease_native_reliability_report_v1`; previews are in `ease_native_reliability_panda_preview_v1` and `ease_native_reliability_piper_preview_v1`. No reserved final-test evaluation occurred.

### 4.35 Panda grasp and release diagnostics (E42)

**Matched design.** Reused the nine E41 expanded Panda placements, identical homes, calibration seeds and frozen E35 checkpoints. A new diagnostic replay logs tool/object rotations, finger opening, native gripper actuator force and object velocity; all 45 baseline trials reproduce E41 traces, phases, step counts, final positions and success exactly. Native actuator gains, force limits, friction, collision models and final success criteria are unchanged. Each of four interventions receives all nine cases for privileged, analytical and three neural controllers. No policy-success filtering or final-test rollout occurs.

**Interventions.** Yaw adds tool-X displacement statistics and a third forward application of the same frozen motion Transformer, alongside position and tool-Z, targeting the nearest world cardinal X direction aligned with the initial cube faces. This experimentally extends the shared task controller to nine rows, without weight training. Hold keeps native arm actuator position targets fixed and pauses estimator updates during close/release/settle. Settle adds a stricter 2 mm / 0.03 direction-error pre-grasp gate to Hold. Neither is adopted as the recommended fix: gains for the privileged controller do not transfer consistently to the neural models.

Telemetry shows cubes sliding within the fingers after the phase-entry offset estimate, causing the fixed lowering target to continue toward the table. The final intervention retains the baseline motion controller and transitions from lower to release when measured cube-center height is 0.215–0.230 m and XY target error is below 20 mm. Release holds the current tool position. The output variant is named `supported`, but it does not sense table support: it permits a short drop of up to about 15 mm. Actual final settling, target accuracy, absence of forbidden contact, release and earlier grasp/lift criteria remain required. This is scripted object-state feedback, not learned grasp sequencing.

| Panda variant | Privileged /9 | Analytical /9 | Neural 0 /9 | Neural 1 /9 | Neural 2 /9 |
|---|---:|---:|---:|---:|---:|
| Exact E41 baseline | 3 | 3 | 4 | 0 | 2 |
| Face-aligned yaw | 6 | 2 | 0 | 0 | 3 |
| Contact-phase joint hold | 9 | 5 | 1 | 0 | 2 |
| Hold plus tighter settling | 5 | 3 | 1 | 0 | 0 |
| Object-height release | 9 | 8 | 7 | 2 | 4 |

**Attribution and limits.** All calibration traces match across interventions. The object-height release run also matches every baseline pre-lowering trajectory exactly, isolating its later intervention on these observed tasks. It improves every controller's success count without changing grasp/lift behavior. Final forbidden-contact counts are 0/0/0/3/1 in table controller order. Analytical has one remaining lift-grasp loss. Neural seed 0 has one lift loss and one close timeout; seed 1 has three earlier contact failures, two lift losses and two lift timeouts; seed 2 has one earlier contact failure, three lift losses and one close timeout. Thus the release failure is substantially reduced, but robust Panda grasp acquisition and retention are unresolved, and analytical still outperforms every neural seed. Sequential tuning on the same nine cases does not constitute independent confirmation.

**Verification and reproducibility.** Audited 225 trials, including the 45 exact baseline replays, all asset/checkpoint hashes, complete case/method coverage and probe/task/telemetry step accounting. Final mean total steps are privileged 449.3, analytical 447.6, neural 391.2/330.9/349.9; early termination prevents interpreting lower means as efficiency gains. All 61 regression tests pass, including full-orientation finite-difference Jacobians and measured-motion recovery. A recovered neural-0 case 9 replays exactly in 445 steps, with rendered stages inspected in `outputs/ease_panda_grasp_preview_v1`. Full diagnostics, negative variants and audit are in `outputs/ease_panda_grasp_report_v1`. The new `panda_grasp` runner defaults to object-height release; earlier runners retain their original defaults, and PiPER has no new rollout claim in E42.

### 4.36 Motion-estimation diagnostics and synthetic long-history fine tuning (E43–E44)

**Prediction diagnostics.** E43 exactly replays 36 E42 object-height-release trials (analytical and three neural models), logging predictions before each executed motion. A separate analytical fit to the same measured Gram/cross statistics predicts that same next displacement for comparison; this shadow estimate never supplies neural actions. Simulator Jacobian access is blocked. Trial-equal mean XYZ prediction errors for neural seeds 0/1/2 are 0.600/1.085/0.671 mm versus shadow 0.216/0.484/0.214 mm. Neural tool-direction prediction errors are 3.75/3.36/3.52 times their shadow counterparts. These measurements concern the executed motion directions, not full-Jacobian correctness or proof of a specific failure mechanism. They identify a plausible estimation bottleneck, especially for seed 1.

Two frozen inference ablations receive 27 Panda trials each. Rescaling feeds unscaled direction cross statistics into the same Transformer, then multiplies predicted direction Jacobians by 0.15, keeping the task controller's units unchanged. It produces 5/3/4 successes out of nine versus baseline 7/2/4. Smoothing uses 0.8 previous plus 0.2 newly predicted Jacobian and gives 8/3/3. Neither consistently improves all seeds; neither is adopted as a general fix. Neural actions remain learned-Jacobian control, without analytical fallback.

**Fixed synthetic training.** E44 starts from each corresponding E35 checkpoint and runs exactly 2,000 AdamW updates at learning rate 0.00005, batch 128, with gradient clipping 1. Updates alternate the original E35 synthetic noisy short-history batch with new synthetic 33–256-transition histories: a 32-step full-rank probe followed by rank-1/2/3 joint motion. Joint-motion amplitudes range 0.0002–0.012 rad; synthetic Jacobian scales range 0.03–0.8. Targets are clean weighted ridge motion estimates with the same 0.98 decay and 1e-6 prior. New batches are mostly noiseless, with occasional position-noise scales 0.0001/0.0005. The E35 half uses only the original training-split excitation profiles. No PiPER/Panda rollout, native geometry, target task or reserved final-test data enters training. Model architecture and parameter count are unchanged; final checkpoints are evaluated without best-seed or best-step selection.

On one fixed, separate 1,024-example synthetic validation batch, MSE changes from 0.012794 to 0.010872, 0.013724 to 0.010596, and 0.012624 to 0.011046 for seeds 0/1/2. This validates the synthetic objective only, not downstream reliability.

**Native gates.** E44 runs 54 manipulation trials with the same per-robot task/controller settings: E41 object-offset placement for PiPER and E42 object-height release for Panda. Calibration records match their earlier references exactly. It also runs the 120-trial E39 native reaching gate, including unchanged privileged and analytical references.

| Task | Earlier neural seeds 0/1/2 | Fine-tuned seeds 0/1/2 | Analytical reference |
|---|---|---|---|
| PiPER manipulation, /9 | 9 / 7 / 8 | 8 / 9 / 6 | 9 |
| Panda manipulation, /9 | 7 / 2 / 4 | 7 / 5 / 6 | 8 |
| PiPER reaching, /12 | 9 / 9 / 11 | 10 / 11 / 11 | 10 |
| Panda reaching, /12 | 12 / 12 / 12 | 12 / 12 / 12 | 12 |

The weaker Panda seeds improve, but PiPER manipulation regresses in seeds 0 and 2. Seed 1 improves both arms (PiPER 7 to 9, Panda 2 to 5), which is a descriptive per-seed result rather than a basis for selecting and reporting only that seed. Fine-tuned PiPER forbidden contacts are 0/0/1; Panda 0/1/0. Remaining failures include missing grasps, lost grasps, lift/close timeouts and one inaccurate Panda placement. Keep the E35 checkpoints as the established reference and E44 as experimental candidates. The original procedural training-control suite has not been rerun for E44; native reaching retention must not be presented as complete regression coverage or broad transfer.

**Verification.** Audited 264 rollout records: 90 E43, 54 E44 manipulation and 120 E44 reaching. Verified checkpoint ancestry/current hashes, training-manifest hash, fixed step counts, finite losses, method/case coverage, step accounting and matched calibration traces. The reused completed reaching protocol has a stale prose label saying E35 weights, but its checkpoint paths/hashes correctly identify E44; the original protocol is preserved and future runner wording corrected. All 61 regression tests pass. A recovered Panda neural-1 case 10 reproduces trace, phases, success and final state exactly in 377 steps; inspected rendered stages are in `outputs/ease_motion_lowrank_preview_v1`. Full diagnostic results and provenance are in `outputs/ease_motion_adaptation_report_v1`. These repeatedly observed native development tasks remain separate from the reserved final test.

### 4.37 Retention-regularized training and fresh native placements (E45–E46)

**Training intervention.** E45 starts again from the three original E35 checkpoints, not E44. It retains the alternating short-/long-history synthetic batches and fixed 2,000-update budget per seed, lowers AdamW learning rate to 0.00002, and penalizes prediction drift from the corresponding frozen E35 teacher. The teacher-prediction penalty has coefficient 2 for six joints and 0.5 for seven; the clean ridge-target loss remains. All examples are synthetic or use the original training-split excitation profiles. No native robot trajectory or geometry enters training. Architecture is unchanged. This is a combined lower-learning-rate/regularization experiment, not an isolated causal test of the penalty.

Fixed synthetic validation MSE changes from 0.012794/0.013724/0.012624 to 0.011503/0.011771/0.011546. All three final candidates pass the original 256-task procedural training-control gate without simulator Jacobian access or neural inverse solves: 768/768 total. This closes the original-training retention check for E45 only; it does not establish native task retention or retroactively validate E44.

**Observed native retention.** Fixed final checkpoints receive the same 54 manipulation and 120 reaching trials used for the earlier native gates.

| Task | Original E35 seeds 0/1/2 | E45 seeds 0/1/2 |
|---|---|---|
| PiPER manipulation, /9 | 9 / 7 / 8 | 8 / 9 / 9 |
| Panda manipulation, /9 | 7 / 2 / 4 | 6 / 3 / 5 |
| PiPER reaching, /12 | 9 / 9 / 11 | 8 / 9 / 9 |
| Panda reaching, /12 | 12 / 12 / 12 | 11 / 12 / 12 |

The penalty helps some manipulation seeds but does not guarantee retention: PiPER manipulation seed 0 and reaching seeds 0/2 regress, as does Panda reaching seed 0. Retain E35 as the reference and do not promote E45 as a replacement.

**Fresh-placement protocol.** Before candidate task results, E46 fixes six new placements per robot: PiPER X = 0.21/0.24/0.27 m and Panda X = 0.32/0.37/0.42 m, crossed with Y = -0.02/0.05 m. Targets transfer +0.04 m in Y from the negative start and -0.04 m from the positive start. IK seed is 45000; all twelve candidates pass all nine nominal endpoint checks. Every accepted case remains regardless of controller outcome. Dynamic object-offset targets are not covered by nominal endpoint checks. Homes and the same 32-step calibration are shared across eight methods: privileged, analytical, all three original seeds and all three fixed candidate seeds. Native mechanics and per-robot task rules remain unchanged. The protocol explicitly permits completing the fresh diagnostic comparison even if the observed retention gate fails; no fresh-task tuning or checkpoint selection occurs.

| Fresh tasks, /6 | Privileged | Analytical | Original seeds 0/1/2 | E45 seeds 0/1/2 |
|---|---:|---:|---|---|
| PiPER | 6 | 6 | 6 / 6 / 6 | 6 / 6 / 5 |
| Panda | 6 | 6 | 5 / 1 / 5 | 6 / 3 / 5 |

The candidate improves fresh Panda seeds 0/1 and retains seed 2, but loses one fresh PiPER grasp in seed 2. Every candidate fresh rollout is free of forbidden contacts, though seed 1 Panda still loses grasps or times out in settling. Both references solve every fresh case. These are new placements on the same two robot models, not new embodiments or the reserved final test. The small paired set supports limited fresh-task evidence, not broad reliability or learned superiority. Improvements here do not erase earlier retention failures.

**Verification and decision.** Audited 1,038 rollout records: 768 original training-control, 54 observed native manipulation, 120 native reaching and 96 fresh comparisons. Asset/checkpoint hashes, ancestry, fixed update counts, training manifest, complete case/method coverage and identical fresh calibration traces are verified. All 61 regression tests pass. Fresh Panda candidate seed-0 case 9, a recovered case relative to the corresponding original seed, reproduces trace, phases, success and final state exactly in 424 steps; rendered lowering was inspected. Artifacts are in `outputs/ease_retention_report_v1` and `outputs/ease_retention_fresh_preview_v1`. The report's `decision.json` records failed six-joint retention and no promotion. Original E35 and both candidate generations remain available; no reserved final-test evaluation occurred.

### 4.38 Local pre-grasp calibration refresh (E47)

**Hypothesis and fixed protocol.** Additional measured excitation near the grasp might improve local motion identification without retraining. Use the original E35 checkpoints and all twelve E46 placements, now explicitly observed development data. Four conditions cover five controllers and twelve cases, for 240 trials: exact historical baseline, waiting, probing with updates, and identical probing with updates frozen. All receive the same initial 32-step calibration. Each intervention adds a shared gated hover 4 cm above the nominal grasp height, followed by up to 32 additional physical steps (1.28 s at 25 Hz). The probe has eight home-relative joint targets drawn uniformly within ±0.006 rad of the measured hover pose, four steps each, with the gripper open. The wait holds that measured joint pose. There is no simulator reset after initial calibration and no weight update. Probe and wait retain the normal decaying statistics and actuator-bias updates; frozen probe freezes both. All actual steps and forbidden contacts count, including hover, calibration and failures. Stop early on forbidden contact. No case is filtered by policy success.

| Condition | PiPER neural seeds 0 / 1 / 2 (each /6) | Panda neural seeds 0 / 1 / 2 (each /6) | Analytical PiPER / Panda (each /6) |
|---|---|---|---|
| Original baseline | 6 / 6 / 6 | 5 / 1 / 5 | 6 / 6 |
| Hover + waiting | 6 / 5 / 6 | 3 / 0 / 4 | 6 / 5 |
| Hover + probe + updates | 6 / 6 / 6 | 4 / 1 / 2 | 6 / 6 |
| Hover + identical probe, updates frozen | 6 / 6 / 6 | 3 / 1 / 4 | 6 / 4 |

The privileged controller solves all twelve tasks in every condition. Its probe and frozen-probe trajectories match exactly, as expected because estimator state is unused. PiPER neural probing preserves all six successes per seed, at mean total costs 772.7/797.7/755.0 steps versus baseline 736.3/759.8/721.7. Panda probing does not preserve reliability: seed 0 loses case 7; seed 1 gains case 7 but loses case 10, leaving its total unchanged; seed 2 loses cases 6, 9 and 11. Neural Panda forbidden-contact counts are baseline 0/1/1, wait 0/3/1, probe 1/1/0 and frozen probe 1/1/0. Lower total steps on some failed conditions reflect early termination, not improved efficiency.

**Interpretation.** On identical physical probe trajectories, enabling updates improves analytical Panda success from 4/6 to 6/6. Neural effects are mixed: seed 0 improves 3 to 4, seed 1 exchanges a gain and a loss, and seed 2 drops 4 to 2. Compared with matched waiting, probe updates improve neural Panda seeds 0/1 by one task each but lose two seed-2 tasks. More local data alone is therefore not a reliable fix for this estimator/controller combination. The historical baseline lacks the hover waypoint, so baseline comparisons combine waypoint and refresh effects. The frozen-update intervention changes both motion statistics and actuator-bias adaptation; it does not isolate the Transformer estimate alone. All three seeds share the same six placements per robot, so they are not independent task samples. These remain scripted state-based simulation diagnostics, not new-embodiment, visual-policy or hardware validation.

**Verification and decision.** All 60 baseline replays reproduce the E46 original/reference traces, initial calibration, outcomes and step counts exactly. All 60 intervention groups have identical pre-refresh trajectories and identical physical probe versus frozen-probe traces. Checkpoint and case hashes are verified, native mechanics are unchanged, and Jacobian/inverse guards remain active. All 63 regression tests pass, including fixed refresh budget, matched excitation, frozen statistics/bias and collision stopping. A newly failed Panda seed-2 case 6 reproduces exactly in 277 steps; the rendered transport frame shows the dropped cube. Raw traces, full success/collision/cost tables, paired gains/losses, audit and no-promotion decision are in `outputs/ease_local_refresh_v1`; the exact failure replay is in `outputs/ease_local_refresh_preview_v1`. Retain the original E35 controller and all failed interventions. The reserved final test remains untouched.

### 4.39 Isolating motion-statistic and actuator-bias updates (E48)

**Question and fixed design.** E47 conflated two adaptations: updates to the Gram/cross statistics used to estimate local motion, and the joint-command tracking correction called actuator bias. E48 crosses both switches during the same 32-step refresh: both, motion only, bias only, and neither. The bias is a moving average, `b <- .8 b + .2 (command - measured dq)`, not a measurement of physical torque or a separately identified gravity parameter. Freezing motion holds both Gram and cross matrices, including their decay and ridge additions; freezing bias holds its pre-refresh value. Normal updates resume immediately after refresh. This is a finite-interval intervention, not permanently disabling either mechanism.

All twelve now-observed E46 placements, original E35 weights, E47 hover/probe amplitudes and random seeds, native mechanics, initial calibration, phase rules and physical-step accounting are fixed. Run four conditions for the analytical controller and all three neural seeds: 192 new trials. The 96 both/neither runs replay the corresponding E47 controls exactly. Privileged E47 references are reused explicitly because neither estimator component enters privileged control. No model training, native-data fitting, parameter tuning or final-test evaluation occurs.

| Updates during refresh | PiPER neural seeds 0 / 1 / 2 (each /6) | Panda neural seeds 0 / 1 / 2 (each /6) | Analytical PiPER / Panda (each /6) |
|---|---|---|---|
| Both | 6 / 6 / 6 | 4 / 1 / 2 | 6 / 6 |
| Motion statistics only | 5 / 6 / 6 | 3 / 2 / 5 | 6 / 6 |
| Actuator bias only | 6 / 5 / 6 | 4 / 2 / 5 | 6 / 6 |
| Neither | 6 / 6 / 6 | 3 / 1 / 4 | 6 / 4 |

**Paired task effects.** For Panda seed 2, either isolated update gives 5/6, while both give 2/6. Relative to motion only, enabling bias loses cases 6, 7 and 11; relative to bias only, enabling motion updates loses cases 6, 9 and 11. The two isolated conditions recover different cases, despite equal totals. Seed 1 also has lower success with both than with either alone. However, motion only loses a PiPER seed-0 case through forbidden contact, while bias only loses a PiPER seed-1 case through transport grasp loss. Panda seed 0 scores lower with motion only than with both. No common switch setting reliably improves all seeds and both robots, and selecting a different setting after observing each seed would not establish an adaptive policy. The original no-hover reference remains PiPER 6/6 per seed and Panda 5/1/5; it is not replaced.

**Matched prediction diagnostics.** Before incorporating each probe transition, predict its measured pose change as `J(previous statistics) @ measured dq`. Log XYZ error and error in the `.15 * tool-Z` coordinates separately; also log `command - previous bias - measured dq`. These are diagnostic conditional motion residuals, not predictions of joint motion from the action. Actual probe motion is identical across the four conditions within each case/controller group. Bias does not affect the motion estimate on that fixed trajectory: both and motion-only errors match exactly, as do bias-only and neither. Across different controllers, hover trajectories can differ, so cross-controller error comparisons are descriptive rather than matched counterfactuals.

| Panda controller | XYZ error, frozen → updated (mm) | Scaled-direction error ×1000, frozen → updated | Bias residual norm, frozen → updated (rad) |
|---|---|---|---|
| Analytical | 0.1274 → 0.0763 | 0.0428 → 0.0260 | 0.02386 → 0.00645 |
| Neural seed 0 | 0.5580 → 0.4940 | 0.1421 → 0.1264 | 0.02101 → 0.00611 |
| Neural seed 1 | 0.5050 → 0.4706 | 0.1553 → 0.1470 | 0.03196 → 0.00761 |
| Neural seed 2 | 0.5444 → 0.4692 | 0.1460 → 0.1287 | 0.02043 → 0.00600 |

Values are mean Euclidean residual norms over probe transitions. Scaled-direction units are neither millimetres of physical displacement nor angles. All corresponding PiPER mean residuals also improve when their component is updated; full values, collisions and total step costs are retained in the experiment report. Better average probe residuals do not imply more reliable subsequent grasping. E48 therefore does not support a simple explanation that extra local data worsens average motion estimation. It instead motivates inspecting how the refreshed Jacobian and tracking correction combine when Cartesian task control resumes. It does not yet identify the precise failure mechanism, and later closed-loop trajectories diverge after the matched intervention.

**Verification and decision.** Audited all 192 records, including 96 exact E47 trajectory/phase/outcome/step replays and 48 four-condition groups with identical calibration and physical trajectories through refresh. Every enabled state component matches its both-update counterpart; every frozen component equals its pre-refresh state. Checkpoint, case and reference-record hashes are unchanged; Jacobian and neural-inverse guards remain active. All physical steps count. All 64 tests pass, including selective-state preservation and resumption of normal updates after refresh. Complete records, state snapshots, residuals, paired gains/losses, costs, failures, source snapshots, audit and no-promotion decision are in `outputs/ease_split_refresh_v1`. Preserve every condition and the original reference; these six placements per robot and shared tasks across seeds remain development diagnostics, not independent reliability evidence.

### 4.40 Probe-to-descent command decomposition (E49)

**Protocol.** Replay every E48 condition/controller/case, stopping after the first sixteen descent steps (0.64 s). These are 192 short diagnostic replays, not new complete task evaluations. Outcomes are explicitly inherited from E48. Retain the original calibration, refresh states, controller and task sequence. An optional observer logs the actual damped-Jacobian command, its bias correction, raw sum, clipped action, executed command, measured joint motion, Jacobian singular values, prediction residuals and pose response. The first command jump is the Euclidean difference between the first bounded descent action and the last executed probe command. Later logged differences to the same probe command are not step-to-step discontinuities. No action is altered, and no tuning or training occurs.

**Results for the strongest E48 interaction.** Panda seed 2 has the following values; command norms are in radians and clipping counts cover six cases × sixteen descent steps:

| Refresh updates | Inherited E48 success /6 | Mean first motion norm | Mean first bias norm | Mean first command jump | First clipped cases /6 | Clipped steps /96 |
|---|---|---|---|---|---|---|
| Both | 2 | 0.01394 | 0.00658 | 0.01415 | 0 | 26 |
| Motion only | 5 | 0.01394 | 0.01912 | 0.03119 | 0 | 27 |
| Bias only | 5 | 0.01283 | 0.00658 | 0.01285 | 0 | 32 |
| Neither | 4 | 0.01283 | 0.01912 | 0.02884 | 0 | 36 |

The both-update failure pattern cannot be explained simply by clipping on the first descent action: none of these seed-2 actions clips. Its mean initial command jump is less than half the motion-only value, while their early clipping totals are similar. Bias-only succeeds more often despite more clipped descent steps. These descriptive comparisons do not establish the absence of a handoff effect, but do not justify assuming that reducing jump magnitude or clipping alone will fix the task. Mean early-descent XYZ residuals are 0.5758/0.4954/0.8149/0.6953 mm for both/motion-only/bias-only/neither; these follow diverging trajectories, so they are not matched-motion estimates like E48's probe residuals.

**Other controllers and limits.** PiPER frozen-bias conditions have first-step clipping in all six cases for every controller, yet many still achieve 6/6; updated-bias conditions have none on their first step. Panda seed 1 has substantially more clipping and larger first motion terms than seed 2, but this is not a demonstrated explanation of its failures. Full per-controller tables and vectors are retained. E49 observes only the first sixteen descent steps, so it does not locate later grasp loss, contact forces or yaw alignment errors. No new successful intervention, generalization improvement or controller promotion is claimed.

**Verification.** All 192 prefixes, calibration traces and refresh states match E48 exactly, and all include sixteen observed descent steps. The decomposed terms reconstruct every bounded action exactly. All actual simulated steps count; checkpoint/reference hashes are unchanged and privileged-Jacobian access remains blocked. All 64 regression tests pass. Artifacts, source snapshots, vectors, summaries and audit are in `outputs/ease_handoff_diagnostic_v1`. Existing E48 outcomes remain the source of all success labels.

### 4.41 Contact failure localization and full angular control (E50–E51)

E50 adds observation-only telemetry to 48 complete original E35 episodes: analytical and three neural seeds on all twelve E46 placements. Full trajectories, outcomes and initial probes reproduce exactly. Logged measurements include current finger/cube normal forces, opening, object position in the tool frame, tool rotation, cube-symmetric yaw mismatch and command components. These measurements do not enter the policy.

Panda neural seed 0, case 9 closes at about 42.8 degrees yaw mismatch with approximately 0.99 N normal force per finger, then loses the cube during transport. However, analytical control succeeds at essentially the same yaw. Seed 1 loses cases 7–9 during lift despite initially aligned, centered grips; maximum tool-direction errors are approximately 0.250, 0.290 and 0.155, and the cube returns to the table. Thus these are physical drops, not merely overly strict contact labels, and corner alignment alone does not explain them. Contact force snapshots are descriptive, not causal proof.

E51 tests measured three-dimensional rotation-vector increments, scaled by 0.15, alongside Cartesian displacement. It retains the two frozen model calls, common Gram statistics, actuator-bias update and controller gains. The full orientation target either preserves initial upright yaw (hold) or selects the nearest world-cardinal cube-face orientation (align). All twelve original homes remain, without new feasibility filtering; all 120 calibrations exactly match the original 32-step probes. Physics, final physical success and phase budgets are unchanged. The intermediate gate now requires full angular error below 0.08 radians. Consequently comparisons against baseline change both representation and controlled objective; hold versus align changes only the target.

| Condition | Robot | Privileged /6 | Analytical /6 | Neural seeds 0 / 1 / 2, each /6 |
|---|---|---:|---:|---|
| Original | PiPER | 6 | 6 | 6 / 6 / 6 |
| Original | Panda | 6 | 6 | 5 / 1 / 5 |
| Hold initial yaw | PiPER | 6 | 6 | 5 / 2 / 5 |
| Hold initial yaw | Panda | 6 | 5 | 3 / 3 / 2 |
| Face alignment | PiPER | 6 | 6 | 5 / 6 / 6 |
| Face alignment | Panda | 6 | 6 | 3 / 1 / 3 |

Neither candidate passes the per-robot/per-seed success and collision retention screen. Keep E35 unchanged. This negative result does not establish that orientation control is intrinsically harmful: the frozen estimator was not retrained for angular increments. It does rule out adopting this particular controller-only intervention. All failures, paired gains/losses and interaction costs are retained. All 66 tests pass, including native finite-difference angular-Jacobian and rotation-target checks; checkpoint hashes are unchanged. These are observed development placements, not independent generalization evidence.

Artifacts: [E50 contact report](../../outputs/ease_contact_diagnostic_v1/report.md), [E51 control report](../../outputs/ease_angular_control_v1/report.md), with source snapshots, trial traces, audits and explicit no-promotion decision. The 168 full rollouts comprise 48 exact baseline replays and 120 angular-control trials.

### 4.42 Matched control-aware estimator training (E52)

This experiment tests a training-objective intervention with fixed architecture and inference controller. Each of three E35 checkpoints initializes two candidates trained for 2,000 updates, batch size 128, AdamW learning rate 2e-5. Both receive exactly the same generated batches and frozen-parent anchor: six-joint weight 2, seven-joint weight 0.5. The prediction arm minimizes clean-ridge Jacobian MSE plus this anchor; the control arm adds coefficient 0.1 times a local task-response loss. Final checkpoints are evaluated without best-step or best-seed selection.

Synthetic statistics jointly represent Cartesian and scaled tool-Z motion, using 1–256 transitions, a full-rank first 32 steps followed by rank-1/2/3 motion, 0.98 decay, 1e-6 ridge and mostly clean observations. Tool-Z Jacobians lie in a randomly oriented tangent plane. No native robot data or geometry enters training. For a predicted joint-by-task matrix P, the differentiable training operator is P(PᵀP + 0.015²I)⁻¹. The loss compares the true synthetic Jacobian's achieved response under that operator against its own damped controller. Matrix solves occur only in this training loss; neural inference remains inverse-guarded. The same original tool-Z controller, not E51 angular control, is used for native evaluation.

| Condition | Original training successes, seeds 0/1/2 (each /256) | PiPER seeds 0/1/2 (each /6) | Panda seeds 0/1/2 (each /6) | Native contact failures /36 |
|---|---|---|---|---:|
| E35 reference | 256 / 256 / 256 | 6 / 6 / 6 | 5 / 1 / 5 | 2 |
| Prediction-only | 256 / 256 / 256 | 5 / 6 / 5 | 3 / 3 / 4 | 1 |
| Control-aware | 255 / 256 / 256 | 0 / 0 / 1 | 1 / 0 / 1 | 28 |

The control-aware training-gate failure is six-joint body 84001, goal 13: no collision, but final error 0.09786 m after 150 steps. All candidates are still evaluated on the prespecified development tasks as diagnostics, with this failed gate retained. Native control-aware succeeds 3/36 versus prediction-only 26/36 and original 29/36. It loses 23 paired successes and gains none relative to prediction-only. Analytical remains 6/6 on each robot. Reference native scores reuse the E46 records already exactly replayed in E50; they are not independent new observations.

On fixed synthetic validation, response losses change from 5.375/5.303/5.734 to 1.503/1.529/1.776 for control-aware, versus 4.136/4.175/4.267 for prediction-only. However, control-aware Jacobian MSE rises to approximately 0.023 versus prediction-only 0.0056. A post-training diagnostic shows the mean achieved-response trace shifts below the ideal: six-joint 3.645 versus ideal 4.937, seven-joint 3.908 versus ideal 4.971. Prediction-only traces are 9.115 and 9.329. This supports a substantial response-strength change, not pure identification improvement; it does not establish the causal mechanism for native collisions.

**Decision:** reject both candidates as replacements and preserve E35. The control-aware objective is an unclipped local surrogate with constant synthetic Jacobians. It omits actuator-bias compensation, action saturation, changing kinematics and contact. This particular loss/weight/distribution fails badly; the experiment does not reject all control-aware learning. Native mean steps are 546.6 for prediction-only and 166.2 for control-aware; the latter reflects early failures and is not an efficiency advantage. Avoid tuning this weight against the same native outcomes without a separately versioned hypothesis and protocol.

Audits verify paired batch hashes, unchanged parents, six checkpoint hashes, all 72 calibration matches and all 1,536 training-gate records. All 68 tests pass. The final test remains untouched. Full [experiment report](../../outputs/ease_control_training_v1/report.md), paired gains/losses, fixed validation diagnostics, training traces and source snapshots are in `outputs/ease_control_training_v1`; native trial traces and checkpoints audit are in `outputs/ease_control_training_native_v1`.

### 4.43 Fixed-history action-ranking audit (E53)

E53 tests whether the training surrogate ranks candidate actions consistently with measured simulator outcomes. It performs no optimization. At the first and twentieth step of each reached phase on all twelve analytical E46 trajectories, each candidate receives identical measured Gram/cross/bias statistics. Candidates are analytical and the three original, prediction-only and control-aware model seeds. There are 202 sampled states: 102 PiPER and 100 Panda. Each candidate runs five control steps from an exact `mj_copyData` clone while the current ramped target and aperture are held fixed. Online estimator/bias updates, actuator dynamics, clipping and joint limits remain active. The task sequencer does not advance during these branches.

Three diagnostic scores are computed with the instantaneous true geometric Jacobian: E52's full response-operator MSE; local unclipped target residual; and local residual using the actual joint-limited command minus measured bias. Privileged geometry is used only to score actions, never to generate candidate commands. The last proxy approximates actuator response and omits contact dynamics. Candidate error is the norm of Cartesian residual concatenated with 0.15 times tool-Z residual. Pairwise ordering agreement excludes differences within 1e-12, averages within each trajectory and then across trajectories; branches are dependent observations, not independent task trials.

| Robot | Horizon | Operator agreement | Local unclipped agreement | Local executed-command agreement |
|---|---:|---:|---:|---:|
| PiPER | 1 step | 45.0% | 65.9% | 73.4% |
| PiPER | 5 steps | 47.8% | 62.4% | 67.9% |
| Panda | 1 step | 47.3% | 73.4% | 74.0% |
| Panda | 5 steps | 46.6% | 58.6% | 58.6% |

These all-candidate comparisons are not driven solely by analytical estimation: neural-only one-step operator/executed agreement is 45.3%/73.1% on PiPER and 45.8%/74.7% on Panda. The explicit clipping/bias treatment adds a larger improvement on PiPER; most of Panda's one-step gain already comes from scoring the particular task action instead of all response directions. The operator metric favors the control-aware Panda model over its paired prediction-only model at 89/89/92 of 100 states, but measured five-step residual favors it at only 30/43/30 states. This demonstrates a mismatch between that metric and local action quality on these histories.

The local proxy is not yet a validated manipulation loss. Panda five-step agreement drops to 50.7% during transport, 41.9% during lowering and 46.9% during release. PiPER transport is 49.8%. No counterfactual has forbidden contact, despite 28 such failures in E52's independent candidate trajectories. Shared successful analytical histories and a five-step fixed target do not cover the candidate-specific history evolution or long-horizon failures. This audit therefore does not identify the cause of those collisions, establish a safe controller, or support a final-task performance gain.

All twelve full driver replays match original traces, phases, steps, probes, outcomes and final object poses exactly. All 202 repeated analytical branches match exactly, and all nine checkpoint hashes remain unchanged. The added optional pre-step observer leaves the default controller unchanged. Tests verify exception-safe simulator/flag restoration, the controller formula and case-weighted ranking statistics; all 71 tests pass. The 2,020 counterfactual branches plus 202 repeats are diagnostic probes, not new task-success evaluations. Detailed phase/candidate/contact tables and per-case rates are in the [E53 report](../../outputs/ease_action_objective_v1/report.md). No model is promoted and the final test is untouched.

### 4.44 Procedural training-arm score validation with model-generated histories (E54)

E54 extends the frozen score audit to all eight procedural training bodies: four six-joint and four seven-joint arms. Analytical and all nine original/E52 models each generate a separate cold-start history, giving 80 drivers. Each tracks the first and last existing training goal for its paired body, alternating every 50 steps with a 2 mm reference ramp. The run lasts 150 steps unless contact occurs; static-goal success termination is ignored because this is scheduled tracking. No native, validation or final-test body is used, and no optimization occurs.

Samples are fixed at steps 9, 29, 49, 50, 79, 99, 100, 129 and 149. At each state, ten candidates receive the driver's identical measured Gram/cross/bias statistics and branch for five steps, continuing the target schedule and updating their own statistics. All 80 drivers finish without contact, yielding all 720 planned states and 7,200 branches. All driver trajectories reproduce exactly in fresh runs without branching; all 720 matching-driver first-step branches equal the actual next driver step. Candidate histories are generated by their models, rather than imposed successful analytical histories as in E53.

The existing position-only controller uses gain 0.5, damping 0.015, command clip 0.04 and joint limits +/-2.5. Scores use the instantaneous geometric Jacobian for diagnostics only. Neural estimation remains inverse-guarded and receives no privileged geometry. This tests three-dimensional reaching, not six-dimensional tool-pose or gripper control. As in E53, pairwise score agreement is averaged within bodies then across bodies, excluding differences within 1e-12. The eight bodies are the relevant clusters; branches are not independent statistical trials.

| DoF | Horizon | Operator agreement | Local unclipped agreement | Local executed-command agreement |
|---|---:|---:|---:|---:|
| 6 | 1 step | 41.3% | 79.7% | 81.2% |
| 6 | 5 steps | 39.8% | 74.7% | 75.7% |
| 7 | 1 step | 40.4% | 78.7% | 81.0% |
| 7 | 5 steps | 39.0% | 71.1% | 72.5% |

Local executed-command agreement exceeds operator agreement for both DoFs and horizons within every driver group (analytical, original, prediction-only and control-aware). On the control-aware histories specifically, five-step agreement is 67.1% versus 33.3% for six joints and 69.3% versus 34.4% for seven. Around target switches, local five-step agreement remains 74.9% and 69.7%, versus operator 36.7% and 34.6%. Thus the E53 local-ranking advantage persists on model-generated training histories with continuing targets, rather than only on fixed successful native analytical histories.

Most of the gain comes from scoring the particular task action; explicit clipping/bias handling adds a smaller aggregate improvement here. First-command saturation occurs in 261/3,600 six-joint and 395/3,600 seven-joint branches. No driver or branch has contact. Joint-limit enforcement is active, but this was not a prescribed near-limit stress test. These data therefore do not validate contact safety or explain E52's native manipulation collisions. They also do not establish that a differentiable version of the score will train a better model: the score uses a privileged local Jacobian for evaluation, and its bias-to-motion approximation remains imperfect.

**Decision:** the action-specific scoring idea merits a separately specified training experiment for position tracking. Before extending a claim to manipulation, cover procedural tool-pose motion and contact-relevant states; do not treat this position-only result as validation of the full E52 task. Keep E35 as the reference. All 73 tests pass, all nine checkpoints are unchanged, and the final test remains untouched. The [E54 report](../../outputs/ease_procedural_action_v1/report.md) contains per-driver, per-body, saturation, target-switch and collision-free results; raw histories, source snapshots and audits are retained alongside it.

### 4.45 Procedural position/tool-direction action-score validation (E55)

E55 extends the audit to the task coordinates used by native manipulation: Cartesian position concatenated with 0.15-scaled tool-Z direction, with yaw unconstrained. Two endpoints per training body come from forward kinematics at home plus seeded uniform +/-0.15-radian joint offsets (seed 55000 + body ID). All sixteen endpoints are retained without controller-outcome or contact filtering. They are kinematically realizable endpoints; this does not guarantee that the ramped intermediate reference path is exactly realizable. Targets switch every 50 steps, ramping position by at most 2 mm and normalized tool direction by approximately 0.01 radians per step.

All eight training bodies and all ten separate analytical/original/prediction/control driver histories are retained, with cold-start statistics, gain 0.35, damping 0.015, action clip 0.04 and enforced joint limits. The frozen model is applied separately to position and direction statistics, exactly as in the native pose controller. Nine fixed sample steps per 150-step driver give 720 states; all ten candidates branch for five steps with continuing reference schedules and online updates. No native or held-out geometry enters this diagnostic and no optimization occurs.

| DoF | Horizon | Operator agreement | Local unclipped agreement | Local executed-command agreement |
|---|---:|---:|---:|---:|
| 6 | 1 step | 48.6% | 72.0% | 80.4% |
| 6 | 5 steps | 43.2% | 60.3% | 64.0% |
| 7 | 1 step | 44.5% | 74.5% | 78.6% |
| 7 | 5 steps | 42.3% | 64.8% | 66.9% |

The action-specific score exceeds operator agreement for both horizons and DoFs within every driver group. On control-aware model histories, local five-step agreement is 61.4% versus operator 40.0% for six joints and 66.6% versus 40.5% for seven. Around target switches, local five-step agreement is 61.8% and 67.0%. Pairwise statistics are averaged within each body and then across bodies, excluding numerical ties within 1e-12. These are dependent descriptive comparisons on eight training bodies, not independent task-success observations.

Clipping affects 1,544/3,600 six-joint first commands and 975/3,600 seven-joint first commands, more frequently than in E54. Local unclipped scores remain better than the operator score, and modeling execution constraints adds a further improvement. Five-step agreement remains substantially weaker than one-step agreement; local score validity is not equivalent to reliable long-horizon tracking. All drivers and branches are contact-free and the fixture has no object/gripper, so grasp and contact-safety conclusions remain unavailable.

Descriptive driver averages retain E52's regression: control-aware mean position error is 30.66 mm (six joints) and 22.04 mm (seven), versus prediction-only 8.65 and 6.89 mm. Mean tool-direction chord errors are 0.14958/0.10097 versus 0.03116/0.03113. These are scheduled tracking errors on known training bodies, not a newly trained-model result or a success gate. Prediction-only's favorable means here do not erase its previously failed native retention checks.

All 80 driver trajectories reproduce exactly without counterfactual branching, and all 720 matching-driver first steps equal their actual driver steps. There are 7,200 diagnostic branches and no missing planned samples. Tests verify finite-difference Cartesian/tool-direction Jacobians on six/seven joints, deterministic target generation, unit reference directions and state restoration; all 75 tests pass. Checkpoint hashes remain unchanged, and privileged Jacobians are diagnostic-only. The [E55 report](../../outputs/ease_procedural_pose_v1/report.md) retains per-driver, per-body, target-switch, saturation, tracking-error and collision-free details.

**Decision:** E53–E55 now provide a basis for a separately specified matched training experiment using an action-specific local objective. No claim is made that optimizing the score will improve closed-loop behavior. Freeze initialization, paired data, loss scaling and training budget before running; preserve an otherwise identical prediction-only arm and the original E35 reference. Require original reaching and six-joint retention, then assess pose/manipulation outcomes with all seeds and failures retained. No additional score sweep against the same native tasks is justified by these results alone.

### 4.46 Matched local-action objective training (E56)

E56 tests optimization of the local action score after E53–E55's ranking audits. Each of three E35 checkpoints initializes a prediction-only and local-action candidate, trained for 2,000 updates at learning rate 2e-5, batch 128. Both use the same clean-ridge MSE and frozen-parent prediction anchor (six-joint weight 2, seven-joint 0.5). The local arm adds a fixed coefficient 0.01 times normalized task-residual squared error after the predicted controller action. No coefficient sweep or best-seed/checkpoint selection is performed.

The local controller uses gain 0.35, damping 0.015, joint command clipping at 0.04 and limits +/-2.5. Estimated displacement subtracts sampled measured bias from the joint-limited command. Residual squared norm is divided by task-error squared norm plus 0.002² before averaging. Synthetic error is J*dq, with joint displacement sampled at standard deviation 0.02 and scaled by exp(U[-2.3,0.7]); bias is Gaussian with standard deviation 0.006 clipped at +/-0.025, and joint positions are uniform within limits. Action contexts use a separate deterministic RNG, preserving E52's exact synthetic state batches. All data are synthetic; there is no native geometry or measured native training data. The approximation still omits contact, changing Jacobians and exact actuator dynamics, and independently sampled bias/positions need not match real history distributions.

All three prediction-only checkpoints, batch hashes and complete fit/anchor loss traces reproduce E52 exactly. Its 768 original training-gate episodes and 36 native trials are therefore reused with checkpoint, source-record and case hashes checked. They are not new independent evaluations. All three local-action candidates receive 768 new original training-gate episodes and 36 new native trials under the unchanged original controller and calibration. Both arms pass 256/256 reaching tasks in every seed.

| Condition | PiPER seeds 0/1/2 (each /6) | Panda seeds 0/1/2 (each /6) | Total native success /36 | Forbidden-contact failures /36 |
|---|---|---|---:|---:|
| E35 original | 6 / 6 / 6 | 5 / 1 / 5 | 29 | 2 |
| Prediction-only | 5 / 6 / 5 | 3 / 3 / 4 | 26 | 1 |
| Local action | 5 / 5 / 5 | 4 / 1 / 5 | 25 | 5 |

Analytical remains 6/6 on both robots. The local-action arm gains six paired tasks and loses seven relative to prediction-only, with Panda aggregate success unchanged at 10/18 and PiPER reduced from 16/18 to 15/18. Relative to E35, every local-action seed loses one PiPER task. Neither candidate condition passes the per-seed/robot success and collision retention screen. The local candidates' remaining failures are two final-placement and four lost-grasp-during-lift cases. Mean total steps are 533.2 versus prediction-only 546.6; this is not evidence of efficiency given differing failure counts.

On fixed synthetic validation, normalized local losses are 1.0603/0.9522/0.9567 versus prediction-only 1.1260/1.2713/1.1690. Prediction MSE is slightly worse at 0.006274/0.006459/0.006501 versus 0.005577/0.005671/0.005676. Thus lower action loss occurs in all three seeds without an aggregate manipulation gain. E53–E55's better ranking on frozen measured histories is not sufficient to establish that optimizing this synthetic surrogate improves deployed control. Compared with E52, the loss formulation and its coefficient both change, so that cross-experiment difference is not a pure objective ablation; the primary matched contrast is within E56.

**Decision:** no promotion; retain E35. Avoid another coefficient sweep on the same native tasks. A next falsifiable test can examine whether coherent measured procedural histories improve training-context transfer, with a fixed dataset and otherwise matched prediction-only/local-action arms. That would remain procedural training only and must not be presented as an established remedy. Original reaching, six-joint retention and later independent validation remain required.

All parent hashes remain unchanged; paired state/context hashes match; reused baseline records and fresh calibration traces are audited. All 77 tests pass, including agreement between the differentiable objective and the NumPy clipped-controller calculation, finite gradients and deterministic masked action contexts. Full [E56 report](../../outputs/ease_local_action_training_v1/report.md), source snapshots, six checkpoints, losses, gate records and explicit no-promotion decision are retained. Fresh native trial traces are in `outputs/ease_local_action_native_v1`; final-test evaluation remains untouched.


### 4.47 Coherent procedural-history training (E57)

E57 fixes a measured-context dataset before training. All 80 E55 trajectories (eight procedural training bodies, ten drivers each) replay exactly, yielding every one of their 12,000 pre-action states without success filtering. Each example keeps measured Gram/cross statistics, joint mask, joint position, tracking bias and XYZ/scaled-tool-Z error from the same state. The simulator's current geometric Jacobian supplies privileged training labels only; it is blocked from driver control. Entire seed-2 driver trajectories are held out: 8,400 training and 3,600 validation states, with a fixed 512-state validation subset. Bodies and goals overlap between these partitions; this is trajectory validation, not unseen-embodiment validation.

Three E35 initializations each produce prediction-only and local-action candidates. Each shared batch contains 64 measured training states and 64 synthetic E56 contexts. Both objectives include Jacobian prediction and the frozen-parent anchor; the local arm adds the unchanged 0.01 normalized clipped, bias-aware task-residual objective. Each run uses 2,000 updates, batch 128 and learning rate 2e-5, with no coefficient sweep or checkpoint selection. Paired batch/context hashes match and parent hashes remain unchanged. Native data and geometry are excluded. Measured labels are current geometric Jacobians while synthetic fit labels remain clean-ridge estimates, so comparison with E56 changes both context and label distributions and cannot isolate coherence alone.

The local-action validation loss on coherent trajectories falls from 1.1746/1.3173/1.0236 to 0.6124/0.7409/0.6452, versus prediction-only 0.7461/0.7974/0.7705. Geometric-J prediction MSE is lower for prediction-only (0.008262/0.008262/0.008125) than local-action (0.009040/0.010360/0.009768). All six checkpoints pass 256/256 original reaching episodes, totaling 1,536 fresh gate rollouts. The 72 native manipulation evaluations are fresh because both conditions' checkpoints changed; earlier candidate evaluations are not reused.

Native manipulation outcomes across seeds are:

| Condition | PiPER successes per seed /6 | Panda successes per seed /6 | Total /36 | Forbidden contacts |
|---|---|---|---:|---:|
| Original E35 | 6 / 6 / 6 | 5 / 1 / 5 | 29 | 2 |
| Coherent prediction-only | 6 / 6 / 4 | 3 / 4 / 2 | 25 | 1 |
| Coherent local-action | 3 / 4 / 4 | 3 / 3 / 1 | 18 | 5 |

Local-action gains four cases but loses eleven versus its matched prediction-only arm. Its failures comprise five forbidden contacts, seven lift grasp losses, two transport grasp losses, two missing bilateral grasps and two close-phase timeouts. Prediction-only has four lift grasp losses, three transport grasp losses, two missing bilateral grasps, one contact and one final-placement failure. Mean interaction counts, including calibration, are 515.4 for prediction-only and 427.5 for local-action; earlier failures make the latter unsuitable as an efficiency improvement claim.

Neither condition passes retention. Keep E35; do not select a favorable seed or promote the lower offline loss as a control improvement. Coherent contexts with current geometric labels are insufficient under this fixed training recipe; the result does not rule out other data coverage, objectives or architectures. The twelve placements are repeatedly observed development cases, and original/analytical reference scores reuse audited E46 records. Analytical control solves all six cases on each robot. No independent generalization or learned-baseline advantage is established.

All 79 regression tests pass. The report audit verifies six immutable candidate hashes, paired batches/contexts, unchanged parents, 80 exact data replays, disjoint trajectory partitions, all 1,536 fresh reaching records, all 72 fresh native trial records and matched calibration. The [E57 report](../../outputs/ease_coherent_training_v1/report.md), explicit no-promotion decision, checkpoints, losses and source snapshots are retained. Dataset: `outputs/ease_coherent_pose_data_v1`; native traces: `outputs/ease_coherent_native_v1`. Final test remains untouched.
### 4.48 Predicted versus actual procedural motion (E58)

E58 holds measured histories and simulator states fixed. The analytical and three original E35 drivers replay 32 complete E55 trajectories exactly on eight training bodies. Nine prespecified sample times per trajectory yield 288 states; each is branched with analytical, original-three, E57 prediction-three and E57 local-action-three estimators. All 2,880 branches run five steps with unchanged clipping, limits, online updates and moving reference. All 288 matching-driver first steps match the unbranched trajectory exactly. Checkpoints remain unchanged; no native data, fitting or final-test evaluation occurs. These histories overlap E57 data and are mechanism diagnostics, not independent validation.

For each first step, predicted displacement minus actual displacement decomposes exactly into `(estimated_J - true_J) * predicted_dq`, `true_J * (predicted_dq - actual_dq)`, and `true_J * actual_dq - actual_dx`. The three terms isolate estimator error, actuator/bias response mismatch and local geometric nonlinearity. Predicted joint displacement is the clipped, limited command minus measured tracking bias. Privileged geometry is used only for diagnostic scoring and is blocked from candidate control. Vector terms can cancel; squared magnitudes are not additive.

| DoF | Horizon | Estimated-J score agreement | True-J/predicted-motion agreement | True-J/realized-motion agreement |
|---|---:|---:|---:|---:|
| Six | 1 | 58.5% | 78.2% | 99.4% |
| Six | 5 | 56.9% | 63.4% | 63.8% |
| Seven | 1 | 51.8% | 76.4% | 99.5% |
| Seven | 5 | 51.0% | 66.2% | 68.8% |

Agreement is averaged within body then across bodies, excludes numerical ties at 1e-12, and does not treat dependent branches as independent trials. The realized-motion score is a post-hoc oracle unavailable before execution. Its near-perfect first-step ranking supports actuator/bias mismatch as a local surrogate limitation. It does not establish a usable response predictor or explain native failures. Five-step ordering remains imperfect because the reference, estimates and actions evolve beyond the scored first step.

Across estimator groups, actuator/bias scaled-displacement RMS ranges from 0.0061–0.0098 for six joints and 0.0042–0.0108 for seven, whereas nonlinear-geometry RMS is 0.000107–0.000160 and 0.000039–0.000061 respectively. Estimated-J errors are also substantial and can cancel response mismatch. Full per-group decomposition is retained in the experiment report.

Local-action beats its paired prediction-only candidate on actual first-step residual in 79/88/79 of 144 six-joint states and 95/111/107 seven-joint states; at five steps the counts are 78/90/90 and 99/104/103. Thus a majority of local comparisons favor local-action for every seed at both horizons, despite its worse E57 native result. These win counts do not establish an average error-magnitude or long-horizon task advantage. All 2,880 branches are contact-free; no conclusion about native grasp/contact mechanisms follows.

The initial v1 stopped on an overly strict float64-versus-float32 command-log assertion (maximum observed discrepancy 7.4e-10); its source and failure status are preserved. Corrected v2 requires exact command equality at recorded float32 precision while keeping physical-motion calculations at full precision. All 81 tests pass, including decomposition cancellation and rectangular six-/seven-joint identities. Audited all saved trial records, 2,880 decomposition identities, source-driver and checkpoint hashes. [Full report](../../outputs/ease_motion_gap_v2/report.md), source snapshots and records are retained. Keep E35. Next, evaluate a procedural joint-response predictor on whole held-out bodies/trajectories before incorporating it into policy training; native data stays diagnostic-only.

### 4.49 Held-out-body joint-response prediction (E59)

E59 collects all 150 transitions from each of 32 exact E55 analytical/E35 driver replays on eight procedural training bodies: 4,800 transitions and 31,200 joint observations. A single shared per-joint linear ridge model predicts a residual correction to command-minus-bias. Inputs are the current limited command, pre-action joint velocity and position, sine/cosine of position, current measured bias, previous command and previous realized displacement. Inputs exclude body/joint identity, geometry, actuator parameters, privileged Jacobians and future states. Initial history values are zero. The command/velocity residual ridge provides a simpler fitted comparison.

Eight leave-one-body-out folds exclude every trajectory of the omitted body from fitting and normalization. The analytical/original_0/original_1 trajectories of the seven other bodies form the fit partition; their original_2 trajectories are separately held out for trajectory validation. The ridge coefficient is fixed at 0.001 with no tuning. All 16 fitted models are saved and independently reproduced from permitted rows. Upstream E35 drivers previously trained on these bodies; the body holdout applies specifically to the response predictor. This is not a novel axis-family or benchmark final-test evaluation.

| DoF | Mean body RMSE: bias baseline | Command/velocity | Full measured-feature predictor | Reduction versus bias baseline |
|---|---:|---:|---:|---:|
| Six | 0.005013 rad | 0.003439 rad | 0.001044 rad | 79.2% |
| Seven | 0.005967 rad | 0.004567 rad | 0.000966 rad | 83.8% |

The full predictor improves all eight held-out bodies and all 32 body/driver strata. Per-body RMSE ranges from 0.000625 to 0.001677 rad; all raw predictions and trajectory-validation metrics are retained. The full model adds position and bias as well as history, so this comparison does not isolate history's causal contribution. Fold training sets overlap, and eight bodies provide few independent units; joint rows are not independent embodiment samples.

All source replays, immutable driver hashes, saved fit parameters, fit-only normalization, 32 body/trajectory scores and held-body predictions were audited. All 83 tests pass. One new test initially required bitwise equality across BLAS batch sizes and failed at 1.4e-17; its assertion now uses 1e-14 tolerance, with no changes to experiment code or fits. Sources, models and [full report](../../outputs/ease_joint_response_v1/report.md) are retained.

This is a positive one-step prediction result, not a closed-loop policy result. No native data, controller changes or policy training were used. Keep E35. Next test whether predicted response improves counterfactual action ranking on cloned held-out-body states before integrating it into action optimization or policy training. Correlated driver trajectories do not establish accuracy for unseen candidate commands or stability over longer horizons.

### 4.50 Held-out response counterfactual action ranking (E60)

E60 uses the sixteen immutable E59 response fits to score exactly the alternatives in E58. For each body, both response models were fitted and normalized without that body. The same analytical/E35 histories supply pre-action velocity, joint position, bias, previous command and previous measured joint displacement. The candidate command is substituted into the response input before its branch is executed. Response scores never influence candidate actions. All 32 driver trajectories, 288 matching-driver first steps and 2,880 full five-step branches reproduce E58 exactly; all original scores match. No model is fitted, no native data is used, and no policy/controller changes occur.

| DoF | Horizon | Bias + true J | Full response + true J | Bias + estimated J | Full response + estimated J |
|---|---:|---:|---:|---:|---:|
| Six | 1 | 78.24% | 88.59% | 58.52% | 57.78% |
| Six | 5 | 63.45% | 62.91% | 56.86% | 55.57% |
| Seven | 1 | 76.38% | 88.50% | 51.85% | 55.69% |
| Seven | 5 | 66.18% | 69.86% | 50.99% | 56.31% |

These are body-macro pairwise ranking agreements, excluding ties at 1e-12. True geometric Jacobians are diagnostic-only; candidate-specific estimated Jacobians represent the composed nonprivileged score. One-step true-J ranking improves on every body. The command/velocity-only response baseline scores 80.00%/71.88% at one step and 61.22%/60.54% at five for six/seven joints. Realized-motion oracle agreement remains 99.41%/99.48% at one step.

Counterfactual response RMSE falls from 0.009273 to 0.002178 rad for six joints and 0.008617 to 0.001445 rad for seven. This extends E59's prediction improvement to commands proposed by other frozen estimators. The full model includes pose/bias as well as history, so no history-specific causal attribution is supported. All 2,880 branches are contact-free. The eight bodies are excluded from response fitting only; upstream drivers/estimators previously used them, and the histories overlap earlier diagnostics. No native grasp/contact or independent unseen-policy claim follows.

The estimated-J composition remains unreliable and regresses slightly on six joints; five-step true-J results are mixed. Therefore do not deploy this score as an action selector or promote a controller. The positive finding supports a bounded matched training-objective experiment using frozen cross-fitted procedural response models, with all original reaching/six-joint/native retention gates required. It does not establish that such training will improve deployed control. Keep E35.

All 83 regression tests pass. Independent verification matches all E58 branches and original scores, 576 matching-driver predictions to saved E59 held-body predictions, archived trial records and unchanged nine policy checkpoints/sixteen response fits. [Full report](../../outputs/ease_response_ranking_v1/report.md), protocol, source snapshots and verification records are retained. Final test remains untouched.

### 4.51 Matched response-corrected local-action training (E61)

E61 compares two local-action objectives from identical E35 initializations across three seeds. Both receive Jacobian fit loss, the same frozen-parent anchor and a fixed 0.01 normalized local-action term, for 2,000 updates at learning rate 2e-5, batch 128. The uncorrected arm uses clipped/limited command minus measured bias. The corrected arm substitutes the frozen E59 leave-one-body-out affine response for measured examples. Synthetic examples retain the uncorrected response in both arms. No coefficient sweep, best-seed selection or native training data is used.

The measured subset contains analytical/original_0/original_1/original_2 histories from E57: 3,600 training states and 1,200 original_2 trajectory-validation states, with a fixed 512-state validation subset. Every batch contains 64 measured and 64 synthetic examples shared between conditions. Policy validation shares bodies and goals with training; response models exclude their queried body and all original_2 trajectories from fitting. E57 used additional E52-driver histories, so cross-experiment comparisons are not pure objective ablations. The primary comparison is within E61. Directory names prediction/control mean uncorrected/response-corrected here; neither is prediction-only.

The frozen linear response is converted into `predicted_dq = alpha * command + offset`, with offset determined by pre-action measured velocity, pose, bias and previous command/displacement. Gradients propagate through the candidate command but do not update response coefficients or context. Independent NumPy checks over five commands and all 31,200 E59 joint observations give 156,000 comparisons, maximum float32 discrepancy 3.4e-9. All 84 tests pass, including the identity-response limit, finite nonzero gradients and masked affine loss calculation.

All six 2,000-update training runs completed. All original reaching gates pass 256/256, for 1,536 fresh episodes. Corrected trajectory-validation losses are 0.6934/0.7715/0.7137 versus uncorrected 0.6938/0.6635/0.6608; thus the correction gives only a negligible seed-0 improvement and worse values in seeds 1/2. There is no consistent offline advantage under this fixed optimization budget.

| Condition | PiPER successes per seed /6 | Panda successes per seed /6 | Total /36 | Forbidden contacts |
|---|---|---|---:|---:|
| Original E35 | 6 / 6 / 6 | 5 / 1 / 5 | 29 | 2 |
| Uncorrected local objective | 4 / 4 / 3 | 4 / 5 / 2 | 22 | 7 |
| Response-corrected local objective | 2 / 6 / 6 | 1 / 1 / 3 | 19 | 7 |

Correction gains seven paired cases and loses ten versus the matched uncorrected objective. Two corrected PiPER seeds regain all six successes, but seed 0 regresses to two; no favorable-seed selection is justified. Corrected failures include seven forbidden contacts, four lift grasp losses, two lift timeouts, and one each of settle timeout, missing bilateral grasp, final placement failure and transport grasp loss. Mean steps including calibration are 439.2 uncorrected and 487.0 corrected, with failure timing limiting efficiency interpretation.

Neither condition passes retention. Keep E35. E59–E60's improved response prediction and first-step true-J ranking do not establish a beneficial policy-training objective under this fixed recipe. E61 also does not consistently optimize its own held-out corrected metric better than the matched baseline, so the result cannot uniquely identify a deployment-distribution cause. Do not proceed to broader promotion gates or tune another loss coefficient against these native outcomes. A useful next diagnostic should separate free-space native pose tracking from contact/grasp failures using frozen checkpoints before another training intervention.

All 72 candidate native trials are fresh; original/analytical scores reuse audited E46 development records. Analytical control solves all six cases per robot. Paired batch/context hashes, all candidate/parent/response hashes, 1,536 reaching records, 72 native records and matched calibration traces are audited. The [full report](../../outputs/ease_response_corrected_training_v1/report.md), explicit no-promotion decision, six checkpoints, losses and source snapshots are retained; native traces are in `outputs/ease_response_corrected_native_v1`. All 84 tests pass. No final-test evaluation.

### 4.52 Native contact versus pose-tracking isolation (E62)

E62 evaluates all twelve observed E46 layouts with analytical, three E35, three E61 uncorrected and three E61 corrected estimators. Each of the 120 original native rollouts exactly reproduces its audited E46/E61 result. An optional observer records the exact ramped reference, aperture and phase before each action. Restoring the calibrated simulator/controller state and replaying that finite schedule with all contacts intact reproduces positions and rotations within 1e-10 in every case. The original object-dependent transport/lower goals are recorded and frozen, preventing altered object feedback from changing the counterfactual task.

A cube-disabled replay sets only the cube geometry contact masks to zero after calibration. The arm, gripper, table, gravity, gains, limits and scheduled apertures remain unchanged. This intervention causes forbidden contacts in all six analytical PiPER cases and all eighteen E35 PiPER cases despite their original success. Consequently, cube removal is not a clean free-space benchmark: residual gripper/table interactions can change with the cube absent.

After these early analytical observations, a separately recorded companion disables all geometry contact masks after identical calibration. It retains masses, gravity, gains, limits and the same recorded reference/aperture schedule. This is an artificial unconstrained diagnostic, not safe control or manipulation success. Across the two runs, all 120 original, 120 intact, 120 cube-disabled and 120 unconstrained executions are retained; no weights are updated.

| Group | Robot | Original task successes | Unconstrained endpoints track | Original failures whose endpoints track unconstrained |
|---|---|---:|---:|---:|
| Analytical | PiPER | 6/6 | 6/6 | 0 |
| Analytical | Panda | 6/6 | 6/6 | 0 |
| E35 | PiPER | 18/18 | 18/18 | 0 |
| E35 | Panda | 11/18 | 14/18 | 3 |
| E61 uncorrected | PiPER | 11/18 | 15/18 | 4 |
| E61 uncorrected | Panda | 11/18 | 14/18 | 3 |
| E61 corrected | PiPER | 14/18 | 16/18 | 2 |
| E61 corrected | Panda | 5/18 | 9/18 | 5 |

Of 38 failed original trials, 17 meet the recorded endpoint tolerance after contact forces are removed and 21 do not. The remaining misses occur at descend (10), lift (9), release (1) and lower (1). One originally successful corrected Panda trial also misses its unconstrained endpoint, showing that removing contact does not monotonically improve tracking. Full paired XYZ/tool-direction traces and RMS metrics are retained. These observations support investigating both tracking error and contact/grasp handling; they do not uniquely assign a cause to each original failure.

Endpoint tracking here means a single recorded reference endpoint is within 6 mm and 0.08 tool-Z error, not an eight-step hold or completion of a full schedule. Original failures truncate some references, and different models have different reference sequences/durations. A missed early endpoint cannot establish inability to recover with more time; aggregate endpoint totals are not a common-task model ranking. No independent unseen-embodiment, native manipulation improvement or hardware claim is made.

The initial v1 failed the intact replay check because reconstructed references and a mathematically equivalent solve ordering introduced numerical drift; its source and failed status are preserved. v2 captures exact references with a passive optional observer and preserves the original direct solve. Default rollout behavior remains unchanged. All 84 tests pass. All archived records, source hashes, unchanged nine checkpoints and paired outcomes are audited. [Combined report](../../outputs/ease_native_contact_isolation_v2/combined_report.md), cube-disabled traces and unconstrained traces (`outputs/ease_native_unconstrained_v1`) are retained. Keep E35 and final test reserved.

Before another training intervention, use one complete native pose schedule shared across models with sufficient dwell and explicit pose/velocity diagnostics, alongside a separate contact evaluation. E62's deliberately matched prefixes cannot supply that full-task comparison.

### 4.53 Complete common native pose schedule (E63)

E63 removes E62's model-specific failure truncation. Each of twelve observed E46 cases receives one complete 1,700-step reference sequence shared by analytical, three E35, three E61 uncorrected and three E61 corrected estimators. The original nine nominal native tool waypoints use fixed budgets 220/260/60/260/260/260/60/260/60. The XYZ reference ramps by at most 2 mm each step, continuously across phases, independently of model observations. The tool-Z target is +Z and the native aperture sequence is retained. There is no object-offset target adjustment, object-height release shortcut or early termination.

All geometry contact masks are disabled after calibration; gravity, masses, gains, joint limits and online estimator updates remain active. This is an artificial tracking diagnostic, not manipulation or safe control. Each phase passes only if all of its final eight samples satisfy goal-position error below 6 mm and tool-Z direction error below 0.08. A complete schedule must pass all nine phases. Joint speeds, clipping and full pose/joint traces are retained without adding a velocity pass threshold.

Initial v1 stopped at a repeated-calibration equality assertion for original_0, Panda case 8, after twenty complete schedules. The mismatched probe was not saved, so its cause is unresolved; those partial results are excluded. v2 verifies one calibration per case against E46, then clones the full calibrated simulator state and Gram/cross/bias statistics for every method. This guarantees shared starts within the comparison. Twelve calibrations are physically simulated (384 steps), with a 32-step interaction cost charged to each of the 120 trials. The failed version and source snapshots are preserved.

All 120 schedules completed without early termination: 204,000 control steps. The analytical reference passes all twelve cases. Learned results are:

| Condition | PiPER per-seed passes /6 | Panda per-seed passes /6 | Complete schedules /36 | Passed phase holds /324 |
|---|---|---|---:|---:|
| E35 | 6 / 6 / 6 | 6 / 3 / 6 | 33 | 317 |
| E61 uncorrected | 6 / 6 / 6 | 6 / 5 / 3 | 32 | 315 |
| E61 response-corrected | 4 / 6 / 6 | 4 / 5 / 3 | 28 | 303 |

The analytical reference passes 108/108 phase holds across its twelve cases. E35 misses seven Panda phase holds: lower/release/retreat twice each and settle once. Uncorrected misses nine Panda holds. Corrected misses two PiPER lift holds and nineteen Panda holds, chiefly retreat/settle/lower/release. The response-corrected six-joint regression therefore persists without contact forces. No learned condition establishes superiority to the analytical reference.

Among the 37 missed learned phase holds, 30 never achieve eight consecutive in-tolerance samples within their phase and seven achieve such a hold but fail the final-eight requirement. E35 contributes six never-acquired/one acquired-then-lost, uncorrected six/three, corrected eighteen/three. Repeated phases and cases are dependent; these are diagnostic counts, not independent statistical samples. For example, corrected seed 0 PiPER case 0 acquires its lift hold at step 136 but ends the 260-step phase at 6.90 mm error, outside the 6 mm criterion. Conversely, some Panda misses retain substantial position/orientation error throughout the available phase. Both convergence and hold stability warrant investigation.

Mean terminal joint-speed norm on Panda is 0.01654 rad/s analytical, 0.02480 E35, 0.03023 uncorrected and 0.04793 corrected, averaged over phase tails. Velocity is diagnostic, not a post-hoc pass gate or established cause. Full phase acquisition, clipping and moving-reference error are retained in `dynamics.json`. Tool-Z control leaves yaw unconstrained; this is not full six-dimensional orientation tracking.

E63 provides a common full-schedule comparison that E62's truncated prefixes could not. It still changes contact forces, timing and object-dependent goals relative to manipulation, so differences from E61 task success cannot be assigned solely to one cause. The findings establish residual unconstrained pose-tracking deficits under this diagnostic and reject a contact-only account of all failures. Keep E35; no checkpoint promotion or new training. The next focused diagnostic should inspect learned motion estimates and bias during failing Panda pose-acquisition/hold phases against the analytical reference before another objective change. Native data remains diagnostic-only.

All 86 tests pass, including model-independent bounded reference construction and the final-eight hold rule. Shared schedule hashes, all 120 archived traces, phase summaries, 204,000 control steps and unchanged nine checkpoints are verified. The [full report](../../outputs/ease_native_common_schedule_v2/report.md), [phase acquisition analysis](../../outputs/ease_native_common_schedule_v2/dynamics.md) and visually inspected [phase figure](../../outputs/ease_native_common_schedule_v2/phase_holds.png) are retained. The failed v1 is excluded but preserved. Final test remains reserved.

## 4.54 E64: isolated motion-estimate and bias interventions

All fifteen failed E63 schedules are replayed, with four full-phase branches at each of their 37 failed phase entries. Full simulator state and measured estimator statistics are cloned; references, control parameters, budgets and contact disabling are unchanged. The analytical estimate uses each branch's evolving measured Gram/cross statistics. Bias removal changes the additive control term while preserving history updates. Privileged Jacobian access is blocked.

| Group | Failed holds | Analytical motion | Zero bias | Both |
|---|---:|---:|---:|---:|
| E35 | 7 | 5 | 3 | 3 |
| E61 uncorrected | 9 | 6 | 4 | 6 |
| E61 corrected | 21 | 13 | 8 | 11 |
| Total | 37 | 24 | 15 | 20 |


Replacing only the motion estimate recovers 24/37 failed holds, compared with 15/37 for removing bias and 20/37 for both. The retained E35 subset is 5/7, 3/7 and 3/7 respectively. Both six-joint PiPER failures recover under every intervention; these two cases alone do not identify a unique cause. Analytical substitution reduces mean clipped-step fraction from 0.8598 to 0.4807. These are phase-weighted, failure-conditioned observations, not an estimate of improvement over all tasks.

The learned estimate contributes to recoverable failures, but disabling bias is not a general fix. Analytical substitution still misses 13 phases: five release, five settle, two close and one lower. Twelve of these have only 60 steps, and all inherit the original learned trajectory's state/history. Thus late replacement cannot separate poor initialization from insufficient recovery time or estimation/control error. All-analytical E63 already passes every complete schedule from calibration.

Next: compare continuous interventions from calibration over all 12 full schedules, including previously successful cases, before selecting any estimator/bias redesign. Retain E35; do not train on these native cases or promote an intervention from failure-only recovery. A learned-only improvement still requires procedural training, six-joint retention, matched analytical baselines and independent embodiment validation.

Artifacts: `outputs/ease_native_phase_interventions_v1/{protocol,records,audit,status}.json`, per-branch traces/telemetry, source snapshots and `report.md`. All 25,500 replay steps and 37 baseline branches match exactly; 148 branch files and recomputed summaries audited, weights unchanged. There are 26,480 branch control steps. The EASE-FM suite passes 90 tests. A broader repository test invocation fails collection in eight unrelated recovery modules because the WSL environment lacks the openpi package. No final-test evaluation.

## 4.55 E65: continuous interventions from shared calibrated starts

All twelve E63 development cases, including prior successes, are evaluated with ten estimators crossed with measured versus zero additive bias for all 1,700 scheduled control steps. The nine learned checkpoints remain frozen; the analytical estimator is a shared control. Each case uses one verified 32-step calibration, then cloned full simulator state and measured controller statistics. References, gains, limits, online history updates, phase budgets and contact disabling are unchanged. No outcome filtering, parameter tuning or early stopping occurs. Privileged Jacobian access is blocked.

| Estimator | Measured bias | Zero bias | Paired gains | Paired losses |
|---|---:|---:|---:|---:|
| Analytical | 12/12 | 8/12 | 0 | 4 |
| E35 | 33/36 | 25/36 | 0 | 8 |
| E61 uncorrected | 32/36 | 22/36 | 0 | 10 |
| E61 corrected | 28/36 | 24/36 | 5 | 9 |


### Interpretation and decision

Bias removal reduces complete-schedule success in every estimator group: analytical 12/12 to 8/12, E35 33/36 to 25/36, E61 uncorrected 32/36 to 22/36, and corrected 28/36 to 24/36. Across the 108 learned-model cases there are five paired gains and 27 losses. This rejects blanket removal of measured additive bias as a controller improvement on these development schedules.

The two corrected-seed-0 PiPER failures from E64 both recover over complete schedules. However, corrected seed 1 loses another PiPER case, and uncorrected seed 2 also loses a PiPER case. Corrected Panda has three gains and eight losses. E35 retains all 18/18 PiPER successes, but Panda falls from 15/18 to 7/18. The local phase recoveries therefore do not establish a generally better deployed controller.

With bias retained, analytical estimation from calibration passes all twelve schedules and exactly reproduces E63. This is the continuous analytical replacement control; it is independent of the discarded learned checkpoint and is counted once per case, not as nine independent replicas. Compared with E64's late substitutions, the result is consistent with trajectory/history accumulation contributing to unrecovered late failures, but does not isolate that mechanism.

Keep the E35 checkpoints and measured bias. Prioritize improving the learned motion estimator's closed-loop behavior while retaining bias correction. A subsequent training comparison should use procedural data and fixed three-seed budgets, pass the original six-/seven-joint retention gates, and then face the complete native suite. These observed native cases remain diagnostics, not training targets or independent confirmation. No selective per-robot/seed bias switch is adopted.

These results measure contact-disabled tool tracking through scripted waypoints. They do not demonstrate reliable grasping, pick-and-place, novel-embodiment generalization or a publishable learned-method advantage. The reserved final test remains untouched.

### Verification

All 240 saved trials and their nine phase summaries were audited. All 120 original-condition traces match E63 exactly, covering 204,000 replayed control steps. Total execution is 408,000 control steps plus 384 physical calibration steps cloned into 240 starts; each trial is charged 32 calibration steps. All nine checkpoint hashes are unchanged. The EASE-FM regression suite passes 90 tests. The comparison figure was visually inspected. No broader-suite pass is claimed.

Artifacts: `outputs/ease_native_full_interventions_v1/` contains the prespecified protocol, all trial traces, records, audit, per-robot/seed results, paired phase transitions, source snapshots and inspected `full_schedule_comparison.png`.

## 4.56 E66: matched estimator-target training

Six fine-tuning runs start from the three E35 parents, with two conditions per seed. Each of 2,000 matched updates uses 64 synthetic contexts and 64 measured procedural contexts, AdamW at 2e-5, masked mean squared fit plus the fixed E35 anchor, and gradient clipping at 1. Only the measured target differs: instantaneous geometric Jacobian versus analytical estimate from the measured Gram/cross statistics. There is no local-action surrogate or native training data. The 3,600 training rows use analytical/original_0/original_1 drivers; 1,200 original_2 rows provide trajectory validation on the same eight bodies. Synthetic labels remain clean-ridge targets. Each final checkpoint is evaluated without selection.

| Condition | Reaching per seed | Tracking PiPER /18 | Tracking Panda /18 | Tracking total /36 | Manipulation PiPER /18 | Manipulation Panda /18 | Manipulation total /36 |
|---|---|---:|---:|---:|---:|---:|---:|
| E35 reference | 256/256 each | 18 | 15 | 33 | 18 | 11 | 29 |
| Geometric target | 256/256 each | 18 | 14 | 32 | 17 | 10 | 27 |
| Analytical target | 256/256 each | 18 | 18 | 36 | 18 | 14 | 32 |


### Interpretation and next decision

Analytical-target distillation produces the stronger candidate in this matched comparison. It passes all 36/36 common schedules (E35 33/36; geometric-target 32/36), recovering all three E35 failures without a tracking regression. Native manipulation improves to 32/36 (E35 29/36; geometric-target 27/36), with four paired recoveries and one regression. Both tasks retain all 18/18 six-joint PiPER successes. All six trained models pass the original reaching gate at 256/256 each (128/128 per joint count). Bias correction stays enabled; inference remains a learned Transformer plus the existing shared controller, with no analytical estimator solve.

The manipulation improvement is not uniform: Panda per-seed scores are 5/4/5 of six versus E35's 5/1/5. The aggregate gain is concentrated in seed 1. Analytical-target seed 0 loses case 7 during lift and recovers case 9; seed 1 recovers cases 8, 9 and 11. Seed 2 has the same success outcomes as its parent. One forbidden-contact trial remains (E35 has two). The separate analytical controller still passes every native case. These results do not demonstrate a learned advantage over that baseline.

Held-trajectory analytical-estimate MSE is worse for analytical-target training than geometric-target training in all three seeds, despite better native control. Both improve over their parents on those prediction metrics. Average fitting error is therefore not a reliable selection rule for these controllers. The validation split shares training bodies and goals; it cannot establish embodiment generalization.

Keep the E66 analytical-target checkpoints as frozen candidates and preserve E35 as the retained reference. Do not claim complete manipulation retention or promote a uniformly better controller while the paired regression remains. The next experiment should evaluate these fixed checkpoints against E35 and the analytical baseline on prespecified additional procedural embodiments and native placements, without tuning on the observed failures. Keep the reserved final test untouched until the evaluation plan is fixed.

The comparison plot was inspected. A recovered Panda trial (analytical-target seed 1, case 8) and the regression (seed 0, case 7) were rendered and replayed exactly: 411 and 320 total steps including calibration. The preview's first invocation failed before execution because its script-directory import path excluded the repository; explicitly setting PYTHONPATH resolved it without changing simulation or checkpoints. Both successful replay audits and images are retained.

This is an improvement on observed development tasks, not evidence of broad real-robot transfer, hardware performance, end-to-end learned grasping or publication readiness. No native-robot samples entered training. Training budgets, paired batches, source hashes, failed trials and all final checkpoints are retained.

Verification: paired batch hashes and unchanged checkpoint/source hashes pass. All 4,800 analytical targets match independent NumPy solutions (maximum absolute difference 2.98e-8); padding and gradient-masking tests pass. Audited 1,536 original reaching records and 144 native records, including 122,400 common-schedule steps and 37,735 manipulation control steps, plus 384 physical calibration steps cloned into matched starts. All 92 EASE-FM tests pass. Two rendered replays match exactly. E35/analytical baseline records are reused from audited E63/E46, not fresh independent baseline trials. No final-test evaluation.

Artifacts: `outputs/ease_estimator_distillation_v1/` contains protocol, six checkpoints, training histories, target audit, gate records, native traces/audit, summary, comparison figure, rendered recovery/regression and report. The candidate weights remain under `analytical/seed_0..2`; the analytic-sounding directory describes training targets, not an analytical inference path.

## 4.57 E67: prospective frozen transfer validation

All three E66 analytical-target checkpoints and E35 parents are frozen before generation, alongside the measured-history analytical baseline. Procedural seed 670000 produces sixteen new axis strings after excluding all registered paired/E33 strings, with four physical variants, two homes and eight goals per home: 64 bodies and 1,024 reaching cases. The unchanged 150-step horizon uses neural cold start or a single fixed +.01 rad analytical first action, with no separate probe. Six checkpoint hashes and the prior manifests remain fixed. No training or final-test evaluation occurs.

Twelve native placements were fixed before nine-endpoint IK screening: PiPER x=.215/.245/.275, Panda x=.33/.38/.43, both y=-.015/.045, with the existing 4cm Y transfer. All pass screening. One 32-step calibration per case is shared through cloned simulator state/statistics across all seven methods and both tracking/manipulation tasks. Bias correction, gains and native mechanics remain unchanged. Common tracking disables contacts; native manipulation retains them. Rejections and failures are not replaced.

| Evaluation | E35 | E66 | Analytical |
|---|---:|---:|---:|
| Procedural reaching per seed /1024 | 959 / 953 / 959 | 956 / 961 / 967 | 951 |
| Native common tracking /36 | 34 | 34 | 12/12 |
| Native manipulation /36 | 28 | 28 | 12/12 |
| Native PiPER manipulation /18 | 17 | 16 | 6/6 |
| Native Panda manipulation /18 | 11 | 12 | 6/6 |


### Interpretation and decision

The E66 native improvement does not replicate as an aggregate gain on these new placements. E35 and E66 both pass 34/36 complete tracking schedules and 28/36 manipulation trials. Tracking has one paired gain and one loss; manipulation has three gains and three losses. PiPER manipulation drops from 17/18 to 16/18, while Panda rises from 11/18 to 12/18. The E66 seed-1 PiPER case-2 lift regression is new; seed 2 shares its parent's PiPER case-3 missing-bilateral-grasp failure. E66 has zero forbidden-contact trials here versus two for E35, but equal overall success and the six-joint regression prevent a broad improvement claim. Analytical native control passes 12/12 in both tasks.

Procedural reaching shows a small, mixed improvement: E35 2871/3072 versus E66 2884/3072 (+13; 54 paired gains, 41 losses). Per-seed changes are -3/+8/+8. Six-joint success changes 1522/1536 to 1520/1536; seven-joint success changes 1349/1536 to 1364/1536. Collisions increase from 84 to 85 across the three seeds. Mean interaction counts fall for every candidate seed, but include both successful and failed trials and are not a safety metric. The analytical controller scores 951/1024; all six neural checkpoints exceed that total on this suite, so this is not a unique advantage introduced by E66. Shared families, bodies, goals and seeds are dependent observations, not thousands of independent embodiments.

Structural screening accepted all twelve new native candidates; all probes were contact-free. These scenarios use new placements, IK homes and calibration random seeds on the same two native designs. They are not new native embodiments and do not isolate placement from initialization effects. Procedural generation yields sixteen new axis arrangements, eight per joint count, with no overlap against the registered paired/E33 catalogs. Rejected structural proposals are retained; goals are endpoint-screened rather than policy-success-filtered. The reserved final test is untouched.

All eight E66 native manipulation failures are grasp-contact failures: five lost bilateral contact after lift, two after transport, and one lacked bilateral contact after closing. This describes the termination criteria, not proof that the estimator is uninvolved or that every object physically dropped. The common tracking schedule has different references and transitions from manipulation. A useful next experiment is procedural grasp-stability/recovery with frozen E35/E66/analytical estimators and a shared, prespecified contact-aware intervention, followed by retention checks. Do not select per-seed settings or tune on these newly observed native failures.

Retain E35 as the reference and preserve E66 as an experimental candidate. E67 narrows E66's earlier promise: tracking/manipulation gains on the original placements did not establish reliable transfer. No model promotion, retraining, hardware claim, or publication-readiness claim is supported here.

### Verification and artifacts

All 7168 procedural records and 168 native records, common phase summaries, source manifests, generated assets and six frozen checkpoint hashes were audited. All 92 EASE-FM regression tests pass. The comparison figure was visually inspected. Rendered candidate-seed-1 replays exactly match the saved Panda case-7 recovery (497 total steps) and PiPER case-2 regression (351 total steps), including their calibration records. Per-family, physical-condition and per-DoF paired results are saved in family_results.json, condition_results.json and paired.json. Both earlier comparison suites and the new failures remain preserved.

Artifacts: `outputs/ease_frozen_transfer_v1/` contains the locked protocol, new manifest/XMLs, structural audit, all frozen policy results, per-family and physical-condition tables, paired changes, native screening/probes/traces, comparison figure, rendered recovery/regression and report. Body identifiers are local to the new manifest. The reused E33 generator retains its schema/experiment label internally; the parent E67 protocol and manifest hash identify this evaluation.

## 4.58 E68: frozen-estimator grasp settling

Development comparison on 12 familiar and four previously IK-screened unseen procedural placements. Seven frozen estimators; shared calibration and simulator state. All 336 outcomes retained. Contact sensing is additional simulator-provided information. No training or reserved final-test evaluation.

| Estimator | Mode | Success /16 | Six-joint /7 | Seven-joint /9 | Collisions | Total control steps |
|---|---|---:|---:|---:|---:|---:|
| analytical | baseline | 11 | 7 | 4 | 5 | 4927 |
| analytical | fixed | 11 | 7 | 4 | 5 | 5519 |
| analytical | contact | 11 | 7 | 4 | 5 | 5089 |
| parent_0 | baseline | 10 | 4 | 6 | 5 | 5226 |
| parent_0 | fixed | 10 | 4 | 6 | 5 | 5792 |
| parent_0 | contact | 10 | 5 | 5 | 5 | 5329 |
| parent_1 | baseline | 9 | 6 | 3 | 6 | 5610 |
| parent_1 | fixed | 9 | 6 | 3 | 6 | 6132 |
| parent_1 | contact | 9 | 5 | 4 | 5 | 5931 |
| parent_2 | baseline | 12 | 6 | 6 | 4 | 4674 |
| parent_2 | fixed | 11 | 7 | 4 | 5 | 5505 |
| parent_2 | contact | 11 | 7 | 4 | 5 | 5102 |
| candidate_0 | baseline | 11 | 6 | 5 | 5 | 4853 |
| candidate_0 | fixed | 11 | 6 | 5 | 5 | 5511 |
| candidate_0 | contact | 11 | 6 | 5 | 5 | 5167 |
| candidate_1 | baseline | 11 | 6 | 5 | 5 | 5131 |
| candidate_1 | fixed | 12 | 6 | 6 | 3 | 5487 |
| candidate_1 | contact | 11 | 6 | 5 | 4 | 5385 |
| candidate_2 | baseline | 12 | 6 | 6 | 4 | 4613 |
| candidate_2 | fixed | 11 | 6 | 5 | 4 | 5407 |
| candidate_2 | contact | 13 | 6 | 7 | 3 | 5002 |

Pooled learned success: baseline 65/96, fixed-duration 64/96, contact-ended 65/96. Six-joint: 34, 35, 35 /42; collisions: 29, 28, 27.

Contact-ended waiting has 4 paired recoveries and 4 paired regressions. Total learned control steps (including calibration) increase from 30107 to 31916 (6.0%). E35 parents lose one aggregate success, while E66 candidates gain one; neither subgroup is a new independently selected deployment model.

Prespecified native advancement gate: FAIL. Retain the original sequencer; no native expansion of this intervention.

The intervention holds the measured XYZ reference before lift and transport, while keeping upright tool-Z control, closed gripper, measured bias and online statistics updates. Fixed control waits 20 steps; contact control ends after 8 consecutive bilateral endpoint contacts or 20 steps. Neither rejects unstable grasps; original task outcomes remain the criterion. Maximum extra dwell is 40 control steps (1.6 seconds).

These are dependent seed/case repeats on a small, observed development fixture. Pooled counts are descriptive. Fixed waiting is a duration control, not exact per-trial time matching. This tests bounded settling, not active regrasp recovery or all contact-aware controllers. Native robot performance cannot be inferred from this procedural experiment.

Audits verify all checkpoint/input hashes, unique trial keys, trace lengths, dwell bounds, and exact trace reproduction of all 64 historical analytical/E35 baselines. All 95 regression tests pass. Per-case gains and losses are saved in paired.json; full traces remain in trials/.

Artifacts: `outputs/ease_grasp_settling_v1/{protocol,summary,paired,audit}.json`, `report.md`, `comparison.png`, and all full trial traces. Implementation: `examples/ease_fm/grasp_settling.py`; three new sequencing tests cover unchanged baseline, contact streak reset and bounded dwell.

## 4.59 E69: grasp and tracking diagnostics

Read-only instrumentation exactly reproduces all 112 E68 baseline trajectories: 16 procedural placements across seven frozen estimators. This is a diagnostic replay of observed development cases, not a new success benchmark. No weights, controller settings, mechanics or actions changed.

| Group | Outcome | Trials | Collision | Stable close | Carry contact loss | Collision before loss or no loss | Median max relative drift (mm) |
|---|---|---:|---:|---:|---:|---:|---:|
| learned | success | 65 | 0 | 65 | 0 | 0 | 6.7 |
| learned | failure | 31 | 29 | 28 | 10 | 29 | 16.4 |
| analytical | success | 11 | 0 | 11 | 0 | 0 | 6.6 |
| analytical | failure | 5 | 5 | 5 | 0 | 5 | 13.8 |

Learned failure first-collision phases: lift: 8, lower: 4, transport: 2, close: 3, descend: 11, retreat: 1.

Stable close means at least eight consecutive bilateral endpoint contacts anywhere in close. Carry loss means three consecutive missing bilateral endpoint contacts in lift, transport or lower; it does not require prior acquisition. Relative drift is tool-frame object translation from the final close sample; it is not a direct physical-slip label. Contact loss during lowering may reflect valid table support.

Collision chronology and tool/object displacement narrow candidate explanations but cannot isolate causal motion-estimator error from contact dynamics, yaw, sequencer decisions or model error. One-step prediction residuals use measured joint displacement under contact, and transition tracking errors include the commanded waypoint jump. Sensor values are diagnostic simulator state, never supplied to the policy.

All input/checkpoint hashes remain unchanged. All 112 full baseline traces and calibration probes reproduce exactly; 97 EASE-FM tests pass. Threshold tests cover contact acquisition, sustained-loss indexing and rejection of brief contact gaps. No reserved final-test rollout or native tuning was performed.

### Interpretation and next experiment

The procedural failure set is dominated by forbidden contact: 29 of 31 learned failures and all five analytical failures. Nine learned failures combine collision and sustained carry-contact loss; all nine collide first. Twenty collision failures have no sustained carry loss. The other two learned failures are separately audited in terminal_audit.json. This is temporal evidence, not proof that collision caused subsequent loss or that the motion estimator is uninvolved.

All 65 learned successes achieve stable close and avoid sustained carry loss, but their median maximum relative translation is 6.7 mm. The prespecified 5 mm drift event therefore cannot be interpreted as a specific slip detector. It includes compliant grasp motion, rolling and motion relative to the measured tool frame.

Next test a shared clearance-aware path/phase intervention on procedural cases, retaining the original controller and all failures as controls. Keep arm/table, finger/table and arm/gripper collisions separate, with six-joint retention and interaction costs. Do not mask forbidden contacts, change success criteria, or tune the native placements. Active regrasp is secondary until clearance failures are addressed. This is not a native-robot conclusion or a new learned-policy performance gain.

Additional artifact: diagnostics.png shows the two noncollision failures, selected exhaustively rather than for favorable outcomes. Two additional terminal replays match exactly.

Timing clarification: parent_1 case 10 has bilateral contact at trace indices 106-117 during close (105-144), then none at the final close sample. The carry-loss marker at index 145 is the first missing sample in the carry window, not the physical onset of loss. The sequencer proceeds to lift despite missed descend/close tolerances. Parent_0 case 4 misses transport tolerance and finishes 41.8 mm from the target. These observations motivate testing phase transitions; they do not prove a new gate will recover the task.

Terminal checks (success requires XY error <25 mm, height error <5 mm, speed norm <0.02, no finger contact, prior grasp/lift and no collision):

- parent_0 case 4: XY error 41.8 mm, height error 0.1 mm, velocity norm 0.0000, final pad contacts 0, lifted=True; missed phase tolerances: transport.
- parent_1 case 10: XY error 71.6 mm, height error 0.1 mm, velocity norm 0.0000, final pad contacts 0, lifted=False; missed phase tolerances: descend, close.


## 4.60 E70: bounded phase-transition retention

All 16 observed procedural placements, seven frozen estimators and three conditions: 336 trials. Calibration, weights, dynamics, task waypoints and forbidden-contact definitions are unchanged. This experiment isolates phase budget and transition decisions; it does not introduce a collision-clearance planner.

| Estimator | Mode | Success /16 | Six /7 | Seven /9 | Collisions | Aborts | Steps including calibration |
|---|---|---:|---:|---:|---:|---:|---:|
| analytical | baseline | 11 | 7 | 4 | 5 | 0 | 4927 |
| analytical | extended | 11 | 7 | 4 | 5 | 0 | 5224 |
| analytical | gated | 11 | 7 | 4 | 5 | 5 | 3829 |
| parent_0 | baseline | 10 | 4 | 6 | 5 | 0 | 5226 |
| parent_0 | extended | 10 | 4 | 6 | 5 | 0 | 5395 |
| parent_0 | gated | 9 | 4 | 5 | 5 | 7 | 4085 |
| parent_1 | baseline | 9 | 6 | 3 | 6 | 0 | 5610 |
| parent_1 | extended | 9 | 6 | 3 | 6 | 0 | 6220 |
| parent_1 | gated | 9 | 6 | 3 | 6 | 7 | 4314 |
| parent_2 | baseline | 12 | 6 | 6 | 4 | 0 | 4674 |
| parent_2 | extended | 12 | 7 | 5 | 4 | 0 | 5279 |
| parent_2 | gated | 11 | 6 | 5 | 4 | 5 | 4387 |
| candidate_0 | baseline | 11 | 6 | 5 | 5 | 0 | 4853 |
| candidate_0 | extended | 11 | 6 | 5 | 4 | 0 | 5322 |
| candidate_0 | gated | 11 | 6 | 5 | 4 | 5 | 4061 |
| candidate_1 | baseline | 11 | 6 | 5 | 5 | 0 | 5131 |
| candidate_1 | extended | 12 | 6 | 6 | 4 | 0 | 5633 |
| candidate_1 | gated | 10 | 6 | 4 | 4 | 6 | 4017 |
| candidate_2 | baseline | 12 | 6 | 6 | 4 | 0 | 4613 |
| candidate_2 | extended | 13 | 6 | 7 | 3 | 0 | 5401 |
| candidate_2 | gated | 12 | 6 | 6 | 3 | 4 | 4299 |

Extended budgets add at most 40 steps to each moving phase and 20 to close (260 maximum additional steps). Original early pose exit remains active. Gated uses identical budgets but aborts at phase end on accumulated collision, missing five final in-tolerance moving-phase samples, or missing eight final in-tolerance bilateral close samples. Aborts always count as failures. This gate does not stop a collision during a phase, nor avoid contact geometrically.

| Learned models pooled | Success /96 | Six /42 | Seven /54 | Collisions | Aborts | Steps |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 65 | 34 | 31 | 29 | 0 | 30107 |
| extended | 67 | 35 | 32 | 26 | 0 | 33250 |
| gated | 62 | 34 | 28 | 26 | 34 | 25163 |

Prespecified retention requires improved pooled success, no pooled collision increase, and no six-joint success loss in any checkpoint.
- extended: PASS; 5 paired recoveries and 3 regressions; per-checkpoint six-joint retention=True.
- gated: FAIL; 4 paired recoveries and 7 regressions; per-checkpoint six-joint retention=True.

Retention refers to per-checkpoint aggregate six-joint counts, not preservation of every successful case. Extended candidate_0 loses familiar six-joint case 4 while recovering unseen six-joint case 12. Five extended recoveries and three losses therefore remain a mixed result despite passing this aggregate rule. Extended control uses 33,250 versus 30,107 total learned control steps, a 10.4% increase.

Learned gated aborts: collision:lift=6, collision:lower=2, collision:release=1, transition:retreat=1, collision:transport=3, transition:transport=2, collision:descend=13, transition:descend=3, transition:close=2, collision:close=1.

Verification: all 112 baseline traces and calibration probes reproduce E68 exactly. Each of 112 gated traces exactly matches the corresponding extended trajectory up to its stopping point. Model/input hashes are unchanged; 99 EASE-FM tests pass. All failures and stopped trajectories are retained. No training, native evaluation or reserved final-test evaluation.

These dependent repeated seeds and small, IK-screened development placements do not establish population-level transfer. Fewer collision cases can result from ending a trial earlier; task success and paired regressions remain essential. A retention pass would justify further validation, not immediate deployment or a paper-ready claim.

### Interpretation and next decision

Retain these as development candidates for broader validation: extended. Keep the baseline until that validation passes.

First freeze the extended-budget candidate and validate it on fresh procedural placements, including casewise six-joint regressions and cost. Stopping a trial at a failed transition changes exposure to later contacts, so lower collision counts cannot be presented as successful collision avoidance. Extended budgets change interaction duration and can alter contact outcomes even without a gate. These trials test timing and stopping; they do not test geometry-aware motion.

A separate subsequent intervention is a shared path-clearance controller on procedural cases, with an explicit accounting of any geometric information it receives. It should change motion before contact, preserve task success and six-joint retention, and be compared with the original estimator/controller and an analytical estimator under the same geometry access. Avoid another threshold or waiting-time sweep on these observed cases. Native transfer and the reserved final test remain deferred.


## 4.61 E71: fresh-placement validation of extended phase budgets

17/18 prespecified placements pass endpoint IK screening across the same six procedural bodies. All screening rejections are retained without replacement; each accepted placement receives all seven estimators and both conditions (238 trials). Positions, IK seeds and probe seeds are new. Bodies and axis families are reused: this is placement transfer, not new-embodiment transfer or the reserved final test.

| Estimator | Budget | Success | Six-joint | Seven-joint | Collisions | Total steps |
|---|---|---:|---:|---:|---:|---:|
| analytical | baseline | 16/17 | 8/8 | 8/9 | 1 | 4752 |
| analytical | extended | 15/17 | 7/8 | 8/9 | 2 | 5090 |
| parent_0 | baseline | 15/17 | 8/8 | 7/9 | 2 | 5184 |
| parent_0 | extended | 15/17 | 8/8 | 7/9 | 2 | 5471 |
| parent_1 | baseline | 17/17 | 8/8 | 9/9 | 0 | 5030 |
| parent_1 | extended | 15/17 | 8/8 | 7/9 | 2 | 5622 |
| parent_2 | baseline | 15/17 | 8/8 | 7/9 | 2 | 4749 |
| parent_2 | extended | 15/17 | 8/8 | 7/9 | 2 | 5117 |
| candidate_0 | baseline | 14/17 | 7/8 | 7/9 | 3 | 5297 |
| candidate_0 | extended | 15/17 | 6/8 | 9/9 | 2 | 5648 |
| candidate_1 | baseline | 15/17 | 7/8 | 8/9 | 2 | 4907 |
| candidate_1 | extended | 15/17 | 7/8 | 8/9 | 2 | 5233 |
| candidate_2 | baseline | 17/17 | 8/8 | 9/9 | 0 | 4723 |
| candidate_2 | extended | 17/17 | 8/8 | 9/9 | 0 | 5094 |

Pooled learned success: 93/102 baseline versus 92/102 extended; 2 recoveries and 3 regressions. Six-joint: 46 to 45/48; seven-joint: 47 to 47/54. Collisions: 9 to 10. Total control steps including calibration: 29890 to 32185 (+7.7%).

Prespecified aggregate replication rule: FAIL. Per-checkpoint aggregate six-joint retention: False. The E70 improvement does not pass the prespecified replication rule. Keep the original phase budgets and retain extended timing as a failed development candidate.

The frozen E70 implementation adds at most 40 steps per moving phase and 20 to close. It leaves the early-exit rule, physics, waypoints and success/contact criteria unchanged. No strict transition gate is used. No thresholds, parameters or checkpoints were chosen using E71 results. All trials share a cloned measured calibration state within each placement; probes with contacts remain failures rather than being filtered out.

| Family | Budget | Learned success | Collision cases |
|---|---|---:|---:|
| six_amb_train_0 | baseline | 16/18 | 2 |
| six_amb_train_0 | extended | 16/18 | 2 |
| six_amb_train_1 | baseline | 18/18 | 0 |
| six_amb_train_1 | extended | 18/18 | 0 |
| seven_amb_train_0 | baseline | 13/18 | 5 |
| seven_amb_train_0 | extended | 15/18 | 3 |
| seven_amb_train_1 | baseline | 18/18 | 0 |
| seven_amb_train_1 | extended | 17/18 | 1 |
| broad_6_0 | baseline | 12/12 | 0 |
| broad_6_0 | extended | 11/12 | 1 |
| broad_7_8 | baseline | 16/18 | 2 |
| broad_7_8 | extended | 15/18 | 3 |

Screening rejections: candidate 13 (6 joints), failed approach after 0 endpoints.

Verification: all model/input/E70 implementation hashes unchanged; unique trial keys, complete nine-phase rollouts and trace lengths audited. The same 99 EASE-FM tests pass. Full protocols, probes, screening records and trial traces are retained. No model training, native tuning or final-test rollout.

Interpretation is limited by endpoint screening, six reused bodies, dependent training seeds and observed development history. Aggregate six-joint retention does not mean every successful six-joint case is preserved; paired.json lists each regression. Extra time and its cost must remain part of the comparison.

### Decision and next step

The replication rule fails; retain the original phase budgets. Do not promote extended timing. Stop tuning waiting times on these placements. The next distinct experiment should change the path before contact using a shared clearance-aware controller, with identical geometry access for learned and analytical estimators and explicit disclosure of that extra information. The reserved final test remains untouched.


## 4.62 E72: shared geometry-based action filtering

Development test on the original 16 procedural manipulation placements. Seven frozen estimators, original phase budgets and matched calibration states; 224 trials. A separate kinematic model checks midpoint/end of joint-target interpolations and scales motion while preserving measured bias. Physics, contact masks, success rules and neural weights are unchanged.

**Information difference:** the filter receives the known robot geometry/kinematics and exact current simulator configuration, object pose and scene geometry. Both learned and analytical estimators receive the identical filter. These are additional privileged observations unavailable to a history-only policy. No dynamic future rollout or task Jacobian is used. This experiment cannot establish improved learned embodiment identification.

| Estimator | Mode | Success /16 | Six /7 | Seven /9 | Collisions | Control steps | Geometry queries | Changed actions | Fallbacks |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| analytical | baseline | 11 | 7 | 4 | 5 | 4927 | 0 | 0 | 0 |
| analytical | geometry | 11 | 7 | 4 | 4 | 4930 | 16346 | 527 | 140 |
| parent_0 | baseline | 10 | 4 | 6 | 5 | 5226 | 0 | 0 | 0 |
| parent_0 | geometry | 12 | 5 | 7 | 3 | 5715 | 22663 | 1183 | 526 |
| parent_1 | baseline | 9 | 6 | 3 | 6 | 5610 | 0 | 0 | 0 |
| parent_1 | geometry | 9 | 6 | 3 | 6 | 5856 | 25022 | 1502 | 1143 |
| parent_2 | baseline | 12 | 6 | 6 | 4 | 4674 | 0 | 0 | 0 |
| parent_2 | geometry | 12 | 6 | 6 | 4 | 4958 | 16654 | 557 | 293 |
| candidate_0 | baseline | 11 | 6 | 5 | 5 | 4853 | 0 | 0 | 0 |
| candidate_0 | geometry | 11 | 6 | 5 | 5 | 4978 | 15716 | 393 | 157 |
| candidate_1 | baseline | 11 | 6 | 5 | 5 | 5131 | 0 | 0 | 0 |
| candidate_1 | geometry | 11 | 6 | 5 | 5 | 5309 | 19877 | 931 | 368 |
| candidate_2 | baseline | 12 | 6 | 6 | 4 | 4613 | 0 | 0 | 0 |
| candidate_2 | geometry | 13 | 7 | 6 | 3 | 4806 | 16186 | 560 | 197 |

Pooled learned success: 65/96 to 68/96 (3 recoveries, 0 regressions). Six-joint: 34 to 36/42; seven-joint: 31 to 32/54. Collision cases: 29 to 26. Control steps including calibration: 30107 to 31622. Geometry queries: 116118; modified actions: 5126; no-feasible-candidate fallbacks: 2684.

Prespecified advancement rule: PASS; per-checkpoint aggregate six-joint retention=True. Freeze this filter for fresh-placement validation before adoption.

The three recoveries are concentrated in E35 seed 0 (cases 2 and 11) and E66 seed 2 (case 1); the other four learned checkpoints have unchanged success counts. Analytical success stays 11/16, while its collision count falls from five to four. These are descriptive repeated development cases, not independent evidence of a uniform model improvement.

Checks are kinematic samples, not a continuous swept-volume or dynamics guarantee. The object is frozen at its current pose in geometry queries, including when carried; actuator transients and untested points along the path may differ. When already within the clearance margin, nonworsening motion is allowed. When no candidate qualifies, maximum sampled clearance is used and the fallback is recorded. Neither collisions nor failures are masked. The effect of geometry access is inseparable from the filter in this comparison.

Verification: all 112 unfiltered baseline traces and probes reproduce E68 exactly; model/input hashes and physical margins are unchanged. Trial uniqueness, filter-log coverage and query totals are audited. All 102 EASE-FM tests pass, including physical-state isolation and exact pass-through of unmodified actions. Query counts are computational work, not a measured real-time latency guarantee.

Version history: v1 was interrupted after detecting that bias arithmetic could introduce roundoff even at full scale; its partial trials and source are retained. V2 passes full-scale actions through bit-exactly. No thresholds were changed in response to aggregate task outcomes. All completed v2 outcomes are retained. No training, native tuning or reserved final-test evaluation.

### Interpretation and next decision

The prespecified development gate passes; freeze the filter for fresh-placement validation, retaining all casewise regressions. Next validate this frozen filter on E71 placements, and subsequently compare against a shared action-scaling control to distinguish geometry-conditioned decisions from simply slowing motion.

The broader research status remains proof of working simulated cross-kinematic control with limited manipulation robustness. Neither this diagnostic nor the prior placement results establishes learned superiority over analytical identification, broad new-embodiment manipulation, or hardware performance. Publication readiness remains unresolved.


## 4.63 E73: geometry-filter transfer and fixed slowing control

All 17 accepted E71 placements on six existing bodies; all seven frozen estimators and three conditions, 357 trials. The 17/18 screening record and rejected case remain unchanged. This set was not used to design the frozen E72 filter, but its baseline outcomes were already observed in E71; it is development validation, not the reserved final test or new-embodiment transfer.

| Estimator | Mode | Success /17 | Six /8 | Seven /9 | Collisions | Steps | Geometry queries |
|---|---|---:|---:|---:|---:|---:|---:|
| analytical | baseline | 16 | 8 | 8 | 1 | 4752 | 0 |
| analytical | geometry | 16 | 8 | 8 | 1 | 4752 | 12644 |
| analytical | slow | 16 | 8 | 8 | 1 | 5654 | 0 |
| parent_0 | baseline | 15 | 8 | 7 | 2 | 5184 | 0 |
| parent_0 | geometry | 15 | 8 | 7 | 2 | 5010 | 13730 |
| parent_0 | slow | 15 | 6 | 9 | 2 | 5901 | 0 |
| parent_1 | baseline | 17 | 8 | 9 | 0 | 5030 | 0 |
| parent_1 | geometry | 17 | 8 | 9 | 0 | 5030 | 13458 |
| parent_1 | slow | 13 | 8 | 5 | 4 | 6189 | 0 |
| parent_2 | baseline | 15 | 8 | 7 | 2 | 4749 | 0 |
| parent_2 | geometry | 15 | 8 | 7 | 2 | 4783 | 13175 |
| parent_2 | slow | 14 | 8 | 6 | 3 | 6116 | 0 |
| candidate_0 | baseline | 14 | 7 | 7 | 3 | 5297 | 0 |
| candidate_0 | geometry | 14 | 7 | 7 | 3 | 5303 | 14281 |
| candidate_0 | slow | 16 | 7 | 9 | 1 | 5789 | 0 |
| candidate_1 | baseline | 15 | 7 | 8 | 2 | 4907 | 0 |
| candidate_1 | geometry | 15 | 7 | 8 | 2 | 4940 | 13720 |
| candidate_1 | slow | 14 | 7 | 7 | 2 | 5751 | 0 |
| candidate_2 | baseline | 17 | 8 | 9 | 0 | 4723 | 0 |
| candidate_2 | geometry | 17 | 8 | 9 | 0 | 4726 | 12548 |
| candidate_2 | slow | 15 | 8 | 7 | 1 | 5691 | 0 |

| Learned models pooled | Success /102 | Six /48 | Seven /54 | Collisions | Steps |
|---|---:|---:|---:|---:|---:|
| baseline | 93 | 46 | 47 | 9 | 29890 |
| geometry | 93 | 46 | 47 | 9 | 29792 |
| slow | 87 | 44 | 43 | 13 | 35437 |

Prespecified geometry replication rule: FAIL. Per-checkpoint aggregate six-joint retention=True. Geometry success exceeds this fixed slowing control=True.

- geometry versus baseline: 0 paired recoveries, 0 regressions, including 0 six-joint regressions.
- slow versus baseline: 5 paired recoveries, 11 regressions, including 3 six-joint regressions.
- geometry versus slow: 11 paired recoveries, 5 regressions, including 1 six-joint regressions.

Geometry control uses -0.3% control steps versus baseline, 80912 geometry queries, 238 changed actions and 40 no-feasible-candidate fallbacks. Fixed slowing uses +18.6% control steps and zero geometry queries.

The filter is unchanged E72v2: known geometry and exact current simulator pose, 2 mm margin, midpoint/endpoint kinematic checks, action scales 1/.5/.25/0 with measured bias preserved. The slowing control uses a fixed .5 scale on the same motion component, preserves bias and leaves finger commands unchanged. All conditions use the original phase budgets and physics. Slowing is not matched to the filter's per-step action distribution or total interaction time; one constant cannot rule out all geometry-free schedules.

Geometry remains extra privileged information shared across all estimators. A result above the slowing control is not learned-identifier superiority, and a filtered task failure may reflect dynamics outside the sampled kinematic checks. These experiments do not evaluate hardware, latency compliance or a safety guarantee.

The filter changes 238 learned-model actions (about 0.9% of task steps) and performs 80,912 geometry queries, yet all learned casewise success outcomes remain identical to baseline. Its roughly 0.3% reduction in simulated control steps is not evidence of lower computation or real-time latency. The slowing comparison has five recoveries and eleven regressions, so its net loss also masks mixed casewise effects.

Verification: 119 baseline traces and calibration probes exactly reproduce E71. E72 code, model and input hashes are unchanged; physical margins remain unchanged. Geometry access is blocked in both baseline and slowing controls. Trial coverage, nine-phase trajectories and filter/query logs are audited. All 103 EASE-FM tests pass. All failures retained; no training, native tuning or reserved final-test rollout.

### Interpretation and next decision

The frozen geometry filter fails the prespecified replication rule. Retain the original controller; no promotion of the filter. Keep the full slowing comparison as evidence and inspect the paired geometry failures before proposing a different mechanism. Do not retune the filter margin or scale list on this validation set.

The geometry intervention still depends on a known robot model and exact simulator scene state. No outcome here establishes a learned advantage over analytical estimation, broad unseen-embodiment manipulation or publication readiness. The slowing control tests one fixed value, not all geometry-free alternatives.


## 4.64 E74: first-collision counterfactual audit

All ten unsuccessful E73 geometry-filter trajectories are replayed exactly: nine learned-controller failures and one analytical failure. Their first forbidden contacts are finger/table contacts in seven descend and three lower phases. The paired E73 unfiltered trajectories fail on the same cases and first-contact control indices. This is outcome-conditioned diagnosis, not a new success-rate evaluation.

| Counterfactual at the pre-contact state | Immediate contact avoided /10 |
|---|---:|
| selected | 0 |
| half_selected | 0 |
| bias_only | 0 |
| analytical_same_history | 5 |
| selected_zero_incoming_velocity | 2 |

| Case | Estimator | Phase | 21-sample minimum (mm) | Actual substep minimum (mm) | Analytical action avoids | Zero-velocity avoids |
|---|---|---|---:|---:|---|---|
| 1 | candidate_0 | descend | -0.042 | -0.476 | False | False |
| 1 | candidate_1 | descend | 2.000 | -0.527 | False | False |
| 6 | analytical | lower | 2.000 | -1.013 | False | False |
| 6 | candidate_0 | lower | 2.000 | -0.271 | True | True |
| 8 | parent_0 | descend | 2.000 | -0.596 | False | False |
| 8 | parent_2 | descend | 2.000 | -1.532 | True | False |
| 8 | candidate_0 | lower | 2.000 | -1.158 | True | False |
| 8 | candidate_1 | descend | 2.000 | -0.974 | True | False |
| 14 | parent_2 | descend | 2.000 | -1.545 | False | False |
| 15 | parent_0 | descend | 2.000 | -0.207 | True | True |

### Interpretation

All ten filtered actions were accepted by the original geometric test. Nine remain at the 2 mm clearance cap even when the commanded joint/finger interpolation is sampled at 21 points, including its start. Reconstructed actual dynamic substep configurations penetrate forbidden geometry in all ten. Thus simply adding interpolation samples does not resolve these nine misses: the commanded straight joint path is not the executed dynamic trajectory. One event already has about 0.042 mm penetration at the start of the diagnosed step; the original contact flag first records it during that step. This contact-timing distinction is retained.

Half motion and bias-only avoid zero immediate collisions. Changing only the estimator used to compute the action—from the same measured Gram/cross/bias and task error—avoids five. This is a controlled local action comparison and implicates the proposed motion direction, but it does not establish full-task recovery or uniquely identify neural-estimation error as the cause. The analytical controller's independent whole-task rollout succeeds on eight of the ten corresponding event cases, with repeated case identities; it is a different trajectory.

Resetting incoming arm/finger velocities avoids two immediate contacts, so incoming velocity matters in those states. This privileged reset also recomputes constraints and is not a realizable braking action. Its limited benefit does not exclude acceleration, unequal joint responses, tracking lag or contact dynamics in the other failures. The analytical-action and zero-velocity successes overlap; they are not additive recoveries.

### Decision

Retain the original controller and keep the static geometry filter unpromoted. Do not perform another geometry-margin or interpolation-density sweep. The next useful comparison should account for the predicted executed joint response and current velocity, while retaining the same-history analytical action as a comparator. Any simulator-dynamics lookahead must be labeled a privileged model-based diagnostic and separated from claims about history-only adaptation. Before claiming a useful fix, require complete-task evaluation and preservation of successful six-joint cases.

### Verification and limitations

Ten full trajectories, filter logs and calibration probes replay exactly; all ten selected-action branches reproduce the original contact-step tool position. Fifty branches run for one 40 ms control step each, with original state/velocity/time restored after each. Checkpoint, input and source hashes are unchanged. All 104 EASE-FM tests pass, including branch state/flag restoration. No weights, phase budgets, collision masks or task success criteria changed.

All chosen states precede an observed failure, so branch counts cannot estimate deployment success. There is no successful-state intervention control or multi-step recovery evaluation here. Actual-path clearance reconstruction uses the current object pose rather than updating it; the first forbidden pair in every event is finger/table, which does not involve the object. Discrete geometry samples do not guarantee continuous clearance. No training, native tuning or reserved final-test evaluation occurred.


## 4.65 E75: measured joint-response and velocity prediction

All 119 E73 geometry-controller trajectories replay exactly, including successful tasks and every failure. Two analytical online ridge models are fitted separately per actuator from completed interactions: command/intercept and command/velocity/intercept. They update through the 32-step calibration and the subsequent trajectory. Predictions are made before observing the next displacement. Neural policy weights and actions remain unchanged.

Geometry evaluation covers every descent/lowering step before the first forbidden contact, including that contact step. These phases were selected from E74 evidence and are development diagnostics. Every path uses the same known geometry, object pose and 21 interpolation points; none uses future dynamic simulator rollout. Current joint velocity is an additional measurement. This is not a trained replacement for the motion Transformer.

| Group | Threshold | Predictor/path | Detected contacts | Missed contacts | False alarms | Noncollision steps |
|---|---|---|---:|---:|---:|---:|
| all | penetration | command | 1/10 | 9 | 0 | 5745 |
| all | penetration | response_linear | 8/10 | 2 | 0 | 5745 |
| all | penetration | velocity_linear | 9/10 | 1 | 1 | 5745 |
| all | penetration | velocity_curve | 9/10 | 1 | 1 | 5745 |
| all | 2mm_warning | command | 1/10 | 9 | 0 | 5745 |
| all | 2mm_warning | response_linear | 9/10 | 1 | 0 | 5745 |
| all | 2mm_warning | velocity_linear | 9/10 | 1 | 1 | 5745 |
| all | 2mm_warning | velocity_curve | 9/10 | 1 | 1 | 5745 |
| learned | penetration | command | 1/9 | 8 | 0 | 4920 |
| learned | penetration | response_linear | 7/9 | 2 | 0 | 4920 |
| learned | penetration | velocity_linear | 8/9 | 1 | 1 | 4920 |
| learned | penetration | velocity_curve | 8/9 | 1 | 1 | 4920 |
| learned | 2mm_warning | command | 1/9 | 8 | 0 | 4920 |
| learned | 2mm_warning | response_linear | 8/9 | 1 | 0 | 4920 |
| learned | 2mm_warning | velocity_linear | 8/9 | 1 | 1 | 4920 |
| learned | 2mm_warning | velocity_curve | 8/9 | 1 | 1 | 4920 |

| Group | Endpoint predictor | Arm displacement RMSE (rad) | Evaluated steps |
|---|---|---:|---:|
| all | command | 0.014040 | 5755 |
| all | response_linear | 0.000853 | 5755 |
| all | velocity_linear | 0.000646 | 5755 |
| learned | command | 0.014332 | 4929 |
| learned | response_linear | 0.000887 | 4929 |
| learned | velocity_linear | 0.000666 | 4929 |

Paths: command is the joint-target straight line. response_linear uses the no-velocity response endpoint. velocity_linear uses the velocity-conditioned endpoint. velocity_curve uses that same endpoint with a quadratic trajectory constrained to the measured initial velocity. Endpoint errors are step-weighted means of arm-joint squared errors; finger joints are also predicted for clearance queries.

The advancement rule requires more detected first contacts than command-only prediction without more false alarms at the same threshold, separately reported for each velocity path.
- velocity_linear at penetration: FAIL.
- velocity_curve at penetration: FAIL.
- velocity_linear at 2mm_warning: FAIL.
- velocity_curve at 2mm_warning: FAIL.

The no-velocity response comparator is promising: on learned trajectories it detects 7/9 contacts at penetration and 8/9 at the fixed 2 mm warning threshold, with zero false alarms among 4,920 noncollision steps at either threshold. Velocity models detect 8/9 but add one false alarm. The prespecified advancement rule targeted velocity-aware paths; selecting the simpler comparator for subsequent work is an exploratory inference, not a passed prespecified gate or a task-success result. Freeze it before further validation rather than adjusting thresholds on these events.

A false alarm here is a predicted collision/margin violation on an executed noncollision step. It is not a demonstrated lost task, but it measures how often an action-changing filter would potentially interfere. Adjacent samples and repeated checkpoints are dependent, and only ten actual first-contact events are available. Higher recall alone is insufficient to justify a new controller.

Verification: all 119 complete traces, geometry logs and probes reproduce exactly; every prediction has exactly 32+step prior updates. Input, model and source hashes remain unchanged. All 105 EASE-FM tests pass, including a read-only prediction check and synthetic held-out identification test. No action changes, task recoveries, native tuning or reserved final-test evaluation occurred.

### Decision and next step

No velocity-aware path passes the prediction/false-alarm gate. Keep the original controller and do not insert this predictor into action selection. The no-velocity comparator detects 8/9 learned first contacts with no false alarms at the already fixed 2 mm threshold. Freeze that simpler model as an exploratory candidate and validate its prediction/false-alarm tradeoff on separate procedural trajectories before enabling action changes. Do not choose new thresholds from these ten events; better mean endpoint prediction alone is insufficient for collision avoidance. Geometry and velocity access must remain explicit; none of these diagnostics establishes learned superiority over analytical control.


## 4.66 E76: frozen response prediction on separate trajectories

All 112 E72 geometry-filter trajectories replay exactly, including successes and failures. These 16 placements differ from E75, but reuse the same bodies and historical development data. This is neither an independent embodiment test nor a reserved final test. The E75 implementation is imported unchanged and verified against its saved source hash. Neural weights, actions, dynamics, probes and success outcomes are unchanged.

Primary comparison: no-velocity response_linear versus command at the fixed clearance < 0.002 - 1e-6 m threshold. The online predictor uses only completed measurements, including 32 calibration steps. Scoring remains limited to descent/lowering before first forbidden contact, inclusive. Known geometry and exact current simulator state remain privileged inputs. Secondary velocity variants do not determine advancement.

| Group | Predictor | Detected first contacts | Missed in scope | False alarms | Noncollision steps |
|---|---|---:|---:|---:|---:|
| all | command | 7/18 | 11 | 8 | 5202 |
| all | response_linear | 16/18 | 2 | 10 | 5202 |
| all | velocity_linear | 18/18 | 0 | 11 | 5202 |
| all | velocity_curve | 18/18 | 0 | 11 | 5202 |
| learned | command | 6/14 | 8 | 5 | 4556 |
| learned | response_linear | 13/14 | 1 | 8 | 4556 |
| learned | velocity_linear | 14/14 | 0 | 9 | 4556 |
| learned | velocity_curve | 14/14 | 0 | 9 | 4556 |
| learned_6 | command | 0/1 | 1 | 0 | 1914 |
| learned_6 | response_linear | 1/1 | 0 | 0 | 1914 |
| learned_6 | velocity_linear | 1/1 | 0 | 0 | 1914 |
| learned_6 | velocity_curve | 1/1 | 0 | 0 | 1914 |
| learned_7 | command | 6/13 | 7 | 5 | 2642 |
| learned_7 | response_linear | 12/13 | 1 | 8 | 2642 |
| learned_7 | velocity_linear | 13/13 | 0 | 9 | 2642 |
| learned_7 | velocity_curve | 13/13 | 0 | 9 | 2642 |
| successful_learned | command | 0/0 | 0 | 0 | 3410 |
| successful_learned | response_linear | 0/0 | 0 | 0 | 3410 |
| successful_learned | velocity_linear | 0/0 | 0 | 0 | 3410 |
| successful_learned | velocity_curve | 0/0 | 0 | 0 | 3410 |

Coverage of actual first contacts:

- all: 18/30 contacts in scored phases; phase counts {'lift': 7, 'transport': 1, 'lower': 3, 'descend': 15, 'close': 2, 'retreat': 2}.
- learned: 14/26 contacts in scored phases; phase counts {'lift': 7, 'transport': 1, 'lower': 3, 'close': 2, 'descend': 11, 'retreat': 2}.
- learned_6: 1/5 contacts in scored phases; phase counts {'lift': 3, 'transport': 1, 'lower': 1}.
- learned_7: 13/21 contacts in scored phases; phase counts {'close': 2, 'descend': 11, 'lift': 4, 'lower': 2, 'retreat': 2}.
- successful_learned: 0/0 contacts in scored phases; phase counts {}.

Primary gate: **FAIL**. More detected contacts and no increase in false-alarm count are both required.

The frozen predictor fails the prespecified prediction gate. Retain the original controller; inspect false alarms and missed/out-of-scope contacts before proposing any action intervention. Do not tune this threshold on these outcomes.

False alarms are warnings on executed noncollision steps, not proven task losses. Samples within trajectories and checkpoints sharing placements are dependent. Contacts outside the selected phases cannot be counted as detected. A prediction gain does not demonstrate collision avoidance or recovered pick-and-place tasks. No action changes, native tuning or final-test evaluation occurred.

Verification: 112 complete traces, filter logs and calibration probes reproduce exactly; all 32968 task steps have 32+step prior updates. Input, checkpoint and frozen predictor hashes are unchanged.

All 105 EASE-FM regression tests pass.

## 4.67 E77: warning timing and six-joint contact coverage

Read-only diagnosis of all 96 learned E76 trajectories. No policy, fitting rule, threshold, action or original scoring label changes. E76 remains a failed prediction gate.

The response predictor has eight one-step false alarms. Seven occur one or two control steps before first contact; one occurs 83 steps before first contact. Four response-only warnings occur one or two steps (40-80 ms) before contact; one command-only warning offsets these to a net increase of three, all on placement 15 with different checkpoints. None occurs on a successful task. These are correlated observations on unchanged trajectories, not demonstrated useful interventions. A longer-horizon metric chosen after seeing this pattern would be exploratory.

| Six-joint case | Estimator | First-contact phase | Step within phase (zero-based) | Contact pairs |
|---|---|---|---:|---|
| 1 | parent_0 | lift | 25 | [['3', 'table'], ['4', 'table']] |
| 1 | parent_1 | lift | 33 | [['3', 'table'], ['4', 'table']] |
| 2 | candidate_1 | transport | 71 | [['4', 'table'], ['5', 'table']] |
| 4 | parent_2 | lower | 17 | [['gripper_right_pad', 'table']] |
| 12 | candidate_0 | lift | 3 | [['gripper_left_pad', 'table']] |

Only one of five six-joint first contacts is in the scored descent/lowering phases. Of the four omitted contacts, three involve arm-link/table contact during lift or transport; one involves a finger/table contact during lift. Exact geometry-to-body mappings and preceding phase completion flags are saved in contacts.json. End-effector clearance alone cannot cover these failures.

Next: run the frozen predictor across every pre-contact phase with successful-run controls. Keep the original one-step metric and fixed threshold, report phase-specific coverage and full warning lead times separately, and require whole-arm geometry. This is a new development diagnostic, not a revision of the failed E76 gate. Only after that audit should a bounded action intervention be tested for full-task recoveries and regressions.

No new simulation or training occurred. All 112 E76 trajectories had previously replayed exactly; this audit reads all 96 learned traces and verifies source hashes. It neither improves success rates nor establishes independent generalization.

Audit execution: the incomplete v1 directory is preserved. Its assertion incorrectly expected three response-only warnings from the net false-alarm increase; v2 correctly records four response-only warnings and one command-only warning. No experiment inputs or thresholds changed.

## 4.68 E78: frozen response prediction on all task phases

All 112 E72 geometry-filter trajectories replay exactly, including successes and failures. These 16 placements differ from E75, but reuse the same bodies and historical development data. This is neither an independent embodiment test nor a reserved final test. The E75 response model is imported unchanged; the step implementation changes only phase eligibility. Every previously evaluated record and every endpoint error must exactly match E76. Neural weights, actions, dynamics, probes and success outcomes are unchanged.

Primary comparison: no-velocity response_linear versus command at the fixed clearance < 0.002 - 1e-6 m threshold. The online predictor uses only completed measurements, including 32 calibration steps. Scoring covers every phase before first forbidden contact, inclusive. Phase rows contain learned trajectories only. Exact warning lead times are descriptive in warnings.json; no longer-horizon relabeling is used. Known geometry and exact current simulator state remain privileged inputs. Secondary velocity variants do not determine advancement.

| Group | Predictor | Detected first contacts | Missed in scope | False alarms | Noncollision steps |
|---|---|---:|---:|---:|---:|
| all | command | 16/30 | 14 | 414 | 23726 |
| all | response_linear | 28/30 | 2 | 419 | 23726 |
| all | velocity_linear | 30/30 | 0 | 416 | 23726 |
| all | velocity_curve | 30/30 | 0 | 416 | 23726 |
| learned | command | 15/26 | 11 | 280 | 20612 |
| learned | response_linear | 25/26 | 1 | 288 | 20612 |
| learned | velocity_linear | 26/26 | 0 | 285 | 20612 |
| learned | velocity_curve | 26/26 | 0 | 285 | 20612 |
| learned_6 | command | 3/5 | 2 | 123 | 9756 |
| learned_6 | response_linear | 5/5 | 0 | 137 | 9756 |
| learned_6 | velocity_linear | 5/5 | 0 | 133 | 9756 |
| learned_6 | velocity_curve | 5/5 | 0 | 133 | 9756 |
| learned_7 | command | 12/21 | 9 | 157 | 10856 |
| learned_7 | response_linear | 20/21 | 1 | 151 | 10856 |
| learned_7 | velocity_linear | 21/21 | 0 | 152 | 10856 |
| learned_7 | velocity_curve | 21/21 | 0 | 152 | 10856 |
| successful_learned | command | 0/0 | 0 | 92 | 17409 |
| successful_learned | response_linear | 0/0 | 0 | 110 | 17409 |
| successful_learned | velocity_linear | 0/0 | 0 | 106 | 17409 |
| successful_learned | velocity_curve | 0/0 | 0 | 106 | 17409 |
| phase_approach | command | 0/0 | 0 | 0 | 544 |
| phase_approach | response_linear | 0/0 | 0 | 0 | 544 |
| phase_approach | velocity_linear | 0/0 | 0 | 0 | 544 |
| phase_approach | velocity_curve | 0/0 | 0 | 0 | 544 |
| phase_close | command | 0/2 | 2 | 0 | 3335 |
| phase_close | response_linear | 2/2 | 0 | 0 | 3335 |
| phase_close | velocity_linear | 2/2 | 0 | 0 | 3335 |
| phase_close | velocity_curve | 2/2 | 0 | 0 | 3335 |
| phase_descend | command | 5/11 | 6 | 3 | 2661 |
| phase_descend | response_linear | 10/11 | 1 | 7 | 2661 |
| phase_descend | velocity_linear | 11/11 | 0 | 8 | 2661 |
| phase_descend | velocity_curve | 11/11 | 0 | 8 | 2661 |
| phase_lift | command | 6/7 | 1 | 153 | 2805 |
| phase_lift | response_linear | 7/7 | 0 | 138 | 2805 |
| phase_lift | velocity_linear | 7/7 | 0 | 138 | 2805 |
| phase_lift | velocity_curve | 7/7 | 0 | 138 | 2805 |
| phase_lower | command | 1/3 | 2 | 2 | 1895 |
| phase_lower | response_linear | 3/3 | 0 | 1 | 1895 |
| phase_lower | velocity_linear | 3/3 | 0 | 1 | 1895 |
| phase_lower | velocity_curve | 3/3 | 0 | 1 | 1895 |
| phase_release | command | 0/0 | 0 | 0 | 2880 |
| phase_release | response_linear | 0/0 | 0 | 23 | 2880 |
| phase_release | velocity_linear | 0/0 | 0 | 19 | 2880 |
| phase_release | velocity_curve | 0/0 | 0 | 19 | 2880 |
| phase_retreat | command | 2/2 | 0 | 16 | 2217 |
| phase_retreat | response_linear | 2/2 | 0 | 13 | 2217 |
| phase_retreat | velocity_linear | 2/2 | 0 | 13 | 2217 |
| phase_retreat | velocity_curve | 2/2 | 0 | 13 | 2217 |
| phase_settle | command | 0/0 | 0 | 0 | 2800 |
| phase_settle | response_linear | 0/0 | 0 | 0 | 2800 |
| phase_settle | velocity_linear | 0/0 | 0 | 0 | 2800 |
| phase_settle | velocity_curve | 0/0 | 0 | 0 | 2800 |
| phase_transport | command | 1/1 | 0 | 106 | 1475 |
| phase_transport | response_linear | 1/1 | 0 | 106 | 1475 |
| phase_transport | velocity_linear | 1/1 | 0 | 106 | 1475 |
| phase_transport | velocity_curve | 1/1 | 0 | 106 | 1475 |

Coverage of actual first contacts:

- all: 30/30 contacts scored across all phases; phase counts {'lift': 7, 'transport': 1, 'lower': 3, 'descend': 15, 'close': 2, 'retreat': 2}.
- learned: 26/26 contacts scored across all phases; phase counts {'lift': 7, 'transport': 1, 'lower': 3, 'close': 2, 'descend': 11, 'retreat': 2}.
- learned_6: 5/5 contacts scored across all phases; phase counts {'lift': 3, 'transport': 1, 'lower': 1}.
- learned_7: 21/21 contacts scored across all phases; phase counts {'close': 2, 'descend': 11, 'lift': 4, 'lower': 2, 'retreat': 2}.
- successful_learned: 0/0 contacts scored across all phases; phase counts {}.
- phase_approach: 0/0 contacts scored across all phases; phase counts {}.
- phase_close: 2/2 contacts scored across all phases; phase counts {'close': 2}.
- phase_descend: 11/11 contacts scored across all phases; phase counts {'descend': 11}.
- phase_lift: 7/7 contacts scored across all phases; phase counts {'lift': 7}.
- phase_lower: 3/3 contacts scored across all phases; phase counts {'lower': 3}.
- phase_release: 0/0 contacts scored across all phases; phase counts {}.
- phase_retreat: 2/2 contacts scored across all phases; phase counts {'retreat': 2}.
- phase_settle: 0/0 contacts scored across all phases; phase counts {}.
- phase_transport: 1/1 contacts scored across all phases; phase counts {'transport': 1}.

Primary gate: **FAIL**. More detected contacts and no increase in false-alarm count are both required.

The frozen predictor fails the prespecified prediction gate. Retain the original controller; inspect false alarms and missed contacts before proposing any action intervention. Do not tune this threshold on these outcomes.

False alarms are warnings on executed noncollision steps, not proven task losses. Samples within trajectories and checkpoints sharing placements are dependent. All first contacts are covered; collision-step detection still differs from demonstrating collision avoidance. A prediction gain does not demonstrate collision avoidance or recovered pick-and-place tasks. No action changes, native tuning or final-test evaluation occurred.

Verification: 112 complete traces, filter logs and calibration probes reproduce exactly; all 32968 task steps have 32+step prior updates. Input, checkpoint and frozen predictor hashes are unchanged.

Geometry caveat: clearance paths move the arm and finger joints while holding the object at its current pose. Carried-object motion is not predicted. Warnings are clearance-margin violations, not guaranteed next-step physical contacts; phase-specific false alarms do not by themselves identify their cause. Successful-run warnings are especially relevant before any action-changing use.

All 105 EASE-FM tests pass. Structural comparison verifies that phase eligibility is the only change to the copied response class.

Retain the controller. Before an action-changing trial, diagnose the missed contact pairs and whether warnings reflect imminent contact or conservative geometry checks; do not alter the fixed threshold or relabel this gate.

## 4.69 E79: geometry attribution of successful-run warnings

All 110 E78 response-predictor warnings on successful learned runs arise in two six-joint trajectories. Both are replayed, together with the single missed learned collision. All three full traces, probes, geometry-filter logs and prediction records reproduce exactly. This is an outcome-selected development diagnostic, not an independent performance test.

| Case / estimator | Outcome | Diagnosed steps | Phase counts | Actual substep margin warnings | Margin already violated at step start |
|---|---|---:|---|---:|---:|
| 1 / candidate_2 | success | 87 | {'lift': 82, 'transport': 5} | 86 | 85 |
| 2 / parent_0 | success | 23 | {'release': 23} | 0 | 0 |
| 10 / candidate_0 | failure | 1 | {'descend': 1} | 1 | 0 |

**Lift/transport: real near-table motion.** Case 1 / candidate_2 has 87 warnings (82 lift, five transport), all involving link1 or link2 against the table. The actual sampled motion enters the fixed 2 mm margin on 86 steps, and 85 start inside it. The smallest measured clearance is about 0.298 mm; none of these steps penetrates. These warnings mostly describe genuinely close but successful motion, rather than a fictional carried-object collision. They remain false alarms under the original next-step physical-contact metric.

**Release: predicted proximity not observed.** Case 2 / parent_0 has 23 warnings involving link4 and the left gripper pad. Predicted minimum clearance ranges from 1.669 to 1.989 mm; measured substep clearance stays at the 2 mm reporting cap throughout. This identifies a prediction/path error for this pair, but does not isolate whether arm or finger response, or path interpolation, causes it.

**Missed collision: velocity matters in this example.** Case 10 / candidate_0 collides during descent at step 53. Command-only and no-velocity response paths stay at the 2 mm reporting cap, while the measured right-pad/table clearance reaches -0.369 mm. Both existing velocity-aware paths predict -0.085 mm and detect it. This single selected event does not establish a velocity-aware controller advantage or reverse its false-alarm results.

The object pose is included in retrospective actual-state clearance reconstruction. None of the closest predicted warning pairs involves the object, so frozen carried-object pose does not explain these particular warning minima. All clearances use the same expanded-margin shadow model; the physical model is unchanged. Twenty substeps plus the starting state are sampled, not a mathematical continuous-time clearance proof.

Decision: preserve E78 as a failed gate and keep actions unchanged. Before deploying a warning-triggered filter, distinguish an existing small clearance from a newly predicted approach to contact. A useful next controlled diagnostic is same-state, one-step action comparison at actual collisions and successful near-margin states, with no threshold retuning. Report avoidance, new collisions and task-direction error; one-step avoidance would still require full-task retention validation. Do not simply stop every time the 2 mm margin is entered.

Verification: three exact full replays, 111 attributed steps, 21 samples per predicted path and 21 measured states per diagnosed step. Prediction records match E78 exactly; input, checkpoint and source hashes verified. No training, action intervention, reserved-test evaluation or task-success improvement.

## 4.70 E80: bounded action counterfactuals with successful controls

Compare five actions from identical states at every E78 learned first contact (26 states) and every response warning on successful learned runs (110 states from two six-joint trajectories). Each branch advances one 40 ms control step with original incoming velocity, finger aperture and physics. The selected action reproduces the original observation and contact label exactly. Full reference trajectories and filter logs also replay exactly.

Actions: selected geometry-filter action; half or quarter of its task-motion component while preserving clipped measured bias; bias-only; analytical Jacobian from the same measured Gram/cross history and bias. All commands remain clipped to +/-0.04. Alternatives bypass the geometry filter, so this evaluates raw bounded alternatives, not a completed new policy. Cloned branches do not update the controller or alter the reference rollout.

| State group | Action | Contacts / states | Mean task-error change vs selected (mm-equivalent) | Worse-error states | Lost all pad contacts vs selected |
|---|---|---:|---:|---:|---:|
| collision | selected | 26/26 | 0.0000 | 0 | 0 |
| collision | half | 26/26 | 0.1286 | 9 | 0 |
| collision | quarter | 25/26 | 0.1901 | 9 | 0 |
| collision | bias_only | 22/26 | 0.2477 | 9 | 0 |
| collision | analytical_same_history | 16/26 | -1.8649 | 5 | 0 |
| collision_6 | selected | 5/5 | 0.0000 | 0 | 0 |
| collision_6 | half | 5/5 | 0.1020 | 1 | 0 |
| collision_6 | quarter | 5/5 | 0.1698 | 1 | 0 |
| collision_6 | bias_only | 5/5 | 0.2380 | 1 | 0 |
| collision_6 | analytical_same_history | 2/5 | -2.7625 | 0 | 0 |
| collision_7 | selected | 21/21 | 0.0000 | 0 | 0 |
| collision_7 | half | 21/21 | 0.1349 | 8 | 0 |
| collision_7 | quarter | 20/21 | 0.1950 | 8 | 0 |
| collision_7 | bias_only | 17/21 | 0.2500 | 8 | 0 |
| collision_7 | analytical_same_history | 14/21 | -1.6511 | 5 | 0 |
| successful_controls | selected | 0/110 | 0.0000 | 0 | 0 |
| successful_controls | half | 0/110 | 0.0434 | 2 | 0 |
| successful_controls | quarter | 0/110 | 0.0658 | 2 | 0 |
| successful_controls | bias_only | 0/110 | 0.0884 | 2 | 0 |
| successful_controls | analytical_same_history | 68/110 | -4.6418 | 1 | 0 |

Task error is sqrt(position_error^2 + (0.15 * tool_direction_error)^2), following the controller task scaling. Positive change means worse progress than the selected action. Worse-error counts use only a 1e-9 numerical tolerance; magnitudes, not these counts alone, determine practical relevance. Pad counts are the existing within-step contact union, not proof of a stable retained grasp.

This is outcome-selected, retrospective diagnosis with privileged simulator cloning. The 110 controls are dependent samples from only two successful six-joint trajectories; they do not establish broad retention. A branch avoiding one contact can fail on the next step. No branch is selected by an executable prediction rule here, and no full task is recovered. E78 remains a failed prediction gate.

Next decision must use both collision avoidance and successful-control interference. A promising action family would still need an explicit causal selection rule and complete-task validation against unchanged controls, including all casewise regressions and analytical comparisons. Do not substitute one-step avoided-contact counts for task success.

### Interpretation and next decision

Half motion avoids 0/26 imminent contacts, quarter motion 1/26, and bias-only 4/26; none avoids any of the five six-joint contacts. These three alternatives introduce no contacts in the 110 successful controls. Analytical same-history control avoids 10/26 (3/5 six-joint), but introduces 68/110 contacts in successful controls. Its better average task-direction error therefore does not make it an acceptable universal fallback.

The existing geometry filter already selects scale zero (clipped measured bias only) in 13/26 collision states and 108/110 successful-control states; additional scaling then cannot change the command. Nine collision states and 85 successful controls also have no feasible filter candidate. This is a restricted candidate-action limitation as well as a possible timing limitation; timing causality is not established here.

Next compare bounded interventions at earlier causally available warning states, keeping all successful controls and reporting the number of collisions without advance warning. Use the same fixed warning rule rather than choosing a lookback from each known collision. Evaluate short multi-step branches to determine whether avoidance merely delays contact, before any complete-task rollout. Do not install the analytical fallback or claim task recoveries from this experiment.

Verification: 28 full reference trajectories and their filter logs replay exactly; all 136 selected-action branches match their recorded positions and contact labels. Five branches per state produce 680 one-step comparisons. Clone restoration and input/source/checkpoint hashes pass. No policy weights or original task outcomes change.

## 4.71 E81: early-warning intervention pulses

The first causal no-velocity response warning at the unchanged 2 mm threshold triggers each local experiment, regardless of eventual outcome. No warning is selected by counting backward from a known collision. Warning values come from the previously exact-replayed E78 run. Collisions without advance warning remain in coverage counts.

Every triggered state has four 20-step branches (0.8 seconds). Half, quarter and bias-only interventions last five steps (0.2 seconds), followed by fifteen steps of baseline control. The selected branch uses baseline control throughout. Controller history and bias update from each branch independently. Phase goal and aperture stay fixed at their triggering values, so these are matched local continuations, not the original phase-switching full task. No future phase information or analytical fallback is used.

| DOF | Original collisions | Advance warning | Warning only on collision step | No warning | Successful warning trajectories |
|---|---:|---:|---:|---:|---:|
| all | 26 | 14 | 11 | 1 | 2 |
| 6 | 5 | 3 | 2 | 0 | 2 |
| 7 | 21 | 11 | 9 | 1 | 0 |

| Group | Horizon (steps) | Action | Contacts / branches | Avoided vs matched | New vs matched | Mean task-error change (mm-equivalent) |
|---|---:|---|---:|---:|---:|---:|
| original_collisions | 5 | selected | 20/25 | 0 | 0 | 0.0000 |
| original_collisions | 5 | half | 20/25 | 0 | 0 | 1.2116 |
| original_collisions | 5 | quarter | 20/25 | 0 | 0 | 1.8571 |
| original_collisions | 5 | bias_only | 21/25 | 0 | 1 | 2.5543 |
| original_collisions | 20 | selected | 24/25 | 0 | 0 | 0.0000 |
| original_collisions | 20 | half | 24/25 | 0 | 0 | -0.8231 |
| original_collisions | 20 | quarter | 24/25 | 0 | 0 | 2.5619 |
| original_collisions | 20 | bias_only | 24/25 | 0 | 0 | 1.8077 |
| original_collisions_6 | 5 | selected | 3/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 5 | half | 3/5 | 0 | 0 | 0.4535 |
| original_collisions_6 | 5 | quarter | 3/5 | 0 | 0 | 0.5991 |
| original_collisions_6 | 5 | bias_only | 3/5 | 0 | 0 | 0.7416 |
| original_collisions_6 | 20 | selected | 5/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 20 | half | 5/5 | 0 | 0 | -1.8676 |
| original_collisions_6 | 20 | quarter | 5/5 | 0 | 0 | 9.9360 |
| original_collisions_6 | 20 | bias_only | 5/5 | 0 | 0 | 10.5417 |
| successful_controls | 5 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | half | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | quarter | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | bias_only | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | half | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | quarter | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | bias_only | 0/2 | 0 | 0 | 0.0000 |

Task error uses sqrt(position_error^2 + (0.15 * tool_direction_error)^2). Positive changes mean worse progress. Contact outcomes are cumulative within each horizon; per-step contact and error traces are retained. Avoidance is relative to the matched phase-held baseline, not necessarily the original recorded collision. A contact after the five-step pulse but before step 20 counts against longer-horizon avoidance.

Verification: 27 full reference trajectories and filter logs reproduce exactly; selected-branch first observations and contact labels match their originals. Cloned state restoration and input/source/checkpoint hashes pass. 108 branches contain 2160 simulated control steps.

Limitations: historical development data, outcome reporting on the same bodies, and successful warnings concentrated in two trajectories. The phase-held horizon can itself create contacts in an originally successful task. These results cannot establish full-task recovery, broad retention, or an independent safety guarantee. No neural training or reserved-test evaluation occurred; the original controller and E78 failed gate remain unchanged.

### Decision

Reject this five-step motion-scaling pulse as a recovery mechanism. Of 26 original collisions, 14 have advance warning, 11 first warn on the collision step, and one has no warning. Among the 27 triggered trajectories (25 original collision runs and two successful controls), matched baseline contact counts at steps 1/5/10/20 are 11/20/22/24. Bias-only gives 7/21/22/24: the four avoided first-step contacts do not survive the five-step horizon, and one additional contact occurs by step five. Every intervention and baseline has the same 24 colliding branches by step 20, including all five six-joint collision cases. Both successful controls remain collision-free, with no terminal task-error difference.

The one triggered original-collision trajectory that stays clear for 20 steps is also clear under the matched baseline, so it is not an intervention recovery. The original collision with no warning is absent from branch comparisons but retained in coverage. No task-success improvement is established.

Next change the candidate action family rather than tune another slowing pulse: evaluate bounded direction corrections based on the frozen measured-response predictor and whole-arm clearance, with explicit progress constraints and matched successful controls. Candidate selection must use only current state/history and predicted geometry, never the outcomes of simulator branches. Keep the existing controller as the reference and require multi-step and then complete-task retention. This is a proposed new development method; its feasibility and benefit are untested.

## 4.72 E82: response-based direction corrections

The first causal no-velocity response warning at the unchanged 2 mm threshold triggers each local experiment, regardless of eventual outcome. No warning is selected by counting backward from a known collision. Warning values come from the previously exact-replayed E78 run. Collisions without advance warning remain in coverage counts.

Each triggering state has three matched 20-step branches: baseline, direction correction for five steps then baseline, and correction for all 20 steps. Candidate actions change one arm joint by +/-0.01 rad, clipped to +/-0.04. Frozen no-velocity response predictions score 21-point whole-arm clearance paths. A correction requires >1e-6 clearance improvement and predicted task error no more than 0.001 above baseline, using the learned Jacobian. The fixed 2 mm warning threshold, candidate set and progress bound are not tuned. Geometry is privileged; object pose stays fixed during path prediction. No future simulator outcomes select actions.

Controller and response history update independently from completed measurements in each branch. The triggering phase goal and aperture remain fixed. Both pulse and sustained variants are reported; neither is selected retrospectively as a validated controller.

| DOF | Original collisions | Advance warning | Warning only on collision step | No warning | Successful warning trajectories |
|---|---:|---:|---:|---:|---:|
| all | 26 | 14 | 11 | 1 | 2 |
| 6 | 5 | 3 | 2 | 0 | 2 |
| 7 | 21 | 11 | 9 | 1 | 0 |

| Group | Horizon (steps) | Action | Contacts / branches | Avoided vs matched | New vs matched | Mean task-error change (mm-equivalent) |
|---|---:|---|---:|---:|---:|---:|
| original_collisions | 5 | selected | 20/25 | 0 | 0 | 0.0000 |
| original_collisions | 5 | direction_pulse | 14/25 | 6 | 0 | -0.3054 |
| original_collisions | 5 | direction_all | 14/25 | 6 | 0 | -0.3054 |
| original_collisions | 20 | selected | 24/25 | 0 | 0 | 0.0000 |
| original_collisions | 20 | direction_pulse | 17/25 | 7 | 0 | -0.8317 |
| original_collisions | 20 | direction_all | 15/25 | 9 | 0 | -1.4137 |
| original_collisions_6 | 5 | selected | 3/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 5 | direction_pulse | 2/5 | 1 | 0 | -0.2485 |
| original_collisions_6 | 5 | direction_all | 2/5 | 1 | 0 | -0.2485 |
| original_collisions_6 | 20 | selected | 5/5 | 0 | 0 | 0.0000 |
| original_collisions_6 | 20 | direction_pulse | 2/5 | 3 | 0 | -3.0751 |
| original_collisions_6 | 20 | direction_all | 2/5 | 3 | 0 | -3.3547 |
| successful_controls | 5 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 5 | direction_pulse | 0/2 | 0 | 0 | 0.1963 |
| successful_controls | 5 | direction_all | 0/2 | 0 | 0 | 0.1963 |
| successful_controls | 20 | selected | 0/2 | 0 | 0 | 0.0000 |
| successful_controls | 20 | direction_pulse | 0/2 | 0 | 0 | 0.2659 |
| successful_controls | 20 | direction_all | 0/2 | 0 | 0 | 0.2659 |

Task error uses sqrt(position_error^2 + (0.15 * tool_direction_error)^2). Positive changes mean worse progress. Contact outcomes are cumulative within each horizon; per-step contact and error traces are retained. Avoidance is relative to the matched phase-held baseline, not necessarily the original recorded collision. A contact after the five-step pulse but before step 20 counts against longer-horizon avoidance.

Verification: 27 full reference trajectories and filter logs reproduce exactly; selected-branch first observations and contact labels match their originals. All 20 matched baseline steps also equal E81 exactly. Cloned state restoration and input/source/checkpoint hashes pass. 81 branches contain 1620 simulated control steps.

Limitations: historical development data, outcome reporting on the same bodies, and successful warnings concentrated in two trajectories. The phase-held horizon can itself create contacts in an originally successful task. These results cannot establish full-task recovery, broad retention, or an independent safety guarantee. No neural training or reserved-test evaluation occurred; the original controller and E78 failed gate remain unchanged.

Correction costs (additional geometry queries, excluding the original filter):

- direction_pulse: 89/135 checked actions changed; 35133 geometry queries.
- direction_all: 199/540 checked actions changed; 98322 geometry queries.

All 107 EASE-FM tests pass, including correction bounds/input preservation and rejection of a clearance improvement that violates the progress constraint. These are implementation checks, not evidence of physical safety.

### Decision and next experiment

Direction changes produce retained local avoidance where motion scaling did not. At 20 steps the baseline has 24/27 colliding branches, the five-step correction pulse 17/27, and sustained correction 15/27. These are seven and nine paired avoided contacts with no new contacts, respectively. Both variants avoid three of five six-joint contacts. Both successful warning controls remain clear, with mean terminal task-error increase of 0.266 mm-equivalent; only two such controls are available.

This supports a complete-task development retention experiment, not a success claim. Freeze the candidate set, no-velocity predictor, clearance threshold and progress constraint. Evaluate sustained correction as the exploratory primary candidate and report the pulse variant as a secondary comparison on all original placements and checkpoints, including tasks that previously produced no warnings. Use online triggering on the altered trajectory, original phase transitions and budgets, the unchanged geometry-filter baseline and analytical comparisons. Require improved complete-task success with no increased collisions and preservation of every baseline-successful six-joint case; record all regressions, interaction and query costs. Do not select or retune using individual outcome labels.

Both variants were examined on these same historical development states; choosing sustained correction for the next test is exploratory. Every actual gain must subsequently replicate on separate placements and embodiments before a generalization claim. Original task success and the failed E78 prediction gate remain unchanged.

## 4.73 E83: complete-task retention with frozen direction correction

All 16 original placements and seven estimators are evaluated in three modes (336 full pick-and-place tasks). Sustained correction is the exploratory primary candidate; one five-step pulse at the first online warning is secondary. All phase transitions, budgets, collision rules and final success criteria are unchanged. Calibration is identical across modes; all 112 baseline traces, probes and filter logs reproduce exactly.

The correction implementation and numerical settings are frozen from E82. Selection uses causal measured-response predictions, whole-arm geometry and the learned Jacobian task-progress constraint; analytical comparison rows use their corresponding analytical estimator. Geometry and exact current simulator state remain privileged information, and object motion is not predicted. No simulation lookahead selects actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 68/96 | 26 | 31622 | 0 | 0 |
| learned | sustained | 71/96 | 18 | 31079 | 797 | 1106931 |
| learned | pulse | 69/96 | 24 | 31315 | 91 | 455658 |
| learned_6 | baseline | 36/42 | 5 | 12440 | 0 | 0 |
| learned_6 | sustained | 37/42 | 2 | 12266 | 233 | 290094 |
| learned_6 | pulse | 37/42 | 3 | 12280 | 22 | 202230 |
| learned_7 | baseline | 32/54 | 21 | 19182 | 0 | 0 |
| learned_7 | sustained | 34/54 | 16 | 18813 | 564 | 816837 |
| learned_7 | pulse | 32/54 | 21 | 19035 | 69 | 253428 |
| analytical | baseline | 11/16 | 4 | 4930 | 0 | 0 |
| analytical | sustained | 11/16 | 4 | 4922 | 55 | 192570 |
| analytical | pulse | 11/16 | 4 | 5000 | 16 | 67410 |
| candidate_0 | baseline | 11/16 | 5 | 4978 | 0 | 0 |
| candidate_0 | sustained | 11/16 | 5 | 5031 | 76 | 162267 |
| candidate_0 | pulse | 11/16 | 5 | 4932 | 13 | 67977 |
| candidate_1 | baseline | 11/16 | 5 | 5309 | 0 | 0 |
| candidate_1 | sustained | 11/16 | 4 | 5340 | 206 | 239190 |
| candidate_1 | pulse | 11/16 | 5 | 5292 | 19 | 78120 |
| candidate_2 | baseline | 13/16 | 3 | 4806 | 0 | 0 |
| candidate_2 | sustained | 14/16 | 2 | 4731 | 30 | 179067 |
| candidate_2 | pulse | 13/16 | 3 | 4840 | 12 | 72891 |
| parent_0 | baseline | 12/16 | 3 | 5715 | 0 | 0 |
| parent_0 | sustained | 12/16 | 2 | 5533 | 159 | 193473 |
| parent_0 | pulse | 12/16 | 2 | 5680 | 7 | 84609 |
| parent_1 | baseline | 9/16 | 6 | 5856 | 0 | 0 |
| parent_1 | sustained | 9/16 | 3 | 5582 | 107 | 161490 |
| parent_1 | pulse | 9/16 | 6 | 5664 | 24 | 76461 |
| parent_2 | baseline | 12/16 | 4 | 4958 | 0 | 0 |
| parent_2 | sustained | 14/16 | 2 | 4862 | 219 | 171444 |
| parent_2 | pulse | 13/16 | 3 | 4907 | 16 | 75600 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- sustained: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- pulse: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| sustained | 4 | parent_2 | 6 | gain | True | False |
| sustained | 11 | parent_2 | 7 | gain | True | False |
| sustained | 11 | candidate_2 | 7 | gain | True | False |
| pulse | 4 | parent_2 | 6 | gain | True | False |

These are pooled results across checkpoints and repeated historical development placements, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

The pulse triggers once per trajectory and continues for five total steps even if a phase transition occurs; it never retriggers. Sustained correction remains available throughout. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline trace and filter-log equality, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

### Decision

**E83 completed:** Full-task sustained correction yields 71/96 learned successes versus baseline 68/96, with 18 versus 26 collisions. Pulse yields 69/96. Sustained has 3 casewise gains and 0 losses; six-joint success is 37/42. The primary retention gate passes. All 112 baselines reproduce exactly; 107 tests pass. These are historical development tasks, not independent generalization. See Section 4.73.

Freeze the sustained controller and replicate it on the separate E73 placement suite, retaining every previously successful six-joint case and reporting all regressions and query costs. Do not retune on this development gain.

All 107 EASE-FM tests pass. Local avoidance and development task gains do not establish superiority on new embodiments.

## 4.74 E84: complete-task retention with frozen direction correction

All 17 separate E73 placements and seven estimators are evaluated in two modes (238 full pick-and-place tasks). Sustained correction is the exploratory primary candidate; no pulse comparison or settings retuning is performed. All phase transitions, budgets, collision rules and final success criteria are unchanged. Calibration is identical across modes; all 119 baseline traces, probes and filter logs reproduce exactly.

The correction implementation and numerical settings are frozen from E82. Selection uses causal measured-response predictions, whole-arm geometry and the learned Jacobian task-progress constraint; analytical comparison rows use their corresponding analytical estimator. Geometry and exact current simulator state remain privileged information, and object motion is not predicted. No simulation lookahead selects actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 93/102 | 9 | 29792 | 0 | 0 |
| learned | sustained | 95/102 | 7 | 29868 | 136 | 605010 |
| learned_6 | baseline | 46/48 | 2 | 13545 | 0 | 0 |
| learned_6 | sustained | 47/48 | 1 | 13544 | 1 | 252924 |
| learned_7 | baseline | 47/54 | 7 | 16247 | 0 | 0 |
| learned_7 | sustained | 48/54 | 6 | 16324 | 135 | 352086 |
| analytical | baseline | 16/17 | 1 | 4752 | 0 | 0 |
| analytical | sustained | 16/17 | 1 | 4752 | 4 | 89460 |
| candidate_0 | baseline | 14/17 | 3 | 5303 | 0 | 0 |
| candidate_0 | sustained | 15/17 | 2 | 5291 | 6 | 101955 |
| candidate_1 | baseline | 15/17 | 2 | 4940 | 0 | 0 |
| candidate_1 | sustained | 16/17 | 1 | 4899 | 44 | 105231 |
| candidate_2 | baseline | 17/17 | 0 | 4726 | 0 | 0 |
| candidate_2 | sustained | 17/17 | 0 | 4726 | 0 | 87822 |
| parent_0 | baseline | 15/17 | 2 | 5010 | 0 | 0 |
| parent_0 | sustained | 15/17 | 2 | 5060 | 48 | 112476 |
| parent_1 | baseline | 17/17 | 0 | 5030 | 0 | 0 |
| parent_1 | sustained | 17/17 | 0 | 5030 | 0 | 94206 |
| parent_2 | baseline | 15/17 | 2 | 4783 | 0 | 0 |
| parent_2 | sustained | 15/17 | 2 | 4862 | 38 | 103320 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- sustained: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| sustained | 1 | candidate_1 | 6 | gain | True | False |
| sustained | 6 | candidate_0 | 7 | gain | True | False |

These are pooled results across checkpoints and repeated historical development placements, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

Sustained correction remains available throughout. These placements differ from E83 but reuse the same robot bodies and were previously used in development; this is not independent final-test or new-embodiment validation. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline trace and filter-log equality, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

### Replication decision

The frozen sustained controller passes the retention gate: learned success increases from 93/102 to 95/102 and collisions fall from nine to seven. Six-joint success rises from 46/48 to 47/48 and seven-joint success from 47/54 to 48/54. Two casewise gains (case1/candidate1 and case6/candidate0) occur with no losses. The analytical comparison remains16/17. Gains are pooled across checkpoint seeds; neither a single policy success rate nor independent trials should be inferred.

Across E83 and E84, sustained correction gives166/198 learned successes versus161/198 for the geometry-filter baseline, with25 versus35 collisions and no casewise success losses. This aggregation describes repeated historical development experiments, not an independent statistical test. The two suites share robot bodies and both have been used previously for diagnostics. No new-embodiment generalization claim is established.

Cost remains material: E84 adds605,010 correction geometry queries for136 changed learned actions. Including original filtering, query count is685,192 versus80,912 (about8.47x). Control steps including calibration rise from29,792 to29,868. This is not an efficiency gain, and query count is not a wall-clock latency measurement.

Next freeze this implementation and define a new-body development evaluation before examining outcomes, preserving the reserved final test. Include a matched command-displacement/no-response ablation with identical candidate set, geometry access and progress constraint, plus analytical controls, to distinguish response-model benefit from additional geometry search. Evaluate all cases and seed-level outcomes, retain failures and cost measurements, and do not tune thresholds on the two recovered trajectories.

Verification: all119 baseline trajectories, filter logs and probes reproduce exactly; causal update counts and input/checkpoint/source hashes verified. All107 EASE-FM tests pass. No neural retraining or reserved final-test evaluation occurred.

## 4.75 E85: complete-task retention with frozen direction correction

Six new parameterizations (same axis families/topology) and two fixed placements per body were saved before screening and policy execution. 11/12 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.

Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 58/66 | 8 | 19117 | 0 | 0 |
| learned | response | 59/66 | 7 | 19077 | 59 | 377601 |
| learned | command | 58/66 | 8 | 19087 | 1 | 382977 |
| learned_6 | baseline | 28/30 | 2 | 8526 | 0 | 0 |
| learned_6 | response | 28/30 | 2 | 8526 | 3 | 159642 |
| learned_6 | command | 28/30 | 2 | 8526 | 0 | 160398 |
| learned_7 | baseline | 30/36 | 6 | 10591 | 0 | 0 |
| learned_7 | response | 31/36 | 5 | 10551 | 56 | 217959 |
| learned_7 | command | 30/36 | 6 | 10561 | 1 | 222579 |
| analytical | baseline | 10/11 | 1 | 3157 | 0 | 0 |
| analytical | response | 10/11 | 1 | 3092 | 8 | 60186 |
| analytical | command | 10/11 | 1 | 3100 | 1 | 62412 |
| candidate_0 | baseline | 9/11 | 2 | 3256 | 0 | 0 |
| candidate_0 | response | 9/11 | 2 | 3300 | 15 | 67158 |
| candidate_0 | command | 9/11 | 2 | 3256 | 0 | 63840 |
| candidate_1 | baseline | 11/11 | 0 | 3092 | 0 | 0 |
| candidate_1 | response | 11/11 | 0 | 3092 | 0 | 57540 |
| candidate_1 | command | 11/11 | 0 | 3092 | 0 | 57540 |
| candidate_2 | baseline | 10/11 | 1 | 3067 | 0 | 0 |
| candidate_2 | response | 10/11 | 1 | 3155 | 11 | 62979 |
| candidate_2 | command | 10/11 | 1 | 3067 | 0 | 60543 |
| parent_0 | baseline | 9/11 | 2 | 3167 | 0 | 0 |
| parent_0 | response | 9/11 | 2 | 3175 | 8 | 62727 |
| parent_0 | command | 9/11 | 2 | 3167 | 0 | 65121 |
| parent_1 | baseline | 9/11 | 2 | 3438 | 0 | 0 |
| parent_1 | response | 9/11 | 2 | 3261 | 24 | 69321 |
| parent_1 | command | 9/11 | 2 | 3408 | 1 | 77994 |
| parent_2 | baseline | 10/11 | 1 | 3097 | 0 | 0 |
| parent_2 | response | 11/11 | 0 | 3094 | 1 | 57876 |
| parent_2 | command | 10/11 | 1 | 3097 | 0 | 57939 |
| body_0 | baseline | 12/12 | 0 | 3329 | 0 | 0 |
| body_0 | response | 12/12 | 0 | 3329 | 0 | 61845 |
| body_0 | command | 12/12 | 0 | 3329 | 0 | 61845 |
| body_1 | baseline | 12/12 | 0 | 3091 | 0 | 0 |
| body_1 | response | 12/12 | 0 | 3091 | 0 | 56847 |
| body_1 | command | 12/12 | 0 | 3091 | 0 | 56847 |
| body_2 | baseline | 9/12 | 3 | 3568 | 0 | 0 |
| body_2 | response | 9/12 | 3 | 3708 | 31 | 81858 |
| body_2 | command | 9/12 | 3 | 3568 | 0 | 77742 |
| body_3 | baseline | 11/12 | 1 | 3453 | 0 | 0 |
| body_3 | response | 11/12 | 1 | 3453 | 0 | 64743 |
| body_3 | command | 11/12 | 1 | 3453 | 0 | 64743 |
| body_4 | baseline | 4/6 | 2 | 2106 | 0 | 0 |
| body_4 | response | 4/6 | 2 | 2106 | 3 | 40950 |
| body_4 | command | 4/6 | 2 | 2106 | 0 | 41706 |
| body_5 | baseline | 10/12 | 2 | 3570 | 0 | 0 |
| body_5 | response | 11/12 | 1 | 3390 | 25 | 71358 |
| body_5 | command | 10/12 | 2 | 3540 | 1 | 80094 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- response: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- command: FAIL; improved success=False, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| response | 9 | parent_2 | 7 | gain | True | False |

These are prospective development results on new parameterizations derived from six existing bodies. They share existing axis families and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

Both correction modes remain available throughout the task. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline result and filter-log equality against a separately instantiated original GeometryArm, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

Measured-response attribution gate: **PASS**. Requires higher learned success than the command ablation without more collisions. Passing only the baseline-improvement gate does not establish response-model benefit.

Screening records:

- Candidate 0, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 1, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 2, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 3, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 4, body 2, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 5, body 2, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 6, body 3, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 7, body 3, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 8, body 4, 6 joints: rejected, 3/9 endpoints; failed phase lift.
- Candidate 9, body 4, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 10, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 11, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.

### Decision and limits of attribution

Measured-response correction passes both the baseline-retention and matched-ablation gates:59/66 learned successes with seven collisions, versus58/66 and eight collisions for both original geometry filtering and command-only correction. The single casewise gain is case9 / parent2, a seven-joint body; there are no success losses. Six-joint outcomes stay28/30 in every mode. Analytical control stays10/11 in every mode. These results do not establish learned superiority over analytical control.

Only one paired trial distinguishes response from command correction. Treat this as preliminary mechanism evidence, not a statistically established advantage. The new bodies are independent parameter perturbations of existing bodies within +/-10%, retaining their axis sequences and topology; this is limited morphology transfer, not arbitrary robot identification or transfer to quadrupeds. All new parameters and placements were saved before screening and outcomes. Candidate8 (six-joint body4) failed the lift endpoint after three accepted endpoints and was not replaced.

Cost remains large: response correction uses377,601 additional geometry queries for59 changed learned actions, versus382,977 queries and one changed action for the command ablation. Including original filtering gives428,554 response-mode queries versus51,237 baseline queries (about8.36x). Candidate evaluation counts can differ because replacing the displacement predictor also changes the fixed warning condition. No latency or efficiency improvement is claimed.

Next expand the frozen paired comparison to multiple independently sampled parameterizations per family, with candidate manifests and screening rules fixed before outcomes and no replacement. Report per-body/per-seed effects and all six-joint regressions; require more than one favorable checkpoint/body outcome before strengthening the mechanism claim. Keep reserved final-test data untouched. New axis arrangements require a separately declared evaluation and must not be conflated with this parameter-perturbation pilot.

Verification:77 baseline results and filter logs exactly match separately instantiated original GeometryArm runs; all231 primary task evaluations share their candidate's identical measured calibration. Causal update counts, input/source/checkpoint hashes and all108 EASE-FM tests pass. No neural retraining, parameter tuning after outcomes or reserved-test evaluation occurred.

## 4.76 E86: complete-task retention with frozen direction correction

Twelve new parameterizations (same axis families/topology) and two fixed placements per body were saved before screening and policy execution. 21/24 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.

Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 114/126 | 11 | 37010 | 0 | 0 |
| learned | response | 116/126 | 9 | 37080 | 197 | 755034 |
| learned | command | 114/126 | 11 | 37035 | 18 | 827253 |
| learned_6 | baseline | 53/54 | 1 | 15155 | 0 | 0 |
| learned_6 | response | 53/54 | 1 | 15192 | 35 | 292320 |
| learned_6 | command | 53/54 | 1 | 15155 | 0 | 321783 |
| learned_7 | baseline | 61/72 | 10 | 21855 | 0 | 0 |
| learned_7 | response | 63/72 | 8 | 21888 | 162 | 462714 |
| learned_7 | command | 61/72 | 10 | 21880 | 18 | 505470 |
| analytical | baseline | 17/21 | 4 | 6014 | 0 | 0 |
| analytical | response | 18/21 | 3 | 6040 | 23 | 119742 |
| analytical | command | 18/21 | 3 | 5997 | 8 | 126693 |
| candidate_0 | baseline | 21/21 | 0 | 6128 | 0 | 0 |
| candidate_0 | response | 21/21 | 0 | 6128 | 0 | 114576 |
| candidate_0 | command | 21/21 | 0 | 6128 | 0 | 114576 |
| candidate_1 | baseline | 19/21 | 2 | 5969 | 0 | 0 |
| candidate_1 | response | 19/21 | 2 | 6028 | 18 | 117390 |
| candidate_1 | command | 19/21 | 2 | 5969 | 0 | 145719 |
| candidate_2 | baseline | 20/21 | 1 | 5948 | 0 | 0 |
| candidate_2 | response | 20/21 | 1 | 5958 | 4 | 112182 |
| candidate_2 | command | 20/21 | 1 | 5948 | 0 | 111972 |
| parent_0 | baseline | 18/21 | 3 | 6377 | 0 | 0 |
| parent_0 | response | 18/21 | 3 | 6432 | 81 | 146496 |
| parent_0 | command | 18/21 | 3 | 6385 | 1 | 168525 |
| parent_1 | baseline | 16/21 | 4 | 6567 | 0 | 0 |
| parent_1 | response | 17/21 | 3 | 6513 | 93 | 151767 |
| parent_1 | command | 16/21 | 4 | 6584 | 17 | 173250 |
| parent_2 | baseline | 20/21 | 1 | 6021 | 0 | 0 |
| parent_2 | response | 21/21 | 0 | 6021 | 1 | 112623 |
| parent_2 | command | 20/21 | 1 | 6021 | 0 | 113211 |
| body_0 | baseline | 12/12 | 0 | 3238 | 0 | 0 |
| body_0 | response | 12/12 | 0 | 3238 | 0 | 59934 |
| body_0 | command | 12/12 | 0 | 3238 | 0 | 59934 |
| body_1 | baseline | 12/12 | 0 | 3678 | 0 | 0 |
| body_1 | response | 12/12 | 0 | 3678 | 0 | 69174 |
| body_1 | command | 12/12 | 0 | 3678 | 0 | 69174 |
| body_2 | baseline | 12/12 | 0 | 2965 | 0 | 0 |
| body_2 | response | 12/12 | 0 | 2965 | 0 | 54201 |
| body_2 | command | 12/12 | 0 | 2965 | 0 | 54201 |
| body_3 | baseline | 12/12 | 0 | 2927 | 0 | 0 |
| body_3 | response | 12/12 | 0 | 2927 | 0 | 53403 |
| body_3 | command | 12/12 | 0 | 2927 | 0 | 53403 |
| body_4 | baseline | 11/12 | 1 | 3488 | 0 | 0 |
| body_4 | response | 11/12 | 1 | 3488 | 0 | 65772 |
| body_4 | command | 11/12 | 1 | 3488 | 0 | 65772 |
| body_5 | baseline | 11/12 | 1 | 3681 | 0 | 0 |
| body_5 | response | 11/12 | 1 | 3729 | 16 | 76125 |
| body_5 | command | 11/12 | 1 | 3681 | 0 | 93933 |
| body_6 | baseline | 11/12 | 1 | 3705 | 0 | 0 |
| body_6 | response | 12/12 | 0 | 3702 | 1 | 69972 |
| body_6 | command | 11/12 | 1 | 3705 | 0 | 70329 |
| body_7 | baseline | 11/12 | 1 | 3530 | 0 | 0 |
| body_7 | response | 11/12 | 1 | 3551 | 3 | 67389 |
| body_7 | command | 11/12 | 1 | 3530 | 0 | 67536 |
| body_9 | baseline | 5/6 | 1 | 2347 | 0 | 0 |
| body_9 | response | 5/6 | 1 | 2384 | 35 | 55608 |
| body_9 | command | 5/6 | 1 | 2347 | 0 | 85071 |
| body_10 | baseline | 8/12 | 4 | 3910 | 0 | 0 |
| body_10 | response | 8/12 | 4 | 3882 | 70 | 94626 |
| body_10 | command | 8/12 | 4 | 3918 | 10 | 104496 |
| body_11 | baseline | 9/12 | 2 | 3541 | 0 | 0 |
| body_11 | response | 10/12 | 1 | 3536 | 72 | 88830 |
| body_11 | command | 9/12 | 2 | 3558 | 8 | 103404 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- response: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- command: FAIL; improved success=False, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| response | 13 | parent_1 | 7 | gain | True | False |
| response | 16 | analytical | 6 | gain | True | False |
| response | 20 | parent_2 | 7 | gain | True | False |
| command | 16 | analytical | 6 | gain | True | False |

These are prospective development results on new parameterizations derived from six existing body templates. They share existing axis families and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

Both correction modes remain available throughout the task. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline result and filter-log equality against a separately instantiated original GeometryArm, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

Measured-response attribution gate: **PASS**. Requires higher learned success than the command ablation without more collisions. Passing only the baseline-improvement gate does not establish response-model benefit.

Screening records:

- Candidate 0, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 1, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 2, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 3, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 4, body 2, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 5, body 2, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 6, body 3, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 7, body 3, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 8, body 4, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 9, body 4, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 10, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 11, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 12, body 6, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 13, body 6, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 14, body 7, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 15, body 7, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 16, body 8, 6 joints: rejected, 0/9 endpoints; failed phase approach.
- Candidate 17, body 8, 6 joints: rejected, 3/9 endpoints; failed phase lift.
- Candidate 18, body 9, 6 joints: rejected, 0/9 endpoints; failed phase approach.
- Candidate 19, body 9, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 20, body 10, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 21, body 10, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 22, body 11, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 23, body 11, 7 joints: accepted, 9/9 endpoints; failed phase None.

Direct response-versus-command paired effects, by accepted body (checkpoint/placement outcomes are dependent within each body):

| Body | Family | Parameter seed | DOF | Response-only successes | Command-only successes |
|---|---|---:|---:|---:|---:|
| 0 | six_amb_train_0 | 860000 | 6 | 0 | 0 |
| 1 | six_amb_train_0 | 860001 | 6 | 0 | 0 |
| 2 | six_amb_train_1 | 860002 | 6 | 0 | 0 |
| 3 | six_amb_train_1 | 860003 | 6 | 0 | 0 |
| 4 | seven_amb_train_0 | 860004 | 7 | 0 | 0 |
| 5 | seven_amb_train_0 | 860005 | 7 | 0 | 0 |
| 6 | seven_amb_train_1 | 860006 | 7 | 1 | 0 |
| 7 | seven_amb_train_1 | 860007 | 7 | 0 | 0 |
| 9 | broad_6_0 | 860009 | 6 | 0 | 0 |
| 10 | broad_7_8 | 860010 | 7 | 0 | 0 |
| 11 | broad_7_8 | 860011 | 7 | 1 | 0 |

Bodies with no accepted placement are absent from rollout comparisons but retained in screening. A pooled success difference is not evidence of consistent effects across independently sampled bodies; inspect concentration and checkpoint identity before strengthening the claim.

### Decision and scope

Response correction passes baseline retention and matched attribution:116/126 learned successes and nine collisions, versus114/126 and eleven for both baseline and command-only correction. There are two response-only recoveries and no success losses. Both recoveries are seven-joint: body6 (seven_amb_train_1, seed860006), placement case13, parent1; and body11 (broad_7_8, seed860011), case20, parent2. Thus this expansion shows gains on two distinct new parameterizations, families and checkpoints, rather than repetitions of one favorable body. The effect remains only two paired outcomes, not a statistically established broad advantage.

Six-joint results are53/54 in every mode; the expansion preserves strong baseline performance but shows no additional six-joint recovery. Analytical control improves17/21 to18/21 with either correction method, so that analytical gain cannot be attributed to measured response.

The protocol sampled12 new bodies and24 placements. Screening accepted21 placements on11 bodies. Six-joint body8 has both placements rejected (approach and lift); body9 has one approach rejection. These are retained reachability failures, not replacement candidates. Rollout success is conditional on endpoint screening. Joint axes and topology remain unchanged from the six templates.

Response correction adds755,034 geometry queries for197 changed learned actions; combined with filtering,855,780 queries versus102,272 baseline (about8.37x). Command correction adds827,253 queries for18 changes. High search cost is a material limitation; these counts do not measure real-time latency.

Across E85 and E86, response correction yields175/192 learned successes versus172/192 for baseline and command correction, with three response-specific seven-joint recoveries and no losses. This is descriptive aggregation across small, related development studies, not an independent significance test. Six-joint performance across these studies is81/84 in every mode.

Next prioritize the original cross-kinematic question: define an arrangement-disjoint development evaluation using training-family manifests to exclude already trained joint-axis sequences, while leaving reserved final-test data untouched. Freeze controller settings and include matched command-only and analytical controls, all screening failures and per-body outcomes. Do not infer transfer to genuinely new axis arrangements from these parameter-perturbation results, and do not spend additional trials on small perturbations solely to accumulate favorable counts.

Verification: all147 baseline results and filter logs match separately instantiated original controllers; all441 primary evaluations use matched calibration. Input, source and checkpoint hashes and causal update counts pass;108 EASE-FM tests pass. No retraining, outcome-driven replacement or reserved-test evaluation occurred.

## 4.77 E87: unseen joint-axis arrangements and response ablation

Eight new parameterizations on four unseen procedural axis sequences and two fixed placements per body were saved before screening and policy execution. 16/16 placements passed nine-endpoint IK screening; no rejected candidate was replaced. Evaluate all accepted cases with seven frozen estimators in three modes. Original phase budgets and success criteria remain unchanged.

Baseline uses the original geometry filter. Response mode uses frozen E82 measured-response direction correction. Command mode uses the identical correction implementation, candidate set, geometry, sampling, output clipping and learned-Jacobian progress bound, replacing predicted displacement with commanded displacement. Response histories still update identically, but are unused in command scoring. Any changed trigger frequency or query cost follows from that sole predictor difference. No dynamic simulator lookahead chooses actions.

| Group | Mode | Success | Collisions | Control steps incl. calibration | Changed actions | Additional correction queries |
|---|---|---:|---:|---:|---:|---:|
| learned | baseline | 43/96 | 40 | 35473 | 0 | 0 |
| learned | response | 46/96 | 27 | 35191 | 1752 | 1883343 |
| learned | command | 44/96 | 35 | 35283 | 114 | 2531697 |
| learned_6 | baseline | 15/48 | 27 | 17971 | 0 | 0 |
| learned_6 | response | 17/48 | 20 | 17951 | 1453 | 1423023 |
| learned_6 | command | 15/48 | 27 | 17969 | 8 | 2016609 |
| learned_7 | baseline | 28/48 | 13 | 17502 | 0 | 0 |
| learned_7 | response | 29/48 | 7 | 17240 | 299 | 460320 |
| learned_7 | command | 29/48 | 8 | 17314 | 106 | 515088 |
| analytical | baseline | 7/16 | 7 | 5493 | 0 | 0 |
| analytical | response | 9/16 | 4 | 5438 | 171 | 340830 |
| analytical | command | 8/16 | 5 | 5504 | 28 | 373128 |
| candidate_0 | baseline | 6/16 | 8 | 5548 | 0 | 0 |
| candidate_0 | response | 9/16 | 4 | 5564 | 250 | 296898 |
| candidate_0 | command | 7/16 | 6 | 5520 | 31 | 368340 |
| candidate_1 | baseline | 9/16 | 6 | 5998 | 0 | 0 |
| candidate_1 | response | 9/16 | 4 | 5812 | 258 | 278460 |
| candidate_1 | command | 9/16 | 5 | 5812 | 19 | 419076 |
| candidate_2 | baseline | 7/16 | 7 | 5974 | 0 | 0 |
| candidate_2 | response | 7/16 | 4 | 5676 | 434 | 331086 |
| candidate_2 | command | 7/16 | 6 | 5871 | 28 | 534597 |
| parent_0 | baseline | 7/16 | 7 | 6116 | 0 | 0 |
| parent_0 | response | 7/16 | 4 | 6027 | 137 | 309981 |
| parent_0 | command | 7/16 | 6 | 6249 | 23 | 437619 |
| parent_1 | baseline | 8/16 | 6 | 5705 | 0 | 0 |
| parent_1 | response | 8/16 | 6 | 5797 | 284 | 327201 |
| parent_1 | command | 8/16 | 6 | 5705 | 2 | 367353 |
| parent_2 | baseline | 6/16 | 6 | 6132 | 0 | 0 |
| parent_2 | response | 6/16 | 5 | 6315 | 389 | 339717 |
| parent_2 | command | 6/16 | 6 | 6126 | 11 | 404712 |
| body_0 | baseline | 5/12 | 1 | 4625 | 0 | 0 |
| body_0 | response | 5/12 | 1 | 4621 | 8 | 91245 |
| body_0 | command | 5/12 | 1 | 4625 | 0 | 91581 |
| body_1 | baseline | 10/12 | 2 | 3524 | 0 | 0 |
| body_1 | response | 10/12 | 2 | 3524 | 0 | 68964 |
| body_1 | command | 10/12 | 2 | 3524 | 0 | 68964 |
| body_2 | baseline | 0/12 | 12 | 4889 | 0 | 0 |
| body_2 | response | 0/12 | 12 | 4891 | 848 | 830235 |
| body_2 | command | 0/12 | 12 | 4888 | 3 | 948612 |
| body_3 | baseline | 0/12 | 12 | 4933 | 0 | 0 |
| body_3 | response | 2/12 | 5 | 4915 | 597 | 432579 |
| body_3 | command | 0/12 | 12 | 4932 | 5 | 907452 |
| body_4 | baseline | 6/12 | 0 | 4523 | 0 | 0 |
| body_4 | response | 6/12 | 0 | 4523 | 0 | 87213 |
| body_4 | command | 6/12 | 0 | 4523 | 0 | 87213 |
| body_5 | baseline | 9/12 | 3 | 3802 | 0 | 0 |
| body_5 | response | 9/12 | 3 | 3800 | 19 | 79380 |
| body_5 | command | 9/12 | 3 | 3802 | 0 | 78540 |
| body_6 | baseline | 3/12 | 8 | 5560 | 0 | 0 |
| body_6 | response | 4/12 | 2 | 5325 | 246 | 201075 |
| body_6 | command | 4/12 | 3 | 5372 | 106 | 247044 |
| body_7 | baseline | 10/12 | 2 | 3617 | 0 | 0 |
| body_7 | response | 10/12 | 2 | 3592 | 34 | 92652 |
| body_7 | command | 10/12 | 2 | 3617 | 0 | 102291 |

The advancement gate requires improved pooled learned success, no increased pooled collisions, and preservation of every baseline-successful six-joint case.

- response: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.
- command: PASS; improved success=True, nonincreasing collisions=True, six-joint casewise losses=0.

| Mode | Case | Estimator | DOF | Outcome change | Baseline collision | New collision |
|---|---:|---|---:|---|---|---|
| response | 6 | candidate_0 | 6 | gain | True | False |
| response | 7 | analytical | 6 | gain | True | False |
| response | 7 | candidate_0 | 6 | gain | True | False |
| response | 12 | candidate_0 | 7 | gain | True | False |
| response | 13 | analytical | 7 | gain | True | False |
| command | 12 | candidate_0 | 7 | gain | True | False |
| command | 13 | analytical | 7 | gain | True | False |

These are prospective development results on new parameterizations using length/dynamics templates with newly assigned axis sequences. Their exact ordered axis sequences are excluded from training and known catalogs and are pooled across dependent checkpoint/placement repetitions, not independent samples or one policy success rate. All casewise losses remain visible, even if a checkpoint retains its aggregate score. A passed gate would justify separate-placement validation, not a new-embodiment or conference-ready generalization claim.

Both correction modes remain available throughout the task. Query costs include all 21-point candidate paths and are separate from original geometry-filter queries; no runtime advantage is claimed.

Verification: full baseline result and filter-log equality against a separately instantiated original GeometryArm, identical calibration probes, one response update per measured control step, unchanged input/checkpoint/source hashes. Original artifacts and model weights are preserved. No reserved final-test evaluation or retraining occurred.

Measured-response attribution gate: **PASS**. Requires higher learned success than the command ablation without more collisions. Passing only the baseline-improvement gate does not establish response-model benefit.

Screening records:

- Candidate 0, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 1, body 0, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 2, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 3, body 1, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 4, body 2, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 5, body 2, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 6, body 3, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 7, body 3, 6 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 8, body 4, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 9, body 4, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 10, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 11, body 5, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 12, body 6, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 13, body 6, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 14, body 7, 7 joints: accepted, 9/9 endpoints; failed phase None.
- Candidate 15, body 7, 7 joints: accepted, 9/9 endpoints; failed phase None.

Direct response-versus-command paired effects, by accepted body (checkpoint/placement outcomes are dependent within each body):

| Body | Family | Parameter seed | DOF | Response-only successes | Command-only successes |
|---|---|---:|---:|---:|---:|
| 0 | axis_dev87_6_ZYXYXY | 870100 | 6 | 0 | 0 |
| 1 | axis_dev87_6_ZYXYXY | 870101 | 6 | 0 | 0 |
| 2 | axis_dev87_6_ZYXYZY | 870102 | 6 | 0 | 0 |
| 3 | axis_dev87_6_ZYXYZY | 870103 | 6 | 2 | 0 |
| 4 | axis_dev87_7_ZYZYXYX | 870104 | 7 | 0 | 0 |
| 5 | axis_dev87_7_ZYZYXYX | 870105 | 7 | 0 | 0 |
| 6 | axis_dev87_7_ZYXZXZY | 870106 | 7 | 0 | 0 |
| 7 | axis_dev87_7_ZYXZXZY | 870107 | 7 | 0 | 0 |

Bodies with no accepted placement are absent from rollout comparisons but retained in screening. A pooled success difference is not evidence of consistent effects across independently sampled bodies; inspect concentration and checkpoint identity before strengthening the claim.

Axis-specific outcomes:

| Sequence | DOF | Estimator group | Mode | Success | Collisions |
|---|---:|---|---|---:|---:|
| ZYXYXY | 6 | learned | baseline | 15/24 | 3 |
| ZYXYXY | 6 | learned | response | 15/24 | 3 |
| ZYXYXY | 6 | learned | command | 15/24 | 3 |
| ZYXYXY | 6 | analytical | baseline | 2/4 | 1 |
| ZYXYXY | 6 | analytical | response | 2/4 | 1 |
| ZYXYXY | 6 | analytical | command | 2/4 | 1 |
| ZYXYZY | 6 | learned | baseline | 0/24 | 24 |
| ZYXYZY | 6 | learned | response | 2/24 | 17 |
| ZYXYZY | 6 | learned | command | 0/24 | 24 |
| ZYXYZY | 6 | analytical | baseline | 0/4 | 4 |
| ZYXYZY | 6 | analytical | response | 1/4 | 3 |
| ZYXYZY | 6 | analytical | command | 0/4 | 4 |
| ZYZYXYX | 7 | learned | baseline | 15/24 | 3 |
| ZYZYXYX | 7 | learned | response | 15/24 | 3 |
| ZYZYXYX | 7 | learned | command | 15/24 | 3 |
| ZYZYXYX | 7 | analytical | baseline | 3/4 | 0 |
| ZYZYXYX | 7 | analytical | response | 3/4 | 0 |
| ZYZYXYX | 7 | analytical | command | 3/4 | 0 |
| ZYXZXZY | 7 | learned | baseline | 13/24 | 10 |
| ZYXZXZY | 7 | learned | response | 14/24 | 4 |
| ZYXZXZY | 7 | learned | command | 14/24 | 5 |
| ZYXZXZY | 7 | analytical | baseline | 2/4 | 2 |
| ZYXZXZY | 7 | analytical | response | 3/4 | 0 |
| ZYXZXZY | 7 | analytical | command | 3/4 | 0 |

Outcome diagnosis (mutually exclusive, collision takes precedence among failures):

| Group | Mode | Success | Collision | No grasp | Grasp without lift | Placement/release failure after lift |
|---|---|---:|---:|---:|---:|---:|
| learned_6 | baseline | 15 | 27 | 1 | 5 | 0 |
| learned_6 | response | 17 | 20 | 3 | 8 | 0 |
| learned_6 | command | 15 | 27 | 1 | 5 | 0 |
| learned_7 | baseline | 28 | 13 | 4 | 2 | 1 |
| learned_7 | response | 29 | 7 | 4 | 2 | 6 |
| learned_7 | command | 29 | 8 | 4 | 2 | 5 |
| analytical | baseline | 7 | 7 | 1 | 1 | 0 |
| analytical | response | 9 | 4 | 1 | 1 | 1 |
| analytical | command | 8 | 5 | 1 | 1 | 1 |

Grasp and lift flags denote whether the event occurred at any point. Placement/release combines the remaining final object-position, stability and finger-contact checks; these records do not isolate which final check failed. Endpoint IK feasibility does not establish successful contact-rich execution.

Baseline phase diagnostics (unreached phases can overlap within a trial and are descriptive, not proven causes):

| Sequence | Estimator group | First contact phase counts | Unreached phase counts |
|---|---|---|---|
| ZYXYXY | learned | {"lower": 3} | {"close": 7, "descend": 7, "lift": 2, "lower": 7, "release": 7, "retreat": 1, "settle": 2, "transport": 3} |
| ZYXYXY | analytical | {"lower": 1} | {"close": 1, "descend": 1, "lower": 1, "release": 1} |
| ZYXYZY | learned | {"close": 3, "descend": 17, "lower": 4} | {"close": 18, "descend": 17, "lift": 2, "lower": 24, "release": 24, "retreat": 5, "settle": 5, "transport": 2} |
| ZYXYZY | analytical | {"close": 1, "descend": 3} | {"close": 3, "descend": 3, "lower": 3, "release": 3, "retreat": 1, "settle": 1} |
| ZYZYXYX | learned | {"descend": 3} | {"close": 6, "descend": 6, "lift": 2, "lower": 4, "release": 4, "retreat": 1, "settle": 1, "transport": 1} |
| ZYZYXYX | analytical | {} | {"close": 1, "descend": 1, "lower": 1, "release": 1} |
| ZYXZXZY | learned | {"descend": 4, "lift": 5, "transport": 1} | {"close": 5, "descend": 4, "lift": 5, "lower": 5, "release": 4, "retreat": 3, "settle": 3, "transport": 9} |
| ZYXZXZY | analytical | {"lift": 1, "retreat": 1} | {"lift": 1, "retreat": 1, "settle": 1, "transport": 1} |

The lineage audit verifies four training axis sequences and excludes 36 exact sequences from the checked training/development catalogs and built-in fixtures. Reserved geometry metadata is used only for exclusion; no reserved task goals or trajectories are evaluated. E30/E35 synthetic random linear-system supervision remains part of training, and E66 measured training contexts come from the eight audited training bodies. Exact sequence disjointness does not prove inequivalent kinematic functions or independence from prior development choices.

### E87 decision and limitations

The frozen response variant passes both advancement and attribution gates, but absolute task success remains low: 46/96 pooled learned trials, including 17/48 six-joint trials. Baseline achieves 43/96 and command correction 44/96. Response has three task gains and no losses against baseline. Its two gains over command are cases 6 and 7, both candidate_0 on body 3 (ZYXYZY), so the measured-response-specific success effect has not replicated across bodies or checkpoints in this study. The seven-joint gain in case 12 occurs in both correction modes.

Learned collision counts are 40 baseline, 27 response and 35 command. Avoiding a collision often leaves an unsuccessful task: response has 23 noncollision failures, compared with 13 baseline. These mutually exclusive outcome counts do not prove that correction caused new grasp failures; many baseline failures already collide before later task checks.

Response adds 1,883,343 geometry queries, for 2,026,782 total versus 153,357 baseline (13.22 times as many). It changes 1,752 actions. No real-time or efficiency claim is supported. Analytical control improves from 7/16 to 9/16 with response, versus 8/16 command; this also argues against attributing all benefit specifically to the Transformer.

Next: run a separately labeled privileged kinematic-control diagnostic on all 16 fixed cases with the same home states, calibration, task budgets and geometry filter, preserving all outcomes. Compare exact Jacobian substitution with measured-history estimates to distinguish motion-estimation limits from shared path/contact limitations. If exact kinematics still fails, inspect path feasibility and phase transitions before retraining or tuning the correction. This diagnostic is not a deployable policy or a final-test result.

Verification completed: 336 primary task evaluations, 112 independently instantiated exact baseline replays, unchanged checkpoint/input/source hashes and calibration probes, causal response updates, and 108 passing EASE tests. The report generator also passes Python syntax compilation. Artifacts: outputs/ease_direction_novelaxis_v1.


## 4.78 E88: exact-Jacobian diagnostic on unseen arm arrangements

All 16 fixed E87 cases were evaluated with the analytical history estimator and an exact instantaneous simulator Jacobian. Only the Jacobian source changes: calibration, adaptive measured bias, geometry filter, phase sequence, task budgets, physics and success rules remain unchanged. Both variants continue updating measured statistics. Later histories differ because actions differ. No direction correction is applied. This is a privileged diagnostic, not a learned-policy result or a guaranteed performance upper bound.

| Group | Controller | Success | Collisions | Steps including calibration |
|---|---|---:|---:|---:|
| all | analytical | 7/16 | 7 | 5493 |
| all | exact_jacobian | 8/16 | 6 | 5115 |
| 6 | analytical | 2/8 | 5 | 2733 |
| 6 | exact_jacobian | 3/8 | 4 | 2646 |
| 7 | analytical | 5/8 | 2 | 2760 |
| 7 | exact_jacobian | 5/8 | 2 | 2469 |
| axis_dev87_6_ZYXYXY | analytical | 2/4 | 1 | 1229 |
| axis_dev87_6_ZYXYXY | exact_jacobian | 3/4 | 0 | 1196 |
| axis_dev87_6_ZYXYZY | analytical | 0/4 | 4 | 1504 |
| axis_dev87_6_ZYXYZY | exact_jacobian | 0/4 | 4 | 1450 |
| axis_dev87_7_ZYXZXZY | analytical | 2/4 | 2 | 1420 |
| axis_dev87_7_ZYXZXZY | exact_jacobian | 2/4 | 2 | 1143 |
| axis_dev87_7_ZYZYXYX | analytical | 3/4 | 0 | 1340 |
| axis_dev87_7_ZYZYXYX | exact_jacobian | 3/4 | 0 | 1326 |

Paired outcome changes:

- {"baseline_collision": true, "baseline_success": false, "case": 2, "dof": 6, "exact_collision": false, "exact_success": true}

Exact kinematics increases success from 7/16 to 8/16 and reduces collisions from 7 to 6. The sole recovery is six-joint case 2; no successes are lost. Six-joint success is 3/8 versus 2/8; seven-joint remains 5/8. All four ZYXYZY cases still collide and fail with exact Jacobians. Cases 1 and 8 fail without collisions in both variants.

Interpretation: local Jacobian estimation error explains at least one recoverable failure under this intervention, but correcting it does not resolve most failures. The shared controller, geometric path selection, actuator dynamics, grasp execution and phase logic remain candidate limitations. This experiment does not isolate which of them is causal, and exact local kinematics does not establish a collision-free global path.

Next: inspect the difficult six-joint family with a separately labeled path-feasibility diagnostic. Compare the actual descent path with feasible endpoint configurations and identify whether an alternate arm posture can reach the grasp without table contact. Keep fixed-case outcomes separate from any subsequently tuned controller. Do not claim broad transfer reliability or initiate more estimator training solely from this result.

Verification: 32 saved task traces; all 16 analytical baselines exactly reproduce E87; all calibration probes match; exact Jacobian call counts equal task steps and are zero during calibration; input, checkpoint and source hashes remain unchanged. All 109 EASE tests pass, including the new test that exact-Jacobian substitution preserves measured bias and statistics. No retraining or reserved-test evaluation occurred.

### E88 saved video evidence

Videos are illustrative, explicitly outcome-selected examples, not a representative success-rate sample. The manifest records reference hashes, video hashes, frame counts and exact replay checks. Playback shows complete tasks at real-time simulation speed, omitting calibration; controller, final outcome, phase and cumulative collision status are labeled.

- Matched E87 candidate_0 case 6: baseline failure and response-corrected success.
- E88 exact-Jacobian case 2: success; case 4: failure.
- Gallery: `outputs/ease_novelaxis_oracle_v1/videos/index.html`.

Ongoing recording convention: save protocol/settings before evaluation, each completed rollout immediately, aggregate results and audits at completion, and labeled video demos of both successes and failures when present. Verify video replays against saved results and disclose example selection. Preserve old artifacts.


## 4.79 E89: static descent-path feasibility on unseen arrangements

All 16 E88 exact-Jacobian rollouts replay exactly. Search starts from each recorded state immediately before descent. Twenty deterministic endpoint IK attempts per case retain the recorded object and finger poses. Every accepted endpoint receives 101 equally spaced joint-space path samples. The original contact rules apply; the separate 2 mm geometry check is also reported. This is a bounded privileged kinematic search, not an executed policy or continuous-path certificate.

| Case | DOF | Collision-free endpoints / 20 | Contact-free sampled routes | Routes with 2 mm margin | Selected minimum clearance (mm) |
|---|---:|---:|---:|---:|---:|
| 0 | 6 | 13 | 13 | 13 | 2.000 |
| 1 | 6 | 9 | 9 | 9 | 2.000 |
| 2 | 6 | 16 | 4 | 4 | 2.000 |
| 3 | 6 | 13 | 13 | 13 | 2.000 |
| 4 | 6 | 5 | 1 | 1 | 2.000 |
| 5 | 6 | 7 | 0 | 0 | -59.807 |
| 6 | 6 | 4 | 0 | 0 | -12.003 |
| 7 | 6 | 9 | 0 | 0 | -2.661 |
| 8 | 7 | 14 | 11 | 11 | 2.000 |
| 9 | 7 | 13 | 12 | 12 | 2.000 |
| 10 | 7 | 14 | 12 | 12 | 2.000 |
| 11 | 7 | 11 | 6 | 6 | 2.000 |
| 12 | 7 | 15 | 6 | 6 | 2.000 |
| 13 | 7 | 14 | 11 | 11 | 2.000 |
| 14 | 7 | 13 | 11 | 11 | 2.000 |
| 15 | 7 | 12 | 12 | 12 | 2.000 |

### E89 findings and next decision

The search finds 182 collision-free endpoints from 320 attempts and 121 sampled contact-free joint-interpolation routes. Thirteen of 16 cases have at least one route; all such routes also satisfy the sampled 2 mm margin. Endpoints may duplicate a branch and are not independent samples.

Within the difficult six-joint ZYXYZY family, case 4 has one clear route among five endpoints. Its local-start IK attempt fails; attempt 17 finds a different endpoint posture. Cases 5, 6 and 7 have seven, four and nine endpoints respectively, but no clear sampled straight joint-space route. This is evidence of posture/path sensitivity, not proof that these cases are infeasible.

Case 4 requires a maximum single-joint displacement of 4.172 radians, or 105 nominal increments of 0.04 radians, exceeding the existing 100-step descent budget under ideal incremental tracking. This is an illustrative planning-cost estimate, not a physical lower bound: actual actuator displacements can differ from commands. The 5.462-radian joint-vector distance and finite geometry samples mean the clear animation cannot be counted as recovered manipulation.

Next evaluate physical tracking of the discovered route as a separately labeled privileged diagnostic, first under the original descent budget and then a disclosed longer budget. Preserve the recorded start state and compare with the original local controller. This will distinguish static path availability from dynamic executability and budget limitations before changing the learned policy. Whole pick-and-place success must still be evaluated separately.

Verification: 16 exact full-rollout replays, identical calibration probes, saved per-attempt endpoints and per-sample contacts/clearances, unchanged input hashes, and 110 passing tests including an interior-collision regression test. Two path animations reproduce all 101 saved contact and clearance samples each. No training, final-test evaluation or new policy-success result.

### E89 video evidence

Gallery: `outputs/ease_novelaxis_paths_v1/videos/index.html`. Case 4 shows a sampled-clear route; case 5 shows a blocked route. Both prominently state KINEMATIC ANIMATION, NOT EXECUTED MOTION. Playback timing is arbitrary and does not represent actuator speed. Original executed successes/failures remain available in the E88 gallery.


## 4.80 E90: physical execution of frozen descent routes

All 16 fixed cases run in three conditions: E88 exact-Jacobian baseline; frozen E89 joint-waypoint tracking during descent with the original100-step budget; and the same tracker with300 descent steps. All other phases, budgets, measured bias updates, physics, geometry filtering and full-task success criteria are unchanged. The13 cases with a sampled-clear selected route receive the intervention; the other three fall back to the original controller with original budgets. No controller tuning occurred after outcomes.

| Group | Mode | Full-task success | Collisions | Descent target reached | Steps including calibration |
|---|---|---:|---:|---:|---:|
| all | baseline | 8/16 | 6 | 12 | 5115 |
| all | route100 | 11/16 | 4 | 13 | 5929 |
| all | route300 | 13/16 | 3 | 15 | 5818 |
| 6 | baseline | 3/8 | 4 | 5 | 2646 |
| 6 | route100 | 4/8 | 4 | 6 | 3053 |
| 6 | route300 | 5/8 | 3 | 7 | 2877 |
| 7 | baseline | 5/8 | 2 | 7 | 2469 |
| 7 | route100 | 7/8 | 0 | 7 | 2876 |
| 7 | route300 | 8/8 | 0 | 8 | 2941 |

Paired success changes:

- {"case": 1, "mode": "route100", "change": "gain"}
- {"case": 1, "mode": "route300", "change": "gain"}
- {"case": 4, "mode": "route300", "change": "gain"}
- {"case": 8, "mode": "route300", "change": "gain"}
- {"case": 12, "mode": "route100", "change": "gain"}
- {"case": 12, "mode": "route300", "change": "gain"}
- {"case": 13, "mode": "route100", "change": "gain"}
- {"case": 13, "mode": "route300", "change": "gain"}

### E90 interpretation and next decision

Full-task success increases from8/16 to11/16 under the original descent budget and13/16 with the longer budget. Route100 recovers cases1,12,13; route300 additionally recovers cases4,8. No baseline successes are lost. The difficult-family case4, previously only a static-clear route, now completes physical pick-and-place with300 descent steps. Under100 steps it reaches waypoint50/100, then proceeds through the unchanged phase sequence and fails with contact. Cases5-7 have no selected clear route and remain unchanged failures.

This establishes dynamic executability of the selected case4 route and shows that the specific waypoint tracker needs more than the original budget. It does not show that every possible controller requires that larger budget. Improved downstream outcomes after only a descent intervention support posture/path dependence, but do not isolate a single cause for every recovered case.

These are privileged development results, not improved learned-policy scores: exact Jacobians and offline geometry/IK route search are used, and E89 route selection is informed by these development cases. The longer-budget condition is explicitly not interaction-matched. Neither condition constitutes new held-out generalization evidence. Learned E87 scores remain46/96 with response correction.

Next separate the two requirements: test whether the frozen learned motion estimators can complete the remaining phases after the same privileged route intervention, with the planner contribution clearly disclosed; independently investigate multi-segment routes for cases5-7. A matched learned comparison should preserve original and longer budgets separately and retain all cases. Do not train on these outcomes and then describe these cases as unseen evaluation.

Verification:48 saved full rollouts;16 exact E88 baseline replays; six additional unchanged no-route runs; exact descent-start qpos/qvel across interventions; unchanged input hashes;112 passing tests, including waypoint waiting/clipping and non-descent action preservation. Six executed full-task videos reproduce every saved result and trace exactly; decoded frame counts and video hashes are recorded.

### E90 saved videos

Gallery: `outputs/ease_novelaxis_path_execution_v1/videos/index.html`. Cases0 and4 in all three modes were selected before evaluation. Labels disclose executed physics, privileged control, final outcome and cumulative collision. These are full-task executions, unlike the E89 static path animations.


## 4.81 E91: frozen learned estimators with privileged route assistance

All16 E87 development cases and all six frozen checkpoints are compared in baseline, route100 and route300 conditions (288 tasks). Baseline is the original geometry-filter learned controller, without E82 response correction. The planner reuses only the single frozen E89 selected endpoint, constructs a101-sample joint-space route from the learned controller's actual descent posture and accepts it only if the sampled clearance is at least2mm. Rejected or unavailable routes retain baseline actions and budgets. Route300 permits300 descent steps only for accepted routes; all other phase budgets and success rules remain unchanged.

The descent tracker uses joint feedback plus measured adaptive bias; it does not use Transformer actions for that phase. Learned Jacobian control handles approach and all later phases. Offline endpoint IK and online exact scene geometry are privileged planning inputs. No task-time simulator Jacobian calls or target-arm weight updates are allowed. These are development interventions, not an independently evaluated history-only policy.

| Group | Mode | Success | Collisions | Routes used | Steps incl. calibration | Filter queries |
|---|---|---:|---:|---:|---:|---:|
| all | baseline | 43/96 | 40 | 0 | 35473 | 153357 |
| all | route100 | 49/96 | 36 | 78 | 39441 | 167991 |
| all | route300 | 57/96 | 33 | 78 | 39284 | 151792 |
| 6 | baseline | 15/48 | 27 | 0 | 17971 | 89709 |
| 6 | route100 | 18/48 | 27 | 30 | 20476 | 106104 |
| 6 | route300 | 22/48 | 22 | 30 | 19619 | 86755 |
| 7 | baseline | 28/48 | 13 | 0 | 17502 | 63648 |
| 7 | route100 | 31/48 | 9 | 48 | 18965 | 61887 |
| 7 | route300 | 35/48 | 11 | 48 | 19665 | 65037 |
| candidate_0 | baseline | 6/16 | 8 | 0 | 5548 | 22190 |
| candidate_0 | route100 | 9/16 | 5 | 13 | 6280 | 24734 |
| candidate_0 | route300 | 11/16 | 4 | 13 | 6128 | 21768 |
| candidate_1 | baseline | 9/16 | 6 | 0 | 5998 | 26890 |
| candidate_1 | route100 | 8/16 | 6 | 13 | 6773 | 29693 |
| candidate_1 | route300 | 9/16 | 6 | 13 | 6791 | 26903 |
| candidate_2 | baseline | 7/16 | 7 | 0 | 5974 | 28216 |
| candidate_2 | route100 | 8/16 | 7 | 13 | 6471 | 27851 |
| candidate_2 | route300 | 9/16 | 6 | 13 | 6441 | 24871 |
| parent_0 | baseline | 7/16 | 7 | 0 | 6116 | 26654 |
| parent_0 | route100 | 10/16 | 5 | 13 | 6732 | 29886 |
| parent_0 | route300 | 11/16 | 5 | 13 | 6780 | 28216 |
| parent_1 | baseline | 8/16 | 6 | 0 | 5705 | 21503 |
| parent_1 | route100 | 6/16 | 7 | 13 | 6986 | 31620 |
| parent_1 | route300 | 7/16 | 7 | 13 | 7138 | 29266 |
| parent_2 | baseline | 6/16 | 6 | 0 | 6132 | 27904 |
| parent_2 | route100 | 8/16 | 6 | 13 | 6199 | 24207 |
| parent_2 | route300 | 10/16 | 5 | 13 | 6006 | 20768 |

| Mode | Case | Body | DOF | Checkpoint | Change | Baseline collision | Intervention collision |
|---|---:|---:|---:|---|---|---|---|
| route100 | 0 | 0 | 6 | parent_2 | loss | False | False |
| route300 | 0 | 0 | 6 | parent_2 | loss | False | False |
| route100 | 0 | 0 | 6 | candidate_0 | loss | False | False |
| route300 | 0 | 0 | 6 | candidate_0 | loss | False | False |
| route100 | 0 | 0 | 6 | candidate_2 | loss | False | True |
| route300 | 0 | 0 | 6 | candidate_2 | loss | False | True |
| route100 | 1 | 0 | 6 | parent_0 | gain | False | False |
| route300 | 1 | 0 | 6 | parent_0 | gain | False | False |
| route100 | 1 | 0 | 6 | parent_1 | gain | False | False |
| route300 | 1 | 0 | 6 | parent_1 | gain | False | False |
| route100 | 1 | 0 | 6 | parent_2 | gain | False | False |
| route300 | 1 | 0 | 6 | parent_2 | gain | False | False |
| route100 | 1 | 0 | 6 | candidate_0 | gain | False | False |
| route300 | 1 | 0 | 6 | candidate_0 | gain | False | False |
| route100 | 1 | 0 | 6 | candidate_1 | gain | False | False |
| route300 | 1 | 0 | 6 | candidate_1 | gain | False | False |
| route100 | 1 | 0 | 6 | candidate_2 | gain | False | False |
| route100 | 2 | 1 | 6 | parent_0 | gain | True | False |
| route300 | 2 | 1 | 6 | parent_0 | gain | True | False |
| route100 | 2 | 1 | 6 | parent_1 | gain | True | False |
| route300 | 2 | 1 | 6 | parent_1 | gain | True | False |
| route100 | 2 | 1 | 6 | candidate_0 | loss | False | True |
| route300 | 2 | 1 | 6 | candidate_0 | loss | False | True |
| route100 | 2 | 1 | 6 | candidate_1 | loss | False | True |
| route300 | 2 | 1 | 6 | candidate_1 | loss | False | True |
| route300 | 4 | 2 | 6 | parent_0 | gain | True | False |
| route300 | 4 | 2 | 6 | parent_1 | gain | True | False |
| route300 | 4 | 2 | 6 | parent_2 | gain | True | False |
| route300 | 4 | 2 | 6 | candidate_0 | gain | True | False |
| route300 | 4 | 2 | 6 | candidate_2 | gain | True | False |
| route300 | 8 | 4 | 7 | parent_2 | gain | False | False |
| route300 | 8 | 4 | 7 | candidate_0 | gain | False | False |
| route100 | 8 | 4 | 7 | candidate_1 | loss | False | False |
| route300 | 8 | 4 | 7 | candidate_2 | gain | False | False |
| route100 | 9 | 4 | 7 | parent_1 | loss | False | True |
| route300 | 9 | 4 | 7 | parent_1 | loss | False | True |
| route100 | 9 | 4 | 7 | parent_2 | gain | False | False |
| route300 | 9 | 4 | 7 | parent_2 | gain | False | False |
| route100 | 10 | 5 | 7 | parent_0 | gain | True | False |
| route300 | 10 | 5 | 7 | parent_0 | gain | True | False |
| route100 | 10 | 5 | 7 | candidate_0 | gain | True | False |
| route300 | 10 | 5 | 7 | candidate_0 | gain | True | False |
| route100 | 10 | 5 | 7 | candidate_2 | gain | True | False |
| route300 | 10 | 5 | 7 | candidate_2 | gain | True | False |
| route100 | 12 | 6 | 7 | parent_1 | loss | False | True |
| route300 | 12 | 6 | 7 | parent_1 | loss | False | True |
| route100 | 12 | 6 | 7 | candidate_0 | gain | True | False |
| route300 | 12 | 6 | 7 | candidate_0 | gain | True | False |
| route100 | 12 | 6 | 7 | candidate_1 | gain | True | False |
| route300 | 12 | 6 | 7 | candidate_1 | gain | True | False |
| route100 | 13 | 6 | 7 | parent_1 | loss | False | True |
| route300 | 13 | 6 | 7 | parent_1 | loss | False | True |
| route100 | 13 | 6 | 7 | candidate_0 | gain | True | False |
| route300 | 13 | 6 | 7 | candidate_0 | gain | True | False |
| route100 | 14 | 7 | 7 | parent_2 | gain | True | False |
| route300 | 14 | 7 | 7 | parent_2 | gain | True | False |
| route100 | 15 | 7 | 7 | parent_1 | loss | False | False |
| route300 | 15 | 7 | 7 | parent_1 | loss | False | False |
| route100 | 15 | 7 | 7 | candidate_0 | gain | True | False |
| route300 | 15 | 7 | 7 | candidate_0 | gain | True | False |
| route100 | 15 | 7 | 7 | candidate_1 | loss | False | True |
| route300 | 15 | 7 | 7 | candidate_1 | loss | False | True |

Six-joint casewise retention checks:

- route100: FAIL, 5 previously successful six-joint cases lost.
- route300: FAIL, 5 previously successful six-joint cases lost.
- route100: 17 gains and 11 losses versus baseline.
- route300: 24 gains and 10 losses versus baseline.

Lost baseline successes: descriptive phase diagnostics (not causal attribution):

| Case | Checkpoint | Mode | Descent reached | First collision phase | Unreached phases |
|---|---|---|---|---|---|
| 0 | parent_2 | route100 | True | None | lower, release |
| 0 | parent_2 | route300 | True | None | lower, release |
| 0 | candidate_0 | route100 | True | None | transport |
| 0 | candidate_0 | route300 | True | None | transport |
| 0 | candidate_2 | route100 | True | settle | retreat, settle |
| 0 | candidate_2 | route300 | True | settle | retreat, settle |
| 2 | candidate_0 | route100 | True | lower | lower, release |
| 2 | candidate_0 | route300 | True | lower | lower, release |
| 2 | candidate_1 | route100 | True | transport | lift, transport, lower, release, retreat, settle |
| 2 | candidate_1 | route300 | True | transport | lift, transport, lower, release, retreat, settle |
| 8 | candidate_1 | route100 | False | None | descend, close |
| 9 | parent_1 | route100 | True | transport | lift, transport, lower, release, retreat, settle |
| 9 | parent_1 | route300 | True | transport | lift, transport, lower, release, retreat, settle |
| 12 | parent_1 | route100 | True | lift | lift, transport, lower, release, retreat, settle |
| 12 | parent_1 | route300 | True | lift | lift, transport, lower, release, retreat, settle |
| 13 | parent_1 | route100 | True | retreat | retreat, settle |
| 13 | parent_1 | route300 | True | retreat | retreat, settle |
| 15 | parent_1 | route100 | True | None | release, retreat |
| 15 | parent_1 | route300 | True | None | release, retreat |
| 15 | candidate_1 | route100 | True | lower | transport |
| 15 | candidate_1 | route300 | True | lower | transport |

Route eligibility: 78/96 checkpoint/case pairs have an E89 eligible endpoint; 78/96 pass the actual-start check; 0 eligible routes are rejected by that check. Each route condition makes 7878 additional geometry queries for admission, separate from per-step filter queries. Both route modes make identical admission decisions.

These counts pool dependent checkpoints and placements. Report every paired loss, not only net improvement. Different descent endpoints/postures can change all subsequent learned histories and behavior. Any gain is attributable to the combined planning intervention, not solely to the motion Transformer. E87 response-corrected scores remain unchanged, and comparisons against E90 exact-Jacobian outcomes are descriptive because the actual learned start states and admission decisions differ.

Verification: all96 baseline rollouts reproduce E87 exactly; 36 fallback rollouts also match. Descent-start qpos/qvel agree across modes for each checkpoint/case. Route checks preserve physical state. Task Jacobian calls are blocked, the neural inverse guard remains enabled, input/checkpoint/source hashes remain unchanged, and114 tests pass. Every completed trial and its trace is saved immediately. No retraining or reserved-test evaluation occurred.

### E91 decision

Both variants fail the six-joint retention gate and are not promoted. Route100 reaches49/96 success versus43/96 baseline, with17 gains and11 losses. Route300 reaches57/96, with24 gains and10 losses. Six-joint counts are15/48 baseline,18/48 route100 and22/48 route300; each intervention loses five baseline-successful six-joint cases. The longer route recovers five of six checkpoint runs on difficult case4, but blanket route assistance regresses easier cases. All78 eligible routes pass actual-start rechecking; the18 pairs without an eligible E89 route remain exact baseline fallbacks in both modes.

Of21 loss records across the two modes,20 reach the descent target before failing later. This motivates a controlled post-descent handoff diagnostic, but does not prove estimator history is the cause. Checkpoint/placement repetitions and the two modes are dependent.

Do not promote route assistance as a drop-in improvement if the casewise retention gate fails, regardless of pooled gains. Inspect handoff from the planned descent to learned Cartesian control on lost baseline successes. A useful next diagnostic compares learned and exact local Jacobians at the same recorded post-descent states, then uses a separately labeled exact-kinematics continuation to distinguish estimation/handoff errors from remaining trajectory or contact problems. Keep privileged diagnostic gains distinct from learned results and preserve every failure.

### E91 video evidence

Gallery: `outputs/ease_learned_path_transfer_v1/videos/index.html`. Candidate_0 cases0 and4 in baseline and route300 are the prespecified illustrative examples; if necessary the first missing success/failure category is appended. Videos show executed full tasks, disclose hybrid privileged-planner use, and must match saved traces exactly. Example selection is not a representative success-rate sample.


## 4.82 E92: same-state post-descent Jacobian intervention

Selection is explicitly outcome-informed: all ten E91 route300 regressions relative to the original learned baseline, plus the earliest retained success in each checkpoint/DOF stratum represented by those regressions (six controls). These16 contexts are not a representative benchmark. Each is run twice with identical learned approach and planned descent, then learned or exact-Jacobian continuation starting at close. No reset or extra probe occurs at the handoff. Adaptive measured bias, controller statistic updates, physics, geometry filter, phase budgets and success criteria remain unchanged.

| Selected group | Continuation | Success | Collisions | Total steps | Median relative Jacobian error at handoff | Error range |
|---|---|---:|---:|---:|---:|---|
| loss | learned | 0/10 | 7 | 5039 | 0.6738 | 0.5927-0.8236 |
| loss | exact_after_descent | 10/10 | 0 | 3509 | 0.6738 | 0.5927-0.8236 |
| retained_success | learned | 6/6 | 0 | 2268 | 0.5899 | 0.3577-0.9686 |
| retained_success | exact_after_descent | 6/6 | 0 | 2102 | 0.5899 | 0.3577-0.9686 |

| Group | Case | Checkpoint | DOF | Learned success | Exact success | Learned collision | Exact collision | Relative error |
|---|---:|---|---:|---|---|---|---|---:|
| loss | 0 | parent_2 | 6 | False | True | False | False | 0.6835 |
| loss | 0 | candidate_0 | 6 | False | True | False | False | 0.6641 |
| loss | 0 | candidate_2 | 6 | False | True | True | False | 0.5927 |
| loss | 2 | candidate_0 | 6 | False | True | True | False | 0.6035 |
| loss | 2 | candidate_1 | 6 | False | True | True | False | 0.6174 |
| loss | 9 | parent_1 | 7 | False | True | True | False | 0.6907 |
| loss | 12 | parent_1 | 7 | False | True | True | False | 0.7246 |
| loss | 13 | parent_1 | 7 | False | True | True | False | 0.6468 |
| loss | 15 | parent_1 | 7 | False | True | False | False | 0.8236 |
| loss | 15 | candidate_1 | 7 | False | True | True | False | 0.7582 |
| retained_success | 3 | candidate_0 | 6 | True | True | False | False | 0.3577 |
| retained_success | 0 | candidate_1 | 6 | True | True | False | False | 0.5934 |
| retained_success | 8 | candidate_1 | 7 | True | True | False | False | 0.9686 |
| retained_success | 2 | candidate_2 | 6 | True | True | False | False | 0.5270 |
| retained_success | 10 | parent_1 | 7 | True | True | False | False | 0.7914 |
| retained_success | 2 | parent_2 | 6 | True | True | False | False | 0.5863 |

### E92 interpretation

All ten regressed cases recover, and all six retained-success controls remain successful under exact continuation. Seven collisions among the regressions become zero. Since the branches share the same prefix, physical state, measured bias and sufficient statistics at handoff, this is strong controlled evidence that changing the local kinematic estimate can resolve these selected failures. The intervention uses exact Jacobians throughout the remaining task, not only at one instant, and changes subsequent actions, observations and bias estimates.

It does not establish that accumulated history is the specific cause, or that a deployable estimator repair will reproduce these recoveries. Model approximation, insufficient local excitation, nonlinear motion accumulated across the route and subsequent estimation errors remain possible mechanisms. Relative Frobenius error of the6-by-n position/scaled-tool-axis Jacobian is descriptive: the loss group median is0.674 and control median0.590, with overlapping ranges (controls extend to0.969). There is no validated failure threshold or independent statistical inference.

No learned score is improved by this diagnostic. Do not report16/16 as generalization performance: the set was selected using E91 outcomes and the successful continuation is privileged. E91 hybrid scores and failed retention gates remain unchanged.

### E92 next decision

Investigate a nonprivileged local-history repair with frozen weights. First reconstruct recent motion statistics at the recorded handoff states and compare fixed short-window estimates against the current exponentially accumulated history, without using exact Jacobians as policy inputs. Predeclare any candidate intervention and evaluate full rollouts on all E91 cases, including successful controls, under explicit interaction budgets. Do not tune a success/failure threshold on these16 selected contexts or claim fresh-test performance.

Verification:32 complete saved rollouts;16 exact E91 learned replays; exact approach/descent trace equality; identical handoff qpos/qvel/ctrl/time, measured gram/cross/bias, counters and task flags across branches; unchanged input/checkpoint/source hashes; guarded simulator Jacobian access before handoff except the disclosed read-only audit. All115 EASE tests pass. Four executed video replays match complete saved trajectories and decoded frame counts; manifests store hashes.

### E92 videos

Gallery: `outputs/ease_learned_handoff_v1/videos/index.html`. Case0 candidate_0 and candidate_2 show matched learned failures and exact-continuation successes. They were prespecified illustrative regressions and are not representative. Labels disclose privileged continuation.


## 4.83 E93: fixed recent-history estimation after planned descent

All96 E91 checkpoint/task pairs receive two full rollouts: exact long-history route300 replay and window32. After an accepted planned descent, window32 reconstructs gram/cross from the32 most recent completed transitions using the original initial ridge, forgetting factor and diagonal floor; it repeats this reconstruction after each subsequent measured transition. Adaptive measured bias remains unchanged. No new probe, policy weight update, window sweep or simulator-Jacobian policy input is introduced. Cases without an admitted route retain the original history.

The privileged E89/E91 route planner and300-step descent allowance remain present in both conditions. This evaluates a history intervention in that hybrid controller on reused development cases, not a history-only robot policy or independent transfer test.

| Group | Mode | Success | Collisions | Control steps including calibration | Window applied |
|---|---|---:|---:|---:|---:|
| all | long_history | 57/96 | 33 | 39284 | 0 |
| all | window32 | 58/96 | 34 | 40139 | 78 |
| 6 | long_history | 22/48 | 22 | 19619 | 0 |
| 6 | window32 | 28/48 | 18 | 19459 | 30 |
| 7 | long_history | 35/48 | 11 | 19665 | 0 |
| 7 | window32 | 30/48 | 16 | 20680 | 48 |
| candidate_0 | long_history | 11/16 | 4 | 6128 | 0 |
| candidate_0 | window32 | 11/16 | 5 | 6579 | 13 |
| candidate_1 | long_history | 9/16 | 6 | 6791 | 0 |
| candidate_1 | window32 | 9/16 | 5 | 7052 | 13 |
| candidate_2 | long_history | 9/16 | 6 | 6441 | 0 |
| candidate_2 | window32 | 12/16 | 4 | 6515 | 13 |
| parent_0 | long_history | 11/16 | 5 | 6780 | 0 |
| parent_0 | window32 | 11/16 | 5 | 6482 | 13 |
| parent_1 | long_history | 7/16 | 7 | 7138 | 0 |
| parent_1 | window32 | 7/16 | 8 | 7071 | 13 |
| parent_2 | long_history | 10/16 | 5 | 6006 | 0 |
| parent_2 | window32 | 8/16 | 7 | 6440 | 13 |

| Case | Checkpoint | DOF | Change vs long history | Old collision | New collision |
|---|---|---:|---|---|---|
| 0 | parent_1 | 6 | gain | False | False |
| 0 | parent_2 | 6 | gain | False | False |
| 0 | candidate_0 | 6 | gain | False | False |
| 0 | candidate_2 | 6 | gain | True | False |
| 1 | candidate_2 | 6 | gain | False | False |
| 2 | parent_1 | 6 | loss | False | False |
| 2 | candidate_0 | 6 | gain | True | False |
| 4 | candidate_1 | 6 | gain | True | False |
| 8 | parent_0 | 7 | gain | True | False |
| 10 | parent_2 | 7 | loss | False | True |
| 11 | parent_1 | 7 | loss | False | True |
| 11 | parent_2 | 7 | loss | False | True |
| 11 | candidate_0 | 7 | loss | False | True |
| 11 | candidate_1 | 7 | loss | False | True |
| 12 | parent_0 | 7 | loss | False | True |
| 12 | candidate_2 | 7 | gain | True | False |
| 13 | candidate_0 | 7 | loss | False | True |
| 13 | candidate_2 | 7 | gain | True | False |
| 14 | candidate_2 | 7 | loss | False | True |
| 15 | parent_1 | 7 | gain | False | False |
| 15 | parent_2 | 7 | loss | False | False |

### E93 decision and interpretation

Do not adopt window32. It produces58/96 successes versus57/96, but increases collisions from33 to34. Six-joint performance improves22/48 to28/48 while seven-joint performance falls35/48 to30/48. The11 gains and10 losses include one lost previously successful six-joint hybrid case. Only five of the ten E91 regressions recover, unlike the privileged exact-Jacobian continuation in E92. These opposing DOF aggregates do not justify selecting a history rule by joint count after observing the results.

The prespecified gate FAILS: success improvement=True, nonincreasing collisions=False, lost successful six-joint hybrid cases=1. There are11 gains and10 losses versus long history. Window32 recovers5/10 E91 regressions diagnosed in E92.

Relative to the original E87 learned geometry-filter baseline, window32 has28 gains and13 losses, including1 six-joint losses. This separate comparison retains the known planner and longer-budget confounds.

A shorter history changes both locality and the amount/directional coverage of excitation. Near a stationary grasp,32 recent transitions may contain very little informative motion; frozen weights may also encounter statistics different from training. Outcomes cannot by themselves isolate stale history, insufficient excitation or distribution shift. Do not interpret aggregate success as estimator calibration or infer a failure threshold from this run.

If this fixed-window intervention fails retention, retain the previous controller and investigate measured excitation/conditioning at handoff before considering another history rule. Any replacement should preserve informative motion rather than blindly discarding older transitions. Predeclare subsequent checks and retain the same successful controls; do not search window sizes on these outcomes and claim independent validation.

Verification:96 exact E91 reference replays and18 exact unavailable-route fallbacks; identical pre-intervention handoff states and statistics; no task simulator Jacobian access; frozen checkpoint/input/source hashes;117 passing tests including exact last32-statistic replay, bias preservation and copied measurement history. Individual trials were saved immediately. No reserved-test evaluation occurred.

### E93 saved videos

Gallery: `outputs/ease_local_history_v1/videos/index.html`. Prespecified candidate_0 cases0 and4 show both modes; append the first missing success/failure category if needed. A disclosed post-result addition shows the first newly introduced regression (case2 parent_1) in both modes, for six videos total. Executed video replays match the saved results exactly. These illustrative examples do not estimate success rate.


## 4.84 E94: measured excitation retained by recent history

This read-only audit covers all96 saved E93 handoffs, including78 admitted-route pairs where history changes. Temporal diagnostics exactly replay all21 changed-outcome pairs and the earliest retained-success control per DOF, in both modes (46 tasks). Per-step statistics are recorded after the measured update. Long shadow and recent32 statistics are computed on the same executed trajectory; the shadow does not affect actions. No controller modification or new policy-success claim is made.

Subtract the1e-6 isotropic ridge before computing measured Gram spectra. Directions above ridge count measured eigenvalues greater than1e-6, tied to the existing regularizer rather than fitted to outcomes. Effective rank is squared total eigenvalue mass divided by sum of squared eigenvalues. These metrics describe excitation; they are not validated success/failure thresholds.

| DOF | History | Handoffs | Median measured energy | Median effective rank | Median directions above ridge | Median largest-direction fraction |
|---|---|---:|---:|---:|---:|---:|
| 6 | long | 30 | 0.00146304 | 1.2053 | 6.0 | 0.909526 |
| 6 | recent | 30 | 0.00072916 | 1.0000 | 1.0 | 1.000000 |
| 7 | long | 48 | 0.00122307 | 1.2509 | 6.0 | 0.892543 |
| 7 | recent | 48 | 0.000596632 | 1.0000 | 1.0 | 1.000000 |

### E94 findings and decision

Recent32 contains only one direction above the ridge at66/78 admitted-route handoffs (24/30 six-joint and42/48 seven-joint). Median effective rank is approximately1 in both groups; the older weighted history has median effective rank1.205/1.251 and six directions above ridge in both groups. The weak directions are small relative to the dominant one, so longer history is not uniformly well-conditioned either. Recent history retains median48.4%/47.9% of long-history measured energy for six/seven joints.

This pattern is consistent with a straight joint-space descent exciting predominantly one joint combination. Sample count alone does not imply coverage of all joint directions. Windowing removes weaker independent directions alongside older potentially nonlocal data. Since this pattern occurs in both DOFs while E93 effects differ, it cannot alone explain every regression or justify a DOF-specific rule.

Next investigate an estimator-side diagnostic that preserves directional coverage while addressing locality. A controlled option is a small, fixed, separately budgeted local excitation probe at the handoff, with every contact and task cost retained, followed by frozen estimation. Predeclare its bounds and retention criteria before rollout; compare probe/no-probe under matched budgets and include old successes. No probe or replacement controller has been implemented or promoted in E94. Merely changing window length would not address the observed directional deficiency.

Verification: all46 full replays match E93 result dictionaries exactly, with96 saved handoff pairs audited and unchanged inputs/checkpoints. All119 tests pass, including zero-motion/ridge exclusion and known rank-one/isotropic cases. Existing six E93 executed videos are referenced with verified hashes rather than relabeled as new successes. No training or reserved-test evaluation occurred.

### E94 artifacts

- Figure: `outputs/ease_history_excitation_v1/excitation.png` and `.pdf`.
- Per-step spectra: `profiles/`; all-handoff statistics: `handoffs.json`; aggregate phase summaries: `summary.json`.
- Gallery with reused success/failure videos: `outputs/ease_history_excitation_v1/videos/index.html`.


## 4.85 E95: local motion probe versus interaction-matched hold

All96 E93 checkpoint/task pairs run with no probe,32-step hold or32-step probe. The78 admitted-route pairs receive the intervention after planned descent and before finger closing; the other18 retain original behavior and budgets. All use the same frozen motion estimators, recent32 statistics, measured bias and privileged route planner. Probe and hold retain the same32-step budget, anchor feedback, geometry filter, open fingers and final success criteria.

The fixed probe commands each joint +0.004rad for two control steps and -0.004rad for two, in joint-index order, then returns toward the anchor for remaining steps. Offsets are targets around the measured handoff posture, not guaranteed actual displacements. Every physical contact, object displacement and extra interaction counts. No early exit, reset, re-screening or outcome tuning occurs.

| Group | Mode | Success | Collisions | Total steps | Extra intervention steps | Median directions before / after | Median effective rank after |
|---|---|---:|---:|---:|---:|---:|---:|
| all | no_probe | 58/96 | 34 | 40139 | 0 | 1.0 / 1.0 | 1.000 |
| all | hold32 | 59/96 | 35 | 42646 | 2496 | 1.0 / 1.0 | 1.004 |
| all | probe32 | 57/96 | 37 | 41896 | 2496 | 1.0 / 7.0 | 6.266 |
| 6 | no_probe | 28/48 | 18 | 19459 | 0 | 1.0 / 1.0 | 1.000 |
| 6 | hold32 | 25/48 | 22 | 20590 | 960 | 1.0 / 1.0 | 1.004 |
| 6 | probe32 | 26/48 | 21 | 20438 | 960 | 1.0 / 6.0 | 5.595 |
| 7 | no_probe | 30/48 | 16 | 20680 | 0 | 1.0 / 1.0 | 1.000 |
| 7 | hold32 | 34/48 | 13 | 22056 | 1536 | 1.0 / 1.0 | 1.004 |
| 7 | probe32 | 31/48 | 16 | 21458 | 1536 | 1.0 / 7.0 | 6.496 |
| candidate_0 | no_probe | 11/16 | 5 | 6579 | 0 | 1.0 / 1.0 | 1.000 |
| candidate_0 | hold32 | 11/16 | 5 | 6952 | 416 | 1.0 / 1.0 | 1.004 |
| candidate_0 | probe32 | 8/16 | 8 | 6793 | 416 | 1.0 / 7.0 | 6.261 |
| candidate_1 | no_probe | 9/16 | 5 | 7052 | 0 | 1.0 / 1.0 | 1.000 |
| candidate_1 | hold32 | 7/16 | 9 | 7147 | 416 | 1.0 / 1.0 | 1.004 |
| candidate_1 | probe32 | 9/16 | 6 | 7163 | 416 | 1.0 / 7.0 | 6.388 |
| candidate_2 | no_probe | 12/16 | 4 | 6515 | 0 | 1.0 / 1.0 | 1.000 |
| candidate_2 | hold32 | 11/16 | 5 | 7044 | 416 | 1.0 / 1.0 | 1.004 |
| candidate_2 | probe32 | 12/16 | 4 | 6879 | 416 | 1.0 / 7.0 | 6.381 |
| parent_0 | no_probe | 11/16 | 5 | 6482 | 0 | 1.0 / 1.0 | 1.000 |
| parent_0 | hold32 | 11/16 | 5 | 7247 | 416 | 1.0 / 1.0 | 1.004 |
| parent_0 | probe32 | 9/16 | 7 | 7340 | 416 | 1.0 / 7.0 | 6.258 |
| parent_1 | no_probe | 7/16 | 8 | 7071 | 0 | 1.0 / 1.0 | 1.000 |
| parent_1 | hold32 | 9/16 | 6 | 7539 | 416 | 1.0 / 1.0 | 1.004 |
| parent_1 | probe32 | 9/16 | 6 | 6951 | 416 | 1.0 / 7.0 | 6.256 |
| parent_2 | no_probe | 8/16 | 7 | 6440 | 0 | 1.0 / 1.0 | 1.000 |
| parent_2 | hold32 | 10/16 | 5 | 6717 | 416 | 1.0 / 1.0 | 1.004 |
| parent_2 | probe32 | 10/16 | 6 | 6770 | 416 | 1.0 / 7.0 | 6.263 |

| Control | Case | Checkpoint | DOF | Probe outcome change | Control collision | Probe collision |
|---|---:|---|---:|---|---|---|
| hold32 | 0 | parent_0 | 6 | gain | True | False |
| no_probe | 0 | parent_1 | 6 | loss | False | True |
| hold32 | 0 | parent_1 | 6 | loss | False | True |
| hold32 | 0 | candidate_1 | 6 | gain | True | False |
| hold32 | 1 | parent_1 | 6 | gain | False | False |
| no_probe | 2 | candidate_0 | 6 | loss | False | True |
| hold32 | 2 | candidate_0 | 6 | loss | False | True |
| no_probe | 8 | parent_0 | 7 | loss | False | True |
| hold32 | 8 | parent_0 | 7 | loss | False | True |
| no_probe | 8 | candidate_0 | 7 | loss | False | True |
| hold32 | 8 | candidate_0 | 7 | loss | False | True |
| no_probe | 9 | parent_1 | 7 | gain | True | False |
| no_probe | 10 | parent_2 | 7 | gain | True | False |
| no_probe | 10 | candidate_1 | 7 | loss | False | True |
| hold32 | 12 | parent_0 | 7 | loss | False | True |
| no_probe | 12 | parent_1 | 7 | gain | True | False |
| no_probe | 12 | candidate_0 | 7 | loss | False | True |
| no_probe | 12 | candidate_1 | 7 | loss | False | True |
| hold32 | 12 | candidate_2 | 7 | gain | True | False |
| no_probe | 13 | parent_1 | 7 | gain | True | False |
| hold32 | 13 | parent_2 | 7 | loss | False | True |
| hold32 | 13 | candidate_0 | 7 | loss | False | True |
| no_probe | 13 | candidate_1 | 7 | gain | True | False |
| hold32 | 13 | candidate_1 | 7 | gain | True | False |
| no_probe | 13 | candidate_2 | 7 | loss | False | True |
| no_probe | 14 | candidate_2 | 7 | gain | True | False |
| no_probe | 15 | parent_0 | 7 | loss | False | True |
| hold32 | 15 | parent_0 | 7 | loss | False | True |
| no_probe | 15 | parent_2 | 7 | gain | False | False |
| hold32 | 15 | parent_2 | 7 | gain | False | False |
| no_probe | 15 | candidate_1 | 7 | gain | False | False |

### E95 Gate and interpretation

Gate: **FAIL**. Probe improves over hold=False; nonincreasing collisions versus hold=False; six-joint loss comparisons=4 (the same case can be a loss against both controls).

- Versus no_probe: 8 gains and 9 losses.
- Versus hold32: 6 gains and 8 losses.
- hold32: 0 trials contact forbidden geometry during the intervention; 0 first collide during it.
- probe32: 0 trials contact forbidden geometry during the intervention; 0 first collide during it.

Excitation metrics subtract the1e-6 ridge, with above-ridge directions counted using that existing regularizer. Increased directional coverage is not sufficient evidence of an accurate Jacobian or useful task behavior. Probe motion also changes joint pose, velocity, object state and bias estimates. The hold comparison separates some effects of extra time and observation updates, but does not isolate information gain from every mechanical effect.

This is reused development data with a privileged planner, not independent generalization or a history-only policy. The gate fails: do not adopt this probe or tune its amplitude or order on these same outcomes. Restoring directional coverage did not improve full-task success: probe57/96 versus hold59/96 and no-probe58/96, with37 versus35 and34 collisions. Two unique six-joint cases successful under both controls are lost. Next audit learned versus analytical and exact motion estimates at identical post-intervention states to distinguish useful measurements from frozen-estimator mismatch before another control intervention.

Verification:96 exact E93 no-probe references and36 unchanged no-route fallbacks; identical pre-intervention physical state and controller statistics; fixed32-step insertion tests; unchanged hashes and123 passing tests. All task-time simulator Jacobians remain blocked. Each completed trial is saved immediately. No training or reserved-test evaluation occurred.

### E95 Videos

Gallery: `outputs/ease_local_probe_v1/videos/index.html`. Prespecified candidate_0 cases0 and4 show all three modes; the first new probe regression against hold is appended with all three modes if not already represented. Selection is disclosed and illustrative. If the prespecified set lacks an outcome category, the first sorted example of that category is added before the paired regression selection. The completed gallery contains ten exact-replay videos, including this extra hold-failure example. Executed video replays must exactly match saved outcomes and traces.

## 4.86 E96: same-state post-probe estimation audit

All 288 E95 full-task rollouts reproduce their saved result and trace exactly. At the first actual close-phase prediction, learned, ridge analytical and exact simulator Jacobians are compared at the same physical state using the same measured history. The diagnostic never changes the policy prediction. Primary summaries include 78 admitted-route pairs in each mode (30 six-joint and 48 seven-joint); 18 no-route pairs per mode are reported separately. These are repeated checkpoints and tasks, not independent samples.

Relative Jacobian error is ||J_hat - J_exact||_F / ||J_exact||_F. Historical residual is ||DQ J_hat^T - DX||_F / ||DX||_F over the 32 measured transitions (unweighted). Linearized task residual is ||0.35 e - J_exact u||_2, where u is the bias-free DLS correction clipped to +/-0.04 rad. Position and scaled tool-Z component errors, matrices, histories and actions are retained in each snapshot.

| Group | Mode | Estimator | N | Median relative Jacobian error | Median in-history residual | Median linearized task residual |
|---|---|---|---:|---:|---:|---:|
| all | no_probe | learned | 78 | 0.98125 | 0.28667 | 0.001836 |
| all | no_probe | analytical | 78 | 0.96537 | 0.04425 | 0.000229 |
| all | no_probe | exact | 78 | 0.00000 | 0.08729 | 0.000009 |
| all | hold32 | learned | 78 | 0.95504 | 0.24870 | 0.001422 |
| all | hold32 | analytical | 78 | 0.95262 | 0.14587 | 0.000334 |
| all | hold32 | exact | 78 | 0.00000 | 0.00191 | 0.000009 |
| all | probe32 | learned | 78 | 0.41939 | 0.38322 | 0.000640 |
| all | probe32 | analytical | 78 | 0.07895 | 0.07720 | 0.000076 |
| all | probe32 | exact | 78 | 0.00000 | 0.00182 | 0.000009 |
| 6 | no_probe | learned | 30 | 0.98251 | 0.30101 | 0.002426 |
| 6 | no_probe | analytical | 30 | 0.96210 | 0.06969 | 0.000320 |
| 6 | no_probe | exact | 30 | 0.00000 | 0.13758 | 0.000009 |
| 6 | hold32 | learned | 30 | 0.95488 | 0.28463 | 0.001621 |
| 6 | hold32 | analytical | 30 | 0.95255 | 0.18761 | 0.000380 |
| 6 | hold32 | exact | 30 | 0.00000 | 0.00248 | 0.000009 |
| 6 | probe32 | learned | 30 | 0.41939 | 0.38777 | 0.000664 |
| 6 | probe32 | analytical | 30 | 0.07013 | 0.06266 | 0.000071 |
| 6 | probe32 | exact | 30 | 0.00000 | 0.00179 | 0.000009 |
| 7 | no_probe | learned | 48 | 0.97982 | 0.28011 | 0.001584 |
| 7 | no_probe | analytical | 48 | 0.96857 | 0.03934 | 0.000097 |
| 7 | no_probe | exact | 48 | 0.00000 | 0.07758 | 0.000010 |
| 7 | hold32 | learned | 48 | 0.95523 | 0.22214 | 0.001398 |
| 7 | hold32 | analytical | 48 | 0.95868 | 0.14530 | 0.000198 |
| 7 | hold32 | exact | 48 | 0.00000 | 0.00172 | 0.000010 |
| 7 | probe32 | learned | 48 | 0.41042 | 0.36876 | 0.000630 |
| 7 | probe32 | analytical | 48 | 0.08285 | 0.07947 | 0.000091 |
| 7 | probe32 | exact | 48 | 0.00000 | 0.00182 | 0.000011 |
| no_route | no_probe | learned | 18 | 0.59231 | 0.42446 | 0.004686 |
| no_route | no_probe | analytical | 18 | 0.18687 | 0.08594 | 0.001375 |
| no_route | no_probe | exact | 18 | 0.00000 | 0.00275 | 0.000114 |
| no_route | hold32 | learned | 18 | 0.59231 | 0.42446 | 0.004686 |
| no_route | hold32 | analytical | 18 | 0.18687 | 0.08594 | 0.001375 |
| no_route | hold32 | exact | 18 | 0.00000 | 0.00275 | 0.000114 |
| no_route | probe32 | learned | 18 | 0.59231 | 0.42446 | 0.004686 |
| no_route | probe32 | analytical | 18 | 0.18687 | 0.08594 | 0.001375 |
| no_route | probe32 | exact | 18 | 0.00000 | 0.00275 | 0.000114 |
| candidate_0 | no_probe | learned | 13 | 0.97909 | 0.30636 | 0.001839 |
| candidate_0 | no_probe | analytical | 13 | 0.96938 | 0.04449 | 0.000229 |
| candidate_0 | no_probe | exact | 13 | 0.00000 | 0.08776 | 0.000009 |
| candidate_0 | hold32 | learned | 13 | 0.95398 | 0.18128 | 0.001404 |
| candidate_0 | hold32 | analytical | 13 | 0.95295 | 0.14580 | 0.000334 |
| candidate_0 | hold32 | exact | 13 | 0.00000 | 0.00192 | 0.000009 |
| candidate_0 | probe32 | learned | 13 | 0.34518 | 0.33410 | 0.000545 |
| candidate_0 | probe32 | analytical | 13 | 0.07891 | 0.07709 | 0.000076 |
| candidate_0 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| candidate_1 | no_probe | learned | 13 | 0.97536 | 0.28342 | 0.002386 |
| candidate_1 | no_probe | analytical | 13 | 0.96208 | 0.04431 | 0.000201 |
| candidate_1 | no_probe | exact | 13 | 0.00000 | 0.08743 | 0.000009 |
| candidate_1 | hold32 | learned | 13 | 0.95598 | 0.25365 | 0.001290 |
| candidate_1 | hold32 | analytical | 13 | 0.95271 | 0.17969 | 0.000358 |
| candidate_1 | hold32 | exact | 13 | 0.00000 | 0.00191 | 0.000009 |
| candidate_1 | probe32 | learned | 13 | 0.35361 | 0.32779 | 0.000541 |
| candidate_1 | probe32 | analytical | 13 | 0.07890 | 0.07737 | 0.000076 |
| candidate_1 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| candidate_2 | no_probe | learned | 13 | 0.98468 | 0.28214 | 0.001664 |
| candidate_2 | no_probe | analytical | 13 | 0.96215 | 0.04426 | 0.000202 |
| candidate_2 | no_probe | exact | 13 | 0.00000 | 0.08732 | 0.000009 |
| candidate_2 | hold32 | learned | 13 | 0.95393 | 0.24055 | 0.001232 |
| candidate_2 | hold32 | analytical | 13 | 0.95246 | 0.18041 | 0.000337 |
| candidate_2 | hold32 | exact | 13 | 0.00000 | 0.00191 | 0.000009 |
| candidate_2 | probe32 | learned | 13 | 0.36572 | 0.34846 | 0.000489 |
| candidate_2 | probe32 | analytical | 13 | 0.07890 | 0.07735 | 0.000076 |
| candidate_2 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| parent_0 | no_probe | learned | 13 | 1.01033 | 0.34547 | 0.001602 |
| parent_0 | no_probe | analytical | 13 | 0.96914 | 0.04424 | 0.000229 |
| parent_0 | no_probe | exact | 13 | 0.00000 | 0.08726 | 0.000009 |
| parent_0 | hold32 | learned | 13 | 0.95098 | 0.23527 | 0.001573 |
| parent_0 | hold32 | analytical | 13 | 0.95264 | 0.14549 | 0.000319 |
| parent_0 | hold32 | exact | 13 | 0.00000 | 0.00191 | 0.000009 |
| parent_0 | probe32 | learned | 13 | 0.45793 | 0.43507 | 0.000686 |
| parent_0 | probe32 | analytical | 13 | 0.07899 | 0.07714 | 0.000076 |
| parent_0 | probe32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| parent_1 | no_probe | learned | 13 | 0.98227 | 0.27966 | 0.002211 |
| parent_1 | no_probe | analytical | 13 | 0.96847 | 0.04228 | 0.000229 |
| parent_1 | no_probe | exact | 13 | 0.00000 | 0.08338 | 0.000009 |
| parent_1 | hold32 | learned | 13 | 0.95231 | 0.27745 | 0.001887 |
| parent_1 | hold32 | analytical | 13 | 0.95241 | 0.14488 | 0.000326 |
| parent_1 | hold32 | exact | 13 | 0.00000 | 0.00184 | 0.000009 |
| parent_1 | probe32 | learned | 13 | 0.46751 | 0.45567 | 0.000861 |
| parent_1 | probe32 | analytical | 13 | 0.07902 | 0.07718 | 0.000078 |
| parent_1 | probe32 | exact | 13 | 0.00000 | 0.00181 | 0.000009 |
| parent_2 | no_probe | learned | 13 | 0.98912 | 0.23494 | 0.001518 |
| parent_2 | no_probe | analytical | 13 | 0.96197 | 0.04199 | 0.000229 |
| parent_2 | no_probe | exact | 13 | 0.00000 | 0.08280 | 0.000009 |
| parent_2 | hold32 | learned | 13 | 0.96786 | 0.28001 | 0.001412 |
| parent_2 | hold32 | analytical | 13 | 0.95207 | 0.14510 | 0.000328 |
| parent_2 | hold32 | exact | 13 | 0.00000 | 0.00182 | 0.000009 |
| parent_2 | probe32 | learned | 13 | 0.49970 | 0.46707 | 0.000869 |
| parent_2 | probe32 | analytical | 13 | 0.07905 | 0.07723 | 0.000077 |
| parent_2 | probe32 | exact | 13 | 0.00000 | 0.00181 | 0.000009 |

### E96 Paired estimator comparisons

| DOF | Mode | N | Analytical lower Jacobian error | Analytical lower task residual |
|---|---|---:|---:|---:|
| 6 | no_probe | 30 | 21 | 30 |
| 6 | hold32 | 30 | 19 | 29 |
| 6 | probe32 | 30 | 30 | 30 |
| 7 | no_probe | 48 | 39 | 48 |
| 7 | hold32 | 48 | 27 | 48 |
| 7 | probe32 | 48 | 48 | 48 |

### E96 Paired probe-versus-control error changes

Negative delta means probe error is lower. Different intervention modes have different physical states; only comparisons between estimators within a mode hold state fixed.

| DOF | Estimator | Control | N | Improved | Median error delta |
|---|---|---|---:|---:|---:|
| 6 | learned | no_probe | 30 | 30 | -0.56558 |
| 6 | learned | hold32 | 30 | 30 | -0.52131 |
| 6 | analytical | no_probe | 30 | 30 | -0.89614 |
| 6 | analytical | hold32 | 30 | 30 | -0.88242 |
| 7 | learned | no_probe | 48 | 48 | -0.56168 |
| 7 | learned | hold32 | 48 | 48 | -0.52579 |
| 7 | analytical | no_probe | 48 | 48 | -0.87590 |
| 7 | analytical | hold32 | 48 | 48 | -0.87034 |

### E96 Descriptive outcome strata

These strata use the original E95 outcomes, with no outcome selection for the audit. They do not define a failure threshold.

| DOF | Mode | Success | N | Median learned error | Median analytical error |
|---|---|---|---:|---:|---:|
| 6 | no_probe | False | 2 | 1.01122 | 0.97166 |
| 6 | no_probe | True | 28 | 0.98251 | 0.96203 |
| 6 | hold32 | False | 5 | 0.95669 | 0.95271 |
| 6 | hold32 | True | 25 | 0.95486 | 0.95241 |
| 6 | probe32 | False | 4 | 0.41158 | 0.07565 |
| 6 | probe32 | True | 26 | 0.42103 | 0.07013 |
| 7 | no_probe | False | 18 | 0.97968 | 0.96887 |
| 7 | no_probe | True | 30 | 0.97982 | 0.96367 |
| 7 | hold32 | False | 14 | 0.95991 | 0.96526 |
| 7 | hold32 | True | 34 | 0.95490 | 0.95228 |
| 7 | probe32 | False | 17 | 0.44506 | 0.08314 |
| 7 | probe32 | True | 31 | 0.38301 | 0.08257 |

### E96 Six-joint regressions retained from E95

Both cases newly lost against both controls are shown below. Better instantaneous matrix and task-direction estimates do not guarantee successful subsequent contact-rich control.

| Case | Checkpoint | Mode | Success | Learned relative error | Learned task residual |
|---|---|---|---|---:|---:|
| 0 | parent_1 | no_probe | True | 0.98775 | 0.002641 |
| 0 | parent_1 | hold32 | True | 0.95231 | 0.001887 |
| 0 | parent_1 | probe32 | False | 0.42006 | 0.000749 |
| 2 | candidate_0 | no_probe | True | 0.98133 | 0.001734 |
| 2 | candidate_0 | hold32 | True | 0.95398 | 0.000929 |
| 2 | candidate_0 | probe32 | False | 0.40309 | 0.000889 |

### E96 Finding and next decision

The probe supplies measurements from which the existing analytical estimator obtains a substantially more accurate local motion map. On all 78 route-assisted post-probe states, analytical error and linearized task residual are lower than the learned estimator at that same state. Median relative Jacobian errors are 0.419 versus 0.070 for six joints (30 pairs) and 0.410 versus 0.083 for seven joints (48 pairs). The pooled medians are 0.419 learned and 0.079 analytical. Every route-assisted learned and analytical estimate also improves its relative Jacobian error after probing versus either control, although those mode comparisons involve different states.

This identifies unused estimation accuracy in the frozen model on these measured histories. It does not show that estimation mismatch alone causes failures: both lost six-joint cases have improved instantaneous matrix and task-direction measures after probing. Do not adopt the probe or retrain on these evaluation histories. Next predeclare a paired continuation diagnostic with an identical learned/planner/probe prefix and learned versus analytical versus exact post-probe prediction, preserving measured bias, updates, contacts and full-task budgets. This will test whether better estimation translates into task recovery before changing training.

Figure: `estimation.png` and `estimation.pdf`. Each point compares estimators at one identical physical state; points below the diagonal favor analytical error. All snapshots and repeated-checkpoint results remain descriptive development evidence.

### E96 Scope and verification

Exact Jacobian is a read-only reference and never an action input. Full replay equality verifies no behavior change. Linearized task residual uses the exact Jacobian applied to the clipped, bias-free DLS correction; it excludes dynamics, contact and geometry filtering. In-history residual is an in-sample fit, not future prediction accuracy. The exact local Jacobian can itself have nonzero historical residual because history spans different postures. Error differences do not establish full-task recovery or a causal failure threshold.

All input and checkpoint hashes are unchanged. Every snapshot retains 32 transitions; admitted-route snapshot excitation matches E95 post-intervention statistics, and no-route snapshots match across modes. No training or reserved-test evaluation occurs. The 288 replay checks are in addition to 125 passing tests, including known-linear-map and zero-motion diagnostic checks. E95 remains a failed intervention: probe57/96, hold59/96, no-probe58/96. The benchmark uses privileged route planning on reused development cases.

### E96 Video evidence

All ten original E95 videos are linked in `videos/index.html` after SHA-256 verification. They show unchanged successes and failures, not new outcomes from this read-only diagnostic. Selection and playback disclosures are copied from E95.

## 4.87 E97: paired post-probe continuation

All 96 checkpoint/task pairs receive learned, analytical and exact continuation (288 full tasks). Every branch has the same learned approach, privileged route descent and fixed32-step local probe. At the first actual close prediction, after all probe measurements update statistics, switch only the motion estimator. The78 admitted-route pairs switch;18 no-route pairs retain original behavior. Bias updates, recent32 history updates, DLS controller, geometry filter, contacts, phase budgets and success criteria remain identical. No state reset or training occurs.

| Group | Continuation | Success | Collisions | Grasped | Lifted | Total steps |
|---|---|---:|---:|---:|---:|---:|
| all | learned | 57/96 | 37 | 85 | 82 | 41896 |
| all | analytical | 68/96 | 27 | 85 | 83 | 39474 |
| all | exact | 78/96 | 18 | 85 | 83 | 38291 |
| 6 | learned | 26/48 | 21 | 37 | 35 | 20438 |
| 6 | analytical | 30/48 | 18 | 37 | 35 | 19552 |
| 6 | exact | 30/48 | 18 | 37 | 35 | 19243 |
| 7 | learned | 31/48 | 16 | 48 | 47 | 21458 |
| 7 | analytical | 38/48 | 9 | 48 | 48 | 19922 |
| 7 | exact | 48/48 | 0 | 48 | 48 | 19048 |
| route | learned | 57/78 | 19 | 78 | 77 | 34533 |
| route | analytical | 68/78 | 9 | 78 | 78 | 32111 |
| route | exact | 78/78 | 0 | 78 | 78 | 30928 |
| no_route | learned | 0/18 | 18 | 7 | 5 | 7363 |
| no_route | analytical | 0/18 | 18 | 7 | 5 | 7363 |
| no_route | exact | 0/18 | 18 | 7 | 5 | 7363 |
| candidate_0 | learned | 8/16 | 8 | 15 | 15 | 6793 |
| candidate_0 | analytical | 10/16 | 6 | 15 | 15 | 6616 |
| candidate_0 | exact | 13/16 | 3 | 15 | 15 | 6291 |
| candidate_1 | learned | 9/16 | 6 | 14 | 12 | 7163 |
| candidate_1 | analytical | 12/16 | 4 | 14 | 13 | 6536 |
| candidate_1 | exact | 13/16 | 3 | 14 | 13 | 6457 |
| candidate_2 | learned | 12/16 | 4 | 14 | 13 | 6879 |
| candidate_2 | analytical | 12/16 | 4 | 14 | 13 | 6622 |
| candidate_2 | exact | 13/16 | 3 | 14 | 13 | 6523 |
| parent_0 | learned | 9/16 | 7 | 13 | 13 | 7340 |
| parent_0 | analytical | 12/16 | 3 | 13 | 13 | 6626 |
| parent_0 | exact | 13/16 | 3 | 13 | 13 | 6485 |
| parent_1 | learned | 9/16 | 6 | 14 | 14 | 6951 |
| parent_1 | analytical | 10/16 | 6 | 14 | 14 | 6521 |
| parent_1 | exact | 13/16 | 3 | 14 | 14 | 6272 |
| parent_2 | learned | 10/16 | 6 | 15 | 15 | 6770 |
| parent_2 | analytical | 12/16 | 4 | 15 | 15 | 6553 |
| parent_2 | exact | 13/16 | 3 | 15 | 15 | 6263 |

### E97 Paired changes against learned continuation

| Case | Checkpoint | DOF | Continuation | Change | Learned collision | Branch collision |
|---|---|---:|---|---|---|---|
| 0 | parent_1 | 6 | analytical | gain | True | False |
| 0 | parent_1 | 6 | exact | gain | True | False |
| 2 | parent_1 | 6 | analytical | gain | True | False |
| 2 | parent_1 | 6 | exact | gain | True | False |
| 2 | candidate_0 | 6 | analytical | gain | True | False |
| 2 | candidate_0 | 6 | exact | gain | True | False |
| 2 | candidate_1 | 6 | analytical | gain | False | False |
| 2 | candidate_1 | 6 | exact | gain | False | False |
| 8 | parent_0 | 7 | analytical | gain | True | False |
| 8 | parent_0 | 7 | exact | gain | True | False |
| 8 | parent_1 | 7 | analytical | gain | False | False |
| 8 | parent_1 | 7 | exact | gain | False | False |
| 8 | candidate_0 | 7 | analytical | gain | True | False |
| 8 | candidate_0 | 7 | exact | gain | True | False |
| 10 | candidate_0 | 7 | analytical | loss | False | True |
| 10 | candidate_1 | 7 | analytical | gain | True | False |
| 10 | candidate_1 | 7 | exact | gain | True | False |
| 11 | parent_1 | 7 | exact | gain | True | False |
| 11 | parent_2 | 7 | exact | gain | True | False |
| 11 | candidate_0 | 7 | exact | gain | True | False |
| 11 | candidate_1 | 7 | analytical | gain | True | False |
| 11 | candidate_1 | 7 | exact | gain | True | False |
| 12 | parent_0 | 7 | analytical | gain | True | False |
| 12 | parent_0 | 7 | exact | gain | True | False |
| 12 | parent_1 | 7 | analytical | loss | False | True |
| 12 | parent_2 | 7 | analytical | gain | True | False |
| 12 | parent_2 | 7 | exact | gain | True | False |
| 12 | candidate_0 | 7 | analytical | gain | True | False |
| 12 | candidate_0 | 7 | exact | gain | True | False |
| 12 | candidate_1 | 7 | analytical | gain | True | False |
| 12 | candidate_1 | 7 | exact | gain | True | False |
| 13 | parent_0 | 7 | analytical | gain | True | False |
| 13 | parent_0 | 7 | exact | gain | True | False |
| 13 | parent_1 | 7 | analytical | loss | False | True |
| 13 | parent_2 | 7 | analytical | gain | True | False |
| 13 | parent_2 | 7 | exact | gain | True | False |
| 13 | candidate_0 | 7 | exact | gain | True | False |
| 13 | candidate_1 | 7 | analytical | loss | False | True |
| 13 | candidate_2 | 7 | exact | gain | True | False |
| 15 | parent_0 | 7 | exact | gain | True | False |

### E97 Gate and retention

Analytical diagnostic gate: **PASS**. Improves success=True; nonincreasing collisions=True; lost learned-successful six-joint cases=0.

- analytical versus learned: 15 gains, 4 losses.
- analytical versus E95 no_probe: 13 gains, 3 losses; 0 six-joint losses.
- analytical versus E95 hold32: 14 gains, 5 losses; 0 six-joint losses.
- exact versus learned: 21 gains, 0 losses.
- exact versus E95 no_probe: 20 gains, 0 losses; 0 six-joint losses.
- exact versus E95 hold32: 19 gains, 0 losses; 0 six-joint losses.
- Exact versus analytical: 10 gains, 0 losses.

### E97 First forbidden-contact phase

| Phase | Learned | Analytical | Exact |
|---|---:|---:|---:|
| close | 3 | 3 | 3 |
| descend | 11 | 11 | 11 |
| lift | 8 | 2 | 0 |
| lower | 9 | 5 | 4 |
| release | 0 | 1 | 0 |
| retreat | 2 | 4 | 0 |
| settle | 3 | 1 | 0 |
| transport | 1 | 0 | 0 |

All paired control changes are saved in summary.json. E95 reference totals remain no-probe58/96 (34 collisions), hold59/96 (35), and learned probe57/96 (37). Analytical/exact branches retain the32-step probe cost; comparisons with no-probe do not isolate estimator quality alone.

### E97 Interpretation and scope

Measured outcome: learned57/96 with37 collisions, analytical68/96 with27, exact78/96 with18. Analytical gains15 and loses4 versus learned; all four losses are seven-joint cases (10 candidate_0,12 parent_1,13 parent_1,13 candidate_1). Both lost six-joint probe cases from E95 recover. Six-joint success is26/48 learned versus30/48 analytical/exact; seven-joint is31/48 versus38/48 versus48/48. The diagnostic gate passes, but four seven-joint regressions prevent an unqualified replacement recommendation.

On the78 route-assisted pairs, analytical reaches68/78 with9 collisions and exact78/78 with0. All18 no-route pairs remain failed and unchanged, so78/96 is the result of this exact-continuation branch, not a universal ceiling or proof that other tasks are infeasible. All ten remaining analytical failures are seven-joint cases. The exact branch demonstrates that successful full-task continuation is possible from every evaluated probe handoff under the same remaining budgets and controller structure.

Next decision: use the analytical estimator as a stronger measured-history training target on training embodiments only, preserving the frozen evaluation split and all observed regressions. A trained replacement must be tested on complete tasks, not selected by handoff matrix error alone. In parallel within the research sequence, inspect estimator drift through the ten analytical failures before claiming that distillation alone closes the exact-reference gap. No weights or deployed controller are changed by E97.

This is a development diagnostic with privileged route planning and repeated checkpoint/task pairs. The analytical branch uses measured history for its motion fit; the exact branch additionally uses privileged simulator Jacobians. Neither branch is a new learned-policy transfer result. Identical prefixes isolate the post-probe estimator choice, but subsequent observations and trajectories diverge. Outcomes do not prove that a single handoff matrix error or stale history is the sole cause. No thresholds or probe parameters were tuned after outcomes.

Verification:96 learned full results and traces match E95;36 no-route fallbacks match; all288 handoffs match E96 physical state, controller statistics and32-transition histories. Branch prefixes through the probe match E95 exactly. Exact Jacobian access is scoped to the post-switch exact branch. All input and checkpoint hashes remain unchanged. All127 EASE tests pass, including switch timing and restoration of the Jacobian access guard. No reserved-test evaluation occurs.

### E97 Videos

Gallery: `outputs/ease_probe_continuation_v1/videos/index.html`. Prespecified case0 parent_1 and case2 candidate_0 (E95 regressions) and case4 candidate_0 (retained success), all three branches. If needed, append first sorted missing success/failure category and first analytical regression against learned with all branches. Selection is disclosed; these are illustrative rather than representative. Every video replays the complete saved result exactly, checks decoded frame count and records SHA-256 hashes.

## 4.88 E98: training-only local-history neural distillation

Fixed three-seed fine tuning from E66 analytical checkpoints using only the eight original training bodies. Generate16 physical trajectories per body:32 settling,32 E95-style probe and96 rank-one sinusoidal motion steps. Save128 pre-action contexts per trajectory. All contacts and trajectories are retained. Whole trajectories0..11 train;12..15 validate on the same bodies. There are16384 contexts:12288 train and4096 validation;23 physical control steps contain contacts. Validation is not unseen-body evaluation.

2000 AdamW updates per seed, learning rate2e-5, batch128, gradient clip1. Each batch mixes64 measured contexts and64 synthetic replay contexts. Measured targets use the existing float64 analytical ridge solve of the input Gram/cross; every target is independently verified in NumPy. Prediction MSE covers both halves; frozen-parent anchor is applied only to synthetic replay (six-joint weight2, seven-joint0.5). Final checkpoints are used without model selection. No target-embodiment history, Jacobian label or outcome enters training.

| Seed | Held-trajectory MSE before | After | Synthetic MSE before | After |
|---|---:|---:|---:|---:|
| 0 | 0.00331488 | 0.00230932 | 0.00651875 | 0.00596832 |
| 1 | 0.00310682 | 0.00218303 | 0.00662603 | 0.00605862 |
| 2 | 0.00338864 | 0.00232708 | 0.00641641 | 0.00590381 |

### E98 Full-task development evaluation

All16 existing E97 tasks x3 E66 parents x2 conditions =96 full tasks. Parent weights control the identical approach and physical probe; only after the probe on admitted routes do student weights replace the motion estimator. No-route behavior is unchanged. This is phase-specific neural substitution with privileged route planning, not a newly trained end-to-end policy. All48 parent results replay E97 exactly; every student shares the original prefix and E96 handoff.

| Group | Mode | Success | Collisions | Total steps |
|---|---|---:|---:|---:|
| all | parent | 29/48 | 18 | 20835 |
| all | student | 31/48 | 17 | 20905 |
| 6 | parent | 13/24 | 10 | 10250 |
| 6 | student | 15/24 | 9 | 10057 |
| 7 | parent | 16/24 | 8 | 10585 |
| 7 | student | 16/24 | 8 | 10848 |
| candidate_0 | parent | 8/16 | 8 | 6793 |
| candidate_0 | student | 12/16 | 4 | 6614 |
| candidate_1 | parent | 9/16 | 6 | 7163 |
| candidate_1 | student | 11/16 | 5 | 7082 |
| candidate_2 | parent | 12/16 | 4 | 6879 |
| candidate_2 | student | 8/16 | 8 | 7209 |
| route | parent | 29/39 | 9 | 17033 |
| route | student | 31/39 | 8 | 17103 |
| no_route | parent | 0/9 | 9 | 3802 |
| no_route | student | 0/9 | 9 | 3802 |

### E98 Matched E97 reference controllers

These saved reference results use the same three candidate-parent prefixes and16 tasks. They are reused context, not new E98 rollouts. The exact reference is privileged.

| Group | Continuation | Success | Collisions |
|---|---|---:|---:|
| all | analytical | 34/48 | 14 |
| all | exact | 39/48 | 9 |
| 6 | analytical | 15/24 | 9 |
| 6 | exact | 15/24 | 9 |
| 7 | analytical | 19/24 | 5 |
| 7 | exact | 24/24 | 0 |

### E98 Paired gains and losses

| Case | Parent seed | DOF | Change | Parent collision | Student collision |
|---|---|---:|---|---|---|
| 2 | candidate_0 | 6 | gain | True | False |
| 2 | candidate_1 | 6 | gain | False | False |
| 8 | candidate_0 | 7 | gain | True | False |
| 10 | candidate_1 | 7 | gain | True | False |
| 10 | candidate_2 | 7 | loss | False | True |
| 11 | candidate_0 | 7 | gain | True | False |
| 11 | candidate_2 | 7 | loss | False | True |
| 12 | candidate_2 | 7 | loss | False | True |
| 13 | candidate_0 | 7 | gain | True | False |
| 15 | candidate_2 | 7 | loss | False | True |

### E98 Retention decision

Gate: **FAIL**. Strict pooled success improvement=True; nonincreasing collisions=True; every parent success retained=False; every seed nondecreasing=False.

There are 6 gains and 4 losses. All three seeds and all failed tasks are retained. Lower held-trajectory MSE does not establish better closed-loop control. Do not promote a checkpoint that fails the prespecified full-task retention gate; no outcome-driven retuning occurs in this experiment.

Observed decision: do not adopt these checkpoints. Students improve29/48 to31/48 and reduce collisions18 to17, but gains6 and losses4 are uneven across seeds. Seed0 improves8/16 to12/16; seed1 improves9/16 to11/16; seed2 drops12/16 to8/16. All four losses are seven-joint seed2 cases10,11,12,15. Six-joint success improves13/24 to15/24 with no losses; this is15/15 on admitted-route six-joint pairs, while nine no-route pairs remain unchanged failures. Seven-joint pooled success remains16/24 because four gains offset four losses. The matched analytical reference remains stronger at34/48; exact reaches39/48.

Held-trajectory prediction MSE improves29.7-31.3 percent in all seeds, and synthetic replay MSE improves too. This does not prevent full-task regression in seed2. The current evidence supports training-only local-history transfer gains for the six-joint subset, with seed-dependent seven-joint control instability. Do not select the best seed or introduce a DOF switch based on these outcomes.

Next: add complete task-trajectory histories from training embodiments and an explicit parent-behavior retention objective, then evaluate every fixed final seed again. Preserve the current four seven-joint failures as development checks and reserve fresh evaluation for confirmation after the method is fixed. This targets the gap between scripted local-motion fitting and closed-loop task behavior; it is a hypothesis to test, not an established cause.

Scientific figure: `training_and_tasks.png` and `training_and_tasks.pdf` show training fit and full-task outcomes. All contexts, provenance, losses, final weights and failures remain saved.

This evaluates only the post-probe use of the new weights. Synthetic replay loss is a retention surrogate, not a full original reaching-suite check. No whole-policy or original-reaching retention claim is made. Training histories are scripted local motions rather than successful pick/place trajectories. The repeated development benchmark remains separate from training bodies, but is not an independent final test.

Verification:127 existing regression tests pass. Training/validation trajectories are disjoint (96/32 trajectories); all target solves agree with independent NumPy; frozen evaluation weight hashes and input dataset hashes verify. No reserved-test evaluation or target-arm adaptation occurs. Raw contexts, provenance, losses, final weights and full trial traces are saved.

### E98 Videos

Gallery: `outputs/ease_probe_distillation_v1/videos/index.html`. Candidate_0 cases2,4,10 are prespecified parent/student pairs; append first new student regression with both modes and first missing outcome category if necessary. Disclosed illustrations, not representative samples. Executed video replays must match complete serialized results; decoded frame counts and hashes are checked.

## 4.89 E99: training-task histories and matched action retention

Collect36 complete parent-driven trajectories on12 existing training placements across four original training bodies. All cases and failures are retained. Save11714 pre-action contexts; fitting uses only post-probe continuation contexts: 5921 training and3419 validation. Third placement per body is validation across every driver, giving24/12 disjoint trajectories. Reuse E98 local-motion data on all eight training bodies (12288 train/4096 validation). No evaluation-arm histories enter training.

Parent data-collection outcomes: training {'n': 24, 'success': 20, 'collisions': 4}; validation {'n': 12, 'success': 7, 'collisions': 5}. These are source-driver outcomes, not trained-student training-task scores. Full phase traces and calibration observations are saved. Original geometry filtering and32-step probe are used; there is no training route planner. Existing IK homes are privileged setup.

Both conditions start from original E66 checkpoints. Each of2000 identical paired batches contains64 synthetic,32 local-motion and32 training-task contexts. AdamW lr2e-5, gradient clip1; analytical ridge prediction target and synthetic parent anchor are identical. Only retained adds0.01 times the mean squared clipped joint-command difference from the frozen parent, divided by0.04 squared, on task contexts. This uses the current measured bias, task error, DLS damping0.015 and gain0.35. It compares commands before geometry filtering and physics; it is not an executed-contact loss. Parent commands from unsuccessful trajectories are also retained, a possible recovery/retention tradeoff.

All six fixed final checkpoints are evaluated without checkpoint/seed selection. Paired batch hashes match within every seed. Local/task targets are independently checked against NumPy solves. Validation holds out trajectories/placements on the same training bodies.

| Seed | Condition | Local MSE before / after | Task MSE before / after | Held task action drift | Synthetic MSE before / after |
|---|---|---:|---:|---:|---:|
| 0 | data_only | 0.0033294 / 0.0023017 | 0.0030902 / 0.0021674 | 0.26684 | 0.0056808 / 0.005066 |
| 0 | retained | 0.0033294 / 0.0026877 | 0.0030902 / 0.0026957 | 0.27655 | 0.0056808 / 0.0052963 |
| 1 | data_only | 0.0031239 / 0.0022316 | 0.0030765 / 0.0021837 | 0.18953 | 0.0057278 / 0.0051118 |
| 1 | retained | 0.0031239 / 0.0027297 | 0.0030765 / 0.0028003 | 0.18315 | 0.0057278 / 0.00539 |
| 2 | data_only | 0.0032851 / 0.0022784 | 0.0030422 / 0.0021779 | 0.20497 | 0.0056559 / 0.0050656 |
| 2 | retained | 0.0032851 / 0.0026706 | 0.0030422 / 0.0027311 | 0.16996 | 0.0056559 / 0.0053294 |

### E99 Full-task evaluation

All16 development tasks x3 seeds x parent/data_only/retained =144 complete rollouts. Same frozen parent approach and probe; only post-probe neural weights change on admitted routes. Every parent replay matches E97, and every handoff/prefix matches the saved references. No-route behavior remains unchanged. Evaluation remains phase-specific and uses privileged route planning.

| Group | Mode | Success | Collisions | Total steps |
|---|---|---:|---:|---:|
| all | parent | 29/48 | 18 | 20835 |
| all | data_only | 31/48 | 17 | 20407 |
| all | retained | 32/48 | 16 | 20716 |
| 6 | parent | 13/24 | 10 | 10250 |
| 6 | data_only | 15/24 | 9 | 10132 |
| 6 | retained | 14/24 | 10 | 10269 |
| 7 | parent | 16/24 | 8 | 10585 |
| 7 | data_only | 16/24 | 8 | 10275 |
| 7 | retained | 18/24 | 6 | 10447 |
| candidate_0 | parent | 8/16 | 8 | 6793 |
| candidate_0 | data_only | 10/16 | 6 | 6541 |
| candidate_0 | retained | 11/16 | 5 | 6736 |
| candidate_1 | parent | 9/16 | 6 | 7163 |
| candidate_1 | data_only | 11/16 | 5 | 6974 |
| candidate_1 | retained | 10/16 | 6 | 7062 |
| candidate_2 | parent | 12/16 | 4 | 6879 |
| candidate_2 | data_only | 10/16 | 6 | 6892 |
| candidate_2 | retained | 11/16 | 5 | 6918 |
| route | parent | 29/39 | 9 | 17033 |
| route | data_only | 31/39 | 8 | 16605 |
| route | retained | 32/39 | 7 | 16914 |
| no_route | parent | 0/9 | 9 | 3802 |
| no_route | data_only | 0/9 | 9 | 3802 |
| no_route | retained | 0/9 | 9 | 3802 |

### E99 All paired changes

| Condition | Control | Case | Seed | DOF | Change | Collision | First collision phase |
|---|---|---:|---|---:|---|---|---|
| data_only | parent | 2 | candidate_0 | 6 | gain | False | None |
| data_only | parent | 2 | candidate_1 | 6 | gain | False | None |
| data_only | parent | 8 | candidate_2 | 7 | loss | True | lower |
| data_only | parent | 10 | candidate_1 | 7 | gain | False | None |
| data_only | parent | 11 | candidate_0 | 7 | gain | False | None |
| data_only | parent | 11 | candidate_2 | 7 | loss | True | retreat |
| data_only | parent | 12 | candidate_0 | 7 | gain | False | None |
| data_only | parent | 12 | candidate_1 | 7 | gain | False | None |
| data_only | parent | 12 | candidate_2 | 7 | loss | True | retreat |
| data_only | parent | 13 | candidate_1 | 7 | loss | True | retreat |
| data_only | parent | 13 | candidate_2 | 7 | gain | False | None |
| data_only | parent | 15 | candidate_0 | 7 | loss | True | lower |
| retained | parent | 2 | candidate_0 | 6 | gain | False | None |
| retained | parent | 8 | candidate_0 | 7 | gain | False | None |
| retained | parent | 10 | candidate_0 | 7 | loss | True | retreat |
| retained | parent | 10 | candidate_1 | 7 | gain | False | None |
| retained | parent | 11 | candidate_0 | 7 | gain | False | None |
| retained | parent | 11 | candidate_2 | 7 | loss | True | lift |
| retained | parent | 12 | candidate_0 | 7 | gain | False | None |
| retained | parent | 12 | candidate_1 | 7 | gain | False | None |
| retained | parent | 13 | candidate_1 | 7 | loss | True | transport |
| retained | data_only | 2 | candidate_1 | 6 | loss | True | retreat |
| retained | data_only | 8 | candidate_0 | 7 | gain | False | None |
| retained | data_only | 8 | candidate_2 | 7 | gain | False | None |
| retained | data_only | 10 | candidate_0 | 7 | loss | True | retreat |
| retained | data_only | 12 | candidate_2 | 7 | gain | False | None |
| retained | data_only | 13 | candidate_2 | 7 | loss | True | retreat |
| retained | data_only | 15 | candidate_0 | 7 | gain | False | None |

### E99 Retention gates

- data_only: **FAIL**. Success improves=True; collisions nonincreasing=True; every parent success retained=False; every seed nondecreasing=False.
- retained: **FAIL**. Success improves=True; collisions nonincreasing=True; every parent success retained=False; every seed nondecreasing=False.
- data_only versus parent: 7 gains, 5 losses.
- retained versus parent: 6 gains, 3 losses.
- retained versus data_only: 4 gains, 3 losses.

### E99 Interpretation and verification

At the fixed coefficient, retained reaches32/48 successes with16 collisions versus parent29/48 with18 and data_only31/48 with17. Retained gains six parent failures but loses three parent successes: case10 seed0 (first collision in retreat), case11 seed2 (lift), and case13 seed1 (transport), all seven-joint. Seed2 declines12/16 to11/16. Both adoption gates fail; no checkpoint replaces the parent. Six-joint retained improves13/24 to14/24 without losing a parent success, but data_only reaches15/24. The action-retention term trades four recoveries against three losses relative to data_only, so its pooled gain is not universal.

Next decision: audit the three retained regressions at matched physical states, measuring parent/student command differences before and after geometry filtering and their accumulation by phase. First use read-only queries without changing executed actions; then predeclare any training-only remedy and repeat the full paired evaluation. Do not tune on these three trajectories or select a successful seed. On-policy student histories on training bodies are a candidate remedy for distribution shift, which this experiment has not established causally.

The matched comparison isolates the added action-retention term at the fixed coefficient. E98 comparisons also change the data mixture, so they do not isolate task-data addition alone. Lower command drift on training-body validation does not guarantee retention on new-body full tasks. No failed gate is overridden by a favorable pooled score, seed selection or post-hoc DOF switch.

129 tests pass, including controller-formula agreement for6/7 joints and finite action-loss gradients. All48 parent full rollouts and144 identical handoffs verify; both trained branches preserve all nine no-route cases. Dataset, parent, paired-batch and final checkpoint hashes are verified. No target-arm training, inference weight updates or reserved-test evaluation occurs. This is not a whole-policy or original-reaching-suite retention claim.

### E99 Videos and figures

Gallery: `outputs/ease_task_retention_training_v1/videos/index.html`. Prespecified case2 candidate_0 and cases10/15 candidate_2, all three modes; first new regression for each condition is appended with its parent if absent, plus a missing outcome category if needed. Outcome selection is disclosed. Full serialized video replays, decoded frame counts and SHA-256 hashes are checked. Scientific figure: `retention.png` and `retention.pdf`.

## 4.90 E100: same-state command divergence audit

All three E99 retained regressions are audited on both parent and retained trajectories. The lowest-seed unchanged-success control on the same task is included where available: case10 seed2 only. Cases11 and13 have no seed successful under both conditions. Eight complete rollouts reproduce their E99 serialized results exactly. This outcome-selected diagnostic is not a new benchmark score.

At each post-probe step both frozen networks see identical measured Gram/cross statistics, bias, task error and physical state. Their clipped DLS commands are separately passed through the original geometry filter using shadow data. The original command executes unchanged. Physical state and controller statistics are checked unchanged by the audit, and executed raw commands match the corresponding queried network exactly. No simulator Jacobian is queried. Geometry filtering and route planning remain privileged.

### E100 Before-contact comparison on retained trajectories

Entries below use the phase containing the first forbidden contact; only steps starting before any forbidden contact are averaged, including the command whose execution first contacts. The successful control uses retreat for comparison. Norms are joint-command L2 in radians, not position errors.

| Case / seed | Outcome | Phase | Before-contact commands | Raw mean L2 | Filtered mean L2 | Scale disagreements | Last pre-contact raw / filtered L2 |
|---|---|---|---:|---:|---:|---:|---:|
| 10 / 0 | regression | retreat | 5 | 0.069265 | 0.065905 | 1 | 0.082660 / 0.082660 |
| 10 / 2 | success control | retreat | 42 | 0.043597 | 0.043433 | 1 | 0.009796 / 0.009796 |
| 11 / 2 | regression | lift | 26 | 0.056308 | 0.056308 | 0 | 0.058412 / 0.058412 |
| 13 / 1 | regression | transport | 4 | 0.031651 | 0.000000 | 0 | 0.031505 / 0.000000 |

### E100 Interpretation

Command differences are present before the first collision, but they do not define a reliable failure threshold. The successful control also exhibits substantial disagreement. Case13 differs from the other two regressions: the collision occurs in transport after only four pre-contact commands, following lift-phase filter disagreements. Both networks produce exactly identical filtered commands on all four of those transport states (mean filtered L2 zero despite raw mean 0.031651 rad). This directs attention to the inherited lift state, without proving a cause. Immediate command disagreement alone cannot explain inherited pose/velocity differences or establish a counterfactual recovery.

The filter does not uniformly suppress disagreement; its discrete scale choice can change between networks on the same state. Cumulative norms in the figure measure repeated command discrepancy and must not be interpreted as accumulated physical displacement. Comparisons between the two executed trajectories cease to be same-state after their first differing command; within each trajectory every network comparison remains same-state.

Next experiment: collect on-policy student continuation histories exclusively on the original training bodies, retaining failures and a whole-placement validation split. Compare a fixed-budget matched replay-only control with the added on-policy data, without selecting seeds or tuning on the three diagnostic failures. This tests a distribution-shift hypothesis; E100 does not establish it as the cause. Repeat all development tasks and the existing strict retention gate before adoption.

### E100 Verification and artifacts

129 existing tests pass. Eight exact full-result replays, unchanged physical/controller state during queries, exact executed raw commands and unchanged input/checkpoint hashes pass. Per-step raw/filtered vectors, filter decisions, states and phase summaries are saved. No training or reserved-test evaluation occurs. All eight parent/retained diagnostic videos are rendered and checked against saved full results, decoded frame counts and SHA-256 hashes; the gallery discloses outcome selection. Figure: command_divergence.png / .pdf.

## 4.91 E101: student-collected histories versus matched replay

One fixed collection round saves 12254 contexts from 36 training-arm trajectories: 6366 eligible training and 3514 eligible validation contexts. All 36 calibration/approach/descent/probe prefixes exactly match E99. Four training bodies are exact original manifest entries; local-motion replay spans eight. The same held-out third placement per body gives 24/12 disjoint training/validation trajectories across all drivers. Failed and colliding trajectories remain included. Source outcomes: {'train': {'n': 24, 'success': 18, 'collisions': 6}, 'validation': {'n': 12, 'success': 4, 'collisions': 8}}. These are data-collection outcomes, not final trained-model training scores.

Both conditions start each seed's E99 retained weights and receive 2000 AdamW updates at 2e-5, batch128, gradient clip1. Each batch contains64 synthetic,32 local,16 parent-task and16 extra-task contexts. Replay draws the extra16 from parent histories; onpolicy draws them from fixed starting-student histories. Common rows and uniform index draws match, but complete batches intentionally differ in those16 rows. The recorded batch hash covers common rows/draws only. All seeds use the pooled three-driver training data.

Both conditions fit analytical ridge targets, retain the synthetic E66 parent anchor (six-joint weight2, seven-joint0.5), and use0.01 normalized pre-filter action retention against E66 on all32 task rows. Every target is independently checked with NumPy. No simulator Jacobian targets or evaluation-arm histories enter fitting. This is a single fixed student-data collection, not iterative aggregation. Six final checkpoints are used without selection.

### E101 Held-placement prediction metrics

The same held subsets are used within every paired condition; these metrics do not select checkpoints. MSE is against analytical ridge targets. Action drift is relative to the original E66 parent, before filtering.

| Seed | Condition | Local MSE | Parent-task MSE | Student-task MSE | Student-task action drift |
|---|---|---:|---:|---:|---:|
| 0 | replay | 0.002961 | 0.002717 | 0.003212 | 0.179851 |
| 0 | onpolicy | 0.002591 | 0.002766 | 0.003280 | 0.158430 |
| 1 | replay | 0.002414 | 0.002594 | 0.003026 | 0.171392 |
| 1 | onpolicy | 0.002579 | 0.002694 | 0.003096 | 0.168616 |
| 2 | replay | 0.002634 | 0.002813 | 0.003330 | 0.223628 |
| 2 | onpolicy | 0.002611 | 0.002738 | 0.003237 | 0.198847 |

### E101 Full development evaluation

144 new complete rollouts: original parent, replay and onpolicy on16 tasks x3 seeds. Start is the48 saved E99 retained results for the identical protocol, not48 additional new evaluations. The original parent/planner/probe prefix is frozen; trained weights activate only afterward on admitted routes. Training arms differ from evaluation arms; this remains privileged-planner, phase-specific transfer on reused development tasks.

| Group | Model | Success | Collisions | Steps |
|---|---|---:|---:|---:|
| all | parent | 29/48 | 18 | 20835 |
| all | start | 32/48 | 16 | 20716 |
| all | replay | 33/48 | 13 | 20786 |
| all | onpolicy | 30/48 | 15 | 20811 |
| 6 | parent | 13/24 | 10 | 10250 |
| 6 | start | 14/24 | 10 | 10269 |
| 6 | replay | 14/24 | 9 | 10143 |
| 6 | onpolicy | 13/24 | 10 | 10312 |
| 7 | parent | 16/24 | 8 | 10585 |
| 7 | start | 18/24 | 6 | 10447 |
| 7 | replay | 19/24 | 4 | 10643 |
| 7 | onpolicy | 17/24 | 5 | 10499 |
| candidate_0 | parent | 8/16 | 8 | 6793 |
| candidate_0 | start | 11/16 | 5 | 6736 |
| candidate_0 | replay | 10/16 | 4 | 6786 |
| candidate_0 | onpolicy | 11/16 | 4 | 6549 |
| candidate_1 | parent | 9/16 | 6 | 7163 |
| candidate_1 | start | 10/16 | 6 | 7062 |
| candidate_1 | replay | 11/16 | 5 | 7060 |
| candidate_1 | onpolicy | 11/16 | 5 | 7089 |
| candidate_2 | parent | 12/16 | 4 | 6879 |
| candidate_2 | start | 11/16 | 5 | 6918 |
| candidate_2 | replay | 12/16 | 4 | 6940 |
| candidate_2 | onpolicy | 8/16 | 6 | 7173 |
| route | parent | 29/39 | 9 | 17033 |
| route | start | 32/39 | 7 | 16914 |
| route | replay | 33/39 | 4 | 16984 |
| route | onpolicy | 30/39 | 6 | 17009 |
| no_route | parent | 0/9 | 9 | 3802 |
| no_route | start | 0/9 | 9 | 3802 |
| no_route | replay | 0/9 | 9 | 3802 |
| no_route | onpolicy | 0/9 | 9 | 3802 |

### E101 Paired changes

Final object XY error is in millimeters; success requires less than25 mm as well as the other unchanged criteria. Grasped/lifted are sticky events, not proof the grasp was maintained throughout transport.

| Model | Reference | Case | Seed | DOF | Change | Grasped / lifted | XY error mm | First collision |
|---|---|---:|---|---:|---|---|---:|---|
| replay | parent | 1 | candidate_0 | 6 | loss | True / True | 44.60 | None |
| replay | parent | 2 | candidate_0 | 6 | gain | True / True | 3.51 | None |
| replay | parent | 2 | candidate_1 | 6 | gain | True / True | 2.46 | None |
| replay | parent | 8 | candidate_0 | 7 | gain | True / True | 6.58 | None |
| replay | parent | 10 | candidate_1 | 7 | gain | True / True | 2.36 | None |
| replay | parent | 11 | candidate_2 | 7 | loss | True / True | 7.15 | release |
| replay | parent | 12 | candidate_0 | 7 | gain | True / True | 3.38 | None |
| replay | parent | 12 | candidate_1 | 7 | gain | True / True | 4.69 | None |
| replay | parent | 13 | candidate_0 | 7 | gain | True / True | 3.21 | None |
| replay | parent | 13 | candidate_1 | 7 | loss | True / True | 47.49 | retreat |
| replay | parent | 13 | candidate_2 | 7 | gain | True / True | 1.76 | None |
| replay | parent | 15 | candidate_0 | 7 | loss | True / True | 73.48 | None |
| replay | start | 1 | candidate_0 | 6 | loss | True / True | 44.60 | None |
| replay | start | 2 | candidate_1 | 6 | gain | True / True | 2.46 | None |
| replay | start | 10 | candidate_0 | 7 | gain | True / True | 12.19 | None |
| replay | start | 11 | candidate_0 | 7 | loss | True / True | 2.44 | lower |
| replay | start | 13 | candidate_0 | 7 | gain | True / True | 3.21 | None |
| replay | start | 13 | candidate_2 | 7 | gain | True / True | 1.76 | None |
| replay | start | 15 | candidate_0 | 7 | loss | True / True | 73.48 | None |
| onpolicy | parent | 0 | candidate_2 | 6 | loss | True / True | 40.44 | None |
| onpolicy | parent | 1 | candidate_2 | 6 | loss | True / True | 0.59 | lower |
| onpolicy | parent | 2 | candidate_0 | 6 | gain | True / True | 3.02 | None |
| onpolicy | parent | 2 | candidate_1 | 6 | gain | True / True | 17.64 | None |
| onpolicy | parent | 8 | candidate_0 | 7 | gain | True / True | 6.42 | None |
| onpolicy | parent | 10 | candidate_0 | 7 | loss | True / True | 26.21 | None |
| onpolicy | parent | 10 | candidate_1 | 7 | gain | True / True | 1.36 | None |
| onpolicy | parent | 11 | candidate_0 | 7 | gain | True / True | 0.33 | None |
| onpolicy | parent | 11 | candidate_2 | 7 | loss | True / True | 6.23 | lower |
| onpolicy | parent | 12 | candidate_1 | 7 | gain | True / True | 4.02 | None |
| onpolicy | parent | 13 | candidate_0 | 7 | gain | True / True | 2.10 | None |
| onpolicy | parent | 13 | candidate_1 | 7 | loss | True / True | 4.75 | transport |
| onpolicy | parent | 15 | candidate_2 | 7 | loss | True / True | 46.25 | None |
| onpolicy | start | 0 | candidate_2 | 6 | loss | True / True | 40.44 | None |
| onpolicy | start | 1 | candidate_2 | 6 | loss | True / True | 0.59 | lower |
| onpolicy | start | 2 | candidate_1 | 6 | gain | True / True | 17.64 | None |
| onpolicy | start | 12 | candidate_0 | 7 | loss | True / True | 3.29 | retreat |
| onpolicy | start | 13 | candidate_0 | 7 | gain | True / True | 2.10 | None |
| onpolicy | start | 15 | candidate_2 | 7 | loss | True / True | 46.25 | None |
| onpolicy | replay | 0 | candidate_2 | 6 | loss | True / True | 40.44 | None |
| onpolicy | replay | 1 | candidate_0 | 6 | gain | True / True | 0.84 | None |
| onpolicy | replay | 1 | candidate_2 | 6 | loss | True / True | 0.59 | lower |
| onpolicy | replay | 10 | candidate_0 | 7 | loss | True / True | 26.21 | None |
| onpolicy | replay | 11 | candidate_0 | 7 | gain | True / True | 0.33 | None |
| onpolicy | replay | 12 | candidate_0 | 7 | loss | True / True | 3.29 | retreat |
| onpolicy | replay | 13 | candidate_2 | 7 | loss | True / True | 78.03 | retreat |
| onpolicy | replay | 15 | candidate_0 | 7 | gain | True / True | 4.21 | None |
| onpolicy | replay | 15 | candidate_2 | 7 | loss | True / True | 46.25 | None |

### E101 Gates and interpretation

- replay: **FAIL**. {'parent': {'success_improves': True, 'collisions_nonincreasing': True, 'all_successes_retained': False, 'per_seed_nondecreasing': True}, 'start': {'success_improves': True, 'collisions_nonincreasing': True, 'all_successes_retained': False, 'per_seed_nondecreasing': False}, 'passes': False}
- onpolicy: **FAIL**. {'parent': {'success_improves': True, 'collisions_nonincreasing': True, 'all_successes_retained': False, 'per_seed_nondecreasing': False}, 'start': {'success_improves': False, 'collisions_nonincreasing': True, 'all_successes_retained': False, 'per_seed_nondecreasing': False}, 'passes': False}
- replay versus parent: 8 gains, 4 losses.
- replay versus start: 4 gains, 3 losses.
- onpolicy versus parent: 7 gains, 6 losses.
- onpolicy versus start: 2 gains, 4 losses.
- onpolicy versus replay: 3 gains, 6 losses.

The matched comparison tests the extra task-context source at fixed optimization budget and objective. A gate failure is not overridden by pooled improvements, seed selection or DOF switching. This experiment alone cannot establish whether distribution shift caused E99 failures, nor broad new-robot generalization.

### E101 Decision

Student-history training reaches30/48 versus matched replay33/48 and the starting student32/48. It gains three and loses six relative to replay; seed2 declines from11/16 at start to8/16. Replay improves the pooled result but loses four original-parent successes and three starting-student successes. Both strict gates fail and no checkpoint is adopted. These data do not support the fixed student-history mixture as an improvement.

Do not expand this fixed student-history mixture or select a favorable seed. Next audit gripper contact and object-relative motion during lowering/release, including the five non-collision regressions and matched successes, then define any task-level correction on training bodies before repeating the full retention evaluation. Prediction loss and pre-filter action drift alone have not established reliable task retention.

All five non-collision regressions finish transport within25 mm of the target, then lose placement accuracy during lowering or release. This is a descriptive phase localization, not proof of grip slip or a causal effect of the estimator. All129 existing tests pass.

### E101 Saved artifacts

All data/provenance, six final checkpoints and loss histories,144 new full evaluations,48 reference copies, per-case changes, verified video gallery and scientific figure are retained. Checkpoint/input hashes,48 parent replays and144 handoffs verify. No-route branches are unchanged. See videos/index.html and onpolicy_comparison.png / .pdf. No reserved-test evaluation or inference-time weight adaptation.

### E101 Non-collision regressions

These rows lost a success from at least one reference without a forbidden collision. Errors use the last saved state of each phase; they diagnose object displacement but do not prove why it occurred.

| Model | Case | Seed | Object XY mm after transport | After lower | Final |
|---|---:|---|---:|---:|---:|
| replay | 1 | candidate_0 | 11.19 | 30.52 | 44.60 |
| replay | 15 | candidate_0 | 4.82 | 41.52 | 73.48 |
| onpolicy | 0 | candidate_2 | 5.63 | 34.31 | 40.44 |
| onpolicy | 10 | candidate_0 | 12.94 | 9.52 | 26.21 |
| onpolicy | 15 | candidate_2 | 3.85 | 46.25 | 46.25 |

**E102 addendum (5 November 2026):** The finer contact audit narrows the earlier phase interpretation. One object is already on the table by the end of transport; case10 seed0 onpolicy remains within tolerance through release and becomes out of tolerance during retreat. The E101 error tables remain correct. See E102 for contact timing and the corrected interpretation.

## 4.92 E102: contact and object motion during lowering/release

All five E101 non-collision regressions are paired with successful original-parent runs on the same task and seed. These ten outcome-selected diagnostic replays exactly match their full saved E101 results. No controller, physics, task criterion or policy weights change. This is not a new success-rate evaluation.

At each existing contact query after a2ms physics substep, record tool/object positions, object position in the gripper frame, finger positions, cube-pad/table contact presence and summed positive normal contact forces. Active contact below uses summed normal force greater than1e-6 N. Geometric contact presence is separately saved. Contact data and poses follow MuJoCo's existing stepping schedule; no extra forward/step call is inserted.

### E102 Contact timing

Times are measured from the beginning of lowering. Relative shift is the Euclidean displacement of the object center in the gripper frame from the final transport sample; it can include rotation/rolling and is not a direct slip measurement. Pre-table statistics end before the first active cube-table contact.

| Case | Seed | Mode | First table ms | Pre-table bilateral fraction | Max pre-table relative shift mm | XY at first table mm |
|---:|---|---|---:|---:|---:|---:|
| 0 | candidate_2 | parent | 594.000 | 1.000 | 3.085 | 19.985 |
| 0 | candidate_2 | onpolicy | 1138.000 | 0.986 | 6.824 | 41.419 |
| 1 | candidate_0 | parent | 730.000 | 1.000 | 2.947 | 25.997 |
| 1 | candidate_0 | replay | 2.000 | none | none | 11.191 |
| 10 | candidate_0 | parent | 402.000 | 1.000 | 2.116 | 20.247 |
| 10 | candidate_0 | onpolicy | 712.000 | 1.000 | 2.677 | 6.263 |
| 15 | candidate_0 | parent | 642.000 | 1.000 | 2.289 | 8.189 |
| 15 | candidate_0 | replay | 114.000 | 0.196 | 70.213 | 9.069 |
| 15 | candidate_2 | parent | 562.000 | 1.000 | 2.307 | 2.911 |
| 15 | candidate_2 | onpolicy | 106.000 | 0.115 | 66.755 | 39.009 |

### E102 Placement trajectory

| Case | Seed | Mode | Transport XY mm | Lower XY mm | Release XY mm | Final XY mm | Max lower relative shift mm |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 0 | candidate_2 | parent | 8.25 | 1.71 | 4.63 | 4.63 | 5.31 |
| 0 | candidate_2 | onpolicy | 5.67 | 34.31 | 40.99 | 40.44 | 35.71 |
| 1 | candidate_0 | parent | 7.39 | 10.71 | 10.22 | 10.22 | 3.53 |
| 1 | candidate_0 | replay | 11.19 | 30.52 | 44.60 | 44.60 | 112.51 |
| 10 | candidate_0 | parent | 7.76 | 2.32 | 2.72 | 2.72 | 15.40 |
| 10 | candidate_0 | onpolicy | 12.91 | 9.52 | 9.63 | 26.21 | 3.56 |
| 15 | candidate_0 | parent | 1.43 | 4.68 | 4.27 | 4.27 | 5.80 |
| 15 | candidate_0 | replay | 4.83 | 41.52 | 73.48 | 73.48 | 77.72 |
| 15 | candidate_2 | parent | 4.74 | 2.78 | 2.78 | 2.78 | 2.31 |
| 15 | candidate_2 | onpolicy | 3.83 | 46.25 | 46.25 | 46.25 | 79.04 |

### E102 Interpretation and next decision

The five regressions do not share one failure mechanism. Case1 seed0 replay already has active table contact at the first lowering sample; during transport it has bilateral contact for77% of samples and ends with the object on the table (z approximately0.215m), despite its sticky grasped/lifted flags. Case15 seed0 replay and seed2 onpolicy lose bilateral contact early in lowering: pre-table bilateral fractions19.6% and11.5%, with maximum gripper-relative displacement70.2mm and66.8mm. All five successful references retain bilateral contact throughout their pre-table lowering intervals.

Case0 seed2 onpolicy maintains bilateral contact for98.6% of pre-table lowering samples, but first table contact occurs41.4mm from the target (parent20.0mm); its error and contact path differ before setdown, and a pure grip-loss explanation is insufficient. Case10 seed0 onpolicy remains within tolerance at the end of release (9.63mm). During retreat its right pad recontacts the object for19 sampled substeps (38ms at2ms sampling); XY error rises from9.63mm to17.19mm across these contact samples and finishes26.21mm. This temporal association motivates a controlled retreat-path intervention, not a proven causal attribution.

Correction to the E101 interpretation: all five had acceptable XY error at the end of transport, but that did not establish a maintained grasp; one had already dropped to the table. Not all five fail during lowering/release: case10 becomes out of tolerance during retreat. The saved phase-error tables were correct; the earlier prose localization was too broad.

Next test a shared, fixed gentle phase-transition controller on the original training bodies, with explicit monitoring of object/gripper contact through transport, lowering and retreat. Compare unchanged control against the intervention with identical checkpoints, success criteria and full trajectories; choose settings on training bodies only before repeating the entire development retention evaluation. Keep contact observations explicitly labeled as additional privileged diagnostic/controller inputs if used. Do not train on these selected evaluation traces or adopt an outcome-selected seed. No policy is changed by E102.

### E102 Verification and artifacts

Ten exact full-result replays; 55020 recorded contact samples; unchanged reference/checkpoint hashes.129 existing tests pass. All ten success/failure videos are replay-verified, decoded-frame-count checked and SHA-256 recorded. See videos/index.html. Figure contact_motion.png / .pdf: circles mark first active table contact, dotted lines mark release start, dashed horizontal line is25mm success threshold. Different modes follow different trajectories. Raw per-substep contact traces, vectors and normal forces remain saved. Privileged diagnostics/planning; no inference weight updates, training, or final test.

## 4.93 E103: gentle phase-transition training-arm screen

All12 original training placements on four original training bodies x3 fixed E99 retained checkpoints x baseline/budget/gentle =108 complete tasks. Third placement per body remains held-placement validation (24 training,12 held-placement pairs). Original E66 prefix and32-step probe are identical; E99 retained continuation weights stay frozen. No route planner is used on these training tasks; the existing privileged geometry filter remains.

Gentle scales the task-command correction around measured bias using cubic smoothstep during the first20 control steps (0.8s) of lift, transport, lower and retreat. Both gentle and budget-only allow20 extra maximum steps per affected phase. Original commands pass through exactly from step20 onward. Targets, gripper apertures, early-exit conditions, collision masks, success criteria and physics are unchanged. No contact feedback or estimator training is added. Ramp limits joint-command onset, not necessarily Cartesian velocity.

| Group | Mode | Success | Collisions | Steps |
|---|---|---:|---:|---:|
| all | baseline | 22/36 | 14 | 13406 |
| all | budget | 23/36 | 13 | 13503 |
| all | gentle | 22/36 | 12 | 14363 |
| train | baseline | 18/24 | 6 | 8786 |
| train | budget | 19/24 | 5 | 8712 |
| train | gentle | 17/24 | 5 | 9106 |
| validation | baseline | 4/12 | 8 | 4620 |
| validation | budget | 4/12 | 8 | 4791 |
| validation | gentle | 5/12 | 7 | 5257 |
| 6 | baseline | 14/18 | 4 | 5803 |
| 6 | budget | 14/18 | 4 | 5819 |
| 6 | gentle | 14/18 | 3 | 6225 |
| 7 | baseline | 8/18 | 10 | 7603 |
| 7 | budget | 9/18 | 9 | 7684 |
| 7 | gentle | 8/18 | 9 | 8138 |
| seed0 | baseline | 7/12 | 5 | 4344 |
| seed0 | budget | 7/12 | 5 | 4349 |
| seed0 | gentle | 8/12 | 3 | 4741 |
| seed1 | baseline | 7/12 | 5 | 4644 |
| seed1 | budget | 8/12 | 4 | 4702 |
| seed1 | gentle | 7/12 | 4 | 4805 |
| seed2 | baseline | 8/12 | 4 | 4418 |
| seed2 | budget | 8/12 | 4 | 4452 |
| seed2 | gentle | 7/12 | 5 | 4817 |

### E103 All gentle paired changes

| Reference | Case | Seed | DOF | Held placement | Change | Collision |
|---|---:|---:|---:|---|---|---|
| baseline | 1 | 0 | 6 | False | gain | False |
| baseline | 1 | 1 | 6 | False | loss | True |
| baseline | 2 | 0 | 6 | True | gain | False |
| baseline | 2 | 1 | 6 | True | gain | False |
| baseline | 2 | 2 | 6 | True | loss | True |
| baseline | 4 | 0 | 6 | False | loss | False |
| baseline | 6 | 1 | 7 | False | gain | False |
| baseline | 7 | 1 | 7 | False | loss | True |
| budget | 1 | 0 | 6 | False | gain | False |
| budget | 1 | 1 | 6 | False | loss | True |
| budget | 2 | 0 | 6 | True | gain | False |
| budget | 2 | 1 | 6 | True | gain | False |
| budget | 2 | 2 | 6 | True | loss | True |
| budget | 4 | 0 | 6 | False | loss | False |
| budget | 6 | 1 | 7 | False | gain | False |
| budget | 7 | 1 | 7 | False | loss | True |
| budget | 10 | 1 | 7 | False | loss | False |

### E103 Prespecified gate

**FAIL**: {'baseline': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': False}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': False}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}}

Transfer evaluation is conditional on strict training improvement versus both controls, no lost successes or increased collisions, no training-seed decline, and held-placement retention. No ramp duration or seed is selected after outcomes. Contact/object traces are saved at40ms control-step endpoints, which cannot resolve every transient contact; E102 used finer2ms diagnostics.

### E103 Verification

All36 original baseline full rollouts and108 prefixes through closing replay exactly. Input/checkpoint hashes are unchanged.131 tests pass, including monotonic ramp onset, bias-only preservation, bounds and exact original-command recovery. A calibration phase-counter initialization error was fixed before any completed trial; the failed-start source and note are preserved. All new video results must match saved full serialized trials, decoded frame counts and hashes. Figure transition_comparison.png / .pdf; gallery videos/index.html. No final-test evaluation or inference-time weight updates.

### E103 Decision

The combined ramp is not adopted. Across all36 training-body pairs, gentle gives22 successes/12 collisions versus baseline22/14 and budget-only23/13. It gains four and loses four versus baseline, including three six-joint losses; versus budget it gains four and loses five. The24 training pairs decline18 to17 successes (budget19); held-placement success improves4 to5 but loses an existing success. Lower collision count does not establish reliable manipulation or retention. Both prespecified gates fail, so full development transfer is deliberately not evaluated. No final test is opened.

Next isolate lift, transport, lowering and retreat ramps one at a time on the original training bodies, keeping the20-step ramp and matched time controls fixed. Report every phase condition and loss rather than choosing a seed or retrospectively changing the ramp duration. This separates effects that the current combined intervention cannot identify. Checkpoint weights, success criteria and geometry assumptions remain fixed; any later transfer run must first satisfy the training/held-placement retention gate.

## 4.94 E104: isolated phase ramps with matched time controls

324 complete rollouts:36 training-body task/seed pairs x one unchanged baseline plus four phases x two controls. The baseline is executed once per pair and reused in each phase comparison. All36 baseline results match E103 exactly. Every intervention trace matches baseline before its selected phase. E99 retained weights, original E66 prefix/probe, four training bodies, task criteria and privileged geometry filter stay fixed. No route planner, weight training or contact feedback is added.

For each phase separately, budget-only allows20 extra maximum control steps; gentle uses the same allowance and the fixed20-step E103 cubic ramp of action minus measured bias. All other phase budgets and commands are unchanged. Original commands resume exactly from step20 onward. The24 training /12 held-placement pairs are unchanged. All four phases are evaluated and reported without ramp-duration or seed selection.

### E104 Outcomes

| Phase | Group | Mode | Success | Collisions | Steps |
|---|---|---|---:|---:|---:|
| lift | all | baseline | 22/36 | 14 | 13406 |
| lift | all | budget | 23/36 | 13 | 13400 |
| lift | all | gentle | 22/36 | 14 | 13480 |
| lift | train | baseline | 18/24 | 6 | 8786 |
| lift | train | budget | 19/24 | 5 | 8687 |
| lift | train | gentle | 18/24 | 6 | 8747 |
| lift | validation | baseline | 4/12 | 8 | 4620 |
| lift | validation | budget | 4/12 | 8 | 4713 |
| lift | validation | gentle | 4/12 | 8 | 4733 |
| lift | 6 | baseline | 14/18 | 4 | 5803 |
| lift | 6 | budget | 14/18 | 4 | 5795 |
| lift | 6 | gentle | 15/18 | 3 | 5813 |
| lift | 7 | baseline | 8/18 | 10 | 7603 |
| lift | 7 | budget | 9/18 | 9 | 7605 |
| lift | 7 | gentle | 7/18 | 11 | 7667 |
| lift | seed0 | baseline | 7/12 | 5 | 4344 |
| lift | seed0 | budget | 7/12 | 5 | 4398 |
| lift | seed0 | gentle | 9/12 | 3 | 4311 |
| lift | seed1 | baseline | 7/12 | 5 | 4644 |
| lift | seed1 | budget | 8/12 | 4 | 4592 |
| lift | seed1 | gentle | 7/12 | 5 | 4779 |
| lift | seed2 | baseline | 8/12 | 4 | 4418 |
| lift | seed2 | budget | 8/12 | 4 | 4410 |
| lift | seed2 | gentle | 6/12 | 6 | 4390 |
| transport | all | baseline | 22/36 | 14 | 13406 |
| transport | all | budget | 22/36 | 14 | 13441 |
| transport | all | gentle | 23/36 | 13 | 13731 |
| transport | train | baseline | 18/24 | 6 | 8786 |
| transport | train | budget | 18/24 | 6 | 8759 |
| transport | train | gentle | 18/24 | 6 | 8934 |
| transport | validation | baseline | 4/12 | 8 | 4620 |
| transport | validation | budget | 4/12 | 8 | 4682 |
| transport | validation | gentle | 5/12 | 7 | 4797 |
| transport | 6 | baseline | 14/18 | 4 | 5803 |
| transport | 6 | budget | 14/18 | 4 | 5803 |
| transport | 6 | gentle | 15/18 | 3 | 6003 |
| transport | 7 | baseline | 8/18 | 10 | 7603 |
| transport | 7 | budget | 8/18 | 10 | 7638 |
| transport | 7 | gentle | 8/18 | 10 | 7728 |
| transport | seed0 | baseline | 7/12 | 5 | 4344 |
| transport | seed0 | budget | 7/12 | 5 | 4309 |
| transport | seed0 | gentle | 8/12 | 4 | 4409 |
| transport | seed1 | baseline | 7/12 | 5 | 4644 |
| transport | seed1 | budget | 7/12 | 5 | 4714 |
| transport | seed1 | gentle | 7/12 | 5 | 4800 |
| transport | seed2 | baseline | 8/12 | 4 | 4418 |
| transport | seed2 | budget | 8/12 | 4 | 4418 |
| transport | seed2 | gentle | 8/12 | 4 | 4522 |
| lower | all | baseline | 22/36 | 14 | 13406 |
| lower | all | budget | 22/36 | 14 | 13406 |
| lower | all | gentle | 26/36 | 10 | 13594 |
| lower | train | baseline | 18/24 | 6 | 8786 |
| lower | train | budget | 18/24 | 6 | 8786 |
| lower | train | gentle | 21/24 | 3 | 8849 |
| lower | validation | baseline | 4/12 | 8 | 4620 |
| lower | validation | budget | 4/12 | 8 | 4620 |
| lower | validation | gentle | 5/12 | 7 | 4745 |
| lower | 6 | baseline | 14/18 | 4 | 5803 |
| lower | 6 | budget | 14/18 | 4 | 5803 |
| lower | 6 | gentle | 17/18 | 1 | 5881 |
| lower | 7 | baseline | 8/18 | 10 | 7603 |
| lower | 7 | budget | 8/18 | 10 | 7603 |
| lower | 7 | gentle | 9/18 | 9 | 7713 |
| lower | seed0 | baseline | 7/12 | 5 | 4344 |
| lower | seed0 | budget | 7/12 | 5 | 4344 |
| lower | seed0 | gentle | 9/12 | 3 | 4463 |
| lower | seed1 | baseline | 7/12 | 5 | 4644 |
| lower | seed1 | budget | 7/12 | 5 | 4644 |
| lower | seed1 | gentle | 9/12 | 3 | 4683 |
| lower | seed2 | baseline | 8/12 | 4 | 4418 |
| lower | seed2 | budget | 8/12 | 4 | 4418 |
| lower | seed2 | gentle | 8/12 | 4 | 4448 |
| retreat | all | baseline | 22/36 | 14 | 13406 |
| retreat | all | budget | 22/36 | 14 | 13491 |
| retreat | all | gentle | 23/36 | 13 | 13490 |
| retreat | train | baseline | 18/24 | 6 | 8786 |
| retreat | train | budget | 18/24 | 6 | 8831 |
| retreat | train | gentle | 19/24 | 5 | 8810 |
| retreat | validation | baseline | 4/12 | 8 | 4620 |
| retreat | validation | budget | 4/12 | 8 | 4660 |
| retreat | validation | gentle | 4/12 | 8 | 4680 |
| retreat | 6 | baseline | 14/18 | 4 | 5803 |
| retreat | 6 | budget | 14/18 | 4 | 5827 |
| retreat | 6 | gentle | 15/18 | 3 | 5801 |
| retreat | 7 | baseline | 8/18 | 10 | 7603 |
| retreat | 7 | budget | 8/18 | 10 | 7664 |
| retreat | 7 | gentle | 8/18 | 10 | 7689 |
| retreat | seed0 | baseline | 7/12 | 5 | 4344 |
| retreat | seed0 | budget | 7/12 | 5 | 4367 |
| retreat | seed0 | gentle | 8/12 | 4 | 4369 |
| retreat | seed1 | baseline | 7/12 | 5 | 4644 |
| retreat | seed1 | budget | 7/12 | 5 | 4664 |
| retreat | seed1 | gentle | 7/12 | 5 | 4711 |
| retreat | seed2 | baseline | 8/12 | 4 | 4418 |
| retreat | seed2 | budget | 8/12 | 4 | 4460 |
| retreat | seed2 | gentle | 8/12 | 4 | 4410 |

### E104 All paired gains and losses

| Phase | Reference | Case | Seed | DOF | Held placement | Change | Collision |
|---|---|---:|---:|---:|---|---|---|
| lift | baseline | 0 | 2 | 6 | False | loss | True |
| lift | baseline | 1 | 0 | 6 | False | gain | False |
| lift | baseline | 1 | 2 | 6 | False | gain | False |
| lift | baseline | 2 | 0 | 6 | True | gain | False |
| lift | baseline | 2 | 2 | 6 | True | loss | True |
| lift | baseline | 6 | 1 | 7 | False | gain | False |
| lift | baseline | 7 | 1 | 7 | False | loss | True |
| lift | baseline | 7 | 2 | 7 | False | loss | True |
| lift | budget | 0 | 2 | 6 | False | loss | True |
| lift | budget | 1 | 0 | 6 | False | gain | False |
| lift | budget | 1 | 2 | 6 | False | gain | False |
| lift | budget | 2 | 0 | 6 | True | gain | False |
| lift | budget | 2 | 2 | 6 | True | loss | True |
| lift | budget | 6 | 1 | 7 | False | gain | False |
| lift | budget | 7 | 1 | 7 | False | loss | True |
| lift | budget | 7 | 2 | 7 | False | loss | True |
| lift | budget | 10 | 1 | 7 | False | loss | True |
| transport | baseline | 1 | 0 | 6 | False | gain | False |
| transport | baseline | 1 | 1 | 6 | False | loss | True |
| transport | baseline | 2 | 0 | 6 | True | gain | False |
| transport | baseline | 6 | 0 | 7 | False | loss | True |
| transport | baseline | 6 | 1 | 7 | False | gain | False |
| transport | budget | 1 | 0 | 6 | False | gain | False |
| transport | budget | 1 | 1 | 6 | False | loss | True |
| transport | budget | 2 | 0 | 6 | True | gain | False |
| transport | budget | 6 | 0 | 7 | False | loss | True |
| transport | budget | 6 | 1 | 7 | False | gain | False |
| lower | baseline | 1 | 0 | 6 | False | gain | False |
| lower | baseline | 1 | 2 | 6 | False | gain | False |
| lower | baseline | 2 | 0 | 6 | True | gain | False |
| lower | baseline | 2 | 1 | 6 | True | gain | False |
| lower | baseline | 2 | 2 | 6 | True | loss | True |
| lower | baseline | 6 | 1 | 7 | False | gain | False |
| lower | budget | 1 | 0 | 6 | False | gain | False |
| lower | budget | 1 | 2 | 6 | False | gain | False |
| lower | budget | 2 | 0 | 6 | True | gain | False |
| lower | budget | 2 | 1 | 6 | True | gain | False |
| lower | budget | 2 | 2 | 6 | True | loss | True |
| lower | budget | 6 | 1 | 7 | False | gain | False |
| retreat | baseline | 1 | 0 | 6 | False | gain | False |
| retreat | budget | 1 | 0 | 6 | False | gain | False |

### E104 Prespecified gates

Each phase must strictly improve training success over both controls, preserve every success and each training seed count, and not increase collisions; held placements must retain every success with nonincreasing collisions. Passing phases only are eligible for transfer.

- lift: **FAIL**. {'baseline': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': False}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': False, 'success_improves': False, 'per_seed_nondecreasing': False}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'passes': False}

- transport: **FAIL**. {'baseline': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'passes': False}

- lower: **FAIL**. {'baseline': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'passes': False}

- retreat: **PASS**. {'baseline': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'passes': True}

### E104 Verification and scope

131 existing tests pass. All36 full baseline replays and288 pre-intervention phase prefixes verify; all324 prefixes through closing match the original reference. Input/checkpoint hashes are unchanged. All trial traces and40ms contact/object diagnostics are saved; these do not resolve every contact transient. Videos are full-result replay verified with frame counts and hashes. Figure phase_comparison.png / .pdf; gallery videos/index.html. This is a training-body development ablation, not a new learned policy or independent generalization result. No final test or inference-time weight adaptation.

### E104 Decision

Only retreat passes the prespecified training and held-placement retention gates: 23/36 successes versus 22/36 for both baseline and matched budget, preserving every prior success. Lowering reaches 26/36 but loses a held-placement success and is not promoted. Lift and transport also fail retention. Evaluate the unchanged retreat intervention on all 48 novel-axis development pairs, with both controls and no seed selection. This selection uses training-body results only.

### E104 Conditional development transfer

Training-gate passing phases only: ['retreat']. All16 novel-axis development tasks x3 seeds are evaluated in baseline and paired budget/gentle modes. There are144 new rollouts. Fixed E99 retained checkpoints, privileged route planning and identical original-parent/probe prefix; no-route cases remain exactly unchanged. The baseline reproduces all48 E99 retained results. This is reused development data, not the reserved final test.

| Phase | Group | Mode | Success | Collisions | Steps |
|---|---|---|---:|---:|---:|
| retreat | all | baseline | 32/48 | 16 | 20716 |
| retreat | all | budget | 32/48 | 16 | 20793 |
| retreat | all | gentle | 35/48 | 13 | 21113 |
| retreat | 6 | baseline | 14/24 | 10 | 10269 |
| retreat | 6 | budget | 14/24 | 10 | 10289 |
| retreat | 6 | gentle | 15/24 | 9 | 10423 |
| retreat | 7 | baseline | 18/24 | 6 | 10447 |
| retreat | 7 | budget | 18/24 | 6 | 10504 |
| retreat | 7 | gentle | 20/24 | 4 | 10690 |
| retreat | seed0 | baseline | 11/16 | 5 | 6736 |
| retreat | seed0 | budget | 11/16 | 5 | 6736 |
| retreat | seed0 | gentle | 13/16 | 3 | 6840 |
| retreat | seed1 | baseline | 10/16 | 6 | 7062 |
| retreat | seed1 | budget | 10/16 | 6 | 7119 |
| retreat | seed1 | gentle | 11/16 | 5 | 7164 |
| retreat | seed2 | baseline | 11/16 | 5 | 6918 |
| retreat | seed2 | budget | 11/16 | 5 | 6938 |
| retreat | seed2 | gentle | 11/16 | 5 | 7109 |
| retreat | route | baseline | 32/39 | 7 | 16914 |
| retreat | route | budget | 32/39 | 7 | 16991 |
| retreat | route | gentle | 35/39 | 4 | 17311 |
| retreat | no_route | baseline | 0/9 | 9 | 3802 |
| retreat | no_route | budget | 0/9 | 9 | 3802 |
| retreat | no_route | gentle | 0/9 | 9 | 3802 |

### Transfer paired changes

| Phase | Reference | Case | Seed | DOF | Change | Collision |
|---|---|---:|---:|---:|---|---|
| retreat | baseline | 2 | 1 | 6 | gain | False |
| retreat | baseline | 10 | 0 | 7 | gain | False |
| retreat | baseline | 13 | 0 | 7 | gain | False |
| retreat | budget | 2 | 1 | 6 | gain | False |
| retreat | budget | 10 | 0 | 7 | gain | False |
| retreat | budget | 13 | 0 | 7 | gain | False |

### Transfer gates

{'retreat': {'baseline': {'success_improves': True, 'collisions_nonincreasing': True, 'retains_all_successes': True, 'per_seed_nondecreasing': True}, 'budget': {'success_improves': True, 'collisions_nonincreasing': True, 'retains_all_successes': True, 'per_seed_nondecreasing': True}, 'passes': True}}

Only full-suite success improvement over both references with no lost success, increased collisions or seed decline can pass. No seed or DOF switching. Transfer video gallery: transfer/videos/index.html.

Figure: transfer_comparison.png / .pdf. Counts are descriptive; placements and model seeds share embodiments.

### E104 Final transfer decision

PASS for the fixed retreat-only intervention versus both original E99 retained baseline and the matched extra-budget control: 35/48 (72.9%) versus 32/48 (66.7%), collisions 13 versus 16, three gains and zero losses. Six-joint results improve 14 to 15/24; seven-joint 18 to 20/24. Seed counts are 13/11/11 versus 11/10/11. Recovered pairs are case2 seed1, case10 seed0, case13 seed0. All48 baselines and96 pre-retreat prefixes match saved references; all18 no-route intervention replays are unchanged. This passes the operational development gate, not a statistical generalization claim: only three discordant pairs, reused development embodiments, shared bodies and seeds, privileged geometry and route planning. Frozen neural weights; this is a controller intervention, not new transformer training. Preserve this exact candidate and next evaluate a preregistered independent body/placement suite with baseline and matched-budget controls, clustered uncertainty by embodiment, and no tuning on outcomes. Do not consume the reserved final test until the protocol is frozen. Save failures, including the unchanged nine no-route failures, in the analysis. No global default policy is changed by this experiment.


## 4.95 E105: prospective body and placement replication

15/16 candidate placements passed endpoint IK screening. All accepted cases, including no-route failures, received all three checkpoint seeds and all three controls: 135 physical task rollouts. No replacement. Nine E104 compatibility runs match full traces and outcomes.

This cohort has eight newly perturbed bodies and two new placements per body. It shares the four E87 axis orders and procedural generator: it tests fresh parameterizations/placements, not new axis arrangements or real robots. E99 weights and E104 retreat intervention were frozen before outcomes. Privileged IK, route planning and geometric filtering remain. No reserved final-test evaluation.

| Group | Mode | Success | Collisions |
|---|---|---:|---:|
| all | baseline | 27/45 | 17 |
| all | budget | 27/45 | 17 |
| all | gentle | 27/45 | 17 |
| 6 | baseline | 10/21 | 11 |
| 6 | budget | 10/21 | 11 |
| 6 | gentle | 10/21 | 11 |
| 7 | baseline | 17/24 | 6 |
| 7 | budget | 17/24 | 6 |
| 7 | gentle | 17/24 | 6 |
| seed0 | baseline | 10/15 | 4 |
| seed0 | budget | 10/15 | 4 |
| seed0 | gentle | 10/15 | 4 |
| seed1 | baseline | 7/15 | 8 |
| seed1 | budget | 7/15 | 8 |
| seed1 | gentle | 8/15 | 7 |
| seed2 | baseline | 10/15 | 5 |
| seed2 | budget | 10/15 | 5 |
| seed2 | gentle | 9/15 | 6 |
| route | baseline | 27/42 | 14 |
| route | budget | 27/42 | 14 |
| route | gentle | 27/42 | 14 |
| no_route | baseline | 0/3 | 3 |
| no_route | budget | 0/3 | 3 |
| no_route | gentle | 0/3 | 3 |
| body0 | baseline | 4/6 | 2 |
| body0 | budget | 4/6 | 2 |
| body0 | gentle | 4/6 | 2 |
| body1 | baseline | 5/6 | 1 |
| body1 | budget | 5/6 | 1 |
| body1 | gentle | 6/6 | 0 |
| body2 | baseline | 1/6 | 5 |
| body2 | budget | 1/6 | 5 |
| body2 | gentle | 0/6 | 6 |
| body3 | baseline | 0/3 | 3 |
| body3 | budget | 0/3 | 3 |
| body3 | gentle | 0/3 | 3 |
| body4 | baseline | 4/6 | 1 |
| body4 | budget | 4/6 | 1 |
| body4 | gentle | 4/6 | 1 |
| body5 | baseline | 5/6 | 1 |
| body5 | budget | 5/6 | 1 |
| body5 | gentle | 5/6 | 1 |
| body6 | baseline | 2/6 | 4 |
| body6 | budget | 2/6 | 4 |
| body6 | gentle | 2/6 | 4 |
| body7 | baseline | 6/6 | 0 |
| body7 | budget | 6/6 | 0 |
| body7 | gentle | 6/6 | 0 |

### E105 Paired changes

[
  {
    "control": "baseline",
    "case": 2,
    "seed": 1,
    "body": 1,
    "dof": 6,
    "change": "gain"
  },
  {
    "control": "baseline",
    "case": 5,
    "seed": 2,
    "body": 2,
    "dof": 6,
    "change": "loss"
  },
  {
    "control": "budget",
    "case": 2,
    "seed": 1,
    "body": 1,
    "dof": 6,
    "change": "gain"
  },
  {
    "control": "budget",
    "case": 5,
    "seed": 2,
    "body": 2,
    "dof": 6,
    "change": "loss"
  }
]

### E105 Prespecified operational gate

**FAIL**

{
  "baseline": {
    "success_improves": false,
    "retains_all_successes": false,
    "collisions_nonincreasing": true,
    "per_seed_nondecreasing": false
  },
  "budget": {
    "success_improves": false,
    "retains_all_successes": false,
    "collisions_nonincreasing": true,
    "per_seed_nondecreasing": false
  }
}

### E105 Body-clustered uncertainty

The fixed candidate pipeline denominator is48 (16 placements x3 seeds); an IK rejection counts as zero pipeline success for every mode. Success differences are resampled over eight bodies, keeping both placements and all three seeds together. 10,000 fixed-seed bootstrap samples give descriptive percentile intervals. Eight bodies and four shared axis orders limit precision; these are not 48 independent embodiment samples.

{
  "baseline": {
    "pipeline_difference": 0.0,
    "percentile_95": [
      -0.0625,
      0.0625
    ],
    "body_differences": [
      0.0,
      0.16666666666666666,
      -0.16666666666666666,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "clusters": 8,
    "resamples": 10000
  },
  "budget": {
    "pipeline_difference": 0.0,
    "percentile_95": [
      -0.0625,
      0.0625
    ],
    "body_differences": [
      0.0,
      0.16666666666666666,
      -0.16666666666666666,
      0.0,
      0.0,
      0.0,
      0.0,
      0.0
    ],
    "clusters": 8,
    "resamples": 10000
  }
}

### E105 Next decision

Do not promote this as a robust transfer improvement. Keep the frozen outcome and diagnose the paired regressions before proposing another intervention; do not retune this cohort and call it independent confirmation.

Raw traces, protocol, screening records, static planner attempts, checkpoint hashes, source snapshots, and replay-verified success/failure videos are saved alongside this report.

### E105 Changed-pair trace diagnostics

First forbidden contact is sampled at control-step boundaries (40ms); it does not establish the substep contact mechanism. Full pre-retreat prefixes match controls.

[
  {
    "case": 2,
    "seed": 1,
    "mode": "baseline",
    "success": false,
    "first_bad_phase": "retreat",
    "first_bad_pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "final_xy_error": 0.002790932163773178
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "budget",
    "success": false,
    "first_bad_phase": "retreat",
    "first_bad_pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "final_xy_error": 0.0027909321638295665
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "gentle",
    "success": true,
    "first_bad_phase": null,
    "first_bad_pairs": [],
    "final_xy_error": 0.002917784212972583
  },
  {
    "case": 5,
    "seed": 2,
    "mode": "baseline",
    "success": true,
    "first_bad_phase": null,
    "first_bad_pairs": [],
    "final_xy_error": 0.00431118871969557
  },
  {
    "case": 5,
    "seed": 2,
    "mode": "budget",
    "success": true,
    "first_bad_phase": null,
    "first_bad_pairs": [],
    "final_xy_error": 0.004311188720416101
  },
  {
    "case": 5,
    "seed": 2,
    "mode": "gentle",
    "success": false,
    "first_bad_phase": "retreat",
    "first_bad_pairs": [
      [
        "gripper_left_pad",
        "table"
      ]
    ],
    "final_xy_error": 0.008926970904773053
  }
]

## 4.96 E106: opposite retreat contact outcomes

Six unchanged replays cover both E105 changed pairs and all three controls. Every complete serialized result matches E105. The diagnostic observes existing contact calls at2ms and existing geometry-filter decisions at40ms; it adds no physics steps or forward calls. All cases are outcome-selected diagnostics, not a new performance benchmark.

| Case / seed | Mode | Success | First retreat contact (s) | Min tool height (mm) | Filter fallbacks |
|---|---|---|---:|---:|---:|
| 2 / 1 | baseline | False | 1.082 | 230.47 | 30 |
| 2 / 1 | budget | False | 1.082 | 230.47 | 30 |
| 2 / 1 | gentle | True | none | 245.00 | 0 |
| 5 / 2 | baseline | True | none | 227.56 | 0 |
| 5 / 2 | budget | True | none | 227.56 | 0 |
| 5 / 2 | gentle | False | 3.150 | 229.00 | 53 |

### E106 Findings

- Case2/seed1: baseline and time-only control first contact the table1.082s after retreat begins. The right pad contacts; gentle retreat has no forbidden contact. At baseline first contact the preceding filter has already found negative current clearance, no feasible sampled candidate, and selects scale0 (measured bias). Baseline minimum tool height is230.47mm versus245.00mm for gentle.
- Case5/seed2: gentle retreat first contacts the table3.150s into retreat, after the0.8s ramp has ended. The left pad contacts. The immediately preceding decision is marked feasible, with both current and selected clearance at the filter cap of2mm; scale0 selects the measured-bias command. Physical contact appears30ms later. This is direct evidence that this static check does not guarantee contact-free execution.
- The successful case5 baseline reaches227.56mm tool height, below gentle at229.00mm. Tool height alone therefore cannot distinguish safe and unsafe retreat; pad pose and the executed joint trajectory matter. Both failed trajectories have at least one joint near the2.5rad limit at first contact. These observations do not isolate a causal role for joint limits, velocity, bias error, history estimation, or orientation.
- The ramp changes the early trajectory and subsequent history. Because the new collision occurs long after the ramp ends, it should not be described as a contact during the directly attenuated commands. A full causal mechanism requires controlled state-matched tests.

### E106 Next experiment

Keep the original controller and failed candidate unchanged. On training-body states, compare static clearance predictions with short physical replay predictions, logging pad clearance, joint velocity and limit proximity. First establish whether a dynamics-aware predictor detects impending contact without rejecting successful motions. Any new control rule must be chosen on training data and pass retention there before a new prospective cohort. E105 is now diagnostic development data and must not be reused as independent confirmation.

### E106 Saved artifacts and limits

20580 contact samples; six exact full-result replays; all checkpoint/input hashes unchanged. Six existing E105 success/failure videos are copied unchanged and retain original hashes. No new control result, training, or reserved final-test evaluation. Static geometry and all contact forces are privileged simulator diagnostics. Contact arrays are sampled at solver states; they do not certify continuous clearance. Figure: retreat_motion.png/pdf; gallery: videos/index.html.

## 4.97 E107: short physics contact predictions on training bodies

All36 unchanged training-body task/seed trajectories exactly replay E104. All1633 predicted40ms qpos/qvel states and forbidden-contact unions exactly match executed physics. No control intervention, new weights or reserved-test evaluation.

The predictor has privileged access to the exact physical model and complete simulator state. Exact one-step agreement verifies implementation fidelity; it is not evidence of a learned dynamics model or successful collision avoidance. All controllers, commands and collision rules remain unchanged. The held split contains new placements on the same four training bodies, not unseen embodiments.

Static warning means an infeasible filter decision or selected clearance below2mm minus1e-6. Physics warning means any forbidden contact while holding the selected absolute joint/finger target for40/80/160ms. Longer-horizon comparison uses the actual replanning policy as the outcome: false alarms and misses can arise from future command changes. Tails without a complete within-retreat comparison horizon are censored.

### E107 Prediction counts

| Split | Clean start only | Horizon ms | Predictor | N | TP | FP | FN | TN | Censored |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| all | False | 40 | static | 1633 | 43 | 96 | 1 | 1493 | 0 |
| all | False | 40 | physics | 1633 | 44 | 0 | 0 | 1589 | 0 |
| all | False | 80 | static | 1597 | 44 | 93 | 1 | 1459 | 36 |
| all | False | 80 | physics | 1597 | 45 | 1 | 0 | 1551 | 36 |
| all | False | 160 | static | 1525 | 46 | 88 | 1 | 1390 | 108 |
| all | False | 160 | physics | 1525 | 44 | 2 | 3 | 1476 | 108 |
| all | True | 40 | static | 1591 | 2 | 96 | 0 | 1493 | 0 |
| all | True | 40 | physics | 1591 | 2 | 0 | 0 | 1589 | 0 |
| all | True | 80 | static | 1556 | 4 | 93 | 0 | 1459 | 35 |
| all | True | 80 | physics | 1556 | 4 | 1 | 0 | 1551 | 35 |
| all | True | 160 | static | 1486 | 8 | 88 | 0 | 1390 | 105 |
| all | True | 160 | physics | 1486 | 5 | 2 | 3 | 1476 | 105 |
| train | False | 40 | static | 1099 | 10 | 41 | 0 | 1048 | 0 |
| train | False | 40 | physics | 1099 | 10 | 0 | 0 | 1089 | 0 |
| train | False | 80 | static | 1075 | 11 | 40 | 0 | 1024 | 24 |
| train | False | 80 | physics | 1075 | 11 | 0 | 0 | 1064 | 24 |
| train | False | 160 | static | 1027 | 13 | 38 | 0 | 976 | 72 |
| train | False | 160 | physics | 1027 | 11 | 1 | 2 | 1013 | 72 |
| train | True | 40 | static | 1090 | 1 | 41 | 0 | 1048 | 0 |
| train | True | 40 | physics | 1090 | 1 | 0 | 0 | 1089 | 0 |
| train | True | 80 | static | 1066 | 2 | 40 | 0 | 1024 | 24 |
| train | True | 80 | physics | 1066 | 2 | 0 | 0 | 1064 | 24 |
| train | True | 160 | static | 1018 | 4 | 38 | 0 | 976 | 72 |
| train | True | 160 | physics | 1018 | 2 | 1 | 2 | 1013 | 72 |
| held | False | 40 | static | 534 | 33 | 55 | 1 | 445 | 0 |
| held | False | 40 | physics | 534 | 34 | 0 | 0 | 500 | 0 |
| held | False | 80 | static | 522 | 33 | 53 | 1 | 435 | 12 |
| held | False | 80 | physics | 522 | 34 | 1 | 0 | 487 | 12 |
| held | False | 160 | static | 498 | 33 | 50 | 1 | 414 | 36 |
| held | False | 160 | physics | 498 | 33 | 1 | 1 | 463 | 36 |
| held | True | 40 | static | 501 | 1 | 55 | 0 | 445 | 0 |
| held | True | 40 | physics | 501 | 1 | 0 | 0 | 500 | 0 |
| held | True | 80 | static | 490 | 2 | 53 | 0 | 435 | 11 |
| held | True | 80 | physics | 490 | 2 | 1 | 0 | 487 | 11 |
| held | True | 160 | static | 468 | 4 | 50 | 0 | 414 | 33 |
| held | True | 160 | physics | 468 | 3 | 1 | 1 | 463 | 33 |

### E107 Interpretation

At40ms, both methods detect the two contact-onset steps from clean states. Physics has zero false positives versus96 static warnings without subsequent contact. There are no static misses in this training-body baseline sample, so it cannot establish improved recall over static checks. The two positives are case1/seed0 on body84000 (train placement) and case8/seed2 on body84004 (held placement); this is sparse positive coverage. Clean means no forbidden contact at the current boundary, not an entirely collision-free episode prefix. The static2mm warning is a margin criterion, not designed as a calibrated binary physical-contact classifier; its false positives need not be useless safety margins.
At80ms, held-target physics has one false positive and no misses; at160ms it has two false positives and three missed overlapping future windows. These windows cover only two eventual contact events and are not independent failures. Longer hold predictions cannot substitute for replanning-aware prediction. All predictions evaluate the command already selected by the static filter, not alternative candidate actions.
The controller suite passes131 tests (13.51s). Runtime invariants additionally verify all36 full baseline replays and1633 exact one-step states/contact unions. No action-avoidance claim follows from the audit.

### E107 Cost

{
  "40": {
    "median_ms": 2.5141880032606423,
    "p95_ms": 2.8765817995008547
  },
  "80": {
    "median_ms": 4.443752000952372,
    "p95_ms": 4.990254198492039
  },
  "160": {
    "median_ms": 8.308811000460992,
    "p95_ms": 9.163165199424839
  }
}

Wall-clock prediction time includes cloning, step simulation and contact logging; values are machine/load specific and exclude neural inference. Later horizons include earlier work. No claim of hardware real-time readiness.

### E107 Failure examples

[
  {
    "horizon_ms": 40,
    "category": "static_miss_physics_hit",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 40,
    "category": "physics_false_alarm",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 40,
    "category": "physics_miss",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 80,
    "category": "static_miss_physics_hit",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 80,
    "category": "physics_false_alarm",
    "count": 1,
    "first_examples": [
      {
        "case": 2,
        "seed": 0,
        "body": 84000,
        "step": 230,
        "validation": true
      }
    ]
  },
  {
    "horizon_ms": 80,
    "category": "physics_miss",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 160,
    "category": "static_miss_physics_hit",
    "count": 0,
    "first_examples": []
  },
  {
    "horizon_ms": 160,
    "category": "physics_false_alarm",
    "count": 2,
    "first_examples": [
      {
        "case": 1,
        "seed": 0,
        "body": 84000,
        "step": 326,
        "validation": false
      },
      {
        "case": 2,
        "seed": 0,
        "body": 84000,
        "step": 230,
        "validation": true
      }
    ]
  },
  {
    "horizon_ms": 160,
    "category": "physics_miss",
    "count": 3,
    "first_examples": [
      {
        "case": 1,
        "seed": 0,
        "body": 84000,
        "step": 312,
        "validation": false
      },
      {
        "case": 1,
        "seed": 0,
        "body": 84000,
        "step": 313,
        "validation": false
      },
      {
        "case": 8,
        "seed": 2,
        "body": 84004,
        "step": 344,
        "validation": true
      }
    ]
  }
]

### E107 Next decision

Use40ms physical lookahead as a candidate training-only action check if it catches static misses. A warning is not an avoidance action: the next controlled experiment must evaluate candidate fallback actions in copied physics, account for progress and joint limits, and retain every successful training/held-placement task. Longer held-target horizons are diagnostic and should not be promoted merely for detecting more future contacts. No controller has been changed by E107.

### E107 Artifacts

Full prediction records, source, checkpoint/input hashes and per-trajectory original results are saved. The video gallery reuses verified E104 baseline success/failure executions, explicitly labelled as unchanged trajectories. Counts are correlated control steps, not independent trials; do not interpret them as independent-sample confidence intervals.

## 4.98 E108: selecting safer actions with short physics predictions

All36 original training-body task/seed pairs execute baseline and the fixed physics selector:72 complete tasks. Fixed neural weights, original task budgets, no route planner, no ramp. Third placement per body remains held validation (24 training/12 held pairs). The selector only acts in retreat; it does not repair earlier collision failures.

When the static-filter-selected command predicts a forbidden contact over40ms and current state is contact-free, test five fixed alternatives: half/quarter/bias blends, hold-current-position target, and a velocity-opposing heuristic. Every candidate must pass the original static nonworsening-clearance threshold and predict no contact. Choose the safe candidate closest in joint-target distance to the original command, with fixed ties; if none qualifies keep original. All candidates share the same40ms budget and aperture. No alternative set or horizon tuning after outcomes.

| Group | Mode | Success | Collisions | Steps | Changed actions | Warnings | No safe alternative | Changes after earlier collision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all | baseline | 22/36 | 14 | 13406 | 0 | 2 | 0 | 0 |
| all | physics | 22/36 | 14 | 13406 | 5 | 7 | 2 | 2 |
| train | baseline | 18/24 | 6 | 8786 | 0 | 1 | 0 | 0 |
| train | physics | 18/24 | 6 | 8786 | 3 | 4 | 1 | 0 |
| validation | baseline | 4/12 | 8 | 4620 | 0 | 1 | 0 | 0 |
| validation | physics | 4/12 | 8 | 4620 | 2 | 3 | 1 | 2 |
| 6 | baseline | 14/18 | 4 | 5803 | 0 | 1 | 0 | 0 |
| 6 | physics | 14/18 | 4 | 5803 | 3 | 4 | 1 | 0 |
| 7 | baseline | 8/18 | 10 | 7603 | 0 | 1 | 0 | 0 |
| 7 | physics | 8/18 | 10 | 7603 | 2 | 3 | 1 | 2 |
| seed0 | baseline | 7/12 | 5 | 4344 | 0 | 1 | 0 | 0 |
| seed0 | physics | 7/12 | 5 | 4344 | 3 | 4 | 1 | 0 |
| seed1 | baseline | 7/12 | 5 | 4644 | 0 | 0 | 0 | 0 |
| seed1 | physics | 7/12 | 5 | 4644 | 0 | 0 | 0 | 0 |
| seed2 | baseline | 8/12 | 4 | 4418 | 0 | 1 | 0 | 0 |
| seed2 | physics | 8/12 | 4 | 4418 | 2 | 3 | 1 | 2 |

### E108 All paired changes

[]

### E108 Prespecified retention gates

**FAIL**

{
  "train": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true,
    "success_improves": false,
    "per_seed_nondecreasing": true
  },
  "validation": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true
  }
}

### E108 Warning-case audit

[
  {
    "case": 1,
    "seed": 0,
    "success": false,
    "collision": true,
    "changed_actions": 3,
    "no_safe_alternative": 1,
    "choices": [
      {
        "step": 315,
        "choice": "hold",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -1.746878173969693e-05,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -1.746878173969693e-05,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -1.746878173969693e-05,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 0.00014894590896809973,
            "pairs": [],
            "distance": 0.022928897039400816
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": 0.00010012882454130817,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.006874501132785076
          }
        ]
      },
      {
        "step": 316,
        "choice": "hold",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00022134171396499466,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00022134171396499466,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00022134171396499466,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 7.020107215003138e-05,
            "pairs": [],
            "distance": 0.019201659254345072
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -9.429452988622342e-05,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.006221340239492503
          }
        ]
      },
      {
        "step": 317,
        "choice": "hold",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00031698058180011646,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00031698058180011646,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00031698058180011646,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 8.977222225790144e-05,
            "pairs": [],
            "distance": 0.016438246428768347
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00020988646325669977,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.005635365011356991
          }
        ]
      },
      {
        "step": 319,
        "choice": "original",
        "prior_episode_collision": false,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00045238308001739266,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00045238308001739266,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00045238308001739266,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 1.3739761164192899e-06,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.014054845122771225
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.0002973632030811614,
            "pairs": [
              [
                "3",
                "table"
              ],
              [
                "4",
                "table"
              ]
            ],
            "distance": 0.0032094070945114872
          }
        ]
      }
    ],
    "phases": [
      {
        "phase": "approach",
        "reached": true,
        "steps": 5
      },
      {
        "phase": "descend",
        "reached": true,
        "steps": 18
      },
      {
        "phase": "probe",
        "reached": true,
        "steps": 32
      },
      {
        "phase": "close",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "lift",
        "reached": true,
        "steps": 88
      },
      {
        "phase": "transport",
        "reached": true,
        "steps": 11
      },
      {
        "phase": "lower",
        "reached": true,
        "steps": 23
      },
      {
        "phase": "release",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "retreat",
        "reached": true,
        "steps": 80
      },
      {
        "phase": "settle",
        "reached": true,
        "steps": 40
      }
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "success": false,
    "collision": true,
    "changed_actions": 2,
    "no_safe_alternative": 1,
    "choices": [
      {
        "step": 347,
        "choice": "hold",
        "prior_episode_collision": true,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.0002296718354263469,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.0002296718354263469,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.0002296718354263469,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 7.666031938170781e-06,
            "pairs": [],
            "distance": 0.027499459895477658
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00014109483460850863,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0026258618321182495
          }
        ]
      },
      {
        "step": 348,
        "choice": "hold",
        "prior_episode_collision": true,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00018422853436433212,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00018422853436433212,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00018422853436433212,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 5.065295702801065e-06,
            "pairs": [],
            "distance": 0.02355210923769029
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00018241891951270572,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.009998881381124236
          }
        ]
      },
      {
        "step": 349,
        "choice": "original",
        "prior_episode_collision": true,
        "candidates": [
          {
            "name": "half",
            "static_ok": false,
            "clearance": -0.00014864801370761266,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "quarter",
            "static_ok": false,
            "clearance": -0.00014864801370761266,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "bias",
            "static_ok": false,
            "clearance": -0.00014864801370761266,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.0
          },
          {
            "name": "hold",
            "static_ok": true,
            "clearance": 2.174354508319154e-06,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.020961815191969876
          },
          {
            "name": "brake",
            "static_ok": false,
            "clearance": -0.00014596387151639273,
            "pairs": [
              [
                "6",
                "gripper_palm"
              ]
            ],
            "distance": 0.009359623581570706
          }
        ]
      }
    ],
    "phases": [
      {
        "phase": "approach",
        "reached": true,
        "steps": 7
      },
      {
        "phase": "descend",
        "reached": true,
        "steps": 17
      },
      {
        "phase": "probe",
        "reached": true,
        "steps": 32
      },
      {
        "phase": "close",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "lift",
        "reached": true,
        "steps": 71
      },
      {
        "phase": "transport",
        "reached": true,
        "steps": 25
      },
      {
        "phase": "lower",
        "reached": true,
        "steps": 36
      },
      {
        "phase": "release",
        "reached": true,
        "steps": 40
      },
      {
        "phase": "retreat",
        "reached": false,
        "steps": 80
      },
      {
        "phase": "settle",
        "reached": false,
        "steps": 40
      }
    ]
  }
]

### E108 Decision and limits

The fixed rule fails the task-success gate. Preserve the baseline and report the full negative result; do not promote collision avoidance alone as task improvement.
The method has exact simulator state and physical-model access. It is a privileged controller component, not a learned dynamics model or history-only policy. A finite candidate set and40ms horizon cannot guarantee progress or safety. The comparison tests this rule, not all possible physics-based controllers. Held placements share training bodies. No fresh E105 state or reserved final test is used.

### E108 Verification

36 full baseline replays and 72 pre-retreat traces match references. All3266 selected-action predicted40ms qpos/qvel/contact unions match execution exactly. Inputs/checkpoints remain unchanged. Full raw results, all action alternatives and outcomes, and exact-replay videos are saved.

### E108 Scope of the retreat intervention

Baseline first-collision phases: descend3, lift4, transport2, lower4, retreat1. Thus13 of14 failed tasks are already invalid before the intervention can act. With the prefix and success rule fixed, a retreat-only change can recover at most one of these14 failures. Case8/seed2 already collided during lift; its two changed retreat actions cannot recover full-task success even if they prevent later contact.
Case1/seed0 is the only baseline failure with first collision in retreat. Three hold actions delay first retreat collision from control step27 to31 (160ms); all five alternatives then predict contact and the rule keeps the original action. Case8/seed2 has two hold actions and first retreat contact shifts from step48 to50 (80ms), but the earlier lift collision remains. These are delays, not recovered tasks. The five action changes occur in two failing pairs; no paired task outcome changes.
Next, audit predictive contact avoidance before the first collision in descend/lift/transport/lower on training bodies. Separate whether the finite candidate set has any contact-free option from whether choosing one preserves task progress. Do not interpret the failure of this retreat-only rule as a general failure of physics lookahead. No new phase-wide control rule or transfer evaluation is performed here.

[
  {
    "case": 1,
    "seed": 0,
    "first_collision_phase": "retreat"
  },
  {
    "case": 1,
    "seed": 2,
    "first_collision_phase": "lower"
  },
  {
    "case": 2,
    "seed": 0,
    "first_collision_phase": "lower"
  },
  {
    "case": 2,
    "seed": 1,
    "first_collision_phase": "lower"
  },
  {
    "case": 6,
    "seed": 1,
    "first_collision_phase": "lower"
  },
  {
    "case": 8,
    "seed": 0,
    "first_collision_phase": "descend"
  },
  {
    "case": 8,
    "seed": 1,
    "first_collision_phase": "transport"
  },
  {
    "case": 8,
    "seed": 2,
    "first_collision_phase": "lift"
  },
  {
    "case": 10,
    "seed": 0,
    "first_collision_phase": "descend"
  },
  {
    "case": 10,
    "seed": 1,
    "first_collision_phase": "transport"
  },
  {
    "case": 10,
    "seed": 2,
    "first_collision_phase": "descend"
  },
  {
    "case": 11,
    "seed": 0,
    "first_collision_phase": "lift"
  },
  {
    "case": 11,
    "seed": 1,
    "first_collision_phase": "lift"
  },
  {
    "case": 11,
    "seed": 2,
    "first_collision_phase": "lift"
  }
]

Regression suite:131 tests pass in12.48s. Prediction and full-replay assertions run on every evaluated task.

## 4.99 E109: safer actions before first collision across motion phases

72 full tasks: all36 original training-body task/seed pairs, baseline and fixed physics selector. Same E108 five alternatives,40ms prediction, static admissibility and nearest-original-target ranking. Same weights, apertures and budgets. Extend eligibility to descend/lift/transport/lower/retreat, but only before any earlier forbidden contact. Approach/calibration unchanged; no route planner. Descent interventions can alter later probe/history states. No tuning after outcomes.
The24 training and12 held-placement pairs share four original training bodies. No E105 or reserved final test. Exact physical-model/state access is privileged; this is not learned dynamics or history-only control.

| Group | Mode | Success | Collisions | Contact-free failures | Grasped | Lifted | Steps | Changes | Warnings | No safe option |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | baseline | 22/36 | 14 | 0 | 36 | 36 | 13406 | 0 | 14 | 0 |
| all | physics | 22/36 | 14 | 0 | 36 | 34 | 13581 | 50 | 61 | 11 |
| train | baseline | 18/24 | 6 | 0 | 24 | 24 | 8786 | 0 | 6 | 0 |
| train | physics | 18/24 | 6 | 0 | 24 | 24 | 8803 | 6 | 12 | 6 |
| validation | baseline | 4/12 | 8 | 0 | 12 | 12 | 4620 | 0 | 8 | 0 |
| validation | physics | 4/12 | 8 | 0 | 12 | 10 | 4778 | 44 | 49 | 5 |
| 6 | baseline | 14/18 | 4 | 0 | 18 | 18 | 5803 | 0 | 4 | 0 |
| 6 | physics | 14/18 | 4 | 0 | 18 | 18 | 5800 | 4 | 7 | 3 |
| 7 | baseline | 8/18 | 10 | 0 | 18 | 18 | 7603 | 0 | 10 | 0 |
| 7 | physics | 8/18 | 10 | 0 | 18 | 16 | 7781 | 46 | 54 | 8 |
| seed0 | baseline | 7/12 | 5 | 0 | 12 | 12 | 4344 | 0 | 5 | 0 |
| seed0 | physics | 7/12 | 5 | 0 | 12 | 11 | 4411 | 20 | 24 | 4 |
| seed1 | baseline | 7/12 | 5 | 0 | 12 | 12 | 4644 | 0 | 5 | 0 |
| seed1 | physics | 7/12 | 5 | 0 | 12 | 12 | 4627 | 16 | 19 | 3 |
| seed2 | baseline | 8/12 | 4 | 0 | 12 | 12 | 4418 | 0 | 4 | 0 |
| seed2 | physics | 8/12 | 4 | 0 | 12 | 11 | 4543 | 14 | 18 | 4 |

### E109 Paired task changes

[]

### E109 Retention gate

**FAIL**

{
  "train": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true,
    "success_improves": false,
    "per_seed_nondecreasing": true
  },
  "validation": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true
  }
}

### E109 Task progress

Phase reached counts use the original position/tool-axis completion rules, including final-state tolerance on timed-out phases. Phases still advance at the original time limit. Grasped/lifted are original sticky flags, not maintained-grasp guarantees.

| Phase | Mode | Reached /36 | Control steps |
|---|---|---:|---:|
| descend | baseline | 34 | 1024 |
| descend | physics | 34 | 1024 |
| lift | baseline | 32 | 1894 |
| lift | physics | 30 | 1927 |
| transport | baseline | 33 | 899 |
| transport | physics | 31 | 986 |
| lower | baseline | 36 | 1134 |
| lower | physics | 36 | 1060 |
| retreat | baseline | 32 | 1633 |
| retreat | physics | 28 | 1762 |

Changed-action progress uses same-state40ms squared position plus .15-scaled tool-axis error. The original action is a simulator counterfactual and may collide; compare it only as a diagnostic. Progress does not rank candidate selection in this experiment. Actions and seeds are correlated, not independent samples.

[
  {
    "phase": "descend",
    "changed": 1,
    "choices": {
      "brake": 1
    },
    "selected_reduces_error": 0,
    "selected_increases_error": 1,
    "original_reduces_error": 0
  },
  {
    "phase": "lift",
    "changed": 41,
    "choices": {
      "hold": 41
    },
    "selected_reduces_error": 0,
    "selected_increases_error": 41,
    "original_reduces_error": 3
  },
  {
    "phase": "transport",
    "changed": 3,
    "choices": {
      "hold": 2,
      "brake": 1
    },
    "selected_reduces_error": 0,
    "selected_increases_error": 3,
    "original_reduces_error": 1
  },
  {
    "phase": "lower",
    "changed": 2,
    "choices": {
      "brake": 1,
      "half": 1
    },
    "selected_reduces_error": 1,
    "selected_increases_error": 1,
    "original_reduces_error": 1
  },
  {
    "phase": "retreat",
    "changed": 3,
    "choices": {
      "hold": 3
    },
    "selected_reduces_error": 0,
    "selected_increases_error": 3,
    "original_reduces_error": 0
  }
]

### E109 Failures

[
  {
    "case": 1,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "retreat",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 1,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "retreat",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 1,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 1,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 2,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 2,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "forbidden contact already at control boundary",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "descend",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "transport",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "transport",
    "selector_status": "forbidden contact already at control boundary",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "transport",
      "retreat"
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "lift",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 10,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "descend",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "transport",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "transport",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "release"
    ]
  },
  {
    "case": 10,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "descend",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "release"
    ]
  },
  {
    "case": 11,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "transport",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": false,
    "unreached_phases": [
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lift",
    "selector_status": "forbidden contact already at control boundary",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 11,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "lift",
    "selector_status": "no admissible contact-free candidate",
    "grasped": true,
    "lifted": false,
    "unreached_phases": [
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  }
]

### E109 Decision

The rule fails the task-success/retention gate. Keep the original controller; collision reduction or intermediate phase progress alone is not sufficient for promotion.
No horizon, candidate or phase tuning after these outcomes. Full traces retain all failures, alternatives and state predictions. Before another control change, identify whether failure is due to no feasible alternative, loss of task progress, or a contact in an unmodified phase.

### E109 Verification

36 full baseline replays, 72 approach prefixes, and 13343 selected next states/contact unions match exactly. All changed actions occur before any prior collision. Inputs/checkpoints unchanged. Videos replay full serialized results; task success/failure videos are executed motion, not lookahead animation.

### E109 Interpretation of this run

Both modes finish22/36 tasks with14 collision failures; no task gains or losses. The physics selector changes50 actions before any previous collision. Of these,49 increase the immediate squared pose error and only one reduces it. The original static-filter-selected commands also reduce this error in only five of these same-state counterfactuals; the problem is not explained solely by replacing otherwise productive commands.
Phase completion worsens: lift32 to30/36, transport33 to31/36, retreat32 to28/36. Descend remains34/36 and lower36/36. Total executed steps rise13406 to13581; sticky lifted flags drop36 to34. These are progress regressions in already-failing tasks, even though binary success retention passes. There are no collision-free full-task failures in either mode.
At the first logged collision of11 physics trials no alternative satisfies both contact-free prediction and static eligibility. In three trials forbidden contact is already present at the control boundary, where the rule intentionally does not intervene. This is a finite candidate/horizon limitation, not proof that no safe trajectory exists. The same40ms model accurately predicts the selected next states; accurate local prediction does not by itself supply a useful action.
Next, use training-state counterfactual audits to test whether a broader set of goal-directed actions can provide both clearance and progress before changing the executed controller. Evaluate predicted progress and continued feasibility, rather than only closeness to the current command. Keep this completed run as a negative result; do not change its candidate set or rescore success.

Regression suite:131 tests passed in12.32s. Runtime assertions cover selected actions and all baseline/full-prefix replays.

## 4.100 E110: availability of clear, goal-progressing actions

Audit of44 frozen training-body states: first eligible E109 baseline collision warning per pair, plus five phase-start states from first successful case in each DOF/seed stratum. All20 source trajectories exactly replay without changing commands. Candidate actions evolve only in copied physics. No new task-success result.

Narrow bank: the five E108 alternatives. Broad bank includes these plus +/- .01/.02/.04 per-joint probes around measured bias,64 seeded sampled directions, and three goal-directed candidates from simulated finite-difference40ms pose response. The latter use exact-model local response, not learned dynamics or an analytic task Jacobian. Candidate count differs by joint count and duplicate clipped actions share simulation results. No parameter tuning on outcomes.
Primary admissibility requires the original static clearance criterion, zero forbidden physical contact at every2ms substep through40ms, and at least max(1e-8,1% initial squared pose error) progress. Pose error uses XYZ and .15-scaled tool-axis difference. The separate2mm column demands2mm minimum clearance across all predicted substeps; the primary metric alone does not guarantee that margin.
Choose the admissible action with lowest40ms error, tie larger dynamic clearance then name. Its absolute target is held through160ms, without re-ranking. This checks short persistence under a held target, not replanning or full-task completion. No success rate may be inferred from these state counts.

| State set | Split | Bank | N | Clear+progress40ms | Also2mm40ms | Chosen retains160ms | Chosen also2mm160ms |
|---|---|---|---:|---:|---:|---:|---:|
| warning | all | narrow | 14 | 1 | 0 | 0 | 0 |
| warning | all | broad | 14 | 13 | 5 | 12 | 5 |
| warning | train | narrow | 6 | 0 | 0 | 0 | 0 |
| warning | train | broad | 6 | 6 | 3 | 6 | 3 |
| warning | validation | narrow | 8 | 1 | 0 | 0 | 0 |
| warning | validation | broad | 8 | 7 | 2 | 6 | 2 |
| warning | 6 | narrow | 4 | 1 | 0 | 0 | 0 |
| warning | 6 | broad | 4 | 4 | 1 | 3 | 1 |
| warning | 7 | narrow | 10 | 0 | 0 | 0 | 0 |
| warning | 7 | broad | 10 | 9 | 4 | 9 | 4 |
| successful_control | all | narrow | 30 | 29 | 29 | 29 | 29 |
| successful_control | all | broad | 30 | 30 | 30 | 30 | 30 |
| successful_control | train | narrow | 30 | 29 | 29 | 29 | 29 |
| successful_control | train | broad | 30 | 30 | 30 | 30 | 30 |
| successful_control | validation | narrow | 0 | 0 | 0 | 0 | 0 |
| successful_control | validation | broad | 0 | 0 | 0 | 0 | 0 |
| successful_control | 6 | narrow | 15 | 14 | 14 | 14 | 14 |
| successful_control | 6 | broad | 15 | 15 | 15 | 15 | 15 |
| successful_control | 7 | narrow | 15 | 15 | 15 | 15 | 15 |
| successful_control | 7 | broad | 15 | 15 | 15 | 15 | 15 |

### E110 Per-state results

| Case / seed / step | Phase | Type | Held | Narrow choices | Broad choices | Broad2mm | Chosen | Gain fraction40ms | Chosen survives160ms |
|---|---|---|---|---:|---:|---:|---|---:|---|
| 0/0/37 | descend | successful_control | False | 3 | 39 | 39 | directed1.0 | 0.22107729242359575 | True |
| 0/0/122 | lift | successful_control | False | 0 | 36 | 36 | directed1.0 | 0.09852559101992509 | True |
| 0/0/155 | transport | successful_control | False | 2 | 43 | 43 | random31 | 0.23737988611939648 | True |
| 0/0/171 | lower | successful_control | False | 5 | 65 | 65 | directed1.0 | 0.220062014195924 | True |
| 0/0/231 | retreat | successful_control | False | 2 | 41 | 41 | directed0.5 | 0.09524094753299032 | True |
| 0/1/37 | descend | successful_control | False | 3 | 43 | 43 | directed1.0 | 0.22081596010916194 | True |
| 0/1/131 | lift | successful_control | False | 1 | 36 | 36 | directed1.0 | 0.09211310482609134 | True |
| 0/1/157 | transport | successful_control | False | 2 | 37 | 37 | directed1.0 | 0.2525933322137884 | True |
| 0/1/169 | lower | successful_control | False | 3 | 41 | 41 | directed0.5 | 0.17588321889396674 | True |
| 0/1/223 | retreat | successful_control | False | 2 | 41 | 41 | directed0.5 | 0.09389692309781984 | True |
| 0/2/37 | descend | successful_control | False | 3 | 33 | 33 | directed1.0 | 0.2204464229853761 | True |
| 0/2/124 | lift | successful_control | False | 2 | 33 | 33 | directed1.0 | 0.11013605364473752 | True |
| 0/2/146 | transport | successful_control | False | 2 | 39 | 39 | directed1.0 | 0.2871844705752109 | True |
| 0/2/158 | lower | successful_control | False | 3 | 41 | 41 | directed1.0 | 0.20757438006558085 | True |
| 0/2/214 | retreat | successful_control | False | 1 | 34 | 34 | directed1.0 | 0.09986951123703275 | True |
| 1/0/315 | retreat | warning | False | 0 | 24 | 0 | directed0.5 | 0.13805909285919554 | True |
| 1/2/233 | lower | warning | False | 0 | 17 | 0 | random47 | 0.19444989804961435 | True |
| 2/0/186 | lower | warning | True | 1 | 4 | 0 | brake | 0.10322559201867709 | False |
| 2/1/183 | lower | warning | True | 0 | 2 | 2 | directed1.0 | 0.7550106975562118 | True |
| 6/0/37 | descend | successful_control | False | 3 | 32 | 32 | directed0.5 | 0.16192026781739227 | True |
| 6/0/139 | lift | successful_control | False | 2 | 37 | 37 | directed0.5 | 0.08672818233720742 | True |
| 6/0/179 | transport | successful_control | False | 2 | 34 | 34 | directed1.0 | 0.23129404245022284 | True |
| 6/0/191 | lower | successful_control | False | 3 | 20 | 20 | directed1.0 | 0.10178342031978518 | True |
| 6/0/276 | retreat | successful_control | False | 2 | 43 | 43 | directed0.5 | 0.09120531050292095 | True |
| 6/1/218 | lower | warning | False | 0 | 12 | 12 | axis2_1_0.04 | 0.18387203549621003 | True |
| 6/2/37 | descend | successful_control | False | 3 | 30 | 30 | directed0.5 | 0.1636753702194632 | True |
| 6/2/131 | lift | successful_control | False | 2 | 38 | 38 | directed1.0 | 0.10533898796906806 | True |
| 6/2/224 | transport | successful_control | False | 2 | 46 | 46 | directed1.0 | 0.3151745095866084 | True |
| 6/2/237 | lower | successful_control | False | 3 | 33 | 33 | directed1.0 | 0.18606735678497363 | True |
| 6/2/308 | retreat | successful_control | False | 2 | 43 | 43 | directed1.0 | 0.1058867247421367 | True |
| 7/1/37 | descend | successful_control | False | 3 | 31 | 31 | directed0.25 | 0.13082155300990825 | True |
| 7/1/153 | lift | successful_control | False | 1 | 27 | 27 | random7 | 0.054227837111737155 | True |
| 7/1/210 | transport | successful_control | False | 4 | 70 | 70 | directed0.5 | 0.26242803844156404 | True |
| 7/1/231 | lower | successful_control | False | 3 | 32 | 32 | directed0.25 | 0.13000117726861343 | True |
| 7/1/298 | retreat | successful_control | False | 2 | 40 | 40 | directed0.25 | 0.07679301433194137 | True |
| 8/0/56 | descend | warning | True | 0 | 12 | 12 | directed1.0 | 0.267165053345806 | True |
| 8/1/216 | transport | warning | True | 0 | 4 | 0 | random49 | 0.13673119826880897 | True |
| 8/2/131 | lift | warning | True | 0 | 0 | 0 | None | None | False |
| 10/0/85 | descend | warning | False | 0 | 8 | 8 | random33 | 0.32915875048971177 | True |
| 10/1/357 | transport | warning | False | 0 | 1 | 0 | random27 | 0.01761571205938013 | True |
| 10/2/65 | descend | warning | False | 0 | 3 | 3 | random1 | 0.03704112847091491 | True |
| 11/0/162 | lift | warning | True | 0 | 3 | 0 | random43 | 0.1020357690445145 | True |
| 11/1/158 | lift | warning | True | 0 | 10 | 0 | random15 | 0.10701920476512358 | True |
| 11/2/147 | lift | warning | True | 0 | 6 | 0 | random9 | 0.09492736585721606 | True |

### E110 Interpretation and next decision

Positive state availability would show that the previous narrow fallback set missed useful local actions. It would not establish that the neural policy can generate them, that selecting them prevents future failures, or that performance transfers to new embodiments. A controller trial must freeze candidate generation and selection using training evidence, preserve task budgets and all prior successes, and test full episodes. State selection is conditional on warning/success histories and repeated bodies/seeds; do not treat44 states as44 independent embodiments.
Clearance is measured on a separate expanded-geometry copy carrying each predicted object/finger state. This copy is never integrated. Models and full integration states are privileged. Held placements use the same training bodies. No E105 or reserved final-test data enter this audit.

### E110 Saved evidence

4905 distinct candidate simulations across states, complete candidate actions/metrics, full MuJoCo integration snapshots and XML, source/checkpoint hashes, and exact unchanged baseline replays. Videos are verified reused baseline successes/failures, explicitly not executions of newly selected candidates.

### E110 Completed findings and geometric sensitivity

Primary warning-state availability is1/14 narrow versus13/14 broad. The broad bank has qualifying candidates in6/6 training-placement warning states and7/8 held-placement warning states. The40ms-selected broad action retains contact-free progress through160ms in12/14 states, versus0/14 narrow. Broad candidates satisfying the full2mm dynamic margin exist in5/14 warning states. All30 successful-control states have broad candidates with2mm clearance and progress, but this is not a full-task success-retention test.
Case8/seed2 lift is the unresolved warning state: every sampled action predicts forbidden physical contact within40ms. This finite search does not establish infeasibility. Other warning-state chosen actions include sampled joint directions, coordinate probes and finite-difference goal-directed responses; the result is not attributable solely to the directed generator.
Supplemental geometric sensitivity (reported after outcomes, without changing the prespecified primary metric): require minimum expanded-model geometric clearance >=0 in addition to the primary criteria. Warning-state availability becomes0/14 narrow and13/14 broad. The primary-selected brake in case2/seed0 has approximately -0.03mm minimum geometric clearance despite no physical solver contact flag by40ms, and fails the160ms hold check. Other broad candidates in that state have nonnegative40ms clearance. Solver contact sampling and geometry at integrated positions differ; a future controller must not equate a missing contact flag with nonpenetration. We do not retrospectively rerank the160ms outcomes.

[
  {
    "kind": "warning",
    "bank": "narrow",
    "states_with_nonnegative_clearance_and_progress40": 0
  },
  {
    "kind": "warning",
    "bank": "broad",
    "states_with_nonnegative_clearance_and_progress40": 13
  },
  {
    "kind": "successful_control",
    "bank": "narrow",
    "states_with_nonnegative_clearance_and_progress40": 29
  },
  {
    "kind": "successful_control",
    "bank": "broad",
    "states_with_nonnegative_clearance_and_progress40": 30
  }
]
This audit identifies useful local alternatives, not an executed improvement. A next trial should explicitly require nonnegative geometric clearance, freeze progress-aware selection before running full training episodes, and retain the2mm-margin breakdown. Candidate generation costs hundreds of simulator substeps per choice; no real-time or hardware claim is made.

Saved-snapshot verification reconstructs116 original/selected candidate branches exactly at40 and160ms. All131 regression tests passed in13.99s.

## 4.101 E111: progress-aware selection with geometric nonpenetration

72 full tasks: all36 original training-body task/seed pairs, baseline and fixed physics selector. Frozen E110 broad bank,40ms prediction, original static admissibility, zero physical contacts, nonnegative minimum geometric clearance and at least1% pose-error reduction (floor1e-8). Rank by smallest predicted pose error, then larger clearance, then name. Preserve original commands unless contact or penetration is predicted; preserve original if no candidate qualifies. Same weights, apertures and budgets. Extend eligibility to descend/lift/transport/lower/retreat, but only before any earlier forbidden contact. Approach/calibration unchanged; no route planner. Descent interventions can alter later probe/history states. No tuning after outcomes.
The24 training and12 held-placement pairs share four original training bodies. No E105 or reserved final test. Exact physical-model/state access is privileged; this is not learned dynamics or history-only control.

| Group | Mode | Success | Collisions | Contact-free failures | Grasped | Lifted | Steps | Changes | Warnings | No safe option |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | baseline | 22/36 | 14 | 0 | 36 | 36 | 13406 | 0 | 14 | 0 |
| all | physics | 31/36 | 3 | 2 | 36 | 36 | 13453 | 54 | 57 | 3 |
| train | baseline | 18/24 | 6 | 0 | 24 | 24 | 8786 | 0 | 6 | 0 |
| train | physics | 24/24 | 0 | 0 | 24 | 24 | 8663 | 21 | 21 | 0 |
| validation | baseline | 4/12 | 8 | 0 | 12 | 12 | 4620 | 0 | 8 | 0 |
| validation | physics | 7/12 | 3 | 2 | 12 | 12 | 4790 | 33 | 36 | 3 |
| 6 | baseline | 14/18 | 4 | 0 | 18 | 18 | 5803 | 0 | 4 | 0 |
| 6 | physics | 17/18 | 1 | 0 | 18 | 18 | 5775 | 4 | 5 | 1 |
| 7 | baseline | 8/18 | 10 | 0 | 18 | 18 | 7603 | 0 | 10 | 0 |
| 7 | physics | 14/18 | 2 | 2 | 18 | 18 | 7678 | 50 | 52 | 2 |
| seed0 | baseline | 7/12 | 5 | 0 | 12 | 12 | 4344 | 0 | 5 | 0 |
| seed0 | physics | 11/12 | 1 | 0 | 12 | 12 | 4319 | 13 | 14 | 1 |
| seed1 | baseline | 7/12 | 5 | 0 | 12 | 12 | 4644 | 0 | 5 | 0 |
| seed1 | physics | 10/12 | 1 | 1 | 12 | 12 | 4535 | 21 | 22 | 1 |
| seed2 | baseline | 8/12 | 4 | 0 | 12 | 12 | 4418 | 0 | 4 | 0 |
| seed2 | physics | 10/12 | 1 | 1 | 12 | 12 | 4599 | 20 | 21 | 1 |

### E111 Paired task changes

[
  {
    "case": 1,
    "seed": 0,
    "dof": 6,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 1,
    "seed": 2,
    "dof": 6,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 2,
    "seed": 1,
    "dof": 6,
    "validation": true,
    "change": "gain",
    "collision": false
  },
  {
    "case": 6,
    "seed": 1,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 8,
    "seed": 0,
    "dof": 7,
    "validation": true,
    "change": "gain",
    "collision": false
  },
  {
    "case": 10,
    "seed": 0,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 10,
    "seed": 1,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 10,
    "seed": 2,
    "dof": 7,
    "validation": false,
    "change": "gain",
    "collision": false
  },
  {
    "case": 11,
    "seed": 0,
    "dof": 7,
    "validation": true,
    "change": "gain",
    "collision": false
  }
]

### E111 Retention gate

**PASS**

{
  "train": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true,
    "success_improves": true,
    "per_seed_nondecreasing": true
  },
  "validation": {
    "retains_all_successes": true,
    "collisions_nonincreasing": true
  }
}

### E111 Task progress

Phase reached counts use the original position/tool-axis completion rules, including final-state tolerance on timed-out phases. Phases still advance at the original time limit. Grasped/lifted are original sticky flags, not maintained-grasp guarantees.

| Phase | Mode | Reached /36 | Control steps |
|---|---|---:|---:|
| descend | baseline | 34 | 1024 |
| descend | physics | 35 | 990 |
| lift | baseline | 32 | 1894 |
| lift | physics | 30 | 1954 |
| transport | baseline | 33 | 899 |
| transport | physics | 31 | 981 |
| lower | baseline | 36 | 1134 |
| lower | physics | 35 | 1049 |
| retreat | baseline | 32 | 1633 |
| retreat | physics | 31 | 1657 |

Changed-action progress uses same-state40ms squared position plus .15-scaled tool-axis error. The original action is a simulator counterfactual and may collide; compare it only as a diagnostic. Predicted progress gates and ranks candidate selection in this experiment; this is a selection criterion, not independent evidence of generalization. Actions and seeds are correlated, not independent samples.

[
  {
    "phase": "descend",
    "changed": 15,
    "choices": {
      "directed1.0": 1,
      "random33": 1,
      "random1": 1,
      "axis1_-1_0.04": 1,
      "random21": 1,
      "random43": 1,
      "random53": 1,
      "random35": 1,
      "random15": 1,
      "random11": 1,
      "random45": 1,
      "random37": 1,
      "random55": 1,
      "random27": 1,
      "random57": 1
    },
    "selected_reduces_error": 15,
    "selected_increases_error": 0,
    "original_reduces_error": 1
  },
  {
    "phase": "lift",
    "changed": 15,
    "choices": {
      "random43": 1,
      "random41": 1,
      "random45": 1,
      "random53": 1,
      "random21": 1,
      "random15": 1,
      "random1": 1,
      "random27": 1,
      "random29": 1,
      "random9": 2,
      "random23": 1,
      "random25": 1,
      "random17": 1,
      "random49": 1
    },
    "selected_reduces_error": 15,
    "selected_increases_error": 0,
    "original_reduces_error": 15
  },
  {
    "phase": "transport",
    "changed": 7,
    "choices": {
      "random49": 1,
      "random35": 2,
      "random9": 1,
      "random27": 2,
      "random63": 1
    },
    "selected_reduces_error": 7,
    "selected_increases_error": 0,
    "original_reduces_error": 4
  },
  {
    "phase": "lower",
    "changed": 6,
    "choices": {
      "random47": 1,
      "random40": 1,
      "directed1.0": 3,
      "axis2_1_0.04": 1
    },
    "selected_reduces_error": 6,
    "selected_increases_error": 0,
    "original_reduces_error": 2
  },
  {
    "phase": "retreat",
    "changed": 11,
    "choices": {
      "directed0.5": 1,
      "random21": 1,
      "random41": 1,
      "random25": 1,
      "random6": 1,
      "random63": 1,
      "random53": 1,
      "random11": 1,
      "random37": 1,
      "random57": 1,
      "random61": 1
    },
    "selected_reduces_error": 11,
    "selected_increases_error": 0,
    "original_reduces_error": 10
  }
]

### E111 Failures

[
  {
    "case": 1,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "retreat",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 1,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 2,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 2,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 2,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "transport",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 8,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "selector_status": "no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "transport"
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 8,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "lift",
    "selector_status": "no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria",
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 10,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "transport",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 10,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "release"
    ]
  },
  {
    "case": 11,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "retreat",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "settle"
    ]
  },
  {
    "case": 11,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": null,
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport"
    ]
  },
  {
    "case": 11,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "lift",
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": []
  },
  {
    "case": 11,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": null,
    "selector_status": null,
    "grasped": true,
    "lifted": true,
    "unreached_phases": [
      "lift",
      "transport",
      "lower",
      "release",
      "retreat",
      "settle"
    ]
  }
]

### E111 Decision

The fixed rule passes the training/held-placement gate; it remains a candidate requiring a separately frozen broader evaluation.
No horizon, candidate or phase tuning after these outcomes. Full traces retain all failures, alternatives and state predictions. Before another control change, identify whether failure is due to no feasible alternative, loss of task progress, or a contact in an unmodified phase.

### E111 Verification

36 full baseline replays, 72 approach prefixes, and 13215 selected next states/contact unions match exactly. All changed actions occur before any prior collision. Inputs/checkpoints unchanged. Videos replay full serialized results; task success/failure videos are executed motion, not lookahead animation.

The broader search uses exact physics and finite-difference simulated response. This is a privileged model-based controller component, not a new learned policy. See protocol.json for frozen candidate counts, random seeds and thresholds.

The inherited held-placement label denotes placement holdout from the original training data, not an untouched evaluation: these placements were inspected in earlier development experiments, including E110. Treat this gate as development retention, not independent generalization confirmation.

### E111 Compute and margin

{
  "physics_all": {
    "n": 6631,
    "median_ms": 4.48717399922316,
    "p95_ms": 5.124834999151062,
    "total_seconds": 59.97994358582582,
    "branches": 12999
  },
  "physics_search": {
    "n": 57,
    "median_ms": 530.0142250016506,
    "p95_ms": 582.4403586018889,
    "total_seconds": 30.281295568998758,
    "branches": 6425
  },
  "physics_check_only": {
    "n": 6574,
    "median_ms": 4.48464999863063,
    "p95_ms": 5.076675300733768,
    "total_seconds": 29.698648016827065,
    "branches": 6574
  }
}

{
  "changed_actions": 54,
  "changed_with_2mm": 19
}
Measured times are CPU wall time for added prediction/search after the original static-filter decision; they exclude neural inference, normal execution and the original static-filter cost. Baseline was instrumented too, so these are not an uninstrumented whole-policy speed comparison. The physical control interval is40ms. A search slower than40ms is not real-time ready. Every changed action has nonnegative predicted clearance; only the separate count maintains2mm throughout prediction.

### E111 Research progress

We have a reproducible state-based simulation pipeline, frozen neural checkpoints, baseline/retention gates, raw trajectories and replay-verified success/failure videos. E104 improved a reused novel-axis development suite from32/48 to35/48, but E105 fresh-body replication tied baseline at27/45 and introduced a paired regression. E108/E109 short physics fallbacks tied the training baseline at22/36; E109 degraded intermediate phase progress. E110 showed broader local action availability in13/14 warning states, which motivated this full-episode trial.
The central unresolved claim is reliable history-conditioned control on unseen embodiments. Current physics-assisted interventions use exact state/model access and are not evidence of history-only adaptation. Before treating this as a conference result, the final method needs reliable full-task improvement, an untouched prospective embodiment evaluation, appropriate analytical/specialist/history ablations, and clear accounting of privileged inputs and realistic-robot scope. This experiment is one controlled step toward that evidence, not publication readiness.
E111 now improves complete-task success from22/36 to31/36 (nine gains, zero losses), including24/24 training-placement successes and7/12 reused held-placement successes. Forbidden-contact episodes fall from14 to3. All54 changed actions reduce immediate pose error, but phase completion is not uniformly improved; five full-task failures remain. This supports the frozen physics-assisted component on development training bodies. The next milestone is a prospectively frozen unseen-body comparison, followed by attribution ablations and reducing search latency.
Regression validation:131 tests passed in12.32 seconds.

### E111 Run integrity

The initial attempt stopped at a tuple-versus-list contact-pair parity assertion before executing its first alternative action. Its source and seven completed trials are preserved in outputs/ease_progress_selector_initial_check. Only representation/check diagnostics were corrected; candidate generation, thresholds and ranking were unchanged. All72 final trials rerun from scratch; no partial outcomes are substituted.

## 4.102 E112: frozen progress-aware selector on prospective unseen bodies

16/16 candidate placements passed endpoint IK screening; 96 complete paired rollouts across all three checkpoint seeds. Every rejection and accepted-task failure is retained. No route planner, extra task budget, weight update or selector tuning.

Eight new parameterizations of four E87 axis orders, two bodies per order and two placements per body. The axis layouts differ from the original training bodies but were inspected in earlier development. This is a prospective body-parameter replication, not an untouched axis-family test or the reserved final test. Procedural simulator arms; exact physics/state/geometry and IK initialization remain privileged.

| Group | Mode | Success | Collisions | Contact-free failures | Changed actions |
|---|---|---:|---:|---:|---:|
| all | baseline | 28/48 | 13 | 7 | 0 |
| all | physics | 31/48 | 10 | 7 | 68 |
| 6 | baseline | 14/24 | 10 | 0 | 0 |
| 6 | physics | 14/24 | 10 | 0 | 65 |
| 7 | baseline | 14/24 | 3 | 7 | 0 |
| 7 | physics | 17/24 | 0 | 7 | 3 |
| seed0 | baseline | 9/16 | 5 | 2 | 0 |
| seed0 | physics | 10/16 | 4 | 2 | 25 |
| seed1 | baseline | 8/16 | 5 | 3 | 0 |
| seed1 | physics | 9/16 | 4 | 3 | 29 |
| seed2 | baseline | 11/16 | 3 | 2 | 0 |
| seed2 | physics | 12/16 | 2 | 2 | 14 |
| body0 | baseline | 4/6 | 2 | 0 | 0 |
| body0 | physics | 4/6 | 2 | 0 | 0 |
| body1 | baseline | 6/6 | 0 | 0 | 0 |
| body1 | physics | 6/6 | 0 | 0 | 0 |
| body2 | baseline | 1/6 | 5 | 0 | 0 |
| body2 | physics | 1/6 | 5 | 0 | 49 |
| body3 | baseline | 3/6 | 3 | 0 | 0 |
| body3 | physics | 3/6 | 3 | 0 | 16 |
| body4 | baseline | 0/6 | 0 | 6 | 0 |
| body4 | physics | 0/6 | 0 | 6 | 0 |
| body5 | baseline | 3/6 | 3 | 0 | 0 |
| body5 | physics | 6/6 | 0 | 0 | 3 |
| body6 | baseline | 5/6 | 0 | 1 | 0 |
| body6 | physics | 5/6 | 0 | 1 | 0 |
| body7 | baseline | 6/6 | 0 | 0 | 0 |
| body7 | physics | 6/6 | 0 | 0 | 0 |

### E112 Paired changes

[
  {
    "case": 10,
    "seed": 0,
    "body": 5,
    "dof": 7,
    "change": "gain"
  },
  {
    "case": 11,
    "seed": 1,
    "body": 5,
    "dof": 7,
    "change": "gain"
  },
  {
    "case": 11,
    "seed": 2,
    "body": 5,
    "dof": 7,
    "change": "gain"
  }
]

### E112 Frozen transfer gate

**PASS**

{
  "success_improves": true,
  "retains_all_successes": true,
  "collisions_nonincreasing": true,
  "per_seed_nondecreasing": true,
  "per_dof_nondecreasing": true
}

### E112 Screening and uncertainty

[
  {
    "candidate": 0,
    "body": 0,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 1,
    "body": 0,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 2,
    "body": 1,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 3,
    "body": 1,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 4,
    "body": 2,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 5,
    "body": 2,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 6,
    "body": 3,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 7,
    "body": 3,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 8,
    "body": 4,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 9,
    "body": 4,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 10,
    "body": 5,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 11,
    "body": 5,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 12,
    "body": 6,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 13,
    "body": 6,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 14,
    "body": 7,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  },
  {
    "candidate": 15,
    "body": 7,
    "accepted": true,
    "endpoints": 9,
    "failed_phase": null
  }
]

Pipeline success uses all48 candidate/seed pairs, with screened-out placements counted as zero for both modes. The bootstrap resamples eight whole bodies, preserving correlated placements and checkpoints; 10,000 fixed-seed resamples provide descriptive intervals, not evidence from48 independent embodiments.

{
  "pipeline_difference": 0.0625,
  "percentile_95": [
    0.0,
    0.1875
  ],
  "body_differences": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.5,
    0.0,
    0.0
  ],
  "clusters": 8,
  "resamples": 10000
}

### E112 Compute

{
  "check_only": {
    "n": 8336,
    "median_ms": 4.408034497828339,
    "p95_ms": 5.049674751717248
  },
  "search": {
    "n": 74,
    "median_ms": 519.8373780003749,
    "p95_ms": 600.3635065488197
  }
}

Added prediction/search wall time excludes neural inference, original static filtering and normal execution. Compare search latency with the40ms control interval; this is not a hardware real-time claim.

### E112 Interpretation

The frozen selector passes the prespecified transfer gate on this prospective cohort. This supports transfer of the privileged physics-assisted component; it does not isolate learned history adaptation or demonstrate history-only control.

Before a paper claim, separate the history-conditioned neural component from exact-model search using analytical/specialist/history ablations, assess compute, and broaden independent embodiment evidence. No method changes were made after outcomes.

Recovered episodes occur on 1/8 bodies (body IDs [5]). The body-bootstrap interval includes zero when its lower endpoint is zero; passing the operational retention gate is not statistical confirmation of broad transfer. Each of the three observed recoveries in this run uses one action substitution. The six-joint controller makes65 substitutions without changing any task outcome; safe local progress is insufficient for those failures.

### E112 Remaining failures

{
  "release / unmodified phase": 4,
  "lower / no candidate meets all frozen criteria": 3,
  "descend / no candidate meets all frozen criteria": 3,
  "no collision / task criteria failed": 7
}

In this cohort, all ten remaining collision episodes are six-joint: six first collide in descend/lower when no candidate satisfies every frozen criterion, and four first collide during the unmodified release phase. The seven contact-free failures are seven-joint; six belong to one body that misses descent/close tolerances. These are trace-level diagnostics, not controlled causal ablations. The next diagnostic should examine those six-joint states and release transitions without tuning this frozen evaluation result.

### E112 Verification and artifacts

{
  "exact_compatibility_replays": 12,
  "exact_paired_prefixes": 48,
  "exact_selected_predictions": 17104,
  "hashes_unchanged": true,
  "reserved_test_evaluated": false
}

{
  "frozen_nested_functions": [
    "predict",
    "select",
    "step"
  ],
  "nested_function_asts_identical": true,
  "e111_source_hashes_identical": true,
  "full_serialized_compatibility_replays": 12,
  "distinct_from_original_training_body_axes": true,
  "new_body_seeds": true,
  "component_tests": {
    "passed": 16,
    "seconds": 4.41,
    "files": [
      "test_ease_fm_geometry_filter.py",
      "test_ease_fm_local_history.py",
      "test_ease_fm_local_probe.py",
      "test_ease_fm_task_retention_training.py",
      "test_ease_fm_pick_place.py",
      "test_ease_fm_arm_pick_place.py"
    ]
  }
}

Saved protocol, source snapshots, checkpoint/input hashes, all raw traces and failures, timing measurements, and replay-verified success/failure videos. Twelve prespecified E111 compatibility runs match full serialized results. All selected predicted next states/contact unions match execution; all paired calibration/approach prefixes match. See failure_diagnostics.json for every failed rollout.

Final artifact verification: 12 full-result video replays and SHA256 hashes verified.

## 4.103 E113: six-joint failure diagnosis and release-phase extension

48 complete rollouts: all24 six-joint E112 task/seed pairs with original selector and release-extended selector. Here baseline means the E112 physics selector, not the earlier unassisted controller. Only release eligibility changes; exact action bank,40ms horizon, thresholds, progress ranking, apertures and task budgets are unchanged. No retraining or tuning after outcomes. These are reused development bodies, not fresh independent transfer evidence.

| Group | Mode | Success | Collisions | Release changes | Eligible release warnings |
|---|---|---:|---:|---:|---:|
| all | baseline | 14/24 | 10 | 0 | 0 |
| all | physics | 16/24 | 8 | 8 | 10 |
| seed0 | baseline | 4/8 | 4 | 0 | 0 |
| seed0 | physics | 5/8 | 3 | 3 | 4 |
| seed1 | baseline | 4/8 | 4 | 0 | 0 |
| seed1 | physics | 4/8 | 4 | 3 | 4 |
| seed2 | baseline | 6/8 | 2 | 0 | 0 |
| seed2 | physics | 7/8 | 1 | 2 | 2 |
| body0 | baseline | 4/6 | 2 | 0 | 0 |
| body0 | physics | 5/6 | 1 | 2 | 2 |
| body1 | baseline | 6/6 | 0 | 0 | 0 |
| body1 | physics | 6/6 | 0 | 0 | 0 |
| body2 | baseline | 1/6 | 5 | 0 | 0 |
| body2 | physics | 1/6 | 5 | 3 | 4 |
| body3 | baseline | 3/6 | 3 | 0 | 0 |
| body3 | physics | 4/6 | 2 | 3 | 4 |

### E113 Paired changes

[
  {
    "case": 0,
    "seed": 2,
    "change": "gain"
  },
  {
    "case": 6,
    "seed": 0,
    "change": "gain"
  }
]

### E113 Frozen gate

**PASS**

{
  "success_improves": true,
  "retains_all_successes": true,
  "collisions_nonincreasing": true,
  "per_seed_nondecreasing": true
}

### E113 Diagnosis

Constraint counts are sequential intersections: all alternatives, static-admissible, no forbidden physical contact, nonnegative geometric clearance, then required goal-error reduction. A zero at a stage identifies why the tested bank has no eligible action; it does not prove that no safe action exists outside the bank. Physical contact is allowed for object-table and object-pads. Other contacts are failures.

All first-collision states are saved in failure_diagnostics.json; every eligible release warning is in release_diagnostics.json. Earlier failures cannot be repaired by this extension because it starts only at release and disables interventions after any previous forbidden contact.

7/8 remaining failures have static-admissible, contact-free, nonpenetrating40ms alternatives at the first-collision decision, but none meets the required progress gain. The frozen fallback consequently keeps the original unsafe action. This identifies a local conflict between safety and strict immediate goal progress; it does not establish that those alternatives would finish the task. One remaining lower-phase case has no static-admissible contact-free alternative in the tested bank.

[
  {
    "case": 0,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "gripper_left_pad",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": [
      "release"
    ]
  },
  {
    "case": 1,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 86,
      "contact_free": 0,
      "nonpenetrating": 0,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.004073995078845598,
      "required_gain": 4.073995078845598e-05,
      "best_safe_gain": null
    },
    "unreached_phases": []
  },
  {
    "case": 1,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "gripper_right_pad",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 86,
      "contact_free": 0,
      "nonpenetrating": 0,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.004073995078845598,
      "required_gain": 4.073995078845598e-05,
      "best_safe_gain": null
    },
    "unreached_phases": []
  },
  {
    "case": 4,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 91,
      "contact_free": 35,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010981534367269468,
      "required_gain": 1.0981534367269468e-06,
      "best_safe_gain": 3.694962766486782e-07
    },
    "unreached_phases": [
      "descend",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 91,
      "contact_free": 35,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010981534367269468,
      "required_gain": 1.0981534367269468e-06,
      "best_safe_gain": 3.694962766486782e-07
    },
    "unreached_phases": [
      "descend",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 87,
      "contact_free": 14,
      "nonpenetrating": 14,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00017898668859814913,
      "required_gain": 1.7898668859814914e-06,
      "best_safe_gain": -1.838808472145369e-06
    },
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 87,
      "contact_free": 14,
      "nonpenetrating": 14,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00017898668859814913,
      "required_gain": 1.7898668859814914e-06,
      "best_safe_gain": -1.838808472145369e-06
    },
    "unreached_phases": [
      "descend",
      "probe",
      "close",
      "lower",
      "release"
    ]
  },
  {
    "case": 4,
    "seed": 2,
    "mode": "baseline",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 89,
      "contact_free": 37,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010202529681817159,
      "required_gain": 1.020252968181716e-06,
      "best_safe_gain": -2.8065011967128484e-06
    },
    "unreached_phases": [
      "close",
      "lift"
    ]
  },
  {
    "case": 4,
    "seed": 2,
    "mode": "physics",
    "first_collision_phase": "descend",
    "pairs": [
      [
        "3",
        "table"
      ],
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 89,
      "contact_free": 37,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00010202529681817159,
      "required_gain": 1.020252968181716e-06,
      "best_safe_gain": -2.8065011967128484e-06
    },
    "unreached_phases": [
      "close",
      "lift"
    ]
  },
  {
    "case": 5,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ],
      [
        "5",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 88,
      "contact_free": 39,
      "nonpenetrating": 39,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00022072122249776002,
      "required_gain": 2.2072122249776002e-06,
      "best_safe_gain": 2.1942090262566302e-06
    },
    "unreached_phases": [
      "lower",
      "release"
    ]
  },
  {
    "case": 5,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ],
      [
        "5",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 88,
      "contact_free": 39,
      "nonpenetrating": 39,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 0.00022072122249776002,
      "required_gain": 2.2072122249776002e-06,
      "best_safe_gain": 2.1942090262566302e-06
    },
    "unreached_phases": [
      "lower",
      "release"
    ]
  },
  {
    "case": 5,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": []
  },
  {
    "case": 5,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 94,
      "contact_free": 35,
      "nonpenetrating": 35,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 5.410090119087839e-06,
      "required_gain": 5.410090119087839e-08,
      "best_safe_gain": -1.7152685337518624e-06
    },
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "baseline",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 84,
      "contact_free": 25,
      "nonpenetrating": 25,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 1.6068680192624934e-05,
      "required_gain": 1.6068680192624936e-07,
      "best_safe_gain": -7.003729006468406e-07
    },
    "unreached_phases": []
  },
  {
    "case": 6,
    "seed": 1,
    "mode": "physics",
    "first_collision_phase": "lower",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 84,
      "contact_free": 25,
      "nonpenetrating": 25,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 1.6068680192624934e-05,
      "required_gain": 1.6068680192624936e-07,
      "best_safe_gain": -7.003729006468406e-07
    },
    "unreached_phases": []
  },
  {
    "case": 7,
    "seed": 0,
    "mode": "baseline",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": null,
    "unreached_phases": []
  },
  {
    "case": 7,
    "seed": 0,
    "mode": "physics",
    "first_collision_phase": "release",
    "pairs": [
      [
        "4",
        "table"
      ]
    ],
    "criteria": {
      "candidates": 108,
      "static": 91,
      "contact_free": 34,
      "nonpenetrating": 34,
      "progress": 0,
      "choice": "original",
      "before_error_sq": 2.4443927407486863e-06,
      "required_gain": 2.4443927407486862e-08,
      "best_safe_gain": -7.579693170879154e-07
    },
    "unreached_phases": [
      "settle"
    ]
  }
]

### E113 Release decisions

[
  {
    "case": 0,
    "seed": 2,
    "step": 249,
    "candidates": 108,
    "static": 100,
    "contact_free": 91,
    "nonpenetrating": 90,
    "progress": 14,
    "choice": "directed1.0",
    "before_error_sq": 0.0023349795841114736,
    "required_gain": 2.3349795841114736e-05,
    "best_safe_gain": 0.0006522615617576136
  },
  {
    "case": 0,
    "seed": 2,
    "step": 250,
    "candidates": 108,
    "static": 100,
    "contact_free": 82,
    "nonpenetrating": 79,
    "progress": 65,
    "choice": "directed0.25",
    "before_error_sq": 0.00168271802235386,
    "required_gain": 1.68271802235386e-05,
    "best_safe_gain": 0.00034061669997597407
  },
  {
    "case": 5,
    "seed": 1,
    "step": 199,
    "candidates": 108,
    "static": 82,
    "contact_free": 32,
    "nonpenetrating": 32,
    "progress": 2,
    "choice": "random2",
    "before_error_sq": 1.1725442236746228e-05,
    "required_gain": 1.1725442236746229e-07,
    "best_safe_gain": 2.3987124906616824e-06
  },
  {
    "case": 5,
    "seed": 1,
    "step": 200,
    "candidates": 108,
    "static": 92,
    "contact_free": 78,
    "nonpenetrating": 78,
    "progress": 13,
    "choice": "directed0.5",
    "before_error_sq": 9.326729746084546e-06,
    "required_gain": 9.326729746084546e-08,
    "best_safe_gain": 3.2041106816363535e-06
  },
  {
    "case": 5,
    "seed": 1,
    "step": 201,
    "candidates": 108,
    "static": 96,
    "contact_free": 27,
    "nonpenetrating": 26,
    "progress": 1,
    "choice": "axis2_1_0.02",
    "before_error_sq": 6.122619064448192e-06,
    "required_gain": 6.122619064448193e-08,
    "best_safe_gain": 7.125289453603539e-07
  },
  {
    "case": 5,
    "seed": 1,
    "step": 202,
    "candidates": 108,
    "static": 94,
    "contact_free": 35,
    "nonpenetrating": 35,
    "progress": 0,
    "choice": "original",
    "before_error_sq": 5.410090119087839e-06,
    "required_gain": 5.410090119087839e-08,
    "best_safe_gain": -1.7152685337518624e-06
  },
  {
    "case": 6,
    "seed": 0,
    "step": 207,
    "candidates": 108,
    "static": 86,
    "contact_free": 58,
    "nonpenetrating": 58,
    "progress": 11,
    "choice": "random61",
    "before_error_sq": 3.291809525911844e-05,
    "required_gain": 3.291809525911844e-07,
    "best_safe_gain": 1.2412041333927836e-05
  },
  {
    "case": 7,
    "seed": 0,
    "step": 190,
    "candidates": 108,
    "static": 88,
    "contact_free": 38,
    "nonpenetrating": 38,
    "progress": 1,
    "choice": "random14",
    "before_error_sq": 6.710609615353138e-06,
    "required_gain": 6.710609615353138e-08,
    "best_safe_gain": 1.70210581467119e-06
  },
  {
    "case": 7,
    "seed": 0,
    "step": 191,
    "candidates": 108,
    "static": 87,
    "contact_free": 76,
    "nonpenetrating": 74,
    "progress": 6,
    "choice": "directed1.0",
    "before_error_sq": 5.008503800681948e-06,
    "required_gain": 5.0085038006819476e-08,
    "best_safe_gain": 2.5641110599332615e-06
  },
  {
    "case": 7,
    "seed": 0,
    "step": 192,
    "candidates": 108,
    "static": 91,
    "contact_free": 34,
    "nonpenetrating": 34,
    "progress": 0,
    "choice": "original",
    "before_error_sq": 2.4443927407486863e-06,
    "required_gain": 2.4443927407486862e-08,
    "best_safe_gain": -7.579693170879154e-07
  }
]

### E113 Interpretation

The release extension passes the prespecified development gate; require a newly frozen prospective replication before claiming general transfer.

The original requirement of at least1% pose-error reduction is unchanged during release. Both remaining release failures have safe alternatives rejected by this requirement. Next test a separately frozen, bounded near-goal safety fallback on training states, measuring full-task progress and preventing indefinite holding. Do not simply remove progress everywhere or relabel this reused cohort as untouched validation.

### E113 Verification

{
  "exact_e112_replays": 24,
  "exact_pre_release_prefixes": 24,
  "exact_predictions": 8339,
  "hashes_unchanged": true
}

All24 original-selector replays match E112 full serialized results; all extended rollouts match before release. Every selected next-state/contact prediction matches execution. Exact simulator access remains privileged; this is not history-only adaptation. Videos are outcome-selected illustrations and must match full serialized replay.

Final verification: 8 full-result video replays and hashes match.

Progress demo: ../ease_progress_demo_v1/progress_demo.mp4 (60 seconds; results through E113; illustrative graphics plus verified footage; final card includes this release experiment).

## 5. Limitations and next decisions

**E113 completed:** Release-only extension on all24 six-joint E112 pairs: original selector 14/24 (10 collisions), release extension 16/24 (8 collisions). Gate PASS; 8 release substitutions. All24 original-selector replays and24 pre-release prefixes exact. Reused development cohort; frozen40ms bank/progress/nonpenetration criteria, no tuning. 8 verified videos. Separate60-second concept/progress reel covers results through E113. See Section4.103.

**E112 completed:** Frozen E111 selector on eight prospectively perturbed bodies from four previously studied novel-axis families: baseline 28/48 (13 collisions), physics 31/48 (10 collisions); 3 gains and 0 losses. 16/16 placements pass prespecified endpoint screening. Transfer gate PASS. No route planner, weight update, budget extension or selector tuning. Exact model/state remain privileged; not history-only adaptation or untouched axis-family evidence. Saved 12 verified success/failure videos. See Section4.102.

**E111 completed:** Frozen broad progress-aware selector with nonnegative predicted geometric clearance, tested in72 paired training-body episodes: baseline 22/36 (14 collisions; 0 contact-free failures), physics 31/36 (3 collisions; 2 contact-free failures). Training/held-placement gate PASS. Same weights and task budgets; no route planner. Reports phase progress,2mm-margin fraction and search compute cost. All36 baseline replays,72 approach prefixes and 13215 selected next states/contact unions verify exactly. Saves 24 verified executed videos. Privileged exact model/state; no unseen-body or final-test evaluation. See Section4.101.

**E110 completed:** Broader copied-physics action audit on 44 training-body states. Warning-state contact-free progress availability at40ms: narrow 1/14, broad 13/14; broad candidates maintaining2mm margin 5/14. The40ms-selected broad action retains contact-free progress at160ms in 12/14. 20 source tasks replay exactly; 116 snapshot branches reconstruct exactly. No new actions executed in policy, no task-success improvement claim, no final test. See Section4.100.

**E109 completed:** Before-first-collision physics selection across descend/lift/transport/lower/retreat, same fixed E108 alternatives and40ms horizon, tested in72 paired training-body tasks: baseline 22/36 (14 collisions; 0 contact-free failures), physics 22/36 (14 collisions; 0 contact-free failures). Training/held-placement gate FAIL. Records phase completion and same-state action progress. Frozen weights and task budgets; no route planner. All36 baselines,72 approach prefixes and 13343 selected next-state/contact predictions verify exactly. Saves 12 verified executed videos. Privileged physics; no unseen-body or final-test evaluation. See Section4.99.

**E108 completed:** Fixed 40ms exact-physics safer-action selection tested in 72 paired training-body tasks: baseline 22/36 (14 collisions), physics 22/36 (14 collisions). Training/held-placement gate FAIL. Frozen weights and unchanged task budgets; no route planner. All36 baselines,72 pre-retreat prefixes and 3266 selected next-state/contact predictions verify exactly. Saves 4 new verified full-task videos. Privileged physics access; no unseen-body or final-test evaluation. See Section 4.98.

**E107 completed:** Unchanged training-body baseline audit: 36 exact task replays and 1633 exact 40ms predicted states/contact unions. Clean-start 40ms counts: static: TP 2, FP 96, FN 0, TN 1493; physics: TP 2, FP 0, FN 0, TN 1589. Reports held-placement split, longer held-target horizons, latency and all errors. Exact physical-model/state access is privileged; this verifies prediction, not collision avoidance or learned-dynamics transfer. No controller change or final test. See Section 4.97.

**E106 completed:** Six exact E105 diagnostic replays save 20580 contact samples. The fresh-body regression contacts the table 3.150s into retreat, after the ramp ends, despite a preceding static filter decision marked feasible at its 2mm clearance cap. The successful baseline reaches a lower tool height, so tool height alone does not explain the failure. No causal mechanism isolated and no controller change. Six original verified videos retained. See Section 4.96.

**E105 completed:** Prospective replication on eight fresh body parameterizations and two new placements per body, using the same four axis orders. 15/16 placements pass endpoint IK screening; no replacement. baseline 27/45 (17 collisions), budget 27/45 (17 collisions), gentle 27/45 (17 collisions). Strict retention gate FAIL. Reports all 48 candidate/seed opportunities separately from the 45 executable pairs, with body-clustered uncertainty. Saves 131 passing tests, nine exact E104 compatibility replays and 9 verified success/failure videos. Privileged planning/geometry remain; no reserved final test. See Section 4.95.

**E104 completed:** All four isolated ramps screened on training bodies (324 tasks). Training-gate passing phases ['retreat'] receive the complete novel-axis development comparison: retreat: baseline 32/48 (16 collisions), budget 32/48 (16 collisions), gentle 35/48 (13 collisions); transfer gate PASS. Every selected phase is reported; no seed or DOF switching. Fixed weights and unchanged privileged planning/geometry; no final test. Saves 131 passing tests and 30 verified videos across training and transfer galleries. See Section 4.94.

**E103 completed:** Fixed training-arm phase-transition screen: baseline 22/36 successes, 14 collisions; budget 23/36 successes, 13 collisions; gentle 22/36 successes, 12 collisions. Gentle ramps the task-command correction over20 steps at lift/transport/lower/retreat onset; budget-only receives the same extra phase allowance. Strict training/held-placement gate FAILS; no transfer evaluation or policy adoption. All108 full tasks,36 exact baseline references and108 prefixes are saved, with131 passing tests and 9 verified videos. No contact feedback or weight changes. Next isolate the four affected phases on training bodies before choosing any further controller change. See Section 4.93.

**E102 completed:** Ten exact diagnostic replays of five placement regressions and matched successful references save 55020 physics-substep contact samples. Failures differ: transport grasp loss, early-lowering loss of bilateral contact, off-target setdown, and post-release pad recontact during retreat. Two seven-joint failures show only 12-20% bilateral contact before table contact and 67-70 mm relative object displacement. All successful references maintain bilateral contact over their pre-table lowering intervals. Corrects E101 prose: acceptable transport XY error does not establish a maintained grasp, and one miss occurs during retreat rather than lowering/release. No controller change or new benchmark result. Saves ten verified videos, figures and129 passing tests. Next test gentle phase transitions on training bodies with explicit contact monitoring and fixed checkpoints. See Section 4.92.

**E101 completed:** parent 29/48 successes, 18 collisions; start 32/48 successes, 16 collisions; replay 33/48 successes, 13 collisions; onpolicy 30/48 successes, 15 collisions. Strict retention gates against original parent and E99 starting student: replay FAIL; onpolicy FAIL. One fixed collection round on original training arms adds 6366 training /3514 held-placement eligible contexts, retaining failures. Both conditions start E99 retained checkpoints and receive 2000 updates; only 16 extra task contexts per batch change source. Saves six checkpoints,144 new evaluations,48 reference copies,129 passing tests and 15 verified new success/failure videos. Privileged-planner development evaluation; no final test. Do not expand this fixed student-history mixture or select a favorable seed. Next audit gripper contact and object-relative motion during lowering/release, including the five non-collision regressions and matched successes, then define any task-level correction on training bodies before repeating the full retention evaluation. Prediction loss and pre-filter action drift alone have not established reliable task retention. See Section 4.91.

**E100 completed:** Read-only same-state command audit covers all three E99 retained regressions and the one available same-task unchanged-success control. Eight full replays remain exact, with unchanged physical/controller state during queries. Case13 seed1 has identical filtered parent/student commands for all four pre-contact transport steps despite raw disagreement; earlier lift-state divergence remains relevant. Other failures and the successful control both show disagreement, so there is no validated failure threshold. No policy change or new benchmark score. Saves per-step vectors/filter decisions, scientific figures, 129 passing tests and eight verified new videos. Next test training-body on-policy histories against a matched replay-only training control. See Section 4.90.

**E99 completed:** Matched task-data training and action retention reach 31/48 and 32/48 successes, respectively, versus parent 29/48; collisions are 17 and 16 versus 18. Retained gains six cases but loses three seven-joint parent successes, and seed2 declines 12/16 to 11/16. Both replacement gates FAIL. Six-joint retained reaches 14/24 without parent-success losses; data_only reaches 15/24. Training adds 36 full trajectories on four original training bodies, with 24/12 disjoint training/validation trajectories and matched batches across conditions. No evaluation-arm data enters training. All 144 evaluations, six checkpoints, 129 passing tests, scientific figures and 13 verified success/failure videos are saved. Evaluation retains privileged planning and frozen parent prefixes. Next audit command divergence in the three retained regressions before choosing a training-only remedy. See Section 4.89.

**E98 completed:** Three fixed-budget Transformer checkpoints trained on16384 local-history contexts from eight training bodies (12288 train/4096 trajectory validation). Post-probe neural substitution improves29/48 to31/48 full-task successes and collisions18 to17, but the retention gate FAILS: six gains/four seven-joint losses, all losses in seed2. Seed results are8->12,9->11,12->8 of16. Six-joint success improves13/24 to15/24 with no losses (15/15 admitted-route pairs); seven-joint stays16/24. Held-trajectory MSE improves29.7-31.3 percent in every seed but does not prevent task regression. No checkpoint is adopted. All48 parent replays and96 handoffs verify;127 tests and eight new video replays pass. Training uses no evaluation-arm histories; evaluation retains privileged planning and frozen parent prefixes. Next add training-arm task trajectories and explicit behavior retention. See Section4.88.

**E97 completed:** With an identical learned/planner/probe prefix, post-probe analytical continuation reaches68/96 successes with27 collisions versus learned57/96 with37; exact continuation reaches78/96 with18. Analytical gains15 and loses4, all losses seven-joint; both E95 six-joint regressions recover and no learned six-joint success is lost. Six-joint success is26/48 versus30/48; seven-joint31/48 versus38/48 (exact48/48). Exact solves all78 route-assisted pairs without collision;18 no-route pairs remain unchanged failures. The analytical diagnostic gate passes, but this is not a new neural-policy result or an unqualified controller replacement. All96 references,36 fallbacks and288 handoffs verify;127 tests and12 new video replays pass. Next improve the neural estimator with training-only measured-history targets while preserving these regressions. See Section4.87.

**E96 completed:** All 288 E95 rollouts replay exactly during a read-only same-state estimator audit. In all 78 route-assisted post-probe states, the analytical fit has lower Jacobian error and lower linearized task residual than the frozen learned estimator. Median relative Jacobian error is 0.419 learned versus 0.070 analytical for six joints, and 0.410 versus 0.083 for seven joints. Both estimates improve after probing, including in the two lost six-joint cases, so instantaneous accuracy does not establish task recovery. No policy change is adopted. All 125 tests pass; scientific figures, snapshots and ten hash-verified reused success/failure videos are saved. Next compare learned, analytical and exact continuation after an identical probe prefix. See Section 4.86.

**E95 completed:** A fixed 32-step local probe restores median above-ridge directional coverage from one to six/seven, but achieves 57/96 successes with 37 collisions, versus matched hold 59/96 with 35 and no probe 58/96 with 34. Two unique six-joint successes are lost against both controls; the adoption gate fails. Neither intervention contacts forbidden geometry during its 32 steps; regressions occur later. All 96 references and 36 no-route fallbacks match, and 123 tests pass. This remains a frozen learned estimator with privileged planning on reused development tasks. Next audit estimation at identical post-intervention states. See Section 4.85.

**E94 completed:** Read-only excitation audit finds only one measured joint-motion direction above the existing ridge scale in66/78 admitted-route recent histories. Longer history retains a median six directions, though motion remains dominated by one. This supports a locality-versus-directional-coverage limitation, not a proven failure rule. All46 diagnostic rollouts replay exactly,96 handoffs are audited and119 tests pass. Spectral figures and a gallery of unchanged, hash-verified E93 success/failure videos are saved. See Section4.84. Next predeclare a bounded local-excitation diagnostic with explicit interaction-matched control and contact accounting; no controller change is promoted by this audit.

**E93 completed:** Fixed32-transition history after planned descent yields58/96 successes versus57/96, but collisions increase33 to34. Six-joint success improves22/48 to28/48; seven-joint falls35/48 to30/48. There are11 gains and10 losses, including one lost successful six-joint hybrid case, so the retention gate fails and window32 is not adopted. Five of ten earlier regressions recover. All96 references and18 fallbacks match;117 tests and six video replay checks pass. See Section4.83. Next inspect measured excitation and conditioning before designing a history-selection rule; do not tune window length or choose by DOF on these outcomes.

**E92 completed:** Exact-Jacobian continuation from identical post-descent states recovers all 10 E91 longer-route regressions and preserves all six selected retained-success controls. This is a privileged, outcome-selected diagnostic, not a new learned success rate. Jacobian-error ranges overlap, so no threshold is justified. All 16 learned replays match; 115 tests and four executed video replay checks pass. Next investigate local-history estimation with frozen weights. See Section 4.82. Preserve outcome-selection caveats and evaluate any subsequent nonprivileged repair across all cases.

**E91 completed:** Frozen learned estimators with disclosed privileged route assistance reach 49/96 successes at the original descent budget and 57/96 with the longer budget, versus 43/96 baseline. Route100 has 17 gains/11 losses; route300 has 24 gains/10 losses. Both lose five baseline-successful six-joint cases and fail retention, so neither is promoted. Five of six checkpoints recover difficult case 4 with longer descent. All 96 baselines and 36 fallbacks replay exactly; 114 tests and four video replay checks pass. See Section 4.81. Next isolate post-descent handoff behavior with same-state Jacobian diagnostics and a privileged exact-kinematics continuation; preserve learned/hybrid distinctions.

**E90 completed:** Physical tracking of frozen E89 descent routes improves privileged full-task success from 8/16 to 11/16 under the original descent budget and 13/16 with a 300-step descent allowance. No baseline successes are lost. Difficult six-joint case 4 succeeds with 213 descent steps; cases 5-7 have no clear selected route and remain failures. Learned-policy scores are unchanged. All 16 baselines and six no-route runs match, 112 tests pass, and six executed success/failure video replays are saved. See Section 4.80. Next test frozen learned estimators with the same disclosed planner, and investigate multi-segment routes for the three remaining cases.

**E89 completed:** A bounded static descent-path search finds sampled-clear routes in 13/16 cases. In the difficult six-joint ZYXYZY family, case 4 has one alternative clear route, while cases 5-7 have reachable endpoints but no clear straight joint-space route among the tested candidates. Case 4 requires 105 nominal command increments versus the 100-step descent budget. This is path-geometry evidence, not recovered task success. All 16 E88 replays match and 110 tests pass. Clear/blocked kinematic videos are saved with explicit non-execution labels. See Section 4.79. Next test physical tracking of the discovered route under original and explicitly longer diagnostic budgets.

**E88 completed:** Exact-Jacobian substitution improves the analytical reference from 7/16 to 8/16 successes, with collisions falling from seven to six. Only case 2 recovers; all four difficult ZYXYZY six-joint cases still fail. This privileged diagnostic points to remaining shared path/contact-control limits, not estimator error alone. All 16 baselines replay exactly and 109 tests pass. Four labeled success/failure videos reproduce their saved trajectories exactly. See Section 4.78. Next diagnose descent-path feasibility and alternate arm postures in the difficult six-joint family.

**E87 completed:** Four exact joint-axis sequences excluded from training and checked catalogs were evaluated on eight new procedural bodies and 16 fixed placements. Response correction yields 46/96 learned successes versus 43/96 baseline and 44/96 command correction; collisions are 27, 40 and 35 respectively. Six-joint success is 17/48 versus 15/48 for both controls. No baseline successes are lost, but the two response-specific gains share one body and checkpoint. All 112 independent baselines match; 108 tests pass. Low absolute success and 13.22x geometry-query cost remain major limitations. See Section 4.77. Next run an explicitly privileged exact-Jacobian diagnostic on all fixed cases to separate estimator error from shared path/contact limitations before further tuning.

**E86 completed:** Expanded frozen comparison samples12 new parameterizations, accepts21/24 placements on11 bodies, and yields116/126 learned successes for response correction versus114/126 for baseline and command-only correction. Collisions fall11 to9 with no losses. Two response-specific recoveries occur on different seven-joint bodies/families/checkpoints; six-joint remains53/54. All147 baseline checks and108 tests pass. Small effect and unchanged joint-axis arrangements limit the claim. See Section4.76. Next define arrangement-disjoint development evaluation against training manifests, preserving reserved final-test data and frozen settings; small parameter perturbations do not answer the new-kinematics question.

**E85 completed:** Prospective new-morphology pilot accepts11/12 fixed placements on six parameter-perturbed bodies (same axis families). Response correction reaches59/66 learned successes versus58/66 for both baseline and matched command-only correction, with seven versus eight collisions and no losses. The single gain is seven-joint; six-joint remains28/30. Both response gates pass, but one favorable paired outcome is preliminary mechanism evidence. All77 independent baselines match exactly;108 tests pass. See Section4.75. Next expand frozen paired comparisons across independently sampled parameterizations per family and report body/seed effects; do not strengthen the mechanism claim from one recovery.

**E84 completed:** Frozen sustained direction correction replicates on 17 separate historical placements:95/102 learned successes versus93/102, seven versus nine collisions, and47/48 six-joint successes versus46/48. Two recoveries, zero casewise losses; analytical remains16/17. All119 baselines replay exactly;107 tests pass. Added geometry-query cost is605,010. Same robot bodies and previously used development placements limit the claim. See Section4.74. Next predefine a new-body development evaluation with matched no-response correction ablation and analytical controls, preserving reserved final-test data.

**E83 completed:** Full-task sustained correction yields 71/96 learned successes versus baseline 68/96, with 18 versus 26 collisions. Pulse yields 69/96. Sustained has 3 casewise gains and 0 losses; six-joint success is 37/42. The primary retention gate passes. All 112 baselines reproduce exactly; 107 tests pass. These are historical development tasks, not independent generalization. See Section 4.73. Freeze the sustained controller and replicate it on the separate E73 placement suite, retaining every previously successful six-joint case and reporting all regressions and query costs. Do not retune on this development gain.

**E82 completed:** Bounded response-based direction corrections reduce colliding 20-step local branches from 24/27 to 17/27 (five-step pulse) or 15/27 (sustained), with seven/nine paired avoidances and no new contacts. Both variants avoid three of five six-joint contacts; two successful controls remain clear. All 27 full replays and 20-step matched baselines reproduce exactly; 107 tests pass. These are local avoidance results, not recovered pick-and-place tasks. See Section 4.72. Next run complete-task retention with frozen correction settings, all placements/checkpoints and casewise six-joint preservation; sustained correction is an exploratory primary candidate, with pulse reported secondarily.

**E81 completed:** First-warning five-step slowing/bias pulses provide no retained avoidance over 20 steps: every strategy and matched baseline has 24/27 colliding local branches, including all five six-joint cases. Bias-only initially avoids four contacts, but that gain disappears by step five. Advance warning exists for 14/26 original collisions; two successful controls remain clear. All 27 full replays and selected first-step checks pass. Reject this pulse mechanism; task success is unchanged. See Section 4.71. Next evaluate bounded direction corrections using measured-response predictions and whole-arm clearance, with explicit progress constraints; stop tuning motion-scaling pulses on these outcomes.

**E80 completed:** Across 26 imminent-contact states, half/quarter/bias-only actions avoid 0/1/4 contacts and none of five six-joint contacts; they cause zero contacts in 110 successful controls. Analytical same-history actions avoid 10/26 but introduce 68/110 contacts in successful controls. Existing filtering already selects bias-only in 13 collision states and 108 controls. All 28 full trajectories and 136 selected-action branches match exactly; 680 one-step comparisons are diagnostic, not task recoveries. See Section 4.70. Next test earlier causally available warnings with short multi-step branches and successful controls; report unheralded collisions. Retain the current controller and reject a universal analytical fallback.

**E79 completed:** Geometry attribution separates real near-table motion from prediction error. Of 87 lift/transport warnings in one successful six-joint run, 86 correspond to measured clearance inside 2 mm without penetration. Another successful six-joint run has 23 release warnings for predicted link4/gripper proximity that is absent in sampled actual motion. The missed descent collision is detected by the existing velocity-aware predictors. Three full traces and prediction logs replay exactly. E78 remains a failed gate; actions and success rates are unchanged. See Section 4.69. Next compare bounded actions from identical collision and successful near-margin states, recording avoidance, new contacts and task-direction error before attempting complete-task intervention.

**E78 completed:** All-phase response prediction detects 25/26 learned first contacts versus command-only 15/26, with 288 versus 280 one-step false alarms. Six-joint detection is 5/5; successful learned runs have 110 warnings across 17,409 noncollision steps. The fixed advancement rule fails. All 112 trajectories and prior E76 predictions match exactly; 105 tests pass. No action changes or task recoveries. See Section 4.68. Retain the controller. Before an action-changing trial, diagnose the missed contact pairs and whether warnings reflect imminent contact or conservative geometry checks; do not alter the fixed threshold or relabel this gate.

**E77 completed:** Seven of eight response-predictor one-step false alarms precede contact by 40-80 ms on unchanged trajectories; four response-only warnings minus one command-only warning explain the net increase of three. All four response-only warnings share placement 15. Three of four unscored six-joint contacts involve arm links hitting the table during lift/transport. E76 remains a failed gate; no actions or success rates changed. See Section 4.67. Next evaluate frozen prediction across every pre-contact phase with successful-run controls, whole-arm geometry and unchanged one-step scoring; report warning lead times separately.

**E76 completed:** Frozen no-velocity response prediction detects 13/14 in-scope learned first contacts, versus command-only 6/14, with 8 versus 5 false alarms across 4,556 noncollision steps. The prespecified prediction gate fails. Only 14/26 first contacts fall in the evaluated descent/lowering phases. All 112 separate historical trajectories replay exactly; 105 tests pass. Task outcomes are unchanged. See Section 4.66. Keep the controller unchanged. Next audit false alarms and contacts outside the selected phases before designing an intervention; preserve the fixed threshold and record any revised method as a new development experiment.

No velocity-aware path passes the prediction/false-alarm gate. Keep the original controller and do not insert this predictor into action selection. The no-velocity comparator detects 8/9 learned first contacts with no false alarms at the already fixed 2 mm threshold. Freeze that simpler model as an exploratory candidate and validate its prediction/false-alarm tradeoff on separate procedural trajectories before enabling action changes. Do not choose new thresholds from these ten events; better mean endpoint prediction alone is insufficient for collision avoidance. The original policy and success rates are unchanged by E75.

E74 shows a gap between commanded kinematic paths and executed dynamic trajectories in the remaining E73 finger/table failures. Keep the original controller; do not retune static margins or waiting times. Next compare velocity/response-aware action prediction with a same-history analytical-action comparator, explicitly labeling any simulator-based lookahead as privileged. One-step contact avoidance must not be presented as full-task recovery.

The frozen geometry filter fails the prespecified replication rule. Retain the original controller; no promotion of the filter. Keep the full slowing comparison as evidence and inspect the paired geometry failures before proposing a different mechanism. Do not retune the filter margin or scale list on this validation set. These are development placements on existing robot bodies, and geometry is extra privileged information.

The prespecified development gate passes; freeze the filter for fresh-placement validation, retaining all casewise regressions. Next validate this frozen filter on E71 placements, and subsequently compare against a shared action-scaling control to distinguish geometry-conditioned decisions from simply slowing motion. E72 provides exact robot/scene geometry to the filter; do not describe its results as learned history-only adaptation.

The replication rule fails; retain the original phase budgets. Do not promote extended timing. Stop tuning waiting times on these placements. The next distinct experiment should change the path before contact using a shared clearance-aware controller, with identical geometry access for learned and analytical estimators and explicit disclosure of that extra information. E71 changes placements, homes and probe seeds on existing bodies; do not describe it as new-embodiment validation.

Retain these as development candidates for broader validation: extended. Keep the baseline until that validation passes. E70 isolates timing and transition stopping, not collision-clearance motion. Next investigate a shared geometry-aware path intervention with matched information access and explicit six-joint/task-success retention. Do not interpret fewer collisions from aborting as recovered tasks.

E69 narrows the immediate procedural priority to collision clearance and phase control. Twenty-nine of 31 learned failures contain forbidden contact; every joint collision/contact-loss event has collision first. Preserve E35 and frozen E66 while testing shared path/phase changes, with all negative cases and six-joint retention. Telemetry does not causally isolate estimator error, and these findings do not establish the cause of native manipulation failures.

E68 tests a specific bounded settling intervention, not active regrasp recovery. The prespecified advancement gate fails. Retain the original sequencer and E35; do not expand this settling rule to native evaluation. Keep the E66 candidate experimental. Further grasp work should distinguish relative object/finger slip from estimator-driven pose error and test an explicit recovery action on procedural cases before further native tuning. No stronger cross-embodiment or publication claim follows from this experiment.

E67 does not confirm a general native improvement from E66. Retain E35 and preserve the candidate. On new native placements both score 34/36 tracking and 28/36 manipulation, with three candidate gains and three losses including a six-joint regression. Procedural reaching improves only 13/3072, with one additional collision and two fewer six-joint successes. All candidate native manipulation failures involve bilateral grasp-contact criteria; next investigate procedural grasp stability/recovery using frozen estimators and shared interventions, without fitting to these native cases. Keep all seeds, failures, contact/interaction costs and the reserved final test.

- Joint count identifies the training axis family in E04–E14. E15 adds three arrangements per count and common joint resets; its training-control gate must be resolved before interpreting transfer.
- Earlier validation uses very few bodies, including only one novel-axis body in E14. E18 expands this to six familiar-family and eight novel-family validation bodies, but one model seed and two bodies per novel family still limit generality. These are development diagnostics, not confirmatory tests.
- Earlier suites use expert-filtered reaching goals; E33 uses structurally valid endpoints without controller-success filtering. E38 adds measured position/direction control for physical manipulation, with scripted sequencing and IK-screened approach poses. It does not test learned grasp decisions, arbitrary-start planning, vision, hardware or legged locomotion.
- Successful demonstrations omit failed episodes from imitation. Transition weighting favors longer episodes; broader state/action excitation and balanced sampling remain untested changes.
- Frozen calibration is not continuous adaptation. The analytical baseline updates online, so differences in interaction use must be controlled in later comparisons.
- Section 5.1 records an initial related-work screen; a comprehensive novelty audit, compatible method comparisons and venue selection remain unfinished. Publication readiness is not established.

E35 gives a modest higher-noise development advantage over fixed analytical identification, while delay-aware analytical control remains stronger and short-memory neural performance is seed-dependent. E38 demonstrates physical pick-and-place with shared frozen motion estimation, but not consistent analytical-baseline superiority or broad manipulation transfer. Expand feasible placements and improve contact reliability before extending the manipulation claim; separately replicate the noise result against stronger controls. Physical excitation restriction is not yet tested independently. Preserve difficult families, screening failures, earlier negative results, calibration costs, and the reserved final test.

### 5.1 Publication readiness and initial related-work screen (22 October 2026)

**Assessment.** This is a working research prototype with substantial reproducibility infrastructure, not yet a defensible full-conference result. Procedural reaching is strong, but native manipulation is seed-dependent, and analytical control remains a strong competitor. The central missing result is a clearly motivated, repeatable advantage of learned adaptation under fair information and interaction budgets, followed by independent validation. This is more than a writing or presentation gap. A narrower workshop study may be possible sooner if it establishes a useful, well-supported insight, but neither workshop nor conference acceptance can be predicted from the present record. These are research judgments, not venue-specific acceptance rules or a calendar estimate.

**Initial primary-source screen, not a complete novelty review.** The following works make broad claims such as “a neural Jacobian controller” or “one model across embodiments” insufficient on their own:

| Work | Relevant overlap and distinction to investigate |
|---|---|
| [Analyzing Neural Jacobian Methods in Applications of Visual Servoing and Kinematic Control](https://arxiv.org/abs/2106.06083) | Studies neural Jacobian approximations, conditioning and Cartesian control, including a physical seven-DoF arm. Learned Jacobian prediction itself is established prior work. |
| [XMoP](https://arxiv.org/html/2409.15585v2) | Reports frozen cross-embodiment planning across six-/seven-DoF manipulators and seven commercial designs, using robot physical descriptions/URDFs. Its planning task and available geometry differ from our online measured-history estimator. |
| [Controlling diverse robots by inferring Jacobian fields with deep networks](https://www.nature.com/articles/s41586-025-09170-0) | Demonstrates learned visuomotor Jacobian fields across diverse physical robots. It uses robot-specific random-command video data for training; our potential distinction is short-history adaptation with frozen weights, while our current sensing is ideal simulated state. |
| [Octo](https://octo-models.github.io/) | Demonstrates a generalist visuomotor policy and adaptation to new observation/action spaces, evaluated on nine real robot setups. It is broader manipulation context, not a directly interchangeable baseline for our state-based identification task. |
| [AdaTracker, April 2026 preprint](https://arxiv.org/abs/2604.20305) | Infers embodiment constraints from history for cross-embodiment active visual tracking. History-conditioned unseen-embodiment adaptation is also prior art; task differences and information assumptions require a full comparison. |

Classical identification must also be represented: [uncalibrated eye-in-hand visual servoing](https://journals.sagepub.com/doi/10.1177/027836490302210002) already estimates Jacobians without known robot kinematics. Our analytical reference is therefore central, not merely a weak sanity check. These sources motivate further investigation; they do not establish that our precise method is or is not novel. No literature benchmark was implemented this turn.

**Candidate contribution, still unproven.** Fast online motion identification across unseen kinematics using a shared frozen model and a small measured interaction budget. The online estimator does not receive a simulator Jacobian, but the overall experiment still uses robot assets, joint limits, tool frames, ideal pose observations and geometry-based offline feasibility screening. Do not claim that the entire setup is geometry-free. The current method is a learned motion estimator coupled to a scripted controller; it is not an end-to-end visual or flow-matching foundation policy.

| Milestone | Present evidence | Required next evidence |
|---|---|---|
| Clear technical contribution | Shared measured-history estimator and careful negative ablations | Establish a distinct method or validated mechanism relative to neural identification and adaptive control literature. |
| Learned benefit | Strong procedural success; some prior noise advantages, incomplete retention | Prespecify the regime of benefit; compare analytical, delay-aware and appropriate classical/learned alternatives with equal sensing and interaction costs. Report effect sizes and uncertainty, not selected seeds. |
| Reliability | Six-/seven-joint native simulation works on some tasks; unstable Panda seeds | A common configuration that improves the intended regime without hiding six-joint regressions; multiple seeds and enough tasks to quantify reliability. |
| Independent transfer | Current native models/placements are repeatedly observed development cases | Freeze the method and protocol, then evaluate genuinely untouched kinematic designs and placements; report adaptation curves and held-out failures. More layouts on the same observed robots do not establish new-embodiment transfer. |
| Claim-aligned scope | Reaching and scripted pick-and-place with ideal state | Prefer a narrow arm-control claim. Hardware would strengthen practical claims; a rigorous simulation study is possible, but current evidence cannot support hardware or visual generality. Quadrupeds are not necessary for this paper. |
| Reproducibility | Versioned data, hashes, exact replays, preserved negatives, tests | Consolidate a reproducible benchmark and final evaluation once the contribution and method are fixed. |

There is no evidence-based completion percentage or submission date yet. The useful checkpoint is whether a stable method demonstrates a repeatable benefit on a frozen development protocol; independent evaluation comes after that. A paper can be outlined now, but completing prose does not substitute for the missing result.

## 6. Reproducibility and reporting practice

From the repository root:

```powershell
python -m unittest discover -s tests -p 'test_ease_fm*.py' -v
python -m examples.ease_fm.auxiliary_experiment --data outputs/ease_mixed_v1 --output outputs/ease_mixed_aux_new --steps 10000 --device cuda
```

Output directories must be new; the runner refuses to overwrite existing experiments. Protocols are written before training. Training metadata includes dataset and checkpoint hashes, source hashes, optimizer-step budget, seed, runtime, and separate flow/response loss traces. The original five policy tensors remain unchanged; auxiliary targets are stored separately by the loader. Regression checks cover mixed-joint masking, causal histories, episode separation, physical fixtures, response-head padding invariance, and gradient flow into the history encoder.

For every subsequent experiment, append a registry entry with hypothesis, dataset hash, fixed settings, comparison conditions, complete results, interpretation, and unresolved confounds. Keep raw results and source snapshots immutable. Update this report's revision and evidence-based conclusions; do not remove negative results. The report is the cumulative research record from which a paper can later be drafted.

## Revision history

- Revision 99, 14 November 2026: E113 release extension: original selector 14/24 (10 collisions), release extension 16/24 (8 collisions); gate True. Added60-second progress demo through E113.
- Revision 98, 13 November 2026: E112 prospective unseen-body replication: baseline 28/48 (13 collisions), physics 31/48 (10 collisions); 3 gains, 0 losses; gate True.
- Revision 97, 13 November 2026: E111 tests frozen broad progress-aware physics selection in72 paired tasks. baseline 22/36 (14 collisions; 0 contact-free failures), physics 31/36 (3 collisions; 2 contact-free failures); gate True. Saves phase progress, failures and 24 verified videos.
- Revision 96, 13 November 2026: E110 tests broad goal-directed actions on copied training states; warning-state availability 1/14 narrow versus 13/14 broad. Preserves margins, hold-horizon qualification and no policy-success claim.
- Revision 95, 13 November 2026: E109 tests earlier-phase physics selection in72 paired tasks. baseline 22/36 (14 collisions; 0 contact-free failures), physics 22/36 (14 collisions; 0 contact-free failures); gate False. Saves phase progress, failures and 12 verified videos.
- Revision 94, 13 November 2026: E108 completes72 paired training-body tasks. baseline 22/36 (14 collisions), physics 22/36 (14 collisions); gate False. Saves all alternatives and 4 verified videos; no transfer test.
- Revision 93, 13 November 2026: E107 completes 36 unchanged training-body replays. static: TP 2, FP 96, FN 0, TN 1493; physics: TP 2, FP 0, FN 0, TN 1589. Saves all prediction traces, latency, error cases and verified reused baseline videos. No controller intervention.
- Revision 92, 13 November 2026: E106 observes six unchanged E105 replays at2ms, diagnoses static-filter/dynamic-contact mismatch, preserves all failures and six verified original videos. No control change.
- Revision 91, 6 November 2026: E105 prospectively replicates frozen retreat on fresh bodies/placements. baseline 27/45 (17 collisions), budget 27/45 (17 collisions), gentle 27/45 (17 collisions); gate False. Saves all outcomes, coverage, clustered intervals and 9 verified videos.
- Revision 90, 6 November 2026: E104 evaluates all four isolated phases on training bodies and conditionally tests passing phases on novel-axis development tasks. retreat: baseline 32/48 (16 collisions), budget 32/48 (16 collisions), gentle 35/48 (13 collisions); transfer gate PASS. Saves all outcomes, 131 tests and 30 verified videos. No final test.
- Revision 89, 5 November 2026: E103 completes108 training-arm tasks with unchanged/budget/gentle controls. baseline 22/36 successes, 14 collisions; budget 23/36 successes, 13 collisions; gentle 22/36 successes, 12 collisions. Strict gate fails; no transfer evaluation. Records131 tests and 9 verified new videos.
- Revision 88, 5 November 2026: E102 saves55020 substep contact samples in ten exact replays, separates transport/lowering/retreat failure patterns, and corrects the overly broad E101 phase localization. Ten verified new videos and129 tests pass. No new policy or benchmark score.
- Revision 87, 5 November 2026: E101 compares fixed student histories with matched parent replay. parent 29/48 successes, 18 collisions; start 32/48 successes, 16 collisions; replay 33/48 successes, 13 collisions; onpolicy 30/48 successes, 15 collisions. replay FAIL; onpolicy FAIL. Saves144 new trials, six checkpoints and 15 verified videos;129 tests pass.
- Revision 86, 5 November 2026: E100 audits three regressions and one unchanged-success control with eight exact replays. Records same-state raw/filtered commands, zero filtered disagreement before the case13 transport collision, scientific figures and eight verified videos. No policy adoption; next test on-policy histories on training bodies.
- Revision 85, 5 November 2026: E99 saves 36 training-arm trajectories, six paired checkpoints, 144 full-task evaluations and 13 verified new videos. Action retention reaches 32/48 versus parent 29/48 but loses three seven-joint successes; both replacement gates fail. Records all negative outcomes and the matched-state command-audit next decision.
- Revision 84, 5 November 2026: E98 generates16384 measured training-only contexts and fits three fixed2000-update Transformer checkpoints. Full-task neural continuation improves29/48 to31/48, but seed2 loses four seven-joint successes despite improved prediction metrics. Retains all checkpoints and failures; gate FAIL. Saves96 rollouts,127 passing tests, scientific figures and eight verified new videos.

- Revision 83, 5 November 2026: E97 completes288 matched continuation tasks. Analytical improves57/96 to68/96 with15 gains/four seven-joint losses; exact solves78/78 route-assisted cases. Preserves18 unchanged no-route failures,96 references,36 fallbacks,127 passing tests and12 verified new videos. No learned-policy promotion.

- Revision 82, 5 November 2026: E96 audits all288 E95 rollouts unchanged. Documents analytical-versus-learned estimation gap on measured probe histories, identical-state safeguards, 125 passing tests, scientific figures and ten hash-verified reused videos. Instantaneous accuracy does not establish task recovery; next test paired post-probe continuation.

- Revision 81, 2 November 2026: E95 completes 288 local-probe/control rollouts. Directional coverage improves but task success does not; retains all negative outcomes, paired video evidence, 96 reference replays, 36 fallbacks and 123 passing tests. Rejects adoption and motivates a same-state estimation audit.

- Revision 80, 2 November 2026: E94 audits directional information in measured histories across96 handoffs and46 unchanged diagnostic replays. Documents dominant one-direction recent motion, weak older independent directions, scientific figures,119 passing tests and reused verified video evidence. No new policy or success claim.

- Revision 79, 2 November 2026: E93 tests fixed recent32 history across all96 checkpoint/task pairs. Records opposing DOF effects, 11 gains/10 losses, increased collisions and failed retention; rejects adoption. Saves192 rollouts,96 exact references,18 fallbacks,117 tests and six verified videos including a disclosed newly introduced regression.

- Revision 78, 2 November 2026: E92 diagnoses all ten longer-route regressions with same-state exact-Jacobian continuation, recovering all and retaining six controls. Records overlapping handoff Jacobian errors, 32 full tasks, 16 exact learned replays, 115 passing tests and four verified executed videos. Learned benchmark scores remain unchanged; next investigate frozen local-history estimation.

- Revision 77, 1 November 2026: E91 evaluates all six frozen estimators with route assistance and both descent budgets. Records aggregate gains but failed casewise six-joint retention, 288 saved trials, 96 exact baselines, 36 fallback replays, 114 tests and four trajectory-verified success/failure videos. No hybrid promotion; next diagnose post-descent handoff.

- Revision 76, 1 November 2026: E90 physically executes frozen sampled-clear descent routes. Reports 11/16 privileged successes with matched phase budgets and 13/16 with extended descent versus 8/16 baseline, zero losses, 213-step difficult-case recovery, 48 saved tasks, 112 tests and six verified executed videos. Keeps learned-policy results unchanged.

- Revision 75, 1 November 2026: E89 records static descent-path search across all 16 cases, one alternative route for the difficult six-joint family, remaining blocked candidate paths, large posture-change cost, 16 exact replays, 110 passing tests and two verified nonphysical path animations. No new manipulation-success claim.

- Revision 74, 1 November 2026: E88 exact-Jacobian intervention recovers one case, leaving most failures and every difficult-family six-joint failure unresolved. Records 32 tasks, 16 exact baseline replays, 109 passing tests, four trajectory-verified video demos, and the ongoing convention to save success and failure videos.

- Revision 73, 1 November 2026: E87 completes prospective axis-disjoint development evaluation on four arrangements and eight bodies. Records 46/96 response successes versus 43/96 baseline and 44/96 command, three baseline gains without losses, two response-specific six-joint gains concentrated in one body/checkpoint, phase failure diagnostics and 13.22x geometry-query cost. All 112 baseline replays and 108 tests pass. Preserves low absolute success and synthetic-training lineage caveats.

- Revision 72, 1 November 2026: E86 expands to two new parameterizations per existing family. Records 21/24 screening acceptance, two response-specific recoveries on different seven-joint bodies/checkpoints, no losses, unchanged six-joint outcomes and query cost. All147 baseline checks and108 tests pass. Novel-axis development evaluation remains pending.

- Revision 71, 1 November 2026: E85 prospectively samples six new parameterizations and evaluates 11/12 accepted placements with response and command-only correction. Records one response-specific seven-joint recovery, no regressions, unchanged six-joint outcomes, screening rejection and query cost. All77 baselines independently match;108 tests pass. Same axes, no reserved-test or broad-generalization claim.

- Revision 70, 1 November 2026: E84 replicates frozen sustained correction on separate E73 placements with two task recoveries and no regressions. Records complete-task and six-joint retention, analytical comparisons and query cost; all119 baselines exact and107 tests pass. New-body validation and matched response ablation remain pending.

- Revision 69, 1 November 2026: E83 evaluates 336 complete tasks with frozen correction settings, original sequencing and success rules. Reports paired gains/losses, six-joint casewise retention, analytical comparisons and query costs. All 112 baseline traces and filter logs reproduce exactly; 107 tests pass. No independent generalization or reserved-test claim.

- Revision 68, 1 November 2026: E82 implements bounded coordinate direction corrections with measured-response clearance and learned-Jacobian progress constraints. Records 81 branches, seven/nine local paired avoidances, six-joint and successful controls, query costs, exact replay/baseline checks and 107 passing tests. No complete-task recovery or independent generalization claim.

- Revision 67, 29 October 2026: E81 tests first-warning five-step interventions followed by baseline recovery over 20 phase-held steps. Records warning coverage, 108 branches and 2,160 simulated steps, transient versus retained avoidance, successful controls and exact replay checks. Rejects slowing-pulse mechanism; no task recovery or neural training.

- Revision 66, 28 October 2026: E80 compares five bounded actions at 26 first-contact states and 110 successful controls. Records 680 branches, exact full replay and selected-branch checks, six-joint limitations and analytical fallback regressions. No full-task recovery or policy change.

- Revision 65, 28 October 2026: E79 attributes all 110 successful-run warnings and the one missed collision using three exact replays, frozen prediction paths and measured substep geometry. Distinguishes near-margin motion from unobserved predicted proximity; preserves E78 failure and unchanged task outcomes.

- Revision 64, 28 October 2026: E78 extends frozen prediction to every pre-contact task phase with exact trajectory and prior-prediction checks, phase-specific and six-joint metrics, successful-run controls and descriptive warning timings. All 105 tests pass; no intervention, training or final-test evaluation.

- Revision 63, 28 October 2026: E77 audits warning timing and contact identities on all 96 learned E76 trajectories. Preserves one-step false-alarm labels and failed E76 gate; identifies arm-link/table coverage gap. No new simulation, training or final-test evaluation.

- Revision 62, 28 October 2026: E76 validates the unchanged E75 predictor on 112 E72 trajectories; records fixed-threshold false alarms, scoped recall and all first-contact phase coverage. Exact traces, probes, update counts and hashes verified; 105 tests pass. No action changes or final-test evaluation.

- Revision 61, 28 October 2026: Completed E75 causal response-prediction audit on 119 exactly reproduced E73 trajectories, including successful trajectories. Recorded velocity/no-velocity endpoint errors and collision recall/false alarms at fixed thresholds, verified prior-update counts and unchanged hashes; inspected figure; 105 tests pass. No action changes, task-recovery claims, native tuning or final-test evaluation.

- Revision 60, 28 October 2026: Completed E74 on every E73 geometry-filter failure: ten exact full replays and fifty matched one-step branches. Audited dense kinematic versus actual dynamic-path clearances, same-history analytical action and incoming-velocity interventions, unchanged hashes and branch-state restoration. Inspected figure; 104 tests pass. No training, task-recovery claim, native tuning or final-test evaluation.

- Revision 59, 28 October 2026: Completed E73 with 357 trials on all 17 E71 placements, frozen geometry filter, original controller and fixed half-motion comparator. Verified 119 exact baseline replays, unchanged code/model/input hashes, physical margins and blocked control-condition geometry queries. Audited paired regressions, joint-count retention and costs; inspected figure; 103 tests pass. No training, native tuning or final-test evaluation.

- Revision 58, 28 October 2026: Completed E72 geometry-filter v2, 224 paired trials with unchanged physics, original phase budgets and frozen estimators. Preserved interrupted v1 after exact no-op action correction. Audited 112 exact baseline replays, source/input weights, filter work and unchanged physical margins; 102 tests pass. Inspected comparison figure. No training, native tuning or final-test evaluation; explicitly disclosed privileged geometry access.

- Revision 57, 27 October 2026: Completed E71 prospective placement validation with frozen E70 timing and checkpoints. Retained all screening rejections, paired outcomes, joint-count/family breakdowns and costs. Verified source/model/input hashes, trial coverage and trace lengths; inspected figure; 99 tests pass. No training, native tuning or final-test evaluation.

- Revision 56, 27 October 2026: Completed E70, 336 frozen-estimator trials comparing original timing, extended phase budgets and matched gated transitions. Recorded paired gains/losses, per-checkpoint six-joint retention, aborts, collisions and costs. Verified 112 exact baseline replays, 112 exact gated prefixes, unchanged hashes and 99 passing tests; inspected comparison figure. No training, native or reserved final-test evaluation.

- Revision 55, 27 October 2026: Completed E69 with 112 exact baseline telemetry replays, two exact terminal audits, unchanged input/model hashes and 97 passing tests. Recorded contact forces, relative object motion, motion residuals and collision chronology; inspected noncollision-failure figure. Prioritized collision clearance over regrasp; no training, native tuning or final-test evaluation.

- Revision 54, 27 October 2026: Completed E68, 336 matched procedural trials across seven frozen estimators and three settling conditions. Recorded all paired gains/losses, joint-count retention, collisions and interaction costs; verified 64 exact historical baseline traces and unchanged model/input hashes. Inspected comparison figure; 95 tests pass. Reserved final test untouched.

- Revision 53, 27 October 2026: Completed E67 with 16 new axis arrangements, 64 bodies, 7,168 procedural trials and 168 native trials on twelve new placements. E66 gains 13/3072 procedural successes with six-joint/collision regressions; native totals tie E35 at 34/36 tracking and 28/36 manipulation. Recorded all paired changes, family/physical breakdowns, unchanged hashes, inspected figure and two exact rendered replays; 92 tests pass. No promotion, training or reserved final-test evaluation.

- Revision 52, 27 October 2026: Completed E66 six matched 2,000-update estimator-target runs, all six reaching gates at 256/256, and 144 native trials. Analytical-target Transformer achieves 36/36 tracking and 32/36 manipulation versus E35 33/36 and 29/36, retaining all PiPER successes but recording one Panda manipulation regression and seed-concentrated gains. Verified target/batch/checkpoint audits, inspected the figure and two exact rendered replays, and passed 92 tests. Preserve E35 and freeze candidates for broader validation; no final test.

- Revision 51, 27 October 2026: Completed E65 with 240 complete schedules and 408,000 control steps. All 120 original-condition traces replay exactly. Bias removal changes analytical 12 to 8/12, E35 33 to 25/36, uncorrected 32 to 22/36 and corrected 28 to 24/36; five learned gains and 27 losses. Saved paired phase analysis and inspected figure, verified unchanged weights, and passed 90 EASE-FM tests. Keep E35/bias; no promotion or final-test evaluation.

- Revision 50, 24 October 2026: Completed E64 with 25,500 exact original replay steps, 37 exact baseline branches and 148 audited interventions. Analytical motion recovers 24/37 failed holds; zero bias 15/37; both 20/37. Both PiPER failures recover in every intervention. Retained failure-conditioning and inherited-state limits; no promotion or final test. EASE-FM 90 tests pass; broader suite collection blocked by missing openpi.

- Revision 49, 23 October 2026: Completed E63 120 common full schedules and 204,000 control steps from shared calibrated states. Analytical 12/12, E35 33/36, uncorrected 32/36, corrected 28/36. Of 37 missed phase holds, 30 never acquire and seven lose an acquired hold. Preserved initial repeated-calibration failure and excluded partial v1 results. Audited traces/hashes, inspected phase figure, and passed 86 tests. Keep E35; no training or final-test evaluation.

- Revision 48, 23 October 2026: Completed E62 with 120 exact original, 120 intact, 120 cube-disabled and 120 all-geometry-contact-disabled executions. Of 38 failed originals, 17 track unconstrained endpoints and 21 miss; one original success also misses without contacts. Preserved cube-only PiPER contact artifact, truncated-prefix limits and initial numerical replay failure. Audited records/hashes and unchanged weights; all 84 tests pass. Keep E35; no final-test evaluation.

- Revision 47, 23 October 2026: Completed E61 six matched 2,000-update runs, 1,536 successful fresh reaching gates and 72 fresh native trials. Corrected 19/36 versus uncorrected 22/36 and original 29/36; both candidates have seven contact failures. Seven paired gains/ten losses, no retention candidate. Audited frozen response conversion, hashes and calibration; 84 tests pass. Keep E35 and final test reserved.

- Revision 46, 23 October 2026: Completed E60: 2,880 exact counterfactual branch replays and 576 independent response checks. True-J one-step ranking improves to 88.6%/88.5% for six/seven joints on held-out response bodies; estimated-J composition remains weak and five-step effects mixed. All 83 tests pass, weights unchanged. No policy training, native data or promotion.

- Revision 45, 23 October 2026: Completed E59 with 32 exact replays, 4,800 transitions, 31,200 joint rows and eight whole-body response holdouts. Full measured-feature ridge reduces body-macro RMSE by 79.2%/83.8% for six/seven joints; all eight bodies and 32 body/driver strata improve. Independently reproduced 16 fits and saved metrics; 83 tests pass. No policy integration, native data or final-test evaluation.

- Revision 44, 23 October 2026: Completed E58: 32 exact driver replays, 288 fixed states, 2,880 branches and exact first-step repeats. Realized-motion oracle raises one-step score agreement to approximately 99.4%; actuator mismatch exceeds geometric nonlinearity. Local-action wins most short procedural comparisons despite native regression; no contact or native causal claim. Preserved precision-check failure and corrected run; all 81 tests pass. No training or promotion.

- Revision 43, 22 October 2026: Completed E57: 12,000 coherent procedural states, six matched 2,000-update runs, 1,536 successful fresh reaching gates and 72 fresh native trials. Prediction-only 25/36 and local-action 18/36 versus original 29/36; one/five candidate contact failures. Neither passes retention. Audited paired batches, hashes, dataset replay and calibration; all 79 tests pass. Keep E35 and final test reserved.

- Revision 42, 22 October 2026: Completed E56 six matched 2,000-update training runs. All local-action reaching gates pass 256/256; native local 25/36 versus prediction-only 26/36 and original 29/36, with five local contact failures. No promotion. Exactly reproduced prediction-only checkpoints/batches/losses; explicitly reused 768 gate episodes and 36 native trials, adding 768 fresh gate episodes and 36 native trials. All 77 tests pass; synthetic improvement does not establish closed-loop benefit.

- Revision 41, 22 October 2026: Completed E55: 80 exact procedural pose drivers, 720 matched first steps and 7,200 branches. Local executed-command one/five-step agreement is 80.4%/64.0% for six joints and 78.6%/66.9% for seven, exceeding operator rankings in every driver group. Recorded saturation, tracking errors and contact-free limits; no training or promotion. All weights unchanged and 75 tests pass.

- Revision 40, 22 October 2026: Completed E54: 80 exact procedural driver replays, 720 matched first steps and 7,200 moving-target counterfactual branches. Local executed-command one/five-step agreement is 81.2%/75.7% for six joints and 81.0%/72.5% for seven, exceeding operator rankings across every driver group. Recorded position-only/contact-free limits; no training or promotion. All weights unchanged and 73 tests pass.

- Revision 39, 22 October 2026: Completed E53: twelve exact analytical driver replays, 202 states, 2,020 counterfactual branches and 202 exact repeat checks. Operator versus executed-command one-step ranking agreement is 45.0%/73.4% on PiPER and 47.3%/74.0% on Panda. Longer-horizon phase limits and absence of contact failures recorded; no causal collision explanation or model promotion. All weights unchanged and 71 tests pass.

- Revision 38, 22 October 2026: Completed E52: six matched 2,000-update training runs, 1,536 training-gate episodes and 72 native trials. Control-aware lowers synthetic response loss but scores 3/36 native successes with 28 contact failures, versus matched prediction-only 26/36 and original 29/36. Neither passes retention; no promotion. Paired batches, original probes and unchanged parents audited; all 68 tests pass. Added response-strength diagnostic and objective-validity next gate.

- Revision 37, 22 October 2026: Completed E50–E51: 48 exact contact replays and 120 full angular-control trials. Genuine grasp loss includes aligned starts; neither orientation intervention passes retention. Verified identical calibration, unchanged weights and 66 passing tests. Retained original controller and specified a matched control-aware learning comparison as the next, not-yet-run experiment.

- Revision 36, 22 October 2026: Completed E49 with 192 exact early-descent prefix replays and action decomposition; no new full-episode evaluation. Panda seed-2 failures are not explained by first-step clipping or larger mean command jumps. All 64 tests pass. Added initial primary-source related-work screen and explicit publication-readiness milestones; no novelty, reliability improvement, promotion or acceptance claim.

- Revision 35, 22 October 2026: Completed E48's four-condition motion/bias decomposition with 192 trials, 96 exact E47 replays and 48 matched groups. Panda both/motion-only/bias-only/neither neural scores are 4/1/2, 3/2/5, 4/2/5 and 3/1/4 of six; isolated updates each regress a PiPER seed. Matched probe residuals improve despite inconsistent task reliability. Verified selective state updates, hashes, all interaction costs and 64 passing tests. No promotion or final-test evaluation.

- Revision 34, 21 October 2026: Completed E47 with 240 matched trials, 60 exact baseline replays and identical paired probe trajectories. Original neural Panda 5/1/5 changes to probe 4/1/2 of six; PiPER remains 6/6 per seed. Waiting and frozen-update controls expose mixed effects; analytical probe passes all twelve tasks. No promotion. Verified unchanged weights, all interaction costs, 63 passing tests and an exact rendered 277-step failure replay. No final-test evaluation.

- Revision 33, 20 October 2026: Completed E45 anchored synthetic training, all three original training-control gates at 256/256, and observed native gates showing incomplete retention. E46 prespecified twelve fresh placements; PiPER old/new scores are 6/6/6 versus 6/6/5 of six, Panda 5/1/5 versus 6/3/5. Both references solve all fresh tasks. Candidate not promoted. Audited 1,038 records, hashes and paired calibration, inspected an exact 424-step recovered replay, and passed 61 tests. No final-test evaluation.

- Revision 32, 20 October 2026: Completed E43's exact prediction-diagnostic replay and mixed rescaling/smoothing ablations; E44 adds 2,000 synthetic training updates per seed and native retention checks. Panda manipulation improves 7/2/4 to 7/5/6, while PiPER changes 9/7/8 to 8/9/6; reaching is PiPER 10/11/11 and Panda 12 each. Preserved regressions, old checkpoints and native-only retention limits. Audited 264 rollout records, hashes, synthetic validation, an exact 377-step rendered recovery and 61 passing tests. No final-test evaluation.

- Revision 31, 20 October 2026: Completed E42's 225 Panda trials with exact baseline replay and added grasp telemetry. Object-height release improves neural 4/0/2 to 7/2/4 of nine, analytical 3 to 8 and privileged 3 to 9; matched pre-lowering traces isolate the intervention. Retained yaw/hold/settle negatives and remaining early grasp failures. Verified hashes, a 445-step exact rendered recovered trial and 61 passing tests. No model training, native-physics modification or final-test evaluation.

- Revision 30, 20 October 2026: Completed E41 with 270 native manipulation trials, retaining the failed bounded-error pilot. Ramped reference/phase gates give expanded PiPER neural 9/8/8 of nine; object-offset placement changes this to 9/7/8 and Panda from 1/0/1 to 4/0/2. Recorded the PiPER regression, persistent Panda weakness and exact matched pre-transport audit. Verified hashes, two exact rendered successes, inspected comparison plot and 59 passing tests. No new training or reserved final-test evaluation.

- Revision 29, 20 October 2026: Added pinned native PiPER/Panda models and adapters, 120 reaching trials and 45 manipulation trials across three preserved scene revisions. Frozen neural PiPER reaching is 9/9/11 of 12; Panda 12 each. Compact PiPER manipulation is 1/2/3 of three versus analytical 3; Panda remains unreliable. Verified asset/checkpoint hashes, exact successful rendered replays and 58 passing tests. No new training, hardware or final-test claim.

- Revision 28, 20 October 2026: Completed E37–E38 physical arm-driven manipulation pilots, corrected gripper mounting and matched clearance/calibration revisions, and reused frozen estimators for position plus tool-direction motion. Corrected familiar neural successes are 7/7/9 of 12 versus analytical 8; unseen 3/2/3 of four versus analytical 3. All models solve the one retained unseen six-joint case; eight other candidates fail endpoint IK and remain recorded. Preserved initial contact failures, audited 80 corrected/reference trials, verified exact rendered six-joint replay and 56 passing tests. No final-test evaluation or end-to-end learned manipulation claim.

- Revision 27, 20 October 2026: Completed E35 three-seed noisy/short-history fine tuning, 29,696 new robustness rollouts, matched delay-aware controls and 768 successful training checks. Recorded the modest higher-noise advantage, inconsistent short-memory gains and analytical delay advantage. Added E36 physical pick-and-place fixtures (3/3 scripted Cartesian trials), 64 arm scene assets and 128-pose rank audit; no learned manipulation claim. Inspected plots/rendered stages, verified hashes, and passed 53 tests. Final test remains reserved.

- Revision 26, 19 October 2026: Completed E34's 28,672 fixed-checkpoint robustness rollouts across seven conditions and four controllers. Recorded lower neural final success in every condition, the higher-noise interaction-cost tradeoff, delay timeouts and short-history contact failures. Clean replay matches E33 exactly; checkpoint hashes unchanged, figure inspected and 50 tests pass. Final test remains reserved.

- Revision 25, 19 October 2026: Completed E33 with frozen checkpoints on 16 novel axis arrangements, 64 bodies and 1,024 tasks without policy-success filtering. Neural successes are 985/1000/995 versus analytical 1003; six-joint successes are 477/489/486 versus 491 out of 512. Recorded interaction costs, shared contact failures, preserved generation failure, full structural/hash audit, inspected figure and 48 passing tests. No final-test evaluation.

- Revision 24, 19 October 2026: Completed the E30 motion-Transformer redesign, E31 three-model/three-probe replication, and E32 cold-start attribution. Calibrated neural estimation solves 3,452/3,456 validation cases without collisions; cold start solves 1,152/1,152 after deduplication, including every unseen six-joint case. Retained all failures, untrained/constant/twin/hybrid controls, and the equally successful fixed-probe analytical baseline. Added the reusable controller, exact 2,304-case inverse-guarded replay, independent 384/384 API check, inspected figure, and 46 passing tests. No final-test policy evaluation.

- Revision 23, 19 October 2026: Completed E28 observation-only kinematic control and E29 isolated-update ablations. Full online identification solves all 1,152 validation cases without contact, including 288/288 unseen six-joint cases; frozen identification solves 287/288 unseen six-joint cases. Recorded analytical-versus-neural distinction, tracking-bias mechanism, interaction costs, imperfect initial Jacobian estimates, exact 1,408-case replay with privileged Jacobian access blocked, 8,448 audited rollout records, inspected figure, and 43 passing tests. Final test untouched.

- Revision 22, 19 October 2026: Completed E27's eight additional fixed-data policy runs and three-seed aggregation. Recovery-trained correct/empty history pass all training gates; reactive passes two of three. Correct-history validation is 312/312/313, reactive 311/311/275, empty 310/320/300 out of 384. Recorded the lack of a consistent unseen-axis calibration advantage, single-use versus reused probe costs, persistent six-joint failures, all 12 checkpoint hashes and 15,360 rollout records, inspected efficiency curves, and 40 passing tests. No final-test evaluation.

- Revision 21, 18 October 2026: Completed E26's audited crossed axis/posture dataset and four directly supervised policy runs. A fixed 3,211-chunk recovery union improves reactive training from 154/256 to 246/256, though one body still fails its gate. Recovery validation totals are history 312/384, reactive 311/384, and empty history 310/384; the latter passes training at 256/256. Recorded no consistent final-success advantage, persistent six-joint transfer failures, verified hashes/provenance, and an inspected factorial plot. Preserved the pre-training logging failure and fresh corrected run. All 38 tests pass; no final-test rollout.

- Revision 20, 18 October 2026: Completed E25's 4,608 frozen diagnostic rollouts, exact intact replays, and unchanged checkpoint audit. Privileged six-joint success is 6/48, 12/48, 2/48; seven-joint success 48/48 for all seeds. Found six-joint reset posture outside training demonstration coverage. Action shuffling hurts 8/9 conditions; whole-transition shuffling has small aggregate effects. Reactive initial-action RMSE is near the deterministic imitation lower bound. All 37 tests pass; no final-test policy evaluation.

- Revision 19, 18 October 2026: Completed E24 seeds 0/1/2, three calibration seeds, and directly supervised reactive baselines at 15,000 updates. History beats controls in all nine aggregate seed/probe cases; teacher/history training is 256/256 in every seed, while pose/reactive gates fail. Verified 12 checkpoints, frozen controllers, 5,472 validation records, and pose/twin invariance; inspected the joint-count comparison plot. Recorded six-joint weakness and dependent-probe limits. All 37 tests pass; final test remains reserved.

- Revision 18, 18 October 2026: Completed E23 paired dataset and gated single-seed pilot. All 256 expert replays and 68 paired calibration audits pass. Teacher/history training succeeds 256/256; validation correct history 62/96 versus pose 47/96, empty 13/96, twin 29/96. Verified frozen controllers, hashes, and exact pose correct/twin invariance. Recorded failed pose gate, six-joint weakness, latent-objective ambiguity, and early-error missingness. No test rollout. All 36 tests pass.

- Revision 17, 18 October 2026: Completed and verified all three E21 seeds, 2,352 paired validation records, unchanged frozen controllers, and checkpoint hashes. Pose-only novel-axis success 30/64, 17/64, 17/64 versus full history 28/64, 20/64, 17/64; retained seed 1's failed gate. No consistent history advantage established. E22 remains a fixture audit, not a trained benchmark.

- Revision 16, 18 October 2026: Launched matched three-seed E21 pose-only training with frozen teachers and unchanged budgets; results pending. Completed E22 paired-fixture feasibility audit: identical initial policy inputs in all twelve pairs, full-rank Cartesian Jacobians, distinct probe responses, and shared expert-reachable goals. All 34 tests pass. No learned policy evaluated on final test.

- Revision 1, 15 October 2026: Consolidated historical experiments and prespecified auxiliary-response pilot; results pending at initial writing.

- Revision 2, 15 October 2026: Completed E07, recorded all task/prediction outcomes and matched-run checks; 21 regression tests pass. No adaptation claim supported.

- Revision 3, 15 October 2026: Added E08 exact training-episode audit and teacher-state action errors; reordered next steps to resolve basic imitation/control before transfer.

- Revision 4, 15 October 2026: Added E09 overfit results and E10 shared-recovery protocol; initial results pending for E10.

- Revision 5, 15 October 2026: Completed E10/E11; recorded direct-chunk tiny-set gate, remaining flow failures, full negative sampling probes, and a verified comparison figure. All 21 existing regression tests pass.

- Revision 6, 15 October 2026: Found and versioned a Cartesian-observation timing fix, regenerated the dataset, and prespecified E12 full-body comparison. Legacy E04-E11 timing limitation documented; 23 tests pass.

- Revision 7, 15 October 2026: Completed E12/E13, passed the direct-regression training-control gate, retained the negative identity-history result, and added verified per-body plots. All 24 regression tests pass.

- Revision 8, 17 October 2026: Completed E14 frozen unseen-arm validation; retained direct regression as working baseline and flow as comparison. Reported correct-versus-empty failure on novel axes, verified immutable checkpoints and untouched final test.

- Revision 9, 17 October 2026: Added E15 diversified-family design and common-reset correction; preserved the preliminary feasibility dataset and prespecified fixed-budget control gate.

- Revision 10, 17 October 2026: Completed E15 and its conditional privileged reference; recorded 870/939 direct, 79/939 flow, and 816/939 privileged successes, all below the training gate. Verified expert replay, checkpoint/data/recovery hashes, per-body figure, and 26 tests. Expanded validation remains deferred.

- Revision 11, 17 October 2026: Completed E16 fixed-data 10k/20k/30k learning curves. Exact 10k replays reproduce both baselines. Privileged control passes at 20k and 30k (907/939 and 915/939); history control fails all budgets (30k: 880/939, weakest body 46/71). Verified all six checkpoints and continuation states, inspected the learning-curve figure, and passed 28 tests. Prioritized history-representation diagnostics; no held-out policy evaluation performed.

- Revision 12, 17 October 2026: Completed E17 frozen-controller distillation (914/939 correct-history training success) and E18 frozen unseen-arm validation (46/48 familiar-family, 28/64 novel-family with correct32). Recorded empty/wrong controls, all history lengths, per-family failures and collisions. Verified unchanged controller/teacher tensors and checkpoint hashes, inspected the validation plot, and passed 29 tests. Preliminary history-conditioned transfer supported; replication and final-test evaluation remain outstanding.

- Revision 13, 17 October 2026: Completed E19 with seeds 1/2 plus the earlier seed-0 pilot. Novel correct32 succeeds 28/64, 20/64, 17/64, beating both controls in every seed. Retained failed training gates in seed 1 and persistent `seven_e` failure. Added true-parameter transfer diagnostics, per-family outcomes, all history lengths, verified 2,688 validation records and frozen controllers, inspected the replication plot, and passed 30 tests. No final-test policy evaluation performed.

- Revision 14, 18 October 2026: Implemented E20 frozen history interventions; Windows Application Control blocks MuJoCo imports and simulator checks. Completed offline audits of 63,622 states per teacher and 96 training calibrations per seed/condition. Recorded static-pose latent approximation, sparse near-goal `six_c` recovery coverage, one passing standalone transformation test, unchanged checkpoint hashes, and pending rollout validation. No new control-success claim.

- Revision 15, 18 October 2026: Completed E20 in Ubuntu WSL2 without changing Windows security policy. All 31 tests pass; intact-history success/collision outcomes reproduce E19. Recorded all 2,016 intervention rollouts, matched teacher audit, verified hashes, and inspected the comparison plot. Repeated initial pose nearly matches intact history across all seeds, requiring a narrower interpretation of earlier history-transfer results. No final-test evaluation performed.

## Appendix D1: known-arm task demonstration

Two configured PiperX/OpenArm physical pick-and-place showcases use the same frozen E99 neural estimator with online measured-history updates and a model-derived home prior. Both succeed and repeat exactly. Demonstration servos and simplified contact pads; not a new independent transfer benchmark. See `outputs/ease_known_arm_task_demo_v1/report.md` for all attempts, assumptions and prediction diagnostics; `ease_pick_place_demo.mp4` is the30-second showcase.

### D1 revision: fingertip clearance and smooth playback

Corrected proximal grasp frames/pads that allowed the native visual fingertips to intersect the tabletop. Enabled gripper mesh/table and floor contacts. Both known-arm tasks still succeed and repeat exactly. All 840 rendered snapshots per arm have positive gripper/table clearance (PiperX 7.76 mm; OpenArm 8.02 mm). New 50 fps physics replay and gentle dissolves replace the 12.5 fps render and abrupt edits. Prior artifacts preserved. See `outputs/ease_known_arm_task_demo_v2/report.md`; no benchmark scores changed.

## E114: observation-only learned-adaptation ablations


Learned online succeeds in 27/48 checkpoint-task runs, versus 22/48 with response frozen and 30/48 with all post-calibration adaptation frozen. Analytical online succeeds in 12/16 distinct tasks. Learned counts pool three checkpoint seeds on the same16 tasks; they are not48 independent body/task samples. The result depends on which component is frozen; it does not support a blanket benefit from all online adaptation. The learned online estimator does not exceed analytical online task success.

16 reused unseen-body development tasks: eight bodies, four previously studied axis families, two placements per body. Three frozen learned checkpoints and one analytical estimator, each in three update conditions, produce 192 complete episodes. The analytical episodes are not replicated as independent seed trials.

Every condition receives the same 32 measured calibration transitions. Online updates both response and command bias; frozen response updates only bias; frozen all updates neither. The same 32-step local probe is executed before grasp, but frozen conditions do not incorporate its motion observations. Online switches to the existing 32-transition recent-history estimator at that point. All learned conditions use a single E99 retained checkpoint throughout the episode, eliminating the earlier parent/student switch.

No geometry filter, physics selector, planner, privileged response Jacobian or gravity feedforward is used. Estimator initialization is 1e-6 I / zero cross statistics, not a model-derived Jacobian. The common simulator reset pose and phase goals are supplied; observations include measured joint positions and tool pose. Physics only executes the commanded actions and scores the outcome.

| Estimator | Updates | DOF | Successful | Collision episodes |
|---|---|---|---:|---:|
| analytical | online | all | 12/16 | 3 |
| analytical | online | 6 | 6/8 | 2 |
| analytical | frozen_response | all | 11/16 | 4 |
| analytical | frozen_response | 6 | 5/8 | 3 |
| analytical | frozen_all | all | 13/16 | 1 |
| analytical | frozen_all | 6 | 7/8 | 1 |
| learned | online | all | 27/48 | 17 |
| learned | online | 6 | 11/24 | 12 |
| learned | frozen_response | all | 22/48 | 24 |
| learned | frozen_response | 6 | 10/24 | 14 |
| learned | frozen_all | all | 30/48 | 16 |
| learned | frozen_all | 6 | 14/24 | 10 |
| analytical | online | 7 | 6/8 | 1 |
| analytical | frozen_response | 7 | 6/8 | 1 |
| analytical | frozen_all | 7 | 6/8 | 0 |
| learned | online | 7 | 16/24 | 5 |
| learned | frozen_response | 7 | 12/24 | 10 |
| learned | frozen_all | 7 | 16/24 | 6 |


Complete paired gains/losses, per-body breakdown and scope in `outputs/ease_adaptation_ablation_v1/report.md`. Reused development bodies; no new final-test, flow baseline or promotion claim.

Final verification: six relevant existing tests passed; eight independent full re-executions matched every saved trace and state exactly under forbidden-selector/geometry guards. Saved 30 labeled success/failure videos; all replayed state positions match their recorded traces.

## E115: complete motion/bias adaptation factorial

64 new motion-only episodes;192 reused E114 controls. Bias fixed at measured post-calibration values, response history updates unchanged. No action selector or geometry filter.

- learned / online: 27/48 successful, 17 collisions.
- learned / frozen_response: 22/48 successful, 24 collisions.
- learned / frozen_all: 30/48 successful, 16 collisions.
- learned / frozen_bias: 27/48 successful, 13 collisions.
- analytical / online: 12/16 successful, 3 collisions.
- analytical / frozen_response: 11/16 successful, 4 collisions.
- analytical / frozen_all: 13/16 successful, 1 collisions.
- analytical / frozen_bias: 13/16 successful, 1 collisions.

Full paired changes, body/seed breakdown and development-set scope in `outputs/ease_frozen_bias_v1/report.md`.

## E116: six-joint learned estimator diagnosis

32 exact baseline replays and48 full-task analytical-substitution interventions; frozen calibrated bias, no selector.

Before first forbidden collision, the episode-macro mean XYZ prediction error is 0.9143 mm for the learned estimator and 0.3614 mm for analytical ridge on the SAME measured transitions and pre-update statistics. Analytical prediction is better in 24/24 learned-driven episodes.

- baseline: 10/24 success, 12 collisions.
- probe: 17/24 success, 5 collisions.
- lift: 18/24 success, 4 collisions.

Same-history errors, phase/conditioning diagnostics and paired outcomes: `outputs/ease_six_motion_diagnosis_v1/report.md`. Reused development tasks; no training or promotion.

Final audit: 4 independent full intervention replays matched exactly; 14 success/failure videos verified against all stored tool/object state positions.

## E117: training-body learned-estimator repair

Candidate full-task success: 26/36 versus parent 21/36; collision episodes 9 versus 14. Strict retention gate FAILED. All three final checkpoints are retained; no seed/checkpoint selection.

48 matched training-body drivers; three fixed1500-update final checkpoints;36 new candidate evaluations without selector or geometry filter. No unseen-body fitting or evaluation. Full offline metrics, paired task outcomes and provenance: `outputs/ease_estimator_repair_v1/report.md`.

## E118: broadened histories and descent/lowering retention

Candidate 50/72 versus E117 reference 49/72; collisions 18 versus 20. Strict retention gate FAILED. No unseen-body evaluation performed.

Eight added fitting placements and four fresh held placements on existing training bodies,48 new reference rollouts, three fixed2000-update runs and72 candidate rollouts. Complete paired outcomes and held-placement diagnostics: `outputs/ease_broad_history_v1/report.md`.

## E119: candidate-reached corrective histories

Task success 50/72 to 47/72; collisions 18 to 21. Strict retention gate FAILED.

48 exact fitting-history replays, three fixed1500-update corrective runs and72 candidate evaluations. No held-history optimization or unseen evaluation. Full paired gains/losses and legacy retention: `outputs/ease_candidate_history_v1/report.md`.

## E120: bounded correction to frozen estimator

Frozen E118: 50/72 successes, 18 collision episodes. Bounded E119 correction: 52/72 successes, 18 collision episodes. Strict retention gate FAILED.

Seven lost-success pairs exactly replayed; fixed25% action residual capped at0.001 rad/joint/step, all72 episodes. Frozen weights, no selector or unseen-body evaluation. See `outputs/ease_bounded_history_v1/report.md`.

## E121: cumulative correction budget

reference: 50/72, 18 collisions; unbounded: 52/72, 18 collisions; candidate: 46/72, 25 collisions. Recovered 3/4 regressions; retained 1/6 gains. Gate False.

Four exact baseline replays, phase-aligned drift diagnosis and72 fixed-budget evaluations. No unseen bodies or physics selector. See `outputs/ease_drift_budget_v1/report.md`.

## E122: selective measured-history correction

reference: 50/72, 18 collision episodes; unbounded: 52/72, 18 collision episodes; candidate: 51/72, 21 collision episodes. Retained3/6 gains, recovered1/4 regressions; retention gateFalse.

Ten exact shadow replays and72 causal-gate episodes; no unseen bodies or physics selector. Details: `outputs/ease_selective_history_v1/report.md`.

## E123: matched-state correction pulses

Across90 phase-entry pulse comparisons: correction-on helps final success in16, hurts in9, and leaves final success unchanged in65. 70/90 starts have no prior forbidden collision. E120 remains unchanged.

Ten exact E120 references and90 phase-entry off-pulse continuations, full integration/history equality asserted. E120 unchanged. Selected development diagnostics, not new benchmark success rates. Details: `outputs/ease_matched_correction_v1/report.md`.

## E124: learned task-value gate

reference: 50/72, 18 collisions; unbounded: 52/72, 18 collisions; candidate: 58/72, 14 collisions. Retained3/6 gains; recovered4/4 regressions; gateFalse.

480 fitting collection episodes (410 new), clean paired-outcome supervision, one shared fixed MLP and72 full-task evaluations. No held optimization or unseen bodies. Full report: `outputs/ease_task_value_gate_v1/report.md`.

## E125: learned task-value gate

reference: 50/72, 18 collisions; unbounded: 52/72, 18 collisions; previousgate: 58/72, 14 collisions; candidate: 59/72, 13 collisions. Retained5/6 gains; recovered4/4 regressions; gateFalse.

1440 fitting collection episodes (960 new), clean paired-outcome supervision, one shared fixed MLP and72 full-task evaluations. No held optimization or unseen bodies. Full report: `outputs/ease_broad_task_value_gate_v1/report.md`.

E125 preserves all46/48 E124 fitting successes and improves held success from12/24 to13/24, still below E120's15/24. Compared with E120, nine fitting gains and two held losses (seven-joint case20, estimator seeds0 and2). Five of six original E120 gains survive and all four regressions recover; the no-loss criterion fails, so E120 remains the baseline. Fitting placements expanded16 to48 across the workspace; exact held cases remain excluded, but negative-Y fitting support now changes the spatial extrapolation setting. All1296 paired starts were checked, with1108 clean labels entering training. No unseen embodiments were tested.

## E126: ancestry-grouped validation and abstention

Four ancestral-placement folds, three gate models per fold,1108 existing clean fitting pairs. Every nontrivial fixed rule gains one but loses two E120 validation successes; full abstention selected before held retest. E126 matches E12052/72 (37/48 fitting;15/24 held), retains6/6 original gains and recovers0/4 regressions. All nine E125 fitting rescues are relinquished. E120 remains baseline; no useful learned intervention rule validated. No held optimization or unseen bodies. Full report: `outputs/ease_abstaining_task_value_gate_v1/report.md`.

## E127: history-similarity abstention

Reused E126 ensemble weights; fitting-family-only history banks and leave-family-out distance calibration. Selected rule{'min_advantage': 2.0, 'max_spread': 0.0, 'quantile': 0.0} gives0 gains/0 losses over144 fitting validation episodes. Final candidate52/72; retained6/6 original E120 gains and recovered0/4 regressions. Strict retentionFalse. No held optimization or unseen bodies. E120 unchanged. Full report: `outputs/ease_history_similarity_gate_v1/report.md`.

The useful descent intervention and a harmful descent intervention share the same validation fold and support group. The useful example has distance3.685777 versus2.871023 for the harmful one. Therefore a single upper-distance cutoff in this metric cannot accept the useful example while rejecting that harmful example. This observed ordering points to the representation/metric, rather than merely a too-tight cutoff. It is a fitting-validation diagnostic, not a new held-tuned rule.

## E128: task-value ranking diagnosis

Diagnosis only on1108 excluded-fitting-family phase pairs: reversed rankings on34/65 beneficial and25/54 harmful pairs. Saved5 exact paired diagnostic cases with training-neighborhood and block-replacement sensitivity analysis. No new model/rule/rollout or held-placement retest. E120 unchanged. Full report: `outputs/ease_task_value_ranking_diagnosis_v1/report.md`.

E128 additional diagnostics: training heads100% accurate; OOF on/off60.3%/59.7%. Fold-training frequency reference78.8%/79.8%. Most labels989/1108 are neutral; selected collisions differ44–246 control steps after pulse entry. Proposed next controlled tests: paired-advantage objective and measured task-state augmentation, not another confidence cutoff. No new policy trained.

## E129: direct paired-advantage objective

Same four fitting-family folds, inputs, seeds, hidden initialization and training budget. Direct tanh-MSE advantage regressor vs two-head BCE: OOF advantage MSE0.535 vs0.200; decisive sign47.9% vs50.4%. Fixed-rule reconstructed task success94/144 vs96/144, E12097/144. Direct7 gains/10 losses; two-head1/2. Both zero-loss selections abstain. No original held-placement retest, unseen bodies, or baseline replacement. Full report: `outputs/ease_paired_advantage_objective_v1/report.md`.

## E130: richer measured task-state inputs

Fixed two-head BCE and fitting-family splits;139 to162 inputs with ideal measured tool/proprioceptive/gripper state, no object/contact truth. Matched original initialization, added weights zero. Fixed-rule reconstructed success97/144 vs original96/144, E12097/144; richer2 gains/2 losses. Advantage MSE0.202 vs0.200; decisive signs52.1% vs50.4%. Both zero-loss rules abstain. No original held-placement test or promotion; E120 unchanged. Full report: `outputs/ease_richer_task_state_v1/report.md`.

## E131: privileged object/contact-state information upper bound

Explicit simulator-only inputs:20 object-pose/contact/pre-action grasp features appended to measured162. Fixed two-head BCE and ancestral folds. Privileged decisive sign56.3% vs measured52.1%, but fixed-rule reconstructed success96/144 (1 gain/2 losses), measured97/144 (2/2), E12097/144. All zero-loss selections abstain. Extra information is not a mathematical performance bound. No held-placement retest, deployment or E120 change. Full report: `outputs/ease_privileged_task_state_v1/report.md`.

## E132: dense short-horizon paired targets

Same privileged182 inputs/family folds;4432 progress and8864 collision-prefix labels at1/2/4/8 steps from1108 matched starts. Learned progress RMSE exceeds zero-effect reference at every horizon (8-step7.28mm vs3.36mm). Reconstructed task success: E12097/144, long-target96/144, learned-short96/144 (2 gains/3 losses), TRUE-short future-informed diagnostic106/144 (13/4). Diagnostic is not learned performance and still loses successes. Selected zero-loss learned rule abstains. No new rollouts, held-placement test or E120 changes. Full report: `outputs/ease_short_horizon_paired_v1/report.md`.

## E133: structured one-step response

Measured-history scalar response: one-step OOF progress RMSE17.69um vs zero50.67 and E132 MLP98.61;90.86% nonzero sign accuracy. Eight-step pulse reconstruction101/144 vs E12097 (12 gains/8 losses), so not promoted. No-loss grid abstains. Same fitting-family splits, held untouched. Report: `outputs/ease_structured_response_v1/report.md`.

## E135: compact regularized task value

32 measured structured features, linear ridge paired advantage. Primary102/144 vs E12097 (9 gains/4 losses). Sign52.1%; zero-loss grid abstains. No promotion or held evaluation. Report: `outputs/ease_compact_response_value_v1/report.md`.

## E134: one-step response control

Fresh144 fitting rollouts, frozen E133 family-excluded scalar response and10um threshold; exactly one off action. Candidate97/144 vs E12097, 10 gains/10 losses. All prefixes and action restoration verified; held untouched. Report: `outputs/ease_one_step_response_control_v1/report.md`.

## E136: combined local-response and task-value gate

Frozen primary conjunction: fitting102/144 vs E12097,5 gains/0 losses; all144 fresh rollouts exactly match counterfactuals. Original development-held15/24 vs E12015, 1 gains/1 losses. Known training bodies; no unseen-body claim. Report: `outputs/ease_combined_response_value_v1/report.md`.

## E137: family-deletion confidence

Primary consensus:97/144, no gains/losses,9 neutral interventions. No-loss selection abstains.32 family-deletion models saved and OOF predictions verified. Fresh48-placement test reserved, unexecuted. Report: `outputs/ease_family_consensus_gate_v1/report.md`.

## E138: expanded fitting counterfactuals

1,440 additional fitting episodes; 1100 new clean paired starts. Fixed ridge/consensus/trees comparisons, same family grouping and margins. Selected abstain: 195/288 vs E120 195, 0 gains/0 losses. Fresh test reserved. Report: `outputs/ease_expanded_structured_pairs_v1/report.md`.

## E139: rich structured value

Selected abstain: 195/288 vs E120195, 0 gains/0 losses. Fresh test not opened. Report: `outputs/ease_rich_structured_value_v1/report.md`.

## E140: paired effect classifier

Selected abstain: 195/288 vs E120195, 0 gains/0 losses. Fresh test not opened. Report: `outputs/ease_paired_effect_classifier_v1/report.md`.

## E141: learned history weights

Direct measured-motion estimation,32-history Transformer weighting inside explicit ridge solve. Learned OOF position RMSE0.904mm vs uniform1.468 and analytical1.392; both joint counts improve. Stronger fitting-tuned constant kernel0.874mm beats learned, so no attention advantage over that reference. Prediction-only, no task rollout; all references and labels verified. Report: `outputs/ease_learned_history_weights_v1/report.md`.

## E142: anchored history prediction

Mass-normalized Transformer weights are anchored to a training-fold-selected constant kernel; ridge and forgetting are frozen. Four family splits and all three prespecified seeds are retained. Pooled position RMSE improves from 0.864 to 0.838 mm, but one seed is slightly worse than the reference, so the individual-model robustness screen fails. No task episodes were executed here. The reference grid is finite and includes a boundary solution, not a globally optimal analytical estimator. Report: `outputs/ease_anchored_history_weights_v1/report.md`.

## E143: anchored history prediction

The fixed mean of all three E142 Jacobians reaches 0.817 mm versus constant-kernel 0.864 mm: 10.5% lower position MSE. Both joint counts improve; each leave-one-model-out diagnostic also beats the constant reference overall. All three models remain in the primary ensemble; no best seed or pair is selected. This passes the prediction screen for a full-controller experiment, not evidence of higher task success. Report: `outputs/ease_history_weight_ensemble_v1/report.md`.

## E145: measured calibration prior control

Success: **67/96**, with 23 collision episodes. Six joints: 35/48; seven joints: 32/48. Uniform32 reference: 74/96. Paired rescues/losses: **1/8**. All outcomes retained. Report: `outputs/ease_calibration_prior_control_v1/report.md`.

## E146: learned calibration-prior history weights

 Report: `outputs/ease_calibration_prior_learning_v1/report.md`.

## E144: complete history-estimator control

{'analytical': {'all': {'n': 96, 'success': 70, 'collisions': 22}, '6': {'n': 48, 'success': 36, 'collisions': 11}, '7': {'n': 48, 'success': 34, 'collisions': 11}}, 'uniform32': {'all': {'n': 96, 'success': 74, 'collisions': 20}, '6': {'n': 48, 'success': 36, 'collisions': 11}, '7': {'n': 48, 'success': 38, 'collisions': 9}}, 'constant': {'all': {'n': 96, 'success': 70, 'collisions': 22}, '6': {'n': 48, 'success': 37, 'collisions': 10}, '7': {'n': 48, 'success': 33, 'collisions': 12}}, 'e120_ensemble': {'all': {'n': 96, 'success': 72, 'collisions': 21}, '6': {'n': 48, 'success': 36, 'collisions': 11}, '7': {'n': 48, 'success': 36, 'collisions': 10}}, 'learned_ensemble': {'all': {'n': 96, 'success': 67, 'collisions': 26}, '6': {'n': 48, 'success': 37, 'collisions': 11}, '7': {'n': 48, 'success': 30, 'collisions': 15}}}

Advancement: False. All regressions retained; E120 unchanged. Report: `outputs/ease_history_estimator_control_v1/report.md`.

## E147: complete calibration-prior comparison

The learned calibration-prior estimator completes71/96, compared with67/96 for its matched analytical prior (8 rescues,4 regressions),72/96 for E120 ensemble and74/96 for uniform32. All learned-versus-prior changes are seven-joint cases. It fails the frozen advancement screen: E120 ensemble is not exceeded and fitting gain placement13 remains unsuccessful. No held/fresh evaluation or E120 replacement. Report: `outputs/ease_learned_calibration_control_v1/report.md`.

## E149: complete calibration-prior comparison

PRIVILEGED REFERENCE ONLY. Exact current kinematics completes76/96 with16 collision episodes:35/48 six-joint,41/48 seven-joint. This is the same fixed task controller and measured frozen bias, not an upper bound on all possible policies. Exact kinematics improves seven-joint performance but does not recover the six-joint failures of the calibration-prior method. It is never used by a learned estimator or training objective. Report: `outputs/ease_exact_map_control_v1/report.md`.

## E148: measured directional probes

Collected **9614** single40ms branches from **738** clean phase-entry histories across all96 fitting placements. Each active joint receives both signs of a fixed.01rad perturbation around measured holding bias. These are training-only reconstructed simulation states, not full task episodes or inference-time probes. All branch labels verified. Report: `outputs/ease_dense_joint_supervision_v1/report.md`.

## E150: dense directional supervision

Mixed directional RMSE0.1762mm versus fixed E1460.1768 and observed-only continuation0.1752; the5% improvement screen fails. No new controller or fresh evaluation. Development folds are adaptive and collection was not nested end-to-end. Report: `outputs/ease_dense_joint_training_v1/report.md`.

## E153: learned weights with fixed uniform kernel

Ensemble position RMSE0.388mm vs uniform0.415,12.6% lowerMSE; all seeds and joint counts improve. Kernel and ridge exactly preserved, no calibration prior. Prediction only; complete control is E154. Report: `outputs/ease_fixed_uniform_learning_v1/report.md`.

## E151: fixed home-posture control

 Report: `outputs/ease_home_posture_control_v1/report.md`.

## E152: matched analytical-prior posture reference

68/96 versus learned70/96; same posture objective. Added after partial E151 outcomes, no parameter changes. No fresh test.

## E154: fixed-kernel continuation

Standard-controller learned weights complete77/96 versus matched uniform74 and E120 ensemble72; collisions17 versus20 and21. Six joints37/48 versus36/48; seven40/48 versus uniform38 and E12036. Paired versus uniform:5 rescues/2 losses; versus E120 ensemble:9 rescues/4 losses. This is an adaptive fitting-development result, not held or unseen-body evidence. Legacy case13 retention fails; no fresh test. Report: `outputs/ease_fixed_uniform_control_v1/report.md`.

## E155: fixed-kernel continuation

All-phase projection rescues learned case13 but loses case6; diagnostic gate fails. No full/fresh evaluation. Report: `outputs/ease_prior_projector_control_v1/report.md`.

## E156: fixed-kernel continuation

Lift-only projection does not recover learned case13 and loses case6. Every pre-lift trace and saved state matches the original standard controller exactly; all non-lift actions are unchanged. Diagnostic gate fails. No full/fresh evaluation. Report: `outputs/ease_lift_posture_control_v1/report.md`.

## E157: fixed-kernel continuation

Fixed conservative learned weights0.404mm vs uniform0.415; no retraining or temperature sweep. Prediction only. Report: `outputs/ease_conservative_history_weights_v1/report.md`.

## E158: conservative lift diagnostic

Fails legacy case13 despite retaining case6. Twelve outcome-selected diagnostic episodes plus four matched standard controls; exact pre-lift equality. No full/fresh test of this variant. Report: `outputs/ease_conservative_lift_control_v1/report.md`.

## E159: amended prospective fresh-placement validation

Fresh known-body placements: learned41/48, uniform41/48, E120ensemble39/48, analytical39/48. The fitting improvement does not meet the prospective net-improvement pattern. The fresh outcomes remain a negative or mixed validation result; no model was retuned or substituted. Amendment registered before fresh outcomes; earlier retention gates remain failed. No unseen-body claim or retuning. Report: `outputs/ease_fresh_fixed_weights_v1/report.md`.

## E161: fitting elevated-release clearance

E120 ensemble success 72 to 77/96; 5 rescues and 0 regressions. Collision episodes 21 to 16. All pre-lower states match exactly. Elevated release allows a short drop; not gentle placement, learned-estimator superiority or unseen-body evidence. Full report: `outputs/ease_release_clearance_control_v1/report.md`.

## E162: matched fitting release-margin transfer

e120: 72 to 77/96, 5 rescues/0 regressions; uniform32: 74 to 79/96, 5 rescues/0 regressions; learned: 77 to 81/96, 4 rescues/0 regressions. Original estimators unchanged, all pre-lower states exactly matched. Fitting only, no held evaluation. Report: `outputs/ease_release_clearance_transfer_v1/report.md`.

## E160: fitting-only task-control rate limiting

e120_ensemble: 72 to 46/96, 4 rescues and 30 regressions; uniform32: 74 to 53/96, 1 rescues and 22 regressions. All criteria and E120 weights unchanged; no held execution. Full report: `outputs/ease_task_contact_control_v1/report.md`.

## E163: early six-joint contact audit and geometry-ID correction

All11 six-joint E161 collisions occur on body84000,10 during approach/descent and1 during lift. Contact labels3/4 refer to geometry IDs attached to link1/link2, not link3/link4. Signed geometry inspected offline only, with no policy input. Millimetre-scale initial task clearance and near-limit posture motivate E164. Report: `outputs/ease_early_contact_diagnosis_v1/report.md`.

## E164: early six-joint task control with release fix retained

high_approach: 37/48 six-joint successes, 2 rescues/2 regressions, 11 collisions; centered_early: 36/48 six-joint successes, 0 rescues/1 regressions, 11 collisions. Baseline37/48 with11 collisions. Qualified variant: None. Fitting only, no privileged policy inputs or held evaluation. Report: `outputs/ease_early_clearance_control_v1/report.md`.

## E165: conditional early six-joint control

38/48 versus37/48 baseline; 2 rescues/1 regressions; 10 collision episodes, 9 first approach/descent contacts. Fixed screen passed: False. Release fix and E120 weights preserved, fitting-only. Report: `outputs/ease_conditional_approach_control_v1/report.md`.

## E166: descent early six-joint control

35/48 versus37/48 baseline; 0 rescues/2 regressions; 12 collision episodes, 9 first approach/descent contacts. Fixed screen passed: False. Release fix and E120 weights preserved, fitting-only. Report: `outputs/ease_descent_reference_control_v1/report.md`.

## E167: privileged exact-current-motion route diagnostic

Exact-map release-only 37/48, conditional high route 37/48; 0 rescues/0 regressions. Simulator Jacobian is privileged and not a learned-policy result; no training labels or held execution. Report: `outputs/ease_early_exact_diagnostic_v1/report.md`.

## E168: privileged link-clearance constraints with frozen E120

joint_projection: 38/48 success, 6 collision episodes, 1 rescues/0 regressions; task_projection: 39/48 success, 7 collision episodes, 2 rescues/0 regressions. Baseline37/48 with11 collisions. Exact link geometry is privileged; fitting diagnostic only, release fix and scoring preserved. Report: `outputs/ease_link_clearance_diagnostic_v1/report.md`.

## E169: privileged probe/gripper coverage

probe_only: 41/48 success, 4 collisions, paired vs E168 {'rescues': [27, 63], 'losses': []}; gripper_only: 39/48 success, 7 collisions, paired vs E168 {'rescues': [], 'losses': []}; both: 41/48 success, 4 collisions, paired vs E168 {'rescues': [27, 63], 'losses': []}

E120+release remains37/48; E168 task projection39/48.144 fitting intervention episodes; no held test or model update. Complete results and saved success/failure videos: `outputs/ease_probe_gripper_clearance_v1/report.md` and `videos/index.html`. Conservative gripper-plane correction documented before intervention outcomes.

## E170: descent execution-error allowances

fixed_allowance: 42/48 success, 2 collisions, {'rescues': [64], 'losses': []} versus E169; history_allowance: 44/48 success, 0 collisions, {'rescues': [42, 43, 64], 'losses': []} versus E169. E169 protected reference41/48 with4 collisions; E120+release37/48. Privileged fitting diagnostic only,96 new episodes,4 exact replays. E120/probe protection/release preserved. Report: `outputs/ease_descent_execution_v1/report.md`; saved success/failure videos: `videos/index.html`.

## E171: descent task-progress objective

direct_task: 43/48 success, 0 collisions, {'rescues': [], 'losses': [43]} versus E170; limit_gated: 43/48 success, 0 collisions, {'rescues': [], 'losses': [47]} versus E170. E170 protected reference44/48 with0 collisions; E120+release37/48. Privileged fitting diagnostic only,96 new episodes,4 exact replays. E120/probe protection/release preserved. Report: `outputs/ease_descent_progress_v1/report.md`; saved success/failure videos: `videos/index.html`.

## E172: selective descent stall recovery

stall_recovery: 44/48 success, 0 collisions, {'rescues': [], 'losses': []} versus E170. E170 protected reference44/48 with0 collisions; E120+release37/48. Privileged fitting diagnostic only,48 new episodes,4 exact replays. E120/probe protection/release preserved. Report: `outputs/ease_descent_stall_v1/report.md`; saved success/failure videos: `videos/index.html`.

All44 successful reference episodes reproduce exactly. Final descent position errors (mm, reference -> recovery): case26: 67.13 -> 19.10; case62: 59.96 -> 46.82; case65: 40.07 -> 39.03; case67: 48.41 -> 40.59. No new completed tasks. Orientation tradeoffs and late plateaus remain; joint5 stays near its upper limit. Broader E171 changes were rejected because each lost one success. Next useful diagnosis is collision-free posture/route feasibility, not an assumed benefit from more time.

## E173: privileged alternative-posture route execution

route_fixed: 44/48 success, 3 collisions, {'rescues': [], 'losses': []} versus E170; 0 extra descent steps; route_extended: 44/48 success, 0 collisions, {'rescues': [], 'losses': []} versus E170; 697 extra descent steps. Static alternatives exist for all four failures.96 full fitting episodes and4 exact replays; unchanged E120/probe protection/release.44 nonactivated trajectories checked exactly per variant. Privileged route assistance and extra interaction budget are explicitly labeled. Report: `outputs/ease_posture_route_v1/report.md`; videos: `videos/index.html`; publication assessment: `publication_readiness.md`. No held test.

Neither route variant is promoted. Extended routes reach the descent criterion for cases26 and67, but their objects move about154mm and73mm during descent and are not grasped. Case65 ends12mm from the pose target with77mm object motion; case62 remains114mm from the target. Static pose feasibility therefore does not resolve dynamic route tracking and grasp handoff. The zero-collision44/48 reference remains unchanged. Publication-critical learned-versus-analytical unseen-body evidence is still missing; a prospective known-body placement comparison previously tied41/48.

## E174: object-preserving routes and grasp handoff

object_route: 47/48 success, 0 collisions, {'rescues': [62, 65, 67], 'losses': []} versus E170; 377 extra descent steps; guarded_handoff: 47/48 success, 0 collisions, {'rescues': [62, 65, 67], 'losses': []} versus E170; 377 extra descent steps.96 fitting episodes and4 exact reference replays. E120, probe protection and release preserved. Strict grasp-first planning finds routes for3 failures; case26 retains the baseline because it starts in object contact. Object pose/contact inputs privileged; no held test. Report: `outputs/ease_object_preserving_v1/report.md`; videos: `videos/index.html`; targeted review: `literature_review.md`.

Cases62,65,67 are rescued in both conditions with zero measured object displacement during descent. All44 earlier successes and the case26 fallback retain complete reference states/traces exactly. Both variants spend377 extra descent steps total (15.08 simulated seconds). The guarded handoff adds no task successes in this comparison; all three routed grasps pass its checks, so no abort/rejection branch was exercised physically. The route bundle combines strict object avoidance, grasp-first vertical approach, current object XY and gain1 tracking; its components are not individually isolated. Literature review identifies task-and-motion planning/state guards, cuRobo, STORM and contact-implicit MPC as relevant established methods, with implementation and privilege caveats.

## E175: earlier object-safe recovery

48/48 successes, 0 collisions. Paired E174 comparison: {'rescues': [26], 'losses': []}. 46 complete trajectories unchanged.48 fitting episodes and5 exact E174 replays. E120, probe protection and release fix unchanged. Early guard designed on fitting histories; privileged geometry/object pose/contact inputs, no held test. Report: `outputs/ease_early_object_recovery_v1/report.md`; verified success/failure videos: `videos/index.html`.

## E176: frozen-controller estimator comparison

E120 48/48, analytical 46/48; collisions 0 versus 1. Paired outcomes: {'both_success': 46, 'E120_only': [26, 64], 'analytical_only': [], 'both_fail': 0}.96 fitting episodes and5 preflight replays; all48 E120 trajectories exactly reproduce E175. Same frozen controller, protections, probes and conditional step caps; actual interaction costs saved. Privileged fitting comparison, no held evaluation or training.

Report: `outputs/ease_frozen_estimator_comparison_v1/report.md`; success/failure videos: `videos/index.html`.

## E177: prospective unseen-body estimator comparison

E120 42/48, analytical 42/48; collisions 1 versus 0. Paired: {'both_success': 39, 'E120_only': [9, 42, 45], 'analytical_only': [0, 8, 37], 'both_fail': 3}.12 prospectively sampled bodies across4 previously used axis families,4 placements each. Frozen E176 controller/budgets;10 exact fitting compatibility replays. No test-based tuning or training. Privileged simulator inputs; not untouched axis-family or hardware evaluation.

Report: `outputs/ease_unseen_estimator_v1/report.md`; success/failure videos: `videos/index.html`.

## E178: matched-history diagnosis of E120 losses

Retrospective diagnosis of E177 losses0,8,37 and wins9,42,45 as controls.12 exact full-episode replays; 4620 matched histories with6 offline one-step diagnostic action candidates each. Source predictions exact and source physical branches verified against recorded states/contacts. No policy tuning or new task-success claims. Report: `outputs/ease_matched_unseen_diagnosis_v1/report.md`; detailed interpretation: `interpretation.md`; videos: `videos/index.html`.

# E178 interpretation: matched-history diagnosis

The controller, E120 checkpoints, protections and interaction budgets remain frozen. This retrospective analysis covers the three E177 analytical-only successes (0, 8, 37) and three E120-only successes (9, 42, 45). It does not change the original cohort result: 42/48 successes for each estimator, with one E120 collision and zero analytical collisions.

Twelve full trajectories replay exactly. Across 4,620 pre-action histories, both estimators receive the same state, history statistics, calibrated bias, goal and controller context; six action candidates are evaluated in restored simulators. Source predictions are exact and source physical steps reproduce recorded states to 1e-12. On collision-free-before-action motion states from the three failed E120 trajectories, reference-network position-change RMSE is 1.04 mm versus 0.48 mm for analytical estimation. Analytical also predicts better on the E120-win controls (1.50 versus 0.74 mm), so prediction accuracy alone does not determine task success. These errors use measured joint displacement, not an assumption that commanded motion was executed exactly.

## Cases 0 and 8: descent, projection and grasp handoff

On the same E120 descent states, analytical actions improve immediate goal progress by an average 1.51 mm/step in case 0 and 1.46 mm/step in case 8. Crossed nominal-action/projection-metric tests implicate the learned metric: holding the E120 nominal action fixed and replacing only its projection metric improves progress by 2.67 and 0.90 mm/step respectively. Removing the bounded correction changes progress by only -0.04 and +0.02 mm/step. All six candidates have zero one-step collisions on these descent starts. These are local counterfactual effects, not demonstrated complete-episode rescues.

Case 0 exhausts descent at 19.87 mm tool error, then closes at 18.05 mm error. It registers both pads but loses contact during lift and leaves the object on the table. Analytical closes at 2.02 mm error and succeeds. Case 8 exhausts descent at 65.77 mm error after moving the object about 33 mm; its recovery route is rejected because robot/object contact already exists. Closing later causes right-pad/table contact at step 179. Analytical obtains a route and succeeds.

The frozen grasp-readiness guard is conditional on an accepted recovery route. Neither E120 case has an accepted route, so the phase logic proceeds to closing despite poor alignment. This is a concrete interaction between estimation-dependent descent progress and the existing controller handoff, rather than evidence that the small correction residual alone causes the failures.

## Case 37: longer-horizon lift convergence

E120 reaches descent and establishes two-pad contact, but its lift times out at 45.18 mm tool error. Transport also times out and the final object position misses the target by about 40.6 mm. Analytical reaches lift and ultimately finishes about 4.9 mm from the target. Both actual trajectories lose pad contact during transport; analytical's success is the original final-position score, not evidence of uninterrupted stable grasp through release.

A 20-step continuation initially made an analytical swap appear worse. Extending the diagnostic to the unchanged 100-step lift budget reverses that conclusion. From the identical E120 lift-entry state and history, analytical reaches the lift goal in 83 steps with 4.06 mm error and object height 321.9 mm; E120 times out at 100 steps with 45.18 mm error and object height 280.4 mm. E120 without its correction is worse: 95.33 mm error and object height 229.0 mm. From analytical's lift-entry state, analytical reaches in 73 steps (4.47 mm), while E120 times out (24.21 mm) and the object falls to table height. All six full-budget branches are collision-free. The two source continuations reproduce 173 recorded steps to 1e-12.

This supplies direct phase-level evidence that estimation-dependent control affects lift convergence even after matching initial history and physical state. It does not demonstrate a full pick-and-place rescue: the counterfactuals stop at the lift boundary. The longer follow-up was selected after inspecting the short-horizon result and is explicitly exploratory.

## Consequences for the next experiment

Do not simply disable E120's bounded correction: it is small and mixed in descent, and helps the difficult lift relative to the reference-only condition. The next fitting-only experiment should separate (1) the estimator used to define the clearance-projection metric, (2) grasp-readiness checks for all descent exits, and (3) full-budget lift convergence. Any changes need fitting-case selection before a fresh unseen-body evaluation; these six outcome-selected cases are diagnostic, not a new validation set.

All raw predictions, action candidates, restored branch states, phase continuations, audits and source snapshots are saved in this directory. Twelve videos retain the actual frozen-policy successes and failures, including the E120-win controls. The privileged local true Jacobian is an offline diagnostic and action-distance proxy, not an optimal-action label or deployed component.


## E179: fitting projection metric and universal readiness

{'baseline': {'episodes': 48, 'success': 48, 'collisions': 0, 'losses': [], 'aborts': 0, 'routed': 5, 'total_steps': 16106, 'median_steps': 309.5}, 'analytical_metric': {'episodes': 48, 'success': 43, 'collisions': 3, 'losses': [27, 43, 62, 63, 67], 'aborts': 2, 'routed': 3, 'total_steps': 16997, 'median_steps': 312.5}, 'universal_readiness': {'episodes': 48, 'success': 46, 'collisions': 1, 'losses': [40, 42], 'aborts': 1, 'routed': 5, 'total_steps': 16664, 'median_steps': 316.5}, 'combined': {'episodes': 48, 'success': 41, 'collisions': 2, 'losses': [26, 27, 43, 47, 62, 63, 67], 'aborts': 5, 'routed': 3, 'total_steps': 17592, 'median_steps': 309.5}}

Prespecified fitting-only2x2 ablation; frozen E120 nominal policy/checkpoints, protections and release fix.192 episodes plus5 exact preflight replays; all48 baseline trajectories exact. Report: `outputs/ease_metric_readiness_v1/report.md`; videos: `videos/index.html`.

# E179 interpretation

The fitting preservation screen failed for all three candidates. Keep E120 and the original protected controller as the baseline. No unseen bodies were evaluated and no weights were updated.

| Condition | Successes | Collision episodes | Lost baseline successes |
|---|---:|---:|---|
| E120 baseline | 48/48 | 0 | None |
| Analytical projection metric | 43/48 | 3 | 27, 43, 62, 63, 67 |
| Universal readiness and existing handoff | 46/48 | 1 | 40, 42 |
| Both | 41/48 | 2 | 26, 27, 43, 47, 62, 63, 67 |

## What changed, and what this means

E120's nominal action, bounded correction, learned weights and frozen calibrated bias were retained in every condition. The metric intervention replaces only the Jacobian used by the clearance projection with the analytical estimate of the current measured history. It is applied wherever the projection is eligible, including probes. Although this improved immediate descent progress on the retrospective unseen failures in E178, it loses five complete fitting episodes here. A local progress improvement is therefore insufficient evidence for a global metric replacement.

For analytical-metric losses 27, 62 and 63, the first forbidden contact is right-pad/table during closing. Their close-entry orientations fail the inherited .08 readiness bound; cases 62 and 63 also exceed the 8 mm horizontal-position bound. Cases 43 and 67 accept recovery routes but fail the existing object-displacement readiness check (29.70 and 38.58 mm drift), then abort without a collision. The metric intervention changes earlier trajectories, route availability and subsequent histories even though the nominal policy implementation is unchanged.

Universal readiness extends the entire existing routed handoff: it checks readiness, holds the close-entry joint posture, and aborts/opens when not ready or without five consecutive final two-pad contacts. This is not a pure threshold-only ablation. All its trajectories match baseline exactly until closing. Case 40 passes readiness and retains two-pad contact for all 40 close steps, but subsequently collides with the table during lift (step 121). It cannot be explained as an overly strict rejection; the changed closing behavior leads to a different lift state/history. Case 42 instead fails readiness at close entry (16.27 mm horizontal error, .1906 axis error) and aborts, although baseline later succeeds. Rejecting that state without a recovery opportunity loses a recoverable success.

Combining the interventions avoids case 63's collision by aborting, but does not recover its task success. It still collides in cases 27 and 62 despite opening/holding; that fallback is not a guaranteed safe stop under the frozen dynamics/projection. It adds collision-free losses 26 and 47 after both passed readiness and accumulated 39/40 final two-pad steps, respectively. These outcomes do not support deploying the combination.

## Next fitting-only work

Separate readiness checks from posture holding: preserve normal closing behavior for ready, non-routed grasps, and diagnose an alignment recovery for rejected states within the original budget. Any such follow-up must also address the dynamic safety of open-and-hold rather than treating an abort flag as a safety guarantee. Independently, investigate a phase-limited projection-metric change instead of replacing the metric throughout the episode. Neither follow-up has been run or validated in E179. Require retention of the 48 fitting successes and zero collisions before another fresh unseen-body evaluation.

## Evidence and scope

All 192 full episodes and five exact preflight replays completed. Every baseline trajectory exactly reproduces the original protected E120 run. The audit checks 61,215 post-calibration steps for metric construction, bias, readiness/abort behavior, identical calibration, guard-only pre-close equality, and phase budgets. Outcomes and executed states/actions/histories are saved for all cases; videos include every discordant case across all four conditions plus fixed comparison cases. These are deterministic, previously used fitting cases with privileged simulator geometry/object/contact inputs, not independent held-body evidence or hardware results. E120 remains the baseline; no candidate is promoted.


## E180 fitting evaluation of grasp-readiness checks and bounded alignment

Each candidate ran 48 complete pick-and-place episodes on the original six-joint fitting placements. The comparator uses the 48 audited E179 baseline trajectories. Every candidate reaches close entry on the exact baseline trajectory. The routed handoff, probe protection, release height, E120 weights and analytical comparison from earlier experiments retain their original definitions.

| Condition | Successes | Collision episodes | Complete trajectories identical to E120 |
|---|---:|---:|---:|
| Archived E120 baseline | 48/48 | 0 | 48 |
| checks_only | 47/48 | 0 | 47 |
| bounded_alignment | 48/48 | 0 | 47 |

Case 40 passes the entry checks and exactly reproduces the original successful trajectory. Case 42 is the only baseline grasp that fails the entry checks. For accepted non-routed grasps, ordinary E120 closing continues. The recovery condition spends up to 12 of the original 40 close steps aligning to the pick pose with the fingers open. It checks position, orientation and object drift before each attempt to close.

Case 42 failed under checks_only. Alignment commands used = 0. First ready close step (zero based) = None. Final consecutive two-pad steps = 0. Abort = True. Collision = False.
Case 42 succeeded under bounded_alignment. Alignment commands used = 5. First ready close step (zero based) = 5. Final consecutive two-pad steps = 35. Abort = False. Collision = False.

## Research status

E177 established simulated pick-and-place transfer to 12 newly sampled six-joint bodies across four previously used axis families. Protected E120 and the analytical estimator each completed 42/48 placements. E120 uniquely solved three placements and the analytical estimator uniquely solved three. E180 measures fitting retention and provides one recovery opportunity. A claim that learned adaptation improves transfer still needs a prospective paired advantage across fresh bodies, with body-level uncertainty and a matched control for history adaptation.


Artifacts in `outputs/ease_readiness_recovery_v1`.

## E181 prospective pick-and-place transfer

All 12 fresh six-joint bodies passed initial-posture screening. Four placements per body give 48 paired cases in four previously used axis families. Body seeds and physical signatures were checked against archived case manifests before scoring. Each condition uses the E180 bounded recovery rule, privileged geometry and contact protections, and the same interaction caps.

| Estimator | Successes | Episodes with collision | Median control steps including calibration | Episodes using alignment |
|---|---:|---:|---:|---:|
| Online E120 | 34/48 | 6 | 316 | 4 |
| Online analytical | 35/48 | 5 | 305.5 | 4 |
| Calibration-only E120 | 32/48 | 7 | 323 | 5 |

## Paired comparisons

Online E120 versus Online analytical had 1 unique successes and 2 unique losses. Paired difference -2.1 percentage points (body-bootstrap 95% interval -8.3 to +4.2).

Online E120 versus Calibration-only E120 had 3 unique successes and 1 unique losses. Paired difference +4.2 percentage points (body-bootstrap 95% interval -4.2 to +12.5).

Online history updates gained placements 6, 38 and 43 and lost placement 25 relative to calibration-only E120. Online E120 uniquely completed placement 35 against analytical, which uniquely completed 9 and 25. Both body-level intervals include zero. This cohort demonstrates transfer with the protected controllers and leaves the comparative benefit of learned updates uncertain.

The frozen-history condition uses both E120 networks and current state feedback. Its initial 32 calibration transitions determine its motion estimate for the entire episode. Its 32 pre-close probe actions still count toward the budget. Audits verified identical calibration and initial state across all three methods, identical first actions for the two E120 conditions, zero post-calibration updates for the frozen condition, one update per step for both online conditions, and the original phase limits.

Intervals resample 12 complete bodies with replacement, including all four placements of each sampled body. The inference applies to this simulated perturbation distribution and the four sampled axis families.

Artifacts in `outputs/ease_fresh_recovery_v1`.

## E182 matched-history diagnosis

E181 cases 6, 38, 43, 25, 9 and 35. Verified 8,321 matched states and 18 success/failure videos. Case 25 displaced the cube before closing. Report in `outputs/ease_fresh_update_diagnosis_v1/report.md`.

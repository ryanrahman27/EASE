# Research direction: one policy, different kinematic embodiments

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

**E63 completed: common full schedules expose residual tracking failures.**
With geometry contacts disabled, analytical control passes 12/12 schedules;
E35 passes 33/36, E61 uncorrected 32/36 and response-corrected 28/36. Learned
PiPER scores are 18/18, 18/18 and 16/18; most misses are on Panda. Thirty of
37 missed phase holds never acquire tolerance, seven acquire then lose it.
All 120 schedules run 1,700 steps from shared calibrated starts; 86 tests pass.
Keep E35. See report 4.53 and `outputs/ease_native_common_schedule_v2/report.md`.

**E62 completed: both tracking and contact handling need investigation.**
Across 480 native diagnostic executions, 17 of 38 original failures meet their
recorded endpoint tolerances without geometry contact forces; 21 still miss.
References can be failure-truncated, so this is not full-task success or a
causal assignment of each failure. Cube-only removal creates PiPER contact
artifacts. All original results replay exactly; 84 tests pass. Keep E35.
See technical report 4.52 and
`outputs/ease_native_contact_isolation_v2/combined_report.md`.

**E61 completed: response-corrected training fails native retention.**
Corrected local training scores 19/36 versus matched uncorrected 22/36 and
E35 29/36; both new conditions have seven contact failures. All six original
reaching gates pass 256/256. Corrected validation loss worsens in two seeds.
All 1,536 reaching episodes and 72 candidate native trials are fresh; 84 tests
pass. Keep E35. See technical report 4.51 and
`outputs/ease_response_corrected_training_v1/report.md`.

**E60 completed: response correction improves local ranking, with limits.**
Held-out response fits improve true-J one-step action ranking to 88.6%/88.5%
for six/seven joints, improving all eight bodies. Estimated-J composition stays
weak and five-step effects are mixed. All 2,880 branches exactly replay E58;
576 response checks and 83 tests pass. Keep E35; no policy training or promotion.
See technical report 4.50 and `outputs/ease_response_ranking_v1/report.md`.

**E59 completed: held-out-body joint-response prediction improves.**
A shared measured-feature linear predictor lowers body-macro joint-motion RMSE
by 79.2% for six joints and 83.8% for seven versus command-minus-bias, improving
all eight omitted bodies and all 32 body/driver strata. All 16 fits reproduce;
83 tests pass. This is one-step response prediction, not better control.
Keep E35; next validate counterfactual action rankings before policy integration.
See technical report 4.49 and `outputs/ease_joint_response_v1/report.md`.

**E58 completed: actuator-response mismatch limits the local score.**
Across 2,880 fixed-state procedural branches, replacing predicted with realized
joint motion raises first-step ranking agreement from 76–78% to approximately
99.4%. This post-hoc oracle is diagnostic, not deployable. Local-action wins
most short procedural comparisons but still fails native retention; grasp and
contact failures remain unexplained here. Keep E35. All 81 tests pass.
See technical report 4.48 and `outputs/ease_motion_gap_v2/report.md`.

**E57 completed: coherent-history training improves offline losses but fails native retention.**
All six candidates pass 256/256 original reaching episodes. Native manipulation
scores are prediction-only 25/36 and local-action 18/36, versus original E35
29/36, with one and five candidate forbidden-contact failures. Keep E35.
The fixed dataset contains 12,000 measured states from 80 exact procedural
replays; batches mix measured and synthetic contexts equally. All 1,536 gate
episodes and 72 candidate native trials are fresh; 79 tests pass. Trajectory
validation shares bodies/goals with training. See technical report 4.47 and
`outputs/ease_coherent_training_v1/report.md`.

**E56 completed: local-action training does not pass native retention.**
All three candidates retain 256/256 original reaching successes. Native local
scores are PiPER 5/5/5 and Panda 4/1/5 (each /6): 25/36 total, versus paired
prediction-only 26/36 and E35 29/36. Local contact failures increase to five
versus prediction-only one. Keep E35. Lower synthetic local loss is not enough.
Baseline checkpoints/batches/losses reproduce exactly; 768 gate episodes and
36 native trials are reused with provenance checks, alongside 768 fresh gate
episodes and 36 fresh native trials. All 77 tests pass. See report 4.46 and
`outputs/ease_local_action_training_v1/report.md`.

**E55 completed: action-specific score validated for procedural pose tracking.**
XYZ plus scaled tool-Z control gives local one/five-step ranking agreement
80.4%/64.0% for six joints and 78.6%/66.9% for seven, versus operator
48.6%/43.2% and 44.5%/42.3%. All 80 drivers and 720 matched first steps replay
exactly; 7,200 branches and all failures are retained. No contacts or gripper
are present; this is score validation, not a new trained policy. All 75 tests
pass. Next: a prespecified matched local-action versus prediction-only training
experiment, with original reaching and six-joint retention required.
See report 4.45 and `outputs/ease_procedural_pose_v1/report.md`.

**E54 completed: action-specific scoring transfers to procedural histories.**
Eight training bodies and ten separate drivers produce 80 exact replays,
720 sampled states and 7,200 moving-target branches. Local executed-command
one/five-step ranking agreement is 81.2%/75.7% for six joints and 81.0%/72.5%
for seven, versus operator 41.3%/39.8% and 40.4%/39.0%. The advantage holds
across every driver group. This is position-only, contact-free diagnosis, not
an improved trained policy. All 73 tests pass; original weights remain unchanged.
See report 4.44 and `outputs/ease_procedural_action_v1/report.md`.

**E53 completed: local action-ranking diagnostic.**
Across 202 shared-history states and 2,020 counterfactual branches, the E52
operator score agrees with one-step outcomes only 45.0%/47.3% on PiPER/Panda.
Action-specific scoring with clipping, limits and measured bias reaches
73.4%/74.0%, but falls to 67.9%/58.6% over five steps. No branch reproduces
E52 contact failures. All twelve driver replays and 202 repeat branches are
exact; 71 tests pass. No training or model promotion. See report 4.43 and
`outputs/ease_action_objective_v1/report.md`. Next validate the local score on
procedural training arms and candidate-generated histories before optimizing it.

**E52 completed: the tested control-aware objective fails native retention.**
Matched three-seed fine tuning gives prediction-only PiPER 5/6/5 and Panda
3/3/4; control-aware gives 0/0/1 and 1/0/1 (each /6), with 28 contact failures.
Original scores remain 6/6/6 and 5/1/5. Lower synthetic response loss did not
translate into better control. Keep E35; neither candidate is promoted.
All 1,536 training-gate episodes and 72 native trials are audited; 68 tests pass.
See technical report 4.42 and `outputs/ease_control_training_v1/report.md`.
Next audit loss/action agreement before another optimization run.

**E50–E51: contact diagnostics and angular-control comparison completed.**
48 exact full baseline replays confirm physical grasp loss, including aligned
starts. Across 120 new trials, neither initial-yaw hold nor face alignment
improves all robot/seed conditions; both fail retention. Keep the original E35
controller. All calibrations match, weights remain frozen and 66 tests pass.
See technical report 4.41 and `outputs/ease_angular_control_v1/report.md`.
That planned comparison is now complete as E52, above.

**E49 and publication status:** Exact command decomposition does not support a
simple first-step clipping/jump explanation for Panda seed 2's regressions.
All 192 early-descent prefixes match E48; 64 tests pass. No fix is promoted.
Localize later grasp divergence before changing the handoff. The paper still
needs a clear learned-adaptation advantage and independent embodiment evaluation;
an initial related-work screen confirms substantial overlap with existing neural
Jacobian control and cross-embodiment adaptation. See report 4.40 and 5.1.

**E48 update-isolation result:** Better average probe prediction residuals do not
guarantee subsequent grasp reliability. Panda seed 2 reaches 5/6 with either
motion-only or bias-only updates but 2/6 with both; neither isolated choice is
uniformly safe across seeds and PiPER. All 192 trials, 96 exact E47 replays and
48 matched state interventions are verified; 64 tests pass. Keep E35. Next
inspect how Jacobian commands and bias corrections combine at the transition
from joint probing to Cartesian descent, before proposing a handoff change.
These remain observed development tasks. Full results: technical report 4.39.

**E47 local-refresh update:** Additional local excitation does not reliably fix
the original neural controller. PiPER remains 6/6 per seed; Panda baseline 5/1/5
changes to probe 4/1/2, versus wait 3/0/4 and frozen updates 3/1/4 (each /6).
Analytical probing retains all twelve successes. All 240 trials, costs and
failures are retained, with 60 exact baseline replays and 63 passing tests.
Keep E35; next distinguish motion-statistic adaptation from actuator-bias updates
on matched trajectories. These are observed development placements, not new
embodiments or final-test evidence. Full results: technical report 4.38.

**E45–E46 retention update:** All teacher-anchored candidates pass 256/256 original
training-control tasks but fail uniform native retention. Fresh PiPER scores are
old 6/6/6 versus candidate 6/6/5; fresh Panda 5/1/5 versus 6/3/5, out of six each.
Analytical and privileged pass all twelve fresh tasks. Keep the original reference;
the candidate is not promoted. Next isolate local excitation near grasping under
matched interaction budgets rather than assuming synthetic prediction retention
guarantees closed-loop control. Full results and failures: report 4.37.

**E43–E44 estimator update:** The measured-motion Transformer has larger one-step
errors than a shadow analytical fit on the same trajectories. Frozen inference
rescaling/smoothing are mixed. Synthetic long-history fine tuning improves Panda
to 7/5/6 of nine but gives PiPER 8/9/6, regressing two PiPER seeds. Native reaching
retains/improves aggregate scores. Keep the prior checkpoints and treat these as
candidates; next require balanced six-joint retention and the original procedural
training gate before a fresh native evaluation. See technical report 4.36.

**E42 Panda update:** Measured object-height release improves neural results to
7/2/4 of nine versus analytical 8 and privileged 9, from 4/0/2, 3 and 3.
Exactly matched pre-lowering traces isolate this scripted placement intervention.
Yaw alignment, joint holds and tighter settling have mixed/negative results and
remain recorded. Native mechanics and learned weights stay fixed. Remaining
failures occur before lowering; investigate descent/closing/lift control before
claiming reliable grasp retention. Full evidence: technical report 4.35.

**E41 reliability update:** Ramped Cartesian references and explicit phase/grasp
gates now run on nine nominally feasible placements per native robot. PiPER neural
success is 9/8/8 of nine; object-offset placement changes this to 9/7/8 and improves
Panda from 1/0/1 to 4/0/2. Analytical final scores are 9 and 3 respectively.
The placement correction is a shared scripted controller change using allowed
object-position feedback, not estimator training. It has one PiPER regression.
Both revisions and all 270 development trials remain separate in report 4.34.
Next resolve Panda grasp loss/contact failures before freezing a fresh evaluation.

**E39–E40 native-model update:** PiPER (six joints) and Panda (seven) now use
pinned Menagerie mechanics, native actuators and their own coupled grippers.
The earlier same-gripper fixture below remains the procedural benchmark.
Frozen neural reaching scores are 9/9/11 of 12 for PiPER and 12 each for Panda.
Compact native PiPER manipulation scores 1/2/3 of three versus analytical 3;
Panda manipulation remains unreliable. Scene revisions are development evidence,
not learning improvements. Next resolve contact stability and define broader
workspace-matched tasks before testing fresh placements. See report 4.33 for
all screening failures, layout versions, unchanged checkpoints and audits.

**E37–E38 implementation update:** The arm-driven hybrid is now running. Existing
frozen weights independently predict XYZ and scaled tool-Z-direction Jacobians;
the latter regulates downward orientation with free yaw. Same physical gripper,
scripted task phases, measured calibration and controller apply to every model.
Familiar-family neural results are 7/7/9 of 12 versus analytical 8; unseen results
are 3/2/3 of four versus analytical 3. All seeds solve the only IK-feasible unseen
six-joint task, but eight of twelve proposed unseen placements fail endpoint IK.
Expand that narrow coverage and fix contact failures before claiming broad
manipulation transfer. Mounting, calibration and clearance corrections and all
negative pilots are in technical report 4.32. The task definition below records
the original staging plan; its mechanics/pose/hybrid gates now have pilot results.

## Pick-and-place extension: current fixture and next task definition

The first manipulation task will move one 3 cm cube between two tabletop locations,
starting from known object/target poses. Use a physical parallel-jaw gripper,
downward grasp with yaw free where feasible, and the same gripper on every arm.
The arm interface remains variable-length; one additional scalar aperture command
drives both jaw actuators. Object/target positions and a measured gripper pose are
allowed state inputs. Robot geometry and true Jacobians remain excluded from the
learned controller. Vision and arbitrary object categories are deferred.

The implemented fixture has frictional finger/object contacts, a free rigid cube,
and no object weld, attachment or teleportation. A scripted Cartesian carriage
passes three grasp/lift/transport/release cases. Success requires bilateral grasp,
an object height above 7.5 cm, placement within 2.5 cm, settled table height and
speed, zero final finger contact, and no gripper/table contact. These are mechanics
checks, not trained-policy or cross-embodiment successes. Version 3 adds readable
rendering and preserves identical physical outcomes; earlier versions remain.
See `outputs/ease_pick_place_fixture_v3`.

`arm_fixture_xml` exports actual six/seven-joint scenes with two jaw actuators,
a grasp-center site, free cube, table and place marker. The 64 exported scene
assets compile. A local kinematic audit over 128 original homes finds 80 rank-six
and 48 rank-five pose Jacobians (six-joint: 32/32; seven-joint: 48/16). Rank alone
does not establish downward-grasp reachability, reset validity over time or a
collision-free approach. These scene assets are not yet a manipulation dataset.

Implementation gates:

1. Define reachable approach/grasp/lift/place poses and contact-aware failure
   rules on training arms; validate the whole sequence with a privileged expert.
   Record structural exclusions before learned-controller evaluation.
2. Extend measured motion estimation to gripper pose/orientation, or an explicitly
   constrained five-dimensional task; the current three-row Jacobian estimator
   cannot regulate gripper orientation. Keep the gripper command separate from
   the arm joint count and log both.
3. Start with a shared scripted task sequencer using learned arm motion control.
   Label that hybrid honestly; only claim a learned task policy after the phase
   and grasp/release decisions are learned and evaluated.
4. Compare analytical and neural arm controllers with the same sequencer,
   gripper, initial states and total interaction budget on withheld axis families.
   Report grasp, lift, transport, release and full-task outcomes separately.

Repeated reaching waypoints alone will not count as pick-and-place success.


**E34: robustness exposes remaining weaknesses.** Frozen neural models do not
exceed analytical final success in any of seven noise/delay/history conditions.
The larger delay primarily causes timeouts, while four-transition histories
increase neural contact failures and variation across seeds. Higher noise shows
a mean-interaction-cost tradeoff rather than a final-success advantage. Next
investigate noisy/short-history training coverage and explicit delay handling
with matched analytical controls; do not tune to E34 and call it a fresh test.
Restricted memory is tested; physical excitation restriction remains separate.
See report 4.29 and `outputs/ease_robustness_v1/report.md`.

**E33: broader frozen-checkpoint transfer is working, with no analytical-baseline advantage.**
Sixteen new axis families and four crossed geometry/dynamics variants per family
yield 64 bodies and 1,024 reaching tasks. Neural success is 985/1000/995 across
three seeds, including 477/489/486 of 512 six-joint tasks. Fixed-probe analytical
control reaches 1003 overall and 491 six-joint successes with lower mean cost.
No controller-success filtering or checkpoint tuning occurred. Contact failures
concentrate in the `XXXZYZ` family; keep these difficult cases. Next prespecify
noise, delay and limited-excitation comparisons at equal interaction budgets,
alongside contact diagnosis. Final test remains reserved. See report 4.28.

**E30–E32: learned six-joint control now matches the analytical baseline.**
The new joint-token Transformer predicts Jacobian columns from measured Gram/cross
statistics, trained on synthetic linear identification problems using training-only
excitation profiles. A shared damped controller produces joint actions. Online
statistics change; neural weights do not. Cold-start control solves 384/384 in all
three model seeds, including every unseen six-joint task at both postures.
Calibrated online control solves 3,452/3,456 across nine model/probe combinations.
The architecture, objective, training data and controller all changed, so this is
not an isolated layer ablation. Guarded replay rules out analytical Jacobian
substitution, but shared analytical control is intentional. A simple fixed-probe
analytical controller also solves 384/384; broader withheld morphology/task tests
and evidence of a learned advantage remain necessary. See report 4.27.

**E28–E29: explicit measured kinematics solves the six-joint cases.** A shared
analytical controller estimates Jacobian columns and command-tracking bias from
the same 32-step history, then updates them from task transitions. It solves
288/288 unseen six-joint cases and 1,152/1,152 total validation cases. Frozen
identification alone solves 287/288 unseen six-joint cases. Thus measured history
is sufficient for these cases; the current Transformer does not use it reliably.
The learned-policy direction is explicit motion prediction plus shared control,
with online updates and uncertainty. It must beat this analytical baseline to
justify a learned contribution. Guarded replay excludes simulator-Jacobian
leakage; no final-test claim is made. See technical report 4.26.

**E27 replication and efficiency:** Three recovery-trained seeds yield correct
history 312/312/313 validation successes, reactive 311/311/275, and empty history
310/320/300 (each /384). Correct history is more stable than the separate reactive
baseline, but unseen-axis success totals 362/576 versus empty 368/576.
Single-use calibration increases mean cost to 74.3 steps; reuse over 12 goals
reduces it to 45.0, approximately empty history's 45.1. Correct-history new
six-joint shifted-posture success remains 3/48, 0/48, 0/48. Diagnose whether
calibration supports accurate local kinematic prediction before attributing
failures to the controller or claiming rapid identification. Preserve strong
reactive/empty controls and the reserved final test. See report 4.25.

**E26 controlled comparison:** New/seen axes are crossed with matched/shifted
postures, keeping twin observations identical. Shared recovery data largely
closes the history-success gap: history 312/384, reactive 311/384, empty history
310/384. Empty history passes training at 256/256; the separate reactive model
reaches 246/256 but fails one body gate. This single-seed result weakens a claim
that calibration history is necessary for terminal success. Replication and
calibration-inclusive efficiency should precede stronger claims. See report 4.24.

**E25 diagnosis:** Six-joint validation changes reset posture as well as axes:
joints 4/5 start outside training demonstration ranges. Even privileged teachers
solve only 6/48, 12/48, 2/48, versus 48/48 seven-joint tasks in every seed.
Preserve existing results but separate posture and kinematic transfer in the
next dataset version. Frozen action shuffling hurts 8/9 cases, while transition
shuffling changes little. Reactive first-action imitation is already near its
ambiguity bound; investigate recovery coverage rather than only more updates.
See technical report 4.23.

**E24 completed replication:** Three model seeds and three probe seeds retain a
correct-history advantage in every aggregate comparison. Validation ranges are
57–62/96, 52–62/96, and 44–48/96. Every teacher/history training gate passes;
every pose/reactive gate fails. Six-joint transfer is poor (2–17/48), while
seven-joint transfer reaches 41–48/48. This supports repeated history benefit
on the fixed benchmark, not general adaptation or an optimal-reactive comparison.
See technical report 4.22; 37 tests pass and final-test policy evaluation is reserved.

**E23 paired benchmark:** Eight training arms, eight validation arms, and eight
reserved test geometries now have unique bookkeeping IDs. Tasks and probes start
at exactly equal observations within each pair; student training uses only probe
prefixes. The 256 expert episodes replay successfully. The single-seed pilot
passes teacher/history training at 256/256; validation history reaches 62/96
versus pose 47/96, empty 13/96, and twin 29/96. Pose fails the training gate;
six-joint transfer is weak. Replication and stronger reactive baselines remain
necessary. See technical report 4.21.

**E21/E22 (18 October):** Matched trained initial-pose encoders achieve 30/64,
17/64, 17/64 novel-axis successes versus full history's 28/64, 20/64, 17/64.
All three seeds are complete and verified; seed 1's failed gate is retained.
There is no consistent full-history advantage. A separate paired-fixture version removes initial
observation differences within pairs while preserving distinct action responses.
Twelve pairs pass initial equality, full-rank Jacobian, probe, and shared expert
goal checks. This does not yet establish a learned history advantage. See report
4.19–4.20; 34 tests pass in WSL.

**Latest diagnostic (E20, completed in WSL, 18 October 2026):** Repeating initial
pose gets 27/64, 21/64, 17/64 novel-axis successes versus intact history's 28/64,
20/64, 17/64. Static cues retain nearly all aggregate control performance;
interaction-based dynamics identification is not established. All 31 tests pass
in the existing WSL environment. See report 4.18 for all interventions and limits.

**Earlier replication (E19, 17 October 2026):** Three teacher/student seeds yield
28/64, 20/64, 17/64 correct32 successes on withheld arrangements, each above
empty and wrong history. Seed 1 fails training gates and is retained as diagnostic.
The `seven_e` student fails in every seed, and its privileged teachers also
transfer poorly (6/16, 0/16, 4/16). This supports a repeated but limited history
benefit, not robust general adaptation. See report 4.16 and `replicate`.

**Earlier experiments (E17/E18, 17 October 2026):** Frozen-controller latent
distillation brings history-based training control to 914/939. The frozen student
then reaches 46/48 on new familiar-family arms and 28/64 on withheld arrangements
with correct32, versus 0/64 empty and 4/64 wrong on withheld arrangements.
Correct8/16 give 29/64. No target-arm training or weight updates occur. This is
preliminary history-conditioned transfer; failures are uneven and replication is
outstanding. See report 4.14–4.15 and `distill_validation`.

**Earlier experiment (E16, 17 October 2026):** At 30,000 updates on the unchanged
recovery dataset, measured-history control reaches 880/939 (93.7%) and the
privileged physical-parameter reference reaches 915/939 (97.4%). The reference
passes both training gates at 20k and 30k; the history policy passes at none of
10k/20k/30k. The next diagnostic targets history-derived conditioning, using the
successful privileged controller as a fixed reference. Unseen-arm adaptation
is still unestablished. Use `budget_experiment` and report section 4.13.

**Earlier experiment (E15, 17 October 2026):** Three training axis arrangements
per joint count, two bodies per arrangement, and common joint resets are now
implemented. On 939 accepted training episodes, recovery-trained direct
regression succeeds 870 times (92.7%), versus flow's 79. Supplying true axes and
physical parameters to the same-sized network gives 816 successes (86.9%). Both
direct variants miss the training gate. Expanded validation is prepared but
deferred at this stage; E16 above completes the longer-fitting diagnostic.
Use `diverse_data` for this expanded dataset and `diverse_validation` once the
training gate passes. The report preserves the full protocol and negative results.

**Earlier validation (E14, 17 October 2026):** The frozen recovery-trained direct
policy succeeds on 16/16 goals for unseen familiar-family arms with correct32
history. On novel six-joint axes it gives 3/8 correct32, 4/8 empty, and 0/8 wrong32.
Flow gives 10/16 familiar-family and 0/8 novel-axis successes with correct32.
Direct regression is the working baseline; flow remains a comparison. Rapid
history-based kinematic adaptation has not been established. Use
`frozen_validation` for the current synchronized validation path.

**Training milestone (E12/E13):** With synchronized Cartesian observations and shared
recovery data, the history-conditioned direct-regression baseline succeeds on
450/470 training episodes (95.7%), with every training body above 90%. Matched
flow achieves 297/470. Correct history gives 450 successes, empty history 427,
and same-DoF wrong-body history 449: body-specific adaptation is not established.
See [the cumulative technical report](TECHNICAL_REPORT.md) for protocols and all
results. Earlier sections below describe the original pilots; legacy timing
results must not be pooled with the synchronized benchmark.

Use `mixed_data --synchronize` to generate the current dataset and `full_control`
for the objective comparison. The full-control runner rejects legacy observation
timing. Older evaluation scripts do not necessarily support this versioned path.

The target is a shared policy trained across six- and seven-DoF arms that can
control a new six-DoF arm after a short interaction history, with no target-arm
gradient updates. Parameter randomization inside one seven-joint arm is a
preliminary control check, not the desired cross-topology result.

## What one policy learns

Training mixes expert demonstrations from multiple bodies. Expert actions are
computed for each body's actual dynamics and kinematics, but physical parameters,
Jacobians, robot IDs and family labels never enter the student network. The
student sees joint positions/velocities, measured end-effector position, the task
goal, and the available-joint mask. The mask communicates how many actuator
channels exist; it does not communicate link lengths or joint axes.

The policy predicts one joint-position increment for each available joint. It
learns the values of those commands. The software interface still identifies
which channel controls which joint, using a consistent base-to-tip ordering for
these serial-arm fixtures. It does not discover an unknown actuator wiring map.

History is intended to identify effective action-to-state behavior implicitly.
The model is not trained to classify a robot name or reconstruct exact geometry.
At inference, all weights remain frozen; only the available transitions and
their encoded latent change. A short history may or may not suffice—the paired
intervention experiments must establish that empirically.

## Implemented variable-joint architecture

`joint_policy.py` replaces the fixed seven-output action projection with shared
joint tokens and a shared scalar output head. For a batch padded to N joints:

| Quantity | Shape / content |
| --- | --- |
| Joint observation | N × 2: q, dq |
| Task observation | 6: end-effector XYZ and goal XYZ |
| Joint-validity mask | N booleans |
| Noisy action chunk | 16 × N scalar commands |
| History | Up to 32 × N × 11 transition features |
| Embodiment representation | N × 64, one latent per joint |
| Predicted flow velocity | 16 × N |

Each history joint token contains q, dq, executed normalized action, delta-q,
delta-dq, end-effector XYZ and delta-end-effector XYZ. Angular differences are
wrapped. Goal and body metadata are absent. A spatial Transformer mixes valid
joint tokens within each past transition. Two causal temporal Transformer layers
then encode past transitions for each joint, with a learned empty-history token.

The action network projects noisy command/q/dq triples into width 64, adds task
and flow-time conditioning and sinusoidal joint/chunk-position encodings, and
optionally adds the per-joint embodiment latent. Two alternating spatial and
temporal attention stages model joint coordination and action-chunk structure.
All attention uses four heads and feed-forward width 128. The scalar output head
is reused for every valid joint. There are no six-DoF-specific or seven-DoF-specific
heads. All weights are trained from scratch; no vision/language model is present.

Loss masks remove both absent joints and padded episode tails. Invalid joints
are excluded from spatial attention and normalization statistics. Tests verify
that arbitrary padded values cannot change valid outputs, and that trimming a
six-DoF example out of a seven-column batch leaves its valid output unchanged.
The same module accepts 12 joints in a shape/gradient test; **this is not a
quadruped policy or evidence for quadruped transfer**.

## Current procedural fixtures and splits

`mixed_sim.py` creates true six-joint and seven-joint MuJoCo chains. Their joint
axes differ, rather than simulating a disabled seventh actuator:

- Six-DoF training family: Z, Y, Y, X, Y, X.
- Seven-DoF training family: Z, Y, X, Y, X, Y, Z.
- Novel six-DoF validation family: Z, Y, X, Y, Y, X.

Axes are specified in local link frames. Lengths, masses, damping and actuator
strength also vary. These are capsule fixtures, not commercial-arm models.

The pilot has three training bodies per training family. Validation includes one
new six-DoF body and one new seven-DoF body from the training families, plus a
six-DoF body with the novel axis sequence. One additional six-DoF body is reserved
for final testing and is not evaluated by the pilot. Candidate goals are screened
by an expert on all bodies to define a common feasible workspace; that screening
uses privileged information only in benchmark construction.

The stored dataset has 40 shared training goals, 8 validation goals and 8 test
goals. Training includes nominal and broad-start demonstrations (±0.35 radians
across all joints), and independent calibration contexts. Of 480 training
episodes, 470 are accepted for imitation, giving 22,259 transitions. Dataset
sampling currently weights transitions, so longer episodes contribute more
examples; six-DoF examples account for 15,079 transitions and seven-DoF examples
for 7,180. Body-balanced sampling remains an experiment to evaluate explicitly.

The student receives neither axis sequences nor family labels. However, in this
small training set joint count identifies the training family. The novel six-DoF
axis sequence is therefore a necessary diagnostic, and broader within-count
kinematic diversity is required before claiming general embodiment inference.

## Run and interpret the pilot

```powershell
python -m unittest discover -s tests -p 'test_ease_fm*.py' -v
python -m examples.ease_fm.mixed_data --output outputs/ease_mixed_new
python -m examples.ease_fm.mixed_experiment pilot --data outputs/ease_mixed_new --output outputs/ease_mixed_run --steps 2000 --device cuda
```

The pilot trains one baseline and one history-conditioned policy, seed 0, batch
size 32, AdamW at 3e-4, with matched training data and budget. Inference uses ten
Euler steps and executes the first action before replanning. Histories come from
separate training episodes of the same body. Normalization is fit on valid
training values only.

Evaluation compares empty context, 8 correct transitions (0.32 s), 32 correct
transitions (1.28 s), and 32 wrong-body transitions. Wrong-history donors have the
same DoF count, isolating body behavior from the padding mask. Calibration is a
separate bounded-excitation episode followed by a task reset; context stays
frozen during this first test. Per-case flow noise is paired across conditions.
These are short-calibration tests, not continuous recovery from a changing body.

The first 2,000-step pilot (`outputs/ease_mixed_v1_pilot`) had zero successes in
all 24 validation tasks for both models and all history conditions. That verifies
the training/evaluation path but does not show working mixed-embodiment control.
A separate fixed 10,000-step pilot is recorded in
`outputs/ease_mixed_v1_pilot_10000`; do not replace the initial failed run or
interpret one seed as statistical evidence of adaptation.

The completed 10,000-step pilot produced the following success counts (8 goals
per validation arm):

| Policy/context | New six-DoF body | New seven-DoF body | Novel six-DoF axes |
| --- | ---: | ---: | ---: |
| History-free | 0/8 | 1/8 | 0/8 |
| History model, empty | 2/8 | 3/8 | 0/8 |
| Correct 8-transition history | 2/8 | 4/8 | 0/8 |
| Correct 32-transition history | 2/8 | 4/8 | 0/8 |
| Wrong-body 32-transition history | 2/8 | 4/8 | 0/8 |

The learned mixed-arm controller is not reliable, and correct versus wrong
history has identical success counts here. This is not evidence of learned
embodiment identification. The baseline has 138,881 parameters and the history
model 244,289; comparisons between different models are not capacity matched.
Full records and final errors are in each run's `validation.json` and in
`outputs/ease_mixed_v1_pilot_10000/summary.txt`. No final-test body was evaluated.

## History-information sanity check

`history_control.py` implements a separate analytical system-identification
controller. It estimates a local Jacobian from measured joint displacement and
measured end-effector displacement using ridge regression. It estimates residual
servo bias from executed commands minus measured movement. It reads no simulator
Jacobians, link geometry, actuator strength or robot IDs for control. Initial
identification uses 32 calibration actions (1.28 seconds); a rolling history
updates the estimate during task execution. It performs online analytical fitting,
not gradient updates to a neural policy.

On this small validation set it succeeded on 7/8 new six-DoF goals, 8/8 new
seven-DoF goals and 8/8 goals for the novel six-DoF axis sequence: 23/24 overall.
Records are in `outputs/ease_mixed_v1/history_control.json`. The shared expert
screening established feasibility for all of these goals before evaluation.

This establishes a useful development baseline: the measured-history interface
can support control across these fixtures. It is **not a learned EASE-FM result,
not a novelty claim, and not a multi-body statistical generalization result**.
It updates its estimate during execution, whereas the first FM calibration
interventions freeze context after calibration, so it is not a budget-matched
comparison of those adaptation mechanisms.

```powershell
python -m examples.ease_fm.history_control --data outputs/ease_mixed_v1 --output outputs/ease_history_control_new.json
python -m examples.ease_fm.mixed_report outputs/ease_mixed_v1_pilot
```

The measured-response auxiliary experiment is now complete. A capacity-matched
10,000-update pilot (seed 0, 282,372 parameters each) compared flow-only training
against flow plus 0.1 times normalized measured-displacement prediction loss.
Correct32 success was 2/24 for the control and 1/24 for the auxiliary model; both
failed all six-joint and novel-axis tasks. The trained motion predictor beat a
zero-motion baseline, but correct history did not consistently beat wrong-body
history. This is a negative control result, not evidence of learned adaptation.
See [technical report E07](TECHNICAL_REPORT.md#43-auxiliary-experiment-results)
for the full protocol, tables, limitations, and artifact links. Earlier runs are
preserved; this matched control has a different initialization from the previous
pilot and must not be substituted for it.

```powershell
python -m examples.ease_fm.auxiliary_experiment --data outputs/ease_mixed_v1 --output outputs/ease_mixed_aux_new --steps 10000 --device cuda
```

The analytical controller remains an explicit baseline. The subsequent training-episode audit (technical report E08) found only 74/470
flow-only and 73/470 auxiliary successes on seen starts, bodies, and goals.
The tiny-subset follow-up (E09-E11) now finds that shared expert recovery data
raises direct-chunk regression to 7/7 successes on one arm and four training goals.
Flow reaches 16/21 over three sampling repeats and remains imperfect. The next
gate is reliable control across all training bodies, before broadening kinematic
diversity or making adaptation claims. See the technical report for all results.

## Extending to several arms and quadrupeds

Joint tokens solve variable input/output dimensionality, only one part of that
problem. A locomotion extension also needs body orientation and velocity, contact
state, a locomotion task command, suitable training objectives and demonstrations,
and tests for stability. Reaching XYZ observations alone cannot represent those
requirements. Sharing actuator tokens does not imply manipulation skill transfers
to locomotion.

The experimental sequence is mixed six/seven-DoF reaching, then broader held-out
arm families and tasks, then an explicitly multitask arm/legged study. The success
criterion for the first stage is consistent improvement from correct short
histories over empty and same-DoF wrong histories on multiple new arms and model
seeds—not just a policy capable of returning different vector lengths.

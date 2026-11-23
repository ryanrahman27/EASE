# EASE-FM development prototype

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

**E49: command handoff inspected.** All 192 E48 prefixes match exactly through
the first 16 descent steps. Panda seed 2 has no first-step clipping in any update
condition, and its worse both-update condition has a smaller mean command jump
than motion only. This does not establish a smoothing fix. All 64 tests pass;
success labels are inherited from E48, not new complete trials. See technical
report 4.40 and `outputs/ease_handoff_diagnostic_v1/report.md`. Report 5.1 now
records publication-readiness gaps and an initial primary-source literature screen.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.handoff_diagnostic --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --reference outputs/ease_split_refresh_v1 --output outputs/ease_handoff_diagnostic_new
```

**E48: motion/bias update isolation completed.** Across 192 trials, Panda neural
scores are both 4/1/2, motion only 3/2/5, bias only 4/2/5 and neither 3/1/4
(each /6). Each isolated choice loses one PiPER task in a different seed. Both
components improve their matched-probe residuals, yet combined task success can
worsen. Keep the original reference; next inspect the probe-to-task transition.
All 96 E47 control replays are exact, all 48 groups are matched, and 64 tests pass.
See technical report 4.39 and `outputs/ease_split_refresh_v1/report.md`.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.split_refresh --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --reference outputs/ease_local_refresh_v1 --output outputs/ease_split_refresh_new
```

**E47: local calibration refresh completed; not adopted.** Across 240 matched
trials, a 32-step pre-grasp probe retains PiPER 6/6 in all neural seeds but changes
Panda from 5/1/5 to 4/1/2 of six. Waiting gives Panda 3/0/4; identical probing with
updates frozen gives 3/1/4. Analytical probing solves all twelve tasks. All 60
baseline replays and paired probe motions match exactly; weights are unchanged
and 63 tests pass. See technical report 4.38 and
`outputs/ease_local_refresh_v1/report.md`. Keep the original reference.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.local_refresh --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --cases outputs/ease_retention_fresh_v1/cases.json --output outputs/ease_local_refresh_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.local_refresh_report --output outputs/ease_local_refresh_new --reference outputs/ease_retention_fresh_v1/records.json
```

**E45–E46: retention attempt and fresh placements completed.** Teacher-anchored
synthetic fine tuning passes 256/256 original training-control tasks per seed, but
native reaching and some manipulation cases regress. On six fresh placements per
arm, PiPER old/new neural scores are 6/6/6 versus 6/6/5; Panda 5/1/5 versus 6/3/5.
Both analytical and privileged solve all fresh tasks. The candidate is not promoted.
All 1,038 rollout records and 61 passing tests are recorded in report 4.37 and
`outputs/ease_retention_report_v1/report.md`; `decision.json` records failed retention.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.motion_lowrank --data outputs/ease_paired_factorial_v2 --parents outputs/ease_robust_training_v1 --output outputs/ease_motion_retention_new --retention
/home/faisa/miniconda3/bin/python -m examples.ease_fm.robust_training_gate --data outputs/ease_paired_factorial_v2 --models outputs/ease_motion_retention_new --output outputs/ease_motion_retention_gate_new
```

`retention_fresh prepare` records candidates/screens before evaluation;
`retention_fresh evaluate` runs all old/new seeds and shared references. These
placements are now observed development data. Final test remains reserved.

**E43–E44: estimator diagnostics and synthetic fine tuning.** Matched one-step
errors expose a neural/analytical prediction gap. Rescaling and smoothing have
mixed results. A fixed 2,000-update synthetic fine-tune per seed improves Panda
manipulation 7/2/4 to 7/5/6 of nine, but PiPER changes 9/7/8 to 8/9/6. Native
reaching is PiPER 10/11/11 of twelve and Panda 12 each. Keep E35 as the reference;
E44 checkpoints are experimental because two PiPER manipulation seeds regress.
No native trajectories enter training; the original procedural training gate has
not been rerun. All 264 rollout records and 61 passing tests are documented in
report 4.36 and `outputs/ease_motion_adaptation_report_v1/report.md`.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.motion_lowrank --data outputs/ease_paired_factorial_v2 --parents outputs/ease_robust_training_v1 --output outputs/ease_motion_lowrank_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.panda_motion --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_motion_lowrank_new --cases outputs/ease_native_reliability_expanded_v2 --reference outputs/ease_panda_grasp_supported_v1 --output outputs/ease_motion_lowrank_manipulation_new --variant trained --robots piper,panda
```

**E42: Panda release failures reduced.** A measured cube-height release trigger
improves neural successes from 4/0/2 to 7/2/4 of nine observed placements;
analytical improves 3 to 8 and privileged 3 to 9. Calibration and pre-lowering
trajectories match exactly. Earlier descent/grasp/lift failures remain, and no
neural seed beats analytical. All 225 trials, including negative yaw/hold/settle
variants, are retained. Weights and native mechanics are unchanged; 61 tests pass.
See technical report 4.35 and `outputs/ease_panda_grasp_report_v1/report.md`.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.panda_grasp --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --cases outputs/ease_native_reliability_expanded_v2 --output outputs/ease_panda_grasp_new --variant supported
```

`supported` is object-height-triggered release and permits a short drop; it does
not require a table-support contact sensor. Final placement/contact scoring is
unchanged. Other native runners retain their previous defaults.

**E41: expanded native manipulation reliability.** Completed 270 development
rollouts with frozen weights. Ramped targets and phase gates give PiPER neural
9/8/8 of nine placements versus analytical 8. Adding measured object-offset
placement gives PiPER 9/7/8 versus analytical 9 and Panda 4/0/2 versus analytical
3. Panda remains unreliable, and one PiPER neural trial regresses. Exact paired
calibration/pre-transport traces isolate the placement intervention; all negative
revisions remain recorded. All 59 tests pass. See technical report 4.34 and
`outputs/ease_native_reliability_report_v1/report.md` for full counts and costs.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.native_reliability --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --output outputs/ease_native_reliability_new --expanded
```

The current runner uses revision 3 (object-offset placement). Earlier versions
remain in their experiment source snapshots. These are observed development
tasks, with scripted sequencing and ideal object-position feedback, not hardware
or a reserved final-test evaluation.

**E39–E40: native PiPER and Panda integrated.** Pinned Menagerie assets retain
native dynamics, actuator limits and physical grippers. Frozen neural seeds solve
9/9/11 of 12 PiPER reaching tasks and 12/12 each on Panda; analytical scores are
10/12 and 12/12. Compact PiPER pick-and-place scores are 1/2/3 of three versus
analytical 3; Panda manipulation remains unreliable. These are simulated robot
designs, not hardware results. All three development layouts and failures remain
separate. All 58 tests pass; no new training or reserved final-test evaluation.
See technical report 4.33 and `outputs/ease_real_arm_report_v1/report.md`.
Exact successful rendered replays are in `outputs/ease_native_piper_preview_v1`
and `outputs/ease_native_panda_preview_v1`.

Run in Ubuntu WSL from the repository root, reusing the verified pinned assets:

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.real_arm_reach --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --output outputs/ease_real_arm_reach_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.real_arm_pick_place --assets outputs/ease_real_arm_assets_v1 --models outputs/ease_robust_training_v1 --base-height .2 --compact --robots piper --output outputs/ease_real_arm_pick_place_new
```

**E37–E38: physical arm-driven pick-and-place now works.** Frozen E35 estimator
weights predict measured XYZ and tool-direction motion separately; a common
scripted sequencer controls grasp/lift/transport/release. Familiar-family neural
scores are 7/7/9 of 12 versus analytical 8. On four IK-screened unseen-family
tasks, neural scores are 3/2/3 versus analytical 3; every seed solves the one
retained unseen six-joint case. Eight of twelve proposed unseen placements failed
endpoint IK, so this is a small feasibility pilot. No learned grasp sequencing
or broad manipulation-generalization claim. All 56 tests pass; final test reserved.
See report 4.32 and `outputs/ease_pick_place_report_v1/report.md`.
`outputs/ease_pick_place_preview_v1` contains an exact rendered neural six-joint replay.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.arm_pick_place --data outputs/ease_paired_factorial_v2 --output outputs/ease_arm_pick_place_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.pick_place_control --cases outputs/ease_arm_pick_place_new --models outputs/ease_robust_training_v1 --output outputs/ease_pick_place_control_new
```

**E35: noisy/short-history training completed.** Updated neural seeds score
988/983/989 of 1,024 under higher sensor noise versus fixed analytical 979, with
lower mean task steps. This is observed development evidence, not independent
confirmation. Short-history gains are inconsistent; matched delay-aware gain
benefits both methods and analytical remains stronger at 120 ms. All three
updated models retain 256/256 training success. See report 4.30 and
`outputs/ease_robust_training_v1/report.md`. All 53 tests pass; final test reserved.

**E36: physical pick-and-place foundation.** A frictional parallel-jaw fixture
passes three scripted Cartesian grasp/lift/transport/release cases. Arm scene
assets compile for six/seven joints; pose-rank audit and task gates are recorded
in `MIXED_DESIGN.md` and report 4.31. No learned arm pick-and-place result yet.
Rendered stages and physical trials: `outputs/ease_pick_place_fixture_v3`.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.robust_training --data outputs/ease_paired_factorial_v2 --evaluation outputs/ease_broad_development_v2/data --replication outputs/ease_motion_replicate_v1 --output outputs/ease_robust_training_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.robust_training_report --output outputs/ease_robust_training_new --previous outputs/ease_robustness_v1
/home/faisa/miniconda3/bin/python -m examples.ease_fm.pick_place_fixture --output outputs/ease_pick_place_fixture_new
```

**E34: robustness evaluation completed; no learned final-success advantage.**
Across seven fixed conditions and 28,672 rollouts, analytical control exceeds
all three neural seeds in final success. Neural scores at 120 ms delay are
678/723/683 versus analytical 725 of 1,024; four-transition histories give
877/949/861 versus 971. Higher noise yields lower neural mean steps but slightly
lower final success. Clean replay exactly reproduces E33. All 50 tests pass,
weights remain unchanged and final test is reserved. See technical report 4.29
and `outputs/ease_robustness_v1/report.md`.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.robustness --data outputs/ease_broad_development_v2/data --replication outputs/ease_motion_replicate_v1 --output outputs/ease_robustness_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.robustness_report --output outputs/ease_robustness_new
```

**E33: broader transfer holds with unchanged checkpoints.** Across 16 new axis
arrangements, 64 physical bodies and 1,024 tasks, neural seeds 0/1/2 solve
985/1,000/995; six-joint scores are 477/489/486 out of 512. Fixed-probe analytical
control solves 1,003 overall and 491 six-joint tasks, with lower mean interaction
cost. Goals were not filtered by controller success. A difficult six-joint family
concentrates contact failures; all cases are retained. All 48 tests pass, checkpoint
hashes are unchanged, and final test remains reserved. See technical report 4.28
and `outputs/ease_broad_development_v2/report.md`.

Reproduce the broader development evaluation in WSL with a fresh directory:

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.broad_embodiment --prior outputs/ease_paired_factorial_v2 --replication outputs/ease_motion_replicate_v1 --output outputs/ease_broad_development_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.broad_report --output outputs/ease_broad_development_new
```

**E30–E32: the redesigned Transformer now controls the unseen six-joint arms.**
Cold-start neural motion estimation solves 384/384 validation tasks in each of
three training seeds, including 48/48 unseen six-joint tasks at both postures.
Calibrated neural online estimation solves 3,452/3,456 across three model seeds
and three probes, with no collisions; two failures are unseen six-joint cases.
The network predicts Jacobian columns from motion statistics; a shared analytical
Cartesian controller generates actions. Network weights remain frozen at inference.
This is a new architecture/training objective, not a fine-tuned old action policy.
All 46 tests pass; guarded replays exclude analytical Jacobian substitution.
A fixed-probe analytical controller also solves the benchmark, so no learned
superiority or novelty claim is established. Final test remains reserved.
See report 4.27 and `outputs/ease_motion_replicate_v1/report.md`.

Reusable simulated-control interface (joint positions in radians, EE/goal in meters):

```python
from examples.ease_fm.motion_transformer import MotionController

# A fresh state container for a new arm; the same model supports dof=6 or 7.
controller = MotionController.from_checkpoint(
    "outputs/ease_motion_transformer_v1", dof=6, history=None, device="cuda")
action = controller.action(observation)  # observation: joints [dof,2], task [EE xyz, goal xyz]
next_observation, done, info = env.step(action)
controller.update(observation, next_observation, info["executed_action"])
# Repeat action/update until done. Create a new state container when changing arms.
other_arm = MotionController(controller.model, dof=7)  # shares network weights
```

Reproduce the fixed-protocol replication in WSL, using a fresh output directory:

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.motion_replication --data outputs/ease_paired_factorial_v2 --pilot outputs/ease_motion_transformer_v1 --output outputs/ease_motion_replicate_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.motion_report --data outputs/ease_paired_factorial_v2 --output outputs/ease_motion_replicate_new --coldstart outputs/ease_motion_coldstart_v2 --api-check outputs/ease_motion_api_check_v1
```

**E28–E29: six-joint control works with explicit measured kinematics.** The
observation-only analytical controller solves every unseen six-joint case at
both postures across three probe seeds (288/288), and all validation cases
(1,152/1,152) without contact. Frozen identification already solves 287/288
unseen six-joint cases. This is not a fixed Transformer checkpoint: the learned
policy still needs redesign. A guarded replay blocks simulator Jacobian access
and reproduces all outcomes exactly. All 43 tests pass. See report 4.26 and
`outputs/ease_identified_ablation_v1/report.md`; final test remains reserved.

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.kinematic_control --data outputs/ease_paired_factorial_v2 --output outputs/ease_identified_control_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.kinematic_control --data outputs/ease_paired_factorial_v2 --output outputs/ease_identified_ablation_new --ablation
/home/faisa/miniconda3/bin/python -m examples.ease_fm.kinematic_report --data outputs/ease_paired_factorial_v2 --identified outputs/ease_identified_control_new --ablation outputs/ease_identified_ablation_new
```

**E27 completed:** All four E26 policies now have three training seeds. With
shared recovery, correct-history validation is 312/312/313 out of 384, reactive
311/311/275, and empty history 310/320/300. Both correct and empty history pass
training at 256/256 in every seed. On unseen axes alone, correct history totals
362/576 versus empty history 368/576: no consistent calibration benefit.
Mean interaction cost is 74.3 steps with a probe per task, or 45.0 when reused
over 12 goals, versus reactive 50.1 and empty 45.1. New six-joint transfer remains
weak. All 40 tests pass; 12 checkpoints and 15,360 rollout records are audited.
See report 4.25 and `outputs/ease_factorial_replicate_v1/report.md`.
Final-test policy evaluation remains reserved.

To reproduce with a fresh output directory in WSL:

```bash
/home/faisa/miniconda3/bin/python -m examples.ease_fm.factorial_replication --data outputs/ease_paired_factorial_v2 --pilot outputs/ease_factorial_recovery_v2 --output outputs/ease_factorial_replicate_new
/home/faisa/miniconda3/bin/python -m examples.ease_fm.factorial_replication_report --data outputs/ease_paired_factorial_v2 --output outputs/ease_factorial_replicate_new
```

**E26 completed:** The new crossed axis/posture dataset preserves twin ambiguity
and separates matched posture from a fixed posture shift. Four directly trained
policies compare original data with the same 3,211-chunk recovery union. Reactive
training rises from 154/256 to 246/256 (one body still fails its gate).
Recovery validation totals: correct history 312/384, reactive 311/384, empty
history 310/384; empty history passes training at 256/256. No consistent final
success advantage remains in this single-seed comparison. New six-joint axes
remain difficult. See report 4.24 and `outputs/ease_factorial_recovery_v2`.
All 38 tests pass; final-test policy evaluation remains reserved.

**E25 diagnostics complete:** Privileged teachers also struggle on six-joint
validation (6/48, 12/48, 2/48) while solving seven-joint tasks (48/48 each).
The six-joint validation reset places joints 4/5 outside training demonstration
ranges, confounding kinematics with posture transfer. Action shuffling hurts
8/9 conditions; whole-transition shuffling stays near intact performance.
Reactive first-action error is near the unavoidable imitation bound for identical
twin observations. See report 4.23 and `outputs/ease_paired_diagnostics_v1`.
All 37 tests pass; next priorities are reset coverage and shared recovery data.

**E24 replication complete:** Correct-history validation ranges across three
probes are 57–62/96, 52–62/96, and 44–48/96 for model seeds 0/1/2. History beats
all controls in every aggregate seed/probe case. All teacher/history training
scores are 256/256; every pose/reactive training gate fails. Six-joint transfer
remains weak. `paired_replication_report` verifies 12 checkpoints and 5,472
validation records in `outputs/ease_paired_replicate_v1`. All 37 tests pass.
See report 4.22 for limits and the inspected joint-count comparison plot.

**E23 paired benchmark (18 October):** `paired_data` builds the new version;
`paired_audit` verifies 256/256 expert replays and 68 calibration pairs.
`paired_experiment` runs a 10k-update privileged teacher and, only after its
training-control gate, two matched 5k-update history/pose students. Student
contexts are calibration prefixes only. The completed pilot in
`outputs/ease_paired_pilot_v1` reaches 256/256 teacher/history training successes;
validation history 62/96, pose 47/96, empty 13/96, twin 29/96. Pose fails its
training gate, six-joint transfer is weak, and only one seed is tested.
`paired_report` verifies hashes and frozen controllers. No test rollouts.
All 36 tests pass. See technical report 4.21.

**E21/E22 (18 October):** Matched three-seed pose-only training is complete in
`outputs/ease_pose_baseline_v1`. Novel-axis success is 30/64, 17/64, 17/64 versus
full history's 28/64, 20/64, 17/64: no consistent history advantage. `pose_report`
verifies frozen controllers, hashes, and all 2,352 paired validation records.
Seed 1 fails the training gate and remains diagnostic.
The new `ambiguous_fixture` audit passes for 12 pairs: exactly equal initial
policy observations, different axes, full-rank Jacobians, 5–12 cm separation
under identical probes, and shared expert-reachable goals. This is fixture
feasibility, not learned adaptation. All 34 tests pass in WSL. See report 4.19–4.20.

**Latest diagnostic (E20, completed in WSL, 18 October):** Repeated initial pose
gets 27/64, 21/64, 17/64 novel-axis successes versus full history's 28/64, 20/64,
17/64. The current benchmark does not establish action-response dynamics
identification. All 31 tests pass in Ubuntu WSL2 using
`/home/faisa/miniconda3/bin/python`; intact-history outcomes reproduce Windows.
See `history_audit`, `history_audit_report`, and report section 4.18. Next: a
matched trained pose-only baseline and reduced static-cue ambiguity.

**Earlier replication (E19):** Across seeds 0/1/2, correct32 reaches 28/64, 20/64,
17/64 on withheld arrangements, versus empty 0/64, 4/64, 4/64 and wrong32 4/64,
4/64, 5/64. The advantage repeats but transfer is modest and variable; seed 1
fails training gates and is retained. Every student fails `seven_e`; privileged
teachers achieve only 6/16, 0/16, 4/16 there. See `replicate`,
`replication_report`, and report section 4.16. Final test remains reserved.

**Earlier experiments (E17/E18):** A measured-history encoder distilled against a
frozen privileged controller passes training control (914/939). With correct32
history, frozen validation reaches 46/48 on new familiar-family arms and 28/64
on withheld axis arrangements, versus 0/64 empty and 4/64 wrong on the latter.
Correct8/16 give 29/64. This is preliminary history-conditioned transfer with no
target-arm training; one novel family still fails every task. See `history_distill`,
`distill_validation`, and report sections 4.14–4.15. Replication remains necessary.

**Earlier experiment (E16):** Increasing training to 30,000 updates on the fixed
recovery set yields 880/939 (93.7%) for measured history and 915/939 (97.4%) for
true physical parameters. The privileged reference passes the overall/per-body
gate at 20k and 30k; the history policy fails at every tested budget. Both 10k
replays match the earlier weights exactly. Next: diagnose history representation
using the successful privileged controller. This is training-only evidence,
not unseen-arm adaptation. See `budget_experiment`, `budget_report`, and report 4.13.

**Earlier experiment (E15):** The expanded benchmark trains across three axis
arrangements per joint count and twelve bodies. Shared recovery data brings
direct regression to 870/939 training successes (92.7%); flow reaches 79/939.
A same-sized privileged physical-parameter reference reaches 816/939 (86.9%).
Neither direct variant clears the 95% overall / 90% per-body gate, so expanded
unseen-arm evaluation is deferred. See `diverse_data`, `full_control`,
`diverse_oracle`, and report sections 4.11–4.12. The prepared
`diverse_validation` runner checks the training gate before normal evaluation.

**Earlier validation:** E14 retains direct regression as the working baseline.
With correct32 history it reaches 16/16 on unseen familiar-family arms, but only
3/8 on novel axes, versus 4/8 with empty history. Flow remains a comparison;
kinematic adaptation is still unresolved. See `frozen_validation` and the report.

**Training milestone:** E12 direct regression with shared recovery data reaches
450/470 seen training episodes across all six bodies. Use the synchronized
dataset and `full_control` runner; see the technical report for reproduction,
legacy simulator timing limitations, and the unresolved history-adaptation test.

**Cumulative research record:** [Technical report TR-001](TECHNICAL_REPORT.md).
It records protocols, negative results, interpretation, and reproducible artifact
locations. New experiments should update its registry and revision history.

**Updated research target:** [mixed six/seven-DoF embodiment adaptation](MIXED_DESIGN.md).
This is now the primary direction. The earlier v2 experiment changed parameters
inside one seven-joint topology and does not establish cross-topology transfer.

**Historical fixed-topology work:** see [the v2 protocol and experiments](PROTOCOL.md). The v2
pipeline implements causal history conditioning and matched multi-seed
interventions. The rest of this file records the original v1 pilot; its metrics
and limitations describe that historical run, not the current implementation.
Use the mixed-design and technical-report commands for current experiments. The simulator now uses sustained success
and checks contacts at every physics step, so these old pilot counts cannot be
reproduced by rerunning the legacy commands against the updated simulator.

This implements the first data-and-baseline milestone of the supplied October
2026 proposal. It is a standalone experiment inside this repository; no OpenPI
weights, JAX runtime, or real robot are needed.

## Run from the repository root

The pilot was run with Python 3.10.11, NumPy 2.2.6, MuJoCo 3.8.0 and PyTorch
2.11.0+cu128. The main OpenPI project has its own Python/dependency requirements;
use an independent environment for this example. CUDA is optional.

```powershell
python -m unittest discover -s tests -p test_ease_fm.py -v
python -m examples.ease_fm.collect --output outputs/ease_fm_data --episodes 100 --train-bodies 5 --eval-bodies 2
python -m examples.ease_fm.baseline train --data outputs/ease_fm_data --output outputs/ease_fm_baseline --steps 3000 --device cuda
python -m examples.ease_fm.baseline evaluate --data outputs/ease_fm_data --checkpoint outputs/ease_fm_baseline/policy.pt --output outputs/ease_fm_baseline/evaluation.json --trials 20
```

Output directories and evaluation files must be new to prevent overwrites.
Collection saves exact model XML, physical parameter manifests, episode seeds,
outcomes and transition arrays. Failed or colliding trajectories remain recorded
but are excluded from behavioral cloning. Held-out trajectories and body metadata
are never inputs to the student training loader. Observation normalization is fit
only on accepted training transitions. Checkpoints include normalization buffers;
training records include losses and a copy of the data manifest.

## Implemented contract

- Procedural seven-joint capsule arm with randomized lengths, masses, servo
  strength and damping; MuJoCo recomputes inertias from geometry and mass.
- 25 Hz control, 500 Hz physics, bounded joint-position increments of 0.04 radians,
  150-step episodes, positional success within 3 cm.
- Nonprivileged student input: joint position/velocity, end-effector XYZ and goal
  XYZ. End-effector XYZ is an assumed external measurement supplied by simulation;
  this sensing assumption must be reported in a paper.
- Privileged damped-least-squares expert with gravity compensation for collection.
- Separate train, interpolation and longer-link extrapolation bodies. Seeds are
  disjoint; interpolation bodies are sampled from the training parameter ranges.
- Two-layer, width-128 conditional Transformer flow policy, 16-action chunks,
  masked episode tails, 10 Euler steps, one executed action before replanning.
- Action records represent the clipped command actually sent to the position
  servo, not realized joint displacement. Both pre- and post-action states are saved.

## Recorded pilot, October 15, 2026

`outputs/ease_fm_pilot`: five training bodies, one interpolation body and one
extrapolation body, 20 episodes each. The expert succeeded on 140/140 episodes;
none reported contact at control-step boundaries. Training used 1,992 transitions
from the 100 training episodes, seed 0, batch 64 and 300 optimization steps.

`outputs/ease_fm_baseline_pilot`: checkpoint, training record and 21 evaluation
episodes using new task seeds. Success: train 12/15, interpolation 1/3,
extrapolation 1/3. This is a pipeline smoke experiment, not evidence for the
EASE-FM hypothesis. One model seed and one body per held-out split are insufficient
for statistical conclusions. The earlier `outputs/ease_fm_smoke` is a failed
low-servo-gain development run and should not be mixed with the pilot dataset.

## Limits and next scientific gates

1. Replace or cross-check this generic arm against a versioned Franka-like model.
   It is not a Franka model, has no gripper or pushing task, and has not undergone
   broad physical validation. Targets are sampled from collision-free joint
   configurations; expert outcomes separately establish dynamic reachability for
   the recorded episodes. Contact is currently checked only at control boundaries,
   so transient contacts can be missed. Success has no hold-time requirement yet.
2. Freeze a benchmark with training, validation and untouched test body manifests,
   shared feasible Cartesian goal sets, sustained-success and collision rules,
   and larger expert feasibility checks. Current seeds pair joint-space target
   draws across bodies, not Cartesian targets. Do not tune on these pilot test bodies.
3. Establish a stable history-free baseline across multiple training seeds before
   adding a causal encoder. All proposed history methods must share its data,
   control interface, action normalization, chunk horizon and solver budget.
4. Implement causal transition histories, separate-episode same-body contexts,
   empty contexts, and correct/wrong-body/frozen-history interventions. The current
   code does **not** implement EASE-FM adaptation, perturbation recovery or an
   auxiliary dynamics head.
5. Audit related work before claiming novelty; implement matched baselines and
   uncertainty reporting before making publication claims. No conference deadline,
   novelty claim or submission readiness has been established in this session.

The first research decision is whether history interventions give repeatable
held-out gains over a strong baseline. Flow-matching loss reduction alone does
not answer that question.

## Results and video recording convention

Save protocols before evaluation, individual rollout results immediately, and final aggregates and audits. For each evaluation, retain labeled success and failure demos when present, verify their replays against saved traces, and disclose how examples were selected. Preserve prior artifacts. Latest gallery: `outputs/ease_novelaxis_oracle_v1/videos/index.html`; video manifest includes hashes and frame counts.

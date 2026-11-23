# E69: grasp and tracking diagnostics

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

## Interpretation and next experiment

The procedural failure set is dominated by forbidden contact: 29 of 31 learned failures and all five analytical failures. Nine learned failures combine collision and sustained carry-contact loss; all nine collide first. Twenty collision failures have no sustained carry loss. The other two learned failures are separately audited in terminal_audit.json. This is temporal evidence, not proof that collision caused subsequent loss or that the motion estimator is uninvolved.

All 65 learned successes achieve stable close and avoid sustained carry loss, but their median maximum relative translation is 6.7 mm. The prespecified 5 mm drift event therefore cannot be interpreted as a specific slip detector. It includes compliant grasp motion, rolling and motion relative to the measured tool frame.

Next test a shared clearance-aware path/phase intervention on procedural cases, retaining the original controller and all failures as controls. Keep arm/table, finger/table and arm/gripper collisions separate, with six-joint retention and interaction costs. Do not mask forbidden contacts, change success criteria, or tune the native placements. Active regrasp is secondary until clearance failures are addressed. This is not a native-robot conclusion or a new learned-policy performance gain.

Additional artifact: diagnostics.png shows the two noncollision failures, selected exhaustively rather than for favorable outcomes. Two additional terminal replays match exactly.

Timing clarification: parent_1 case 10 has bilateral contact at trace indices 106-117 during close (105-144), then none at the final close sample. The carry-loss marker at index 145 is the first missing sample in the carry window, not the physical onset of loss. The sequencer proceeds to lift despite missed descend/close tolerances. Parent_0 case 4 misses transport tolerance and finishes 41.8 mm from the target. These observations motivate testing phase transitions; they do not prove a new gate will recover the task.

Terminal checks (success requires XY error <25 mm, height error <5 mm, speed norm <0.02, no finger contact, prior grasp/lift and no collision):

- parent_0 case 4: XY error 41.8 mm, height error 0.1 mm, velocity norm 0.0000, final pad contacts 0, lifted=True; missed phase tolerances: transport.
- parent_1 case 10: XY error 71.6 mm, height error 0.1 mm, velocity norm 0.0000, final pad contacts 0, lifted=False; missed phase tolerances: descend, close.

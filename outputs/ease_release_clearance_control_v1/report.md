# E161: elevated release on fitting cases

The E120 ensemble and all estimator checkpoints remain unchanged. Only the lower/release target height changes from 245 mm to 265 mm, a fixed 20 mm margin. No rate limiter from E160 is used. This permits a short free fall and must not be described as gentle placement. The original strict collision, grasp/lift, final object location, settling speed and released-gripper criteria are retained.

| Metric | Original E120 ensemble | Elevated release |
|---|---:|---:|
| Success | 72/96 | 77/96 |
| Six joints | 36/48 | 37/48 |
| Seven joints | 36/48 | 40/48 |
| Collision episodes | 21 | 16 |

Paired outcomes: **5 rescues, 0 regressions**. Positive fitting screen: True. All prior successes retained: True.

All 96 fitting placements are included, including those already failing before lowering. Each candidate trajectory is exactly identical to the baseline in every saved state and trace entry before its first lowering action. Calibration is identical, bias remains frozen, and checkpoint/source hashes are unchanged. No simulator Jacobian, object pose/contact input, geometry-distance query, physics lookahead or selector is used.

This isolates a task waypoint change under E120, not a learned-estimator improvement. It is adaptive development on four known bodies, not a held or unseen-morphology result. No held placements were executed. Success/failure videos are outcome-selected paired executed-state replays, verified against saved tool/object traces and decoded frame counts.

First forbidden-contact phases: {'6:descend': 7, '7:lift': 2, '7:descend': 3, '6:approach': 3, '6:lift': 1}.

All changed placements are retained in summary.json; all complete trajectories, actions, histories and simulator states are saved.

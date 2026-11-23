# E163: offline first-contact audit

All 11 six-joint collision episodes under E120 plus the E161 release fix occur on body84000. Ten first contacts occur in approach/descent, and one in lift. The other six-joint body has no collision episodes.

**Geometry-label correction:** the trace strings `3` and `4` are unnamed MuJoCo geometry IDs. In these saved models they attach to `link1` and `link2`. Earlier informal descriptions calling them link3/link4 should be read as geometry IDs, not anatomical link indices. The task scores and contact criteria are unchanged.

The inspected first saved task states have only millimetres of elbow/table clearance, and the supplied posture places joints near their ±2.5-rad limits. This motivates a higher approach waypoint and joint-limit centering rather than another home-posture attractor. It does not prove either will succeed.

The audit uses privileged signed geometry distances at saved states, only for offline diagnosis. Neither E164 controller receives these distances, a kinematic model, a body ID, a link label, or simulator lookahead. Saved states occur after40ms control steps; distances can miss within-step transient contact and can reflect a newly formed contact before its next integration-step log.

| Case | First contact phase | Step index | Clearance at first saved task state (mm) |
|---|---|---:|---:|
| 13 | descend | 10 | 6.398 |
| 26 | approach | 4 | 6.839 |
| 27 | descend | 19 | 8.607 |
| 42 | descend | 26 | 8.376 |
| 43 | descend | 18 | 8.300 |
| 62 | descend | 17 | 7.288 |
| 63 | descend | 26 | 8.638 |
| 64 | descend | 35 | 7.220 |
| 65 | approach | 4 | 5.067 |
| 66 | lift | 110 | 5.573 |
| 67 | approach | 5 | 6.146 |

Complete snapshots, joint states, supplied home postures and all six link distances are in diagnosis.json. Source failure videos: [E161 gallery](../ease_release_clearance_control_v1/videos/index.html). No new rollouts were executed for this audit.

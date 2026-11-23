# E24: paired-arm replication and reactive baseline

Fixed dataset. Seed 0 includes the earlier pilot; seeds 1/2 are new. All failures retained.

| Seed | Teacher training | History training | Pose training | Reactive training |
|---|---:|---:|---:|---:|
| 0 | 256/256 (pass) | 256/256 (pass) | 214/256 (FAIL) | 96/256 (FAIL) |
| 1 | 256/256 (pass) | 256/256 (pass) | 107/256 (FAIL) | 83/256 (FAIL) |
| 2 | 256/256 (pass) | 256/256 (pass) | 209/256 (FAIL) | 131/256 (FAIL) |

| Seed | Probe seed | History | Empty | Twin | Pose | Reactive |
|---|---|---:|---:|---:|---:|---:|
| 0 | 930000 | 62/96 | 13/96 | 29/96 | 47/96 | 20/96 |
| 0 | 940000 | 57/96 | 13/96 | 35/96 | 47/96 | 20/96 |
| 0 | 950000 | 61/96 | 13/96 | 38/96 | 47/96 | 20/96 |
| 1 | 930000 | 52/96 | 26/96 | 15/96 | 16/96 | 15/96 |
| 1 | 940000 | 61/96 | 26/96 | 19/96 | 16/96 | 15/96 |
| 1 | 950000 | 62/96 | 26/96 | 22/96 | 16/96 | 15/96 |
| 2 | 930000 | 44/96 | 6/96 | 10/96 | 34/96 | 32/96 |
| 2 | 940000 | 45/96 | 6/96 | 10/96 | 34/96 | 32/96 |
| 2 | 950000 | 48/96 | 6/96 | 12/96 | 34/96 | 32/96 |

Reactive values repeat across probe rows because reactive inference uses no probe.
Repeated goals/bodies/probes are dependent observations, not independent embodiment samples.
Update counts are matched, not FLOPs or effective parameter counts. Failed training gates limit transfer interpretation.
No final-test evaluation. Checkpoint hashes, frozen student controllers, and pose correct/twin invariance verified.

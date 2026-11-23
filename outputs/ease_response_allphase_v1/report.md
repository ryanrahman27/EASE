# E78: frozen response prediction on all task phases

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

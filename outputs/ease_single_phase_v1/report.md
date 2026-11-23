# E104: isolated phase ramps with matched time controls

324 complete rollouts:36 training-body task/seed pairs x one unchanged baseline plus four phases x two controls. The baseline is executed once per pair and reused in each phase comparison. All36 baseline results match E103 exactly. Every intervention trace matches baseline before its selected phase. E99 retained weights, original E66 prefix/probe, four training bodies, task criteria and privileged geometry filter stay fixed. No route planner, weight training or contact feedback is added.

For each phase separately, budget-only allows20 extra maximum control steps; gentle uses the same allowance and the fixed20-step E103 cubic ramp of action minus measured bias. All other phase budgets and commands are unchanged. Original commands resume exactly from step20 onward. The24 training /12 held-placement pairs are unchanged. All four phases are evaluated and reported without ramp-duration or seed selection.

## Outcomes

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

## All paired gains and losses

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

## Prespecified gates

Each phase must strictly improve training success over both controls, preserve every success and each training seed count, and not increase collisions; held placements must retain every success with nonincreasing collisions. Passing phases only are eligible for transfer.

- lift: **FAIL**. {'baseline': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': False}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': False, 'success_improves': False, 'per_seed_nondecreasing': False}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'passes': False}

- transport: **FAIL**. {'baseline': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': False, 'collisions_nonincreasing': True, 'success_improves': False, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'passes': False}

- lower: **FAIL**. {'baseline': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': False, 'collisions_nonincreasing': True}}, 'passes': False}

- retreat: **PASS**. {'baseline': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'budget': {'train': {'retains_all_successes': True, 'collisions_nonincreasing': True, 'success_improves': True, 'per_seed_nondecreasing': True}, 'validation': {'retains_all_successes': True, 'collisions_nonincreasing': True}}, 'passes': True}

## Verification and scope

131 existing tests pass. All36 full baseline replays and288 pre-intervention phase prefixes verify; all324 prefixes through closing match the original reference. Input/checkpoint hashes are unchanged. All trial traces and40ms contact/object diagnostics are saved; these do not resolve every contact transient. Videos are full-result replay verified with frame counts and hashes. Figure phase_comparison.png / .pdf; gallery videos/index.html. This is a training-body development ablation, not a new learned policy or independent generalization result. No final test or inference-time weight adaptation.

## Decision

Only retreat passes the prespecified training and held-placement retention gates: 23/36 successes versus 22/36 for both baseline and matched budget, preserving every prior success. Lowering reaches 26/36 but loses a held-placement success and is not promoted. Lift and transport also fail retention. Evaluate the unchanged retreat intervention on all 48 novel-axis development pairs, with both controls and no seed selection. This selection uses training-body results only.

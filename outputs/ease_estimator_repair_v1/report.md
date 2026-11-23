# E117: Training-body estimator diagnosis and repair

Candidate full-task success: 26/36 versus parent 21/36; collision episodes 9 versus 14. Strict retention gate FAILED. All three final checkpoints are retained; no seed/checkpoint selection.

48 selector-free training-body trajectories provide pre-update histories, next measured motion and analytical reference actions. Three E99 parents and one analytical driver use identical calibration. First two placements per body are training, third is held out for every driver; all bodies themselves are training bodies. No unseen-body data enters training or checkpoint selection. All clean training contexts are eligible for motion supervision; probe actions are excluded from the action objective because the probe overrides nominal DLS commands.

Three fixed1500-update fine-tunes use ridge-matrix targets, same-history analytical-action targets and measured-response loss, plus synthetic replay. The network architecture and inference remain inverse-free; analytical solves generate training labels only. Bias remains frozen at calibrated values, with existing recent32-history update and phase budgets. No physics selector, geometry filter or planner.

## Matched held-placement history errors

| Seed | Metric | Parent | Candidate |
|---|---|---:|---:|
| 0 | matrix_mse | 0.00332477 | 0.00296949 |
| 0 | action_rmse | 0.00967349 | 0.00947929 |
| 0 | xyz_error | 0.000868512 | 0.000893077 |
| 1 | matrix_mse | 0.00344244 | 0.00301969 |
| 1 | action_rmse | 0.00959695 | 0.00956176 |
| 1 | xyz_error | 0.000869606 | 0.000895513 |
| 2 | matrix_mse | 0.00334852 | 0.00293752 |
| 2 | action_rmse | 0.00961432 | 0.00939177 |
| 2 | xyz_error | 0.000848703 | 0.00087911 |

Action RMSE is joint-target radians against analytical DLS on identical histories/goals/bias (probe excluded); XYZ error is metres against the next observed tool transition. Both models use the same fixed held histories. These offline metrics do not establish closed-loop improvement.

## Full episodes

| Method | Placement split | DOF | Success | Collision episodes |
|---|---|---|---:|---:|
| learned | all | all | 21/36 | 14 |
| learned | all | 6 | 12/18 | 6 |
| learned | all | 7 | 9/18 | 8 |
| learned | train | all | 15/24 | 8 |
| learned | train | 6 | 7/12 | 5 |
| learned | train | 7 | 8/12 | 3 |
| learned | held | all | 6/12 | 6 |
| learned | held | 6 | 5/6 | 1 |
| learned | held | 7 | 1/6 | 5 |
| candidate | all | all | 26/36 | 9 |
| candidate | all | 6 | 15/18 | 3 |
| candidate | all | 7 | 11/18 | 6 |
| candidate | train | all | 20/24 | 3 |
| candidate | train | 6 | 10/12 | 2 |
| candidate | train | 7 | 10/12 | 1 |
| candidate | held | all | 6/12 | 6 |
| candidate | held | 6 | 5/6 | 1 |
| candidate | held | 7 | 1/6 | 5 |
| analytical | all | all | 10/12 | 1 |
| analytical | all | 6 | 6/6 | 0 |
| analytical | all | 7 | 4/6 | 1 |
| analytical | train | all | 7/8 | 0 |
| analytical | train | 6 | 4/4 | 0 |
| analytical | train | 7 | 3/4 | 0 |
| analytical | held | all | 3/4 | 1 |
| analytical | held | 6 | 2/2 | 0 |
| analytical | held | 7 | 1/2 | 1 |

## Paired outcomes

[
  {
    "seed": 0,
    "parent_success": 8,
    "candidate_success": 7,
    "gains": 0,
    "losses": 1,
    "changes": [
      {
        "case": 8,
        "dof": 7,
        "held": true,
        "gain": false
      }
    ]
  },
  {
    "seed": 1,
    "parent_success": 8,
    "candidate_success": 10,
    "gains": 3,
    "losses": 1,
    "changes": [
      {
        "case": 1,
        "dof": 6,
        "held": false,
        "gain": true
      },
      {
        "case": 7,
        "dof": 7,
        "held": false,
        "gain": true
      },
      {
        "case": 8,
        "dof": 7,
        "held": true,
        "gain": true
      },
      {
        "case": 10,
        "dof": 7,
        "held": false,
        "gain": false
      }
    ]
  },
  {
    "seed": 2,
    "parent_success": 5,
    "candidate_success": 9,
    "gains": 5,
    "losses": 1,
    "changes": [
      {
        "case": 0,
        "dof": 6,
        "held": false,
        "gain": true
      },
      {
        "case": 1,
        "dof": 6,
        "held": false,
        "gain": true
      },
      {
        "case": 2,
        "dof": 6,
        "held": true,
        "gain": false
      },
      {
        "case": 5,
        "dof": 6,
        "held": true,
        "gain": true
      },
      {
        "case": 7,
        "dof": 7,
        "held": false,
        "gain": true
      },
      {
        "case": 10,
        "dof": 7,
        "held": false,
        "gain": true
      }
    ]
  }
]

## Scope and next decision

The gains are concentrated in fitted placements:15/24 to20/24, while held placements remain6/12. Across seeds there are eight paired gains and three losses, and seed0 decreases8/12 to7/12. The strict retention gate therefore fails despite the pooled improvement. Preserve the earlier checkpoints; the candidates are useful experimental artifacts, not promoted replacements.

Matched held-history diagnostics show a tradeoff. Averaged across the three checkpoints, action RMSE decreases approximately9.63 to9.48mrad, but next-step XYZ prediction error rises approximately0.862 to0.889mm. Phase analysis in held_phase_metrics.json localizes improvements and regressions: six-joint transport action error decreases8.40 to7.03mrad, while seven-joint lowering action error increases21.69 to23.00mrad. Descent prediction error worsens for both joint counts. Thus the fitted objective has improved some task-relevant commands without delivering uniformly more accurate motion prediction or better held-placement task success. The combined objective changes do not isolate which loss term causes each change.

The next diagnostic should focus on generalizing the training-body history coverage and preserving descent/lowering behavior, with new training-body validation trajectories and explicit retention checks. These data do not justify a claim of improved unseen-embodiment adaptation.

This is a training-body repair experiment with held placements on those same bodies. It does not evaluate new morphology transfer. The strict gate requires pooled task improvement, no parent success lost, no seed decline, and no collision increase. No unseen or reserved final-test evaluation is performed here. Offline improvement alone cannot promote a checkpoint. If retention fails, preserve the parents and diagnose changed trajectories before further tuning.

Every baseline and candidate outcome, including collisions and failures, is retained with full traces, states and scene XML. Source and checkpoint hashes, row provenance and independent target checks are saved. Videos are outcome-selected illustrations.

Final audit: 10 independent full trajectory replays matched exactly; 18 trace-verified success/failure videos saved.

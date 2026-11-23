"""E178: freeze retrospective matched-history diagnostic, no policy modification."""
from pathlib import Path
import shutil,ast
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_matched_unseen_diagnosis_v1')
SOURCE=Path('outputs/ease_unseen_estimator_v1')
CASES=(0,8,37,9,42,45)

def run():
    assert not ROOT.exists();ROOT.mkdir();(ROOT/'source').mkdir();base=Path(__file__).parent
    for p,h in read(SOURCE/'frozen.json').items():assert fingerprint(p)==h
    for f in ('cases.json','axis_catalog.json'):shutil.copy2(SOURCE/f,ROOT/f)
    s=(base/'unseen_estimator.py').read_text().replace("REFERENCE=Path('outputs/ease_frozen_estimator_comparison_v1')", "REFERENCE=Path('outputs/ease_unseen_estimator_v1')").replace("ROOT=Path('outputs/ease_unseen_estimator_v1')", "ROOT=Path('outputs/ease_matched_unseen_diagnosis_v1')").replace('REPLAYS=(0,26,43,64,67)','REPLAYS=(0,8,37,9,42,45)').replace("['exact_replays']==5","['exact_replays']==6").replace('(5 if replay else len(tasks))','(6 if replay else len(tasks))').replace('E177','E178')
    marker='                before=self.observe();estimator_jacobian,'
    replacement="                integration_state=np.empty(mujoco.mj_stateSize(self.model,mujoco.mjtState.mjSTATE_INTEGRATION));mujoco.mj_getState(self.model,self.data,integration_state,mujoco.mjtState.mjSTATE_INTEGRATION)\n                before=self.observe();estimator_jacobian,"
    assert marker in s;s=s.replace(marker,replacement)
    s=s.replace('logs.append(dict(estimator_jacobian=', 'logs.append(dict(integration_state=integration_state.tolist(),estimator_jacobian=')
    (base/'matched_unseen_replay.py').write_text(s)
    # Removing the two instrumentation additions exactly recovers the adapter's controller.
    clean=s.replace(replacement,marker).replace('logs.append(dict(integration_state=integration_state.tolist(),estimator_jacobian=','logs.append(dict(estimator_jacobian=')
    original=ast.parse((base/'unseen_estimator.py').read_text());instrumented=ast.parse(clean)
    for name in ('ProtectedArm','route_rollout','geometry_ids','distances','distance_gradient','targets','rollout'):
        a=next(n for n in ast.walk(original) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name);b=next(n for n in ast.walk(instrumented) if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name);assert ast.dump(a)==ast.dump(b),name
    protocol=dict(experiment='E178',scope='Retrospective diagnostic of three E177 E120 losses0,8,37; three E120 wins9,42,45 as controls.12 exact full-episode replays capture complete MuJoCo integration state. No training, tuning, deployment or new task-success evaluation.',cases=CASES,source=str(SOURCE),controller='E177 controller/estimators frozen; only passive mj_getState instrumentation added. Both original success/failure trajectories must reproduce exactly.',matched_histories='On every saved pre-action history from both source policies, recompute reference/correction neural ensembles and analytical solve. Score all against the same measured next dq/dx, separating kinematic J*dq error from command-to-motion error. Compare against exact local kinematic Jacobian only as a privileged offline diagnostic, never as policy input.',actions='From identical state/history/goal/context, reconstruct E120, analytical, E120 without its bounded correction, and a privileged local-Jacobian reference. Keep bias, action bounds, aperture, route/guard state, clearance gradient and allowance fixed; use the unchanged safety projection with each candidate metric. Privileged local action is a diagnostic reference, not an optimal-action label.',branches='One40ms step per candidate from the complete saved integration state, including velocities, controls, warm start and other integration fields. Source-policy branch must reproduce saved post-state and contact union. Measure task error reduction, axis error, object displacement, collision and prediction errors. This is a local intervention with source controller context, not a full alternative rollout or proof of eventual task success.',analysis='Primary summaries on states clean before the action, split by source trajectory, phase and loss/win group. All states retained. Report base versus correction and action-projection differences, first divergences, failure chronology and same-state one-step comparisons. No outcome-based controller adjustment.',videos='Full source trajectories for all six cases and both estimators, outcome-labeled and trace/frame/hash verified. Diagnostic branches are not presented as successful full tasks.',checkpoints=read(SOURCE/'protocol.json')['checkpoints'],source_summary_sha256=fingerprint(SOURCE/'summary.json'),prospective_claim=False)
    save(ROOT/'protocol.json',protocol)
    for method in ('e120_ensemble','analytical'):
        root=ROOT/f'replay_{method}';root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    for p in (SOURCE/'source').glob('*.py'):shutil.copy2(p,ROOT/'source'/p.name)
    for name in ('matched_unseen_prepare.py','matched_unseen_replay.py'):shutil.copy2(base/name,ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',ROOT/'axis_catalog.json',*(ROOT/'source').glob('*.py')]})
    save(ROOT/'status.json',dict(status='prepared',controller_changed=False));print('Prepared12 instrumented exact replays and matched-history diagnostics',flush=True)

if __name__=='__main__':run()

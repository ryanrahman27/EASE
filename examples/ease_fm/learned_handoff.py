"""E92: same-state local Jacobian audit and privileged post-descent substitution."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .learned_path_transfer import ROOT as PREVIOUS,models,route_check
from .novelaxis_path_execution import PathArm,PATHS,PRIOR
from .arm_pick_place import rollout,targets
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .benchmark import fingerprint

ROOT=Path('outputs/ease_learned_handoff_v1')


class HandoffPose(GuardedPose):
    def __init__(self,env,model):
        super().__init__(env.n,model);self.env=env;self.use_exact=False;self.diagnostic=False
    def predict(self):
        return (self.env.jacobian(),self.bias) if self.use_exact else super().predict()


def evaluate(case,t,method,model,mode,record_step=None):
    old=json.loads((PREVIOUS/'trials'/f'{case}_{method}_route300.json').read_text())
    path=json.loads((PATHS/'cases'/f'{case}.json').read_text());selected=path['attempts'][path['selected_attempt']]
    assert old['route_used'] and selected['margin_2mm']
    env=PathArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.route=None;env.path_log=[];env.waypoint=0
    c=HandoffPose(env,model);assert calibrate(env,t['home'],c,872000+t['candidate'])==json.loads((PRIOR/f'probe_{case}.json').read_text())
    env.controller=c;env.filter_enabled=True;env.filter_log=[];handoff={};metrics={}
    if record_step is not None:
        original=env.step
        def captured(action,aperture):
            result=original(action,aperture);record_step(env);return result
        env.step=captured
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name
            if name=='descend':
                route,distances=route_check(env,np.array(selected['endpoint']))
                assert distances==old['route_check']['clearances'] and min(distances)>=.002-1e-6
                env.route=route;budget=300
            if name=='close':
                handoff.update(qpos=env.data.qpos.copy().tolist(),qvel=env.data.qvel.copy().tolist(),ctrl=env.data.ctrl.copy().tolist(),time=float(env.data.time),
                    gram=c.gram.copy().tolist(),cross=c.cross.copy().tolist(),bias=c.bias.copy().tolist(),steps=env.steps,bad=env.bad,grasped=env.grasped,lifted=env.lifted)
                learned,_=c.predict();c.diagnostic=True
                try:exact=env.jacobian()
                finally:c.diagnostic=False
                difference=learned-exact
                metrics.update(learned=learned.tolist(),exact=exact.tolist(),frobenius_error=float(np.linalg.norm(difference)),relative_error=float(np.linalg.norm(difference)/max(np.linalg.norm(exact),1e-12)),
                    position_error=float(np.linalg.norm(difference[:3])),scaled_direction_error=float(np.linalg.norm(difference[3:])),
                    gram_eigenvalues=np.linalg.eigvalsh(c.gram).tolist())
                assert handoff['gram']==c.gram.tolist() and handoff['cross']==c.cross.tolist() and handoff['bias']==c.bias.tolist()
                c.use_exact=mode=='exact_after_descent'
            yield name,goal,aperture,budget
    original_jac=mujoco.mj_jacSite
    def guarded_jac(*args,**kwargs):
        assert c.use_exact or c.diagnostic,'Unexpected task Jacobian access'
        return original_jac(*args,**kwargs)
    with patch.object(mujoco,'mj_jacSite',side_effect=guarded_jac):
        result=rollout(env,t['home'],c,reset=False,sequence=phases())
    prefix=lambda trace:[r for r in trace if r['phase'] in ('approach','descend')]
    assert prefix(result['trace'])==prefix(old['trace'])
    if mode=='learned':assert all(result[k]==old[k] for k in result)
    return dict(case=case,method=method,mode=mode,dof=env.n,body=t['body'],handoff=handoff,metrics=metrics,**result)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    prior=json.loads((PREVIOUS/'summary.json').read_text());records=json.loads((PREVIOUS/'records.json').read_text())
    losses=[r for r in prior['changes'] if r['mode']=='route300' and r['change']=='loss'];assert len(losses)==10
    selection=[dict(case=r['case'],method=r['method'],group='loss') for r in losses]
    strata=sorted({(r['method'],r['dof']) for r in losses})
    for method,dof in strata:
        candidates=[r for r in records if r['method']==method and r['dof']==dof and r['mode']=='route300' and r['route_used'] and r['success'] and next(b['success'] for b in records if b['case']==r['case'] and b['method']==method and b['mode']=='baseline')]
        if candidates:selection.append(dict(case=min(r['case'] for r in candidates),method=method,group='retained_success'))
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()));loaded,hashes=models()
    paths=[PREVIOUS/'summary.json',PREVIOUS/'records.json',PRIOR/'cases.json',PRIOR/'axis_catalog.json']+[PREVIOUS/'trials'/f"{s['case']}_{s['method']}_route300.json" for s in selection]+[PATHS/'cases'/f"{s['case']}.json" for s in selection]
    inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E92',selection=selection,inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All ten E91 route300 losses relative to baseline plus earliest retained success per checkpoint/DOF stratum represented among losses, when available. Outcome-selected diagnostic, not generalization evaluation. Two complete rollouts each: learned E91 exact replay, and identical learned/planner prefix followed by exact-Jacobian substitution starting at close (immediately after descent). No state resets or reprobes; same adaptive measured bias and statistic updates. Original remaining budgets and success criteria. One read-only learned-vs-exact local Jacobian audit at handoff.',
        verification='Prefix trace equality to E91, complete learned replay equality, identical handoff physical state and controller sufficient statistics across branches. Exact task Jacobians blocked before handoff except labeled read-only audit. No checkpoint modifications.',
        interpretation='Exact continuation is privileged and changes subsequent trajectories; it tests the local kinematic estimator/controller contribution, not whether history corruption specifically caused failures. Relative Jacobian error is descriptive and may not align with task-relevant action directions. Selected failures and controls cannot estimate overall success rate.',
        videos='Save candidate_0 case0 learned/exact pair and candidate_2 case0 learned/exact pair when selected; if no success or failure represented, append first missing category. Verify full video replays.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('learned_handoff.py','learned_path_transfer.py','novelaxis_path_execution.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for s in selection:
        state=None;metric=None
        for mode in ('learned','exact_after_descent'):
            r=evaluate(s['case'],tasks[s['case']],s['method'],loaded[s['method']],mode)
            if state is None:state=r['handoff'];metric=r['metrics']
            assert state==r['handoff'] and metric==r['metrics']
            (ROOT/'trials'/f"{s['case']}_{s['method']}_{mode}.json").write_text(json.dumps(r))
            rows.append(dict(group=s['group'],**{k:v for k,v in r.items() if k not in ('trace','handoff','metrics')},relative_jacobian_error=r['metrics']['relative_error']))
        print(s,[(r['mode'],r['success'],r['collision']) for r in rows[-2:]],flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=len(selection),identical_handoffs=True,unchanged_hashes=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()

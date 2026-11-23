"""E53: counterfactual actions on fixed measured native histories, no training."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .real_arms import verify_assets
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .native_reliability import rollout


@contextmanager
def cloned_state(env):
    original=env.data
    counters={k:getattr(env,k) for k in ['steps','bad','grasped','lifted']}
    clone=mujoco.MjData(env.model);mujoco.mj_copyData(clone,env.model,original)
    env.data=clone
    try:yield
    finally:
        env.data=original
        for k,v in counters.items():setattr(env,k,v)


def vector(obs):return np.r_[obs['position'],.15*obs['rotation'][:,2]]


def operator(jac):return jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),np.eye(6))


def copy_controller(source,model):
    c=GuardedPose(source.n,model)
    c.gram=source.gram.copy();c.cross=source.cross.copy();c.bias=source.bias.copy()
    return c


def branch(env,controller,error,aperture,horizon=5):
    with cloned_state(env):
        obs=env.observe();start=vector(obs);target=start+error;initial_object=env.data.xpos[env.obj].copy()
        result={};bad=False
        for step in range(horizon):
            jac,bias=controller.predict();err=target-vector(obs)
            motion=operator(jac)@(.35*err);action=np.clip(motion+bias,-.04,.04)
            before=obs;obs,command,info=env.step(action,aperture)
            controller.update(before,obs,command);bad |= bool(info['bad'])
            if step==0 or step==horizon-1:
                result[str(step+1)]=dict(error_squared=float(np.sum((target-vector(obs))**2)),collision=bad,
                    displacement=(vector(obs)-start).tolist(),object_displacement=(env.data.xpos[env.obj]-initial_object).tolist(),
                    pads=len(env.contacts()[0]),command=command.tolist())
        return result


def run(assets,parents,training,reference,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    verify_assets(assets);ref=Path(reference);tasks=json.loads((ref/'cases.json').read_text())
    old=json.loads((ref/'records.json').read_text());models={'analytical':None};hashes={}
    for condition,folder in [('original',parents),('prediction',str(Path(training)/'prediction')),('control',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{condition}_{seed}']=mm[f'neural_{seed}']
    protocol=dict(experiment='E53',checkpoints=hashes,cases_sha256=fingerprint(ref/'cases.json'),reference_sha256=fingerprint(ref/'records.json'),test_evaluated=False,
        sampling='First and twentieth control step of every reached phase on all twelve analytical E46 trajectories. All candidates receive identical measured Gram/cross/bias statistics. No outcome-based sample selection. Exact full analytical replay required.',
        candidates='Analytical and all three original/prediction/control seeds. Frozen weights; no training or native data export to training.',
        counterfactual='Exact mj_copyData branches, restoring all Python flags. Fixed current ramped position/tool-Z reference and aperture for five control steps; online measured estimator/bias updates, real joint clipping and limits. No sequencer advancement. One-step and five-step errors are diagnostic, not task success.',
        proxies='E52 full response-operator MSE; local unclipped residual e-J_true*motion; execution-aware residual e-J_true*(joint-limited command-measured bias). All squared scaled XYZ/tool-Z norms. Privileged geometric Jacobian is diagnostic only and never enters candidate commands.',
        statistics='Pairwise ordering agreement versus measured one/five-step residuals within each common state, grouped by robot/phase; 12 trajectories are clusters, not thousands of independent trials. Contacts reported separately. Native observations are development diagnostics, not independent validation.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['action_objective_audit.py','native_reliability.py','real_arms.py','real_arm_pick_place.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    rows=[];replays=[];repeat_count=0
    real_jac=mujoco.mj_jacSite
    for case,t in enumerate(tasks):
        env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True);driver=GuardedPose(env.n,None)
        def observe(phase,index,obs,error,aperture):
            nonlocal repeat_count
            if index not in [0,19]:return
            # Access privileged geometry only for scores, not prediction/control.
            with patch.object(mujoco,'mj_jacSite',real_jac):truth=env.jacobian()
            ideal=truth@operator(truth);sample=[]
            for name,model in models.items():
                c=copy_controller(driver,model);jac,bias=c.predict();op=operator(jac);motion=op@(.35*error)
                action=np.clip(motion+bias,-.04,.04)
                executed=np.clip(obs['q']+action,env.limits[:,0],env.limits[:,1])-obs['q']
                scores=dict(operator_mse=float(np.mean((truth@op-ideal)**2)),
                    local_unclipped=float(np.sum((error-truth@motion)**2)),
                    local_executed=float(np.sum((error-truth@(executed-bias))**2)))
                result=branch(env,c,error,aperture)
                if name=='analytical':
                    assert result==branch(env,copy_controller(driver,model),error,aperture);repeat_count+=1
                sample.append(dict(method=name,scores=scores,branch=result,clipped_joints=int(np.sum(np.abs(motion+bias)>.04)),
                    command_norm=float(np.linalg.norm(executed)),bias_norm=float(np.linalg.norm(bias))))
            rows.append(dict(case=case,robot=t['robot'],phase=phase,index=index,initial_error_squared=float(error@error),candidates=sample))
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden outside diagnostic')):
            probe=calibrate(env,t['home'],driver,41000+case)
            result=rollout(env,driver,object_release=t['robot']=='panda',before_step_observer=observe)
        expected=next(r for r in old if r['method']=='analytical' and r['case']==case)
        assert probe==expected['probe']
        for key in ['trace','phases','steps','success','failure','final_object']:assert result[key]==expected[key],(case,key)
        replays.append(dict(case=case,exact=True,success=result['success']))
        (root/'trials'/f'case_{case}.json').write_text(json.dumps([r for r in rows if r['case']==case]))
        print('case',case,t['robot'],'exact replay; states',sum(r['case']==case for r in rows),flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows))
    (root/'audit.json').write_text(json.dumps(dict(exact_replays=replays,states=len(rows),branches=10*len(rows),repeat_checks=repeat_count,unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','parents','training','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

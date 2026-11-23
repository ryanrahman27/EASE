"""E54: frozen action-ranking audit on procedural training-arm histories."""
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
from .paired_data import register
from .mixed_sim import Arm,MixedReach
from .motion_transformer import MotionController
from .real_arm_reach import load_models
from .native_reliability import bounded_error
from .action_objective_report import agreement


def operator(jac):
    return jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(3),np.eye(3))


class Estimator(MotionController):
    def __init__(self,model,n):
        if model is not None:super().__init__(model,n)
        else:
            self.model=None;self.dof=n;self.online=True;self.gram=1e-6*np.eye(n)
            self.cross=np.zeros((n,3));self.bias=np.zeros(n);self.cached_jac=None

    def predict(self):
        if self.model is None:return np.linalg.solve(self.gram,self.cross).T
        g=torch.zeros((1,7,7),device=self.device);c=torch.zeros((1,7,3),device=self.device)
        g[0,:self.dof,:self.dof]=torch.tensor(self.gram,device=self.device,dtype=torch.float32)
        c[0,:self.dof]=torch.tensor(self.cross,device=self.device,dtype=torch.float32)
        mask=torch.arange(7,device=self.device)[None]<self.dof
        with torch.no_grad(),patch.object(np.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
            return self.model(g,c,mask)[0,:self.dof].cpu().numpy().T.astype(float)


def copied(source,model):
    c=Estimator(model,source.dof);c.gram=source.gram.copy();c.cross=source.cross.copy();c.bias=source.bias.copy()
    return c


@contextmanager
def clone(env):
    original=env.data;state=(env.steps,env.streak,env.collision,env.goal.copy())
    env.data=mujoco.MjData(env.model);mujoco.mj_copyData(env.data,env.model,original)
    try:yield
    finally:
        env.data=original;env.steps,env.streak,env.collision,env.goal=state


def reference_step(reference,goals,step):
    return reference+bounded_error(np.asarray(goals[(step//50)%2]),reference,.002)


def execute(env,c,reference,goals,step):
    reference=reference_step(reference,goals,step);env.goal=reference.copy()
    obs=env.observe();error=obs['task'][3:]-obs['task'][:3]
    jac=c.predict();motion=operator(jac)@(.5*error)
    action=np.clip(motion+c.bias,-.04,.04)
    after,_,info=env.step(action);c.update(obs,after,info['executed_action'])
    return reference,dict(error_squared=float(np.sum((reference-after['task'][:3])**2)),collision=bool(info['collision']),
        command=info['executed_action'].tolist(),position=after['task'][:3].tolist(),q=after['joints'][:,0].tolist())


def branch(env,c,reference,goals,step):
    with clone(env):
        result={}
        for offset in range(5):
            reference,info=execute(env,c,reference,goals,step+offset)
            if offset in [0,4]:result[str(offset+1)]=info
        return result


def run(data,parents,training,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);arms={a['seed']:Arm(**a) for a in manifest['arms'] if a['split']=='train'}
    tasks=[]
    for pair in manifest['pairs']:
        if pair['split']!='train':continue
        goals=sorted([g for g in manifest['goals']['train'] if g['pair']==pair['id']],key=lambda g:g['id'])
        for body in pair['bodies']:
            assert body in arms
            tasks.append(dict(body=body,goals=[goals[0]['xyz'],goals[-1]['xyz']],goal_ids=[goals[0]['id'],goals[-1]['id']]))
    assert len(tasks)==8
    models={'analytical':None};hashes={}
    for condition,folder in [('original',parents),('prediction',str(Path(training)/'prediction')),('control',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{condition}_{seed}']=mm[f'neural_{seed}']
    protocol=dict(experiment='E54',checkpoints=hashes,manifest_sha256=fingerprint(Path(data)/'manifest.json'),test_evaluated=False,tasks=tasks,
        scope='Training-split procedural arms only. Position-only reaching, not native XYZ/tool-Z manipulation. Frozen models; no training, selection or native data.',
        drivers=list(models),horizon=150,sample_steps=[9,29,49,50,79,99,100,129,149],
        targets='First/last training goals per paired body; alternate every 50 steps with .002 m reference ramp. Cold start, no probes; each of ten drivers produces its own online history. Stop driver at first collision; retain missing later states and failures. Ignore legacy static-goal success termination because task is scheduled tracking.',
        branches='Ten candidates per fixed state, same driver Gram/cross/bias; five steps of continuing scheduled moving reference, online updates and native procedural clipping/limits. Record contact, one/five-step errors. Copy all simulator data and Python state; exact driver replay without branching required.',
        proxies='E53 operator, local unclipped and local executed-command scores generalized to 3D, gain .5 and limits +/-2.5. Privileged geometric Jacobian diagnostic-only; forbidden in candidate inference.',
        statistics='Case-macro pairwise ranking, stratified by DoF and driver family; eight bodies are clusters. Record collision-free pairs separately. No independent task-success claims.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['procedural_action_audit.py','mixed_sim.py','motion_transformer.py','action_objective_report.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    rows=[];drivers=[];exact_branches=0
    real_jac=mujoco.mj_jacSite
    for task in tasks:
        arm=arms[task['body']]
        for driver_name,driver_model in models.items():
            env=MixedReach(arm,horizon=1000,synchronize=True);obs=env.reset(0,task['goals'][0],spread=0)
            reference=env.data.site_xpos[0].copy();controller=Estimator(driver_model,arm.dof);trace=[]
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                for step in range(150):
                    obs=env.observe();target=reference_step(reference,task['goals'],step)
                    error=target.astype(np.float32)-obs['task'][:3]
                    selected=step in protocol['sample_steps'];driver_branch=None
                    if selected:
                        truth=np.zeros((3,arm.dof));real_jac(env.model,env.data,truth,None,0)
                        ideal=truth@operator(truth);candidates=[]
                        for name,model in models.items():
                            c=copied(controller,model);jac=c.predict();op=operator(jac);motion=op@(.5*error)
                            command=np.clip(env.data.qpos+np.clip(motion+c.bias,-.04,.04),-2.5,2.5)-env.data.qpos
                            clipped=int(np.sum(np.abs(motion+c.bias)>.04))
                            scores=dict(operator_mse=float(np.mean((truth@op-ideal)**2)),local_unclipped=float(np.sum((error-truth@motion)**2)),local_executed=float(np.sum((error-truth@(command-c.bias))**2)))
                            result=branch(env,c,reference.copy(),task['goals'],step)
                            if name==driver_name:driver_branch=result['1']
                            candidates.append(dict(method=name,scores=scores,branch=result,clipped_joints=clipped))
                        rows.append(dict(case=arm.seed,robot=f'{arm.dof}dof',driver=driver_name,phase='tracking',index=step,initial_error_squared=float(error@error),candidates=candidates))
                    reference,info=execute(env,controller,reference,task['goals'],step);trace.append(info)
                    if selected:assert driver_branch==info;exact_branches+=1
                    if info['collision']:break
                # A completely fresh no-branch replay must match every driver step.
                replay=MixedReach(arm,horizon=1000,synchronize=True);replay.reset(0,task['goals'][0],spread=0)
                replay_ref=replay.data.site_xpos[0].copy();c=Estimator(driver_model,arm.dof)
                for step,expected in enumerate(trace):
                    replay_ref,actual=execute(replay,c,replay_ref,task['goals'],step);assert actual==expected
            record=dict(body=arm.seed,dof=arm.dof,driver=driver_name,steps=len(trace),collision=trace[-1]['collision'],exact_replay=True,samples=sum(r['case']==arm.seed and r['driver']==driver_name for r in rows),trace=trace)
            drivers.append(record);(root/'trials'/f'{arm.seed}_{driver_name}.json').write_text(json.dumps(dict(driver=record,samples=[r for r in rows if r['case']==arm.seed and r['driver']==driver_name])))
            print(arm.seed,driver_name,'steps',len(trace),'collision',record['collision'],'exact',flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows));(root/'drivers.json').write_text(json.dumps(drivers))
    (root/'audit.json').write_text(json.dumps(dict(driver_replays=len(drivers),exact_first_steps=exact_branches,states=len(rows),branches=10*len(rows),unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','parents','training','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

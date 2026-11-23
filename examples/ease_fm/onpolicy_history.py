"""E101: fixed student-collected histories versus matched parent replay."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import probe_distillation as prior
from . import task_retention_training as e99
from .local_probe import ProbeArm,probe_rollout
from .local_history import LocalPose
from .arm_pick_place import targets,ROT_SCALE,rotation_error
from .pick_place_control import calibrate
from .mixed_sim import AXES,Arm
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer
from .control_training import batch,predict
from .estimator_distillation import analytical_targets

ROOT=Path('outputs/ease_onpolicy_history_v1')
TASKS=Path('outputs/ease_arm_pick_place_v2')
CONDITIONS=('replay','onpolicy')
previous=prior.previous


def action_tensor(pred,error,bias,mask):
    correction=pred@torch.linalg.solve(pred.transpose(1,2)@pred+.015**2*torch.eye(6,device=pred.device,dtype=pred.dtype)[None],(.35*error)[:,:,None])
    return (correction[:,:,0]+bias).clamp(-.04,.04).masked_fill(~mask,0)


def prepare():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();(ROOT/'training_trials').mkdir()
    source=json.loads((prior.ROOT/'protocol.json').read_text());audit=json.loads((prior.ROOT/'data_audit.json').read_text());assert fingerprint(prior.ROOT/'contexts.npz')==audit['contexts_sha256']
    tasks=json.loads((TASKS/'cases.json').read_text());AXES.update(json.loads((TASKS/'axis_catalog.json').read_text()));parents,hashes=previous.models()
    training_bodies=set(source['training_bodies']);assert len(tasks)==12 and all(t['arm']['split']=='train' and t['arm']['seed'] in training_bodies for t in tasks)
    drivers,driver_hashes=e99.load_students();hashes.update(driver_hashes)
    inputs=[prior.ROOT/'contexts.npz',prior.ROOT/'provenance.json',e99.ROOT/'task_contexts.npz',e99.ROOT/'provenance.json',TASKS/'cases.json',TASKS/'axis_catalog.json']
    spec=dict(experiment='E101',inputs={str(p):fingerprint(p) for p in inputs},parents=hashes,training_bodies=sorted(training_bodies),test_evaluated=False,
        data='36 full trajectories on the same12 original training placements, original E66 parent prefix and32-step probe. Fixed E99 retained student drives post-probe continuation. All failures/contacts retained. Train only post-probe contexts; third placement per body held out across every driver. This is one collection round from the starting student, not iterative on-policy aggregation. No evaluation-arm data.',
        training='Both conditions start E99 retained checkpoint for each seed. Fixed2000 updates, AdamW lr2e-5, clip1, batch128:64 synthetic+32 local+16 parent-task+16 extra-task. Replay extra-task comes from E99 parent contexts; onpolicy extra-task comes from newly collected student contexts. Same synthetic/local/parent rows and identical extra-index uniform draws per pair. Analytical ridge fit and synthetic E66 parent anchor (six2/seven.5) plus .01 action retention against E66 on all32 task rows in BOTH conditions. Final checkpoints only. Extra source distribution is the only training intervention.',
        evaluation='All16 development tasks x3 seeds x original parent/replay/onpolicy=144 full tasks, unchanged original-parent/planner/probe prefix. Also compare saved E99 retained starting checkpoints. Strict gate: improve pooled success, nonincreasing collisions, retain every original parent AND starting-student success, no seed success decline against either reference. Report all outcomes, no seed/DOF selection.',
        videos='Cases2 seed0,10 seed2,15 seed2 allthree modes; append first regression against either reference for each condition with corresponding reference video. Include success and failure. Full replay/hash/frame verification.',
        limits='Training tasks cover four original training bodies, local data eight. Privileged geometry/route planning at evaluation. Reused development tasks, no final test, no broad policy promotion.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    samples=[];provenance=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            env=ProbeArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.route=None;env.path_log=[];env.waypoint=0;env.probe_step=0
            class Recorder(LocalPose):
                def predict(self):
                    obs=env.observe();error=np.r_[env.goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])];g,c,m=prior.pack(self);b=np.zeros(7,np.float32);b[:self.n]=self.bias
                    samples.append((g,c,m,error.astype('float32'),b));provenance.append(dict(case=case,body=t['arm']['seed'],driver=seed,phase=env.phase,step=env.steps,validation=case%3==2,eligible=self.local and env.phase!='probe'))
                    saved=self.model
                    if self.local and env.phase!='probe':self.model=drivers[('retained',seed)]
                    try:return super().predict()
                    finally:self.model=saved
            c=Recorder(env.n,parents[f'candidate_{seed}']);calibration=calibrate(env,t['home'],c,38000+case);env.controller=c;env.filter_enabled=True;env.filter_log=[]
            def phases():
                for name,goal,aperture,budget in targets(env.xy,env.target):
                    env.phase=name;env.goal=goal
                    if name=='close':
                        c.local=True;c.rebuild();env.anchor=env.observe()['q'].copy();env.phase='probe';env.goal=env.observe()['position'].copy()
                        yield 'probe',env.goal,0.,32
                        assert env.probe_step==32;env.phase=name;env.goal=goal
                    yield name,goal,aperture,budget
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('No task Jacobians in data generation')):r=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
            (ROOT/'training_trials'/f'{case}_{seed}.json').write_text(json.dumps(dict(case=case,driver=seed,validation=case%3==2,calibration=calibration,**r)))
        print('training data case',case,'complete',flush=True)
    for p,h in {**spec['inputs'],**hashes}.items():assert fingerprint(p)==h
    arrays={key:np.stack([s[i] for s in samples]) for i,key in enumerate(('g','c','mask','error','bias'))};arrays['validation']=np.array([r['validation'] for r in provenance]);arrays['eligible']=np.array([r['eligible'] for r in provenance])
    np.savez_compressed(ROOT/'task_contexts.npz',**arrays);(ROOT/'provenance.json').write_text(json.dumps(provenance))
    audit=dict(rows=len(samples),train=int((~arrays['validation']&arrays['eligible']).sum()),validation=int((arrays['validation']&arrays['eligible']).sum()),trajectories=36,task_bodies=sorted({t['arm']['seed'] for t in tasks}),contexts_sha256=fingerprint(ROOT/'task_contexts.npz'),provenance_sha256=fingerprint(ROOT/'provenance.json'))
    (ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='data_ready')))


def train():
    torch.set_num_threads(4);spec=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text());assert fingerprint(ROOT/'task_contexts.npz')==audit['contexts_sha256']
    values={}
    for name,path in [('local',prior.ROOT/'contexts.npz'),('task',e99.ROOT/'task_contexts.npz'),('onpolicy',ROOT/'task_contexts.npz')]:
        with np.load(path) as f:v={k:torch.tensor(f[k],device='cuda') for k in f.files}
        eligible=v.get('eligible',torch.ones_like(v['validation']));v['train_idx']=torch.where(~v['validation']&eligible)[0];v['val_idx']=torch.where(v['validation']&eligible)[0];v['target']=analytical_targets(v['g'],v['c'],v['mask']);values[name]=v
        for i in range(len(v['target'])):
            n=int(v['mask'][i].sum());np.testing.assert_allclose(v['target'][i,:n].cpu().numpy(),np.linalg.solve(v['g'][i,:n,:n].cpu().numpy().astype(float),v['c'][i,:n].cpu().numpy().astype(float)),rtol=1e-5,atol=1e-6)
    vg=torch.Generator(device='cuda').manual_seed(99999);synthetic=batch(512,'cuda',vg)
    for v in values.values():v['held']=v['val_idx'][torch.randperm(len(v['val_idx']),device='cuda',generator=vg)[:512]]
    for seed in range(3):
        torch.manual_seed(99000+seed);teacher=MotionTransformer().cuda();teacher.load_state_dict(torch.load(prior.PARENTS/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        models={};opts={};history={k:[] for k in CONDITIONS}
        for condition in CONDITIONS:
            m=MotionTransformer().cuda();m.load_state_dict(torch.load(e99.ROOT/'retained'/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));models[condition]=m;opts[condition]=torch.optim.AdamW(m.parameters(),lr=2e-5)
        def metrics(m):
            was=m.training;m.eval();out={}
            with torch.no_grad():
                for name,v in values.items():
                    ix=v['held'];mask=v['mask'][ix];pred=predict(m,v['g'][ix],v['c'][ix],mask);out[name+'_mse']=float(((pred-v['target'][ix]).square()*mask[:,:,None]).sum()/(6*mask.sum()))
                    if name in ('task','onpolicy'):
                        old=predict(teacher,v['g'][ix],v['c'][ix],mask);out[name+'_action_drift']=float(((action_tensor(pred,v['error'][ix],v['bias'][ix],mask)-action_tensor(old,v['error'][ix],v['bias'][ix],mask)).square()*mask).sum()/(mask.sum()*.04**2))
                g,c,mask,t,_=synthetic;out['synthetic_mse']=float(((predict(m,g,c,mask)-t).square()*mask[:,:,None]).sum()/(6*mask.sum()))
            m.train(was);return out
        initial=metrics(models['replay']);gen=torch.Generator(device='cuda').manual_seed(99100+seed);digest=hashlib.sha256()
        for step in range(2000):
            ids={name:v['train_idx'][torch.randint(len(v['train_idx']),(32 if name=='local' else 16,),device='cuda',generator=gen)] for name,v in values.items() if name!='onpolicy'}
            uniform=torch.rand(16,device='cuda',generator=gen);sg,sc,sm,st,_=batch(64,'cuda',gen)
            for x in (sg,sc,sm,st,ids['local'],ids['task'],uniform):digest.update(x.detach().cpu().numpy().tobytes())
            for condition,m in models.items():
                extra=values['task' if condition=='replay' else 'onpolicy'];ei=extra['train_idx'][(uniform*len(extra['train_idx'])).long()]
                chunks=[(values['local'],ids['local']),(values['task'],ids['task']),(extra,ei)]
                g=torch.cat([sg,*[v['g'][ix] for v,ix in chunks]]);c=torch.cat([sc,*[v['c'][ix] for v,ix in chunks]]);mask=torch.cat([sm,*[v['mask'][ix] for v,ix in chunks]]);wanted=torch.cat([st,*[v['target'][ix] for v,ix in chunks]])
                error=torch.cat([v['error'][ix] for v,ix in chunks[1:]]);bias=torch.cat([v['bias'][ix] for v,ix in chunks[1:]]);tm=mask[96:]
                with torch.no_grad():old=predict(teacher,g,c,mask);old_action=action_tensor(old[96:],error,bias,tm)
                pred=predict(m,g,c,mask);fit=((pred-wanted).square()*mask[:,:,None]).sum()/(6*mask.sum());weight=torch.where(sm.sum(-1)==6,2.,.5)[:,None,None];anchor=((pred[:64]-old[:64]).square()*sm[:,:,None]*weight).sum()/(6*sm.sum())
                retain=((action_tensor(pred[96:],error,bias,tm)-old_action).square()*tm).sum()/(tm.sum()*.04**2);loss=fit+anchor+.01*retain
                assert torch.isfinite(loss);opts[condition].zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opts[condition].step();history[condition].append([float(fit.detach()),float(anchor.detach()),float(retain.detach())])
            if (step+1)%500==0:print('train',seed,step+1,{k:v[-1] for k,v in history.items()},flush=True)
        for condition,m in models.items():
            folder=ROOT/condition/f'seed_{seed}';folder.mkdir(parents=True);m.eval();torch.save(m.state_dict(),folder/'policy.pt');meta=dict(seed=seed,condition=condition,steps=2000,initial=initial,final=metrics(m),losses=history[condition],batch_sha256=digest.hexdigest(),policy_sha256=fingerprint(folder/'policy.pt'),starting_sha256=fingerprint(e99.ROOT/'retained'/f'seed_{seed}/policy.pt'),parent_sha256=fingerprint(prior.PARENTS/f'seed_{seed}/policy.pt'));(folder/'training.json').write_text(json.dumps(meta,indent=2));print('trained',seed,condition,meta['final'],flush=True)
    for p,h in {**spec['inputs'],**spec['parents']}.items():assert fingerprint(p)==h
    (ROOT/'status.json').write_text(json.dumps(dict(status='trained')))


def evaluate(case,task,method,parent,student,mode,record_step=None):
    r=prior.evaluate(case,task,method,parent,student,'parent' if mode=='parent' else 'student',record_step);r['mode']=mode;return r


def load_students():
    models={};hashes={}
    for condition in CONDITIONS:
        for seed in range(3):
            folder=ROOT/condition/f'seed_{seed}';meta=json.loads((folder/'training.json').read_text());assert fingerprint(folder/'policy.pt')==meta['policy_sha256'];m=MotionTransformer().cuda();m.load_state_dict(torch.load(folder/'policy.pt',map_location='cuda',weights_only=True));models[(condition,seed)]=m.eval();hashes[str(folder/'policy.pt')]=meta['policy_sha256']
    return models,hashes


def evaluation():
    torch.set_num_threads(4);assert json.loads((ROOT/'status.json').read_text())['status']=='trained';(ROOT/'trials').mkdir()
    tasks=json.loads((previous.PRIOR/'cases.json').read_text());AXES.update(json.loads((previous.PRIOR/'axis_catalog.json').read_text()));parents,ph=previous.models();students,hashes=load_students();assert set(json.loads((ROOT/'protocol.json').read_text())['training_bodies']).isdisjoint(t['arm']['seed'] for t in tasks)
    (ROOT/'evaluation_protocol.json').write_text(json.dumps(dict(checkpoints=hashes,parents=ph,cases_sha256=fingerprint(previous.PRIOR/'cases.json'),test_evaluated=False),indent=2));rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            method=f'candidate_{seed}';handoff=None
            for mode in ('parent',*CONDITIONS):
                r=evaluate(case,t,method,parents[method],students[(mode if mode!='parent' else 'replay',seed)],mode)
                if handoff is None:handoff=r['continuation_handoff']
                assert handoff==r['continuation_handoff'];(ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','handoff','continuation_handoff')})
        print('evaluation case',case,'complete',flush=True)
    for p,h in {**ph,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate']);a=p.parse_args();{'prepare':prepare,'train':train,'evaluate':evaluation}[a.stage]()

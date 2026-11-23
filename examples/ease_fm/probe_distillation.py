"""E98: training-only local histories, fixed-budget neural continuation distillation."""
import argparse
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .benchmark import fingerprint
from .mixed_sim import Arm,AXES
from .paired_data import register
from .arm_pick_place import PickPlace
from .local_history import LocalPose
from .local_probe import probe_offset
from .motion_transformer import MotionTransformer
from .control_training import batch,predict
from .estimator_distillation import analytical_targets
from . import probe_continuation as previous

ROOT=Path('outputs/ease_probe_distillation_v1')
DATA=Path('outputs/ease_paired_factorial_v2')
PARENTS=Path('outputs/ease_estimator_distillation_v1/analytical')


def pack(c):
    g=np.zeros((7,7));cross=np.zeros((7,6));g[:c.n,:c.n]=c.gram;cross[:c.n]=c.cross
    return g.astype('float32'),cross.astype('float32'),np.arange(7)<c.n


def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();manifest=register(DATA)
    arms=[Arm(**a) for a in manifest['arms'] if a['split']=='train'];assert len(arms)==8
    hashes={str(PARENTS/f'seed_{s}/policy.pt'):fingerprint(PARENTS/f'seed_{s}/policy.pt') for s in range(3)}
    spec=dict(experiment='E98',training_manifest_sha256=fingerprint(DATA/'manifest.json'),parents=hashes,training_bodies=[a.seed for a in arms],test_evaluated=False,
        data='Eight manifest training bodies only,16 scripted physical trajectories each. Initial q=arm.home+Uniform(-.6,.6), RNG980000+body_index*100+trajectory.32 anchor-settling steps,32 fixed E95 joint probe steps (+/-.004rad,2steps/sign),96 rank-one sinusoidal target steps (amplitude.08rad, period24*pi steps). Feedback target-q+measured bias clipped.04, open fingers, unchanged physical simulation. Local32 updates throughout. Save every pre-action context at steps32..159:16384 rows. Retain contacts and all trajectories; no success filtering. No task Jacobians or target-embodiment histories. Trajectories0..11 train,12..15 validation on same bodies.',
        training='Start all three E66 analytical checkpoints; 2000 AdamW updates per seed,lr2e-5,batch128,clip gradient1. Each batch64 measured training contexts and64 synthetic E52 replay contexts. Measured target is float64 ridge solve of the exact input Gram/cross, cast float32. Fit MSE on all rows plus existing six-weight2/seven-weight.5 parent anchor on synthetic rows only. No anchor on new measured rows. Fixed final checkpoint; no seed/step selection or parameter tuning.',
        evaluation='All16 E97 tasks x3 candidate parents x parent/student continuation =96 full tasks, irrespective of training metrics. Frozen original E66 prefix and same physical probe; swap only model weights at first post-probe close prediction on admitted routes. No-route behavior remains original. Compare to exact E97 candidate references and report every gain/loss. Adoption gate: strictly improve pooled success, no increased collisions, retain EVERY parent success for both DOFs and no per-seed success decrease. No reserved-test evaluation.',
        videos='Prespecified candidate_0 cases2 (E95 six-joint failure),4 (difficult retained success),10 (E97 analytical regression), parent/student pairs. Append first new student regression with both modes and first missing success/failure category if needed. Exact serialized replay checks.',
        limitations='New data are scripted local motions rather than full pick/place training rollouts. Validation is trajectory holdout on training bodies. Evaluation is reused development data with privileged planning and frozen parent prefix; not whole-policy retraining or independent generalization.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    samples=[];provenance=[];contact_steps=0
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('No Jacobians during training data generation')):
        for bi,arm in enumerate(arms):
            for episode in range(16):
                rng=np.random.default_rng(980000+bi*100+episode);env=PickPlace(arm);anchor=np.array(arm.home)+rng.uniform(-.6,.6,arm.dof);obs=env.reset(anchor)
                c=LocalPose(arm.dof,None);c.local=True;direction=rng.normal(size=arm.dof);direction/=max(abs(direction))
                for step in range(160):
                    if step>=32:
                        samples.append(pack(c));provenance.append(dict(body=arm.seed,episode=episode,step=step,validation=episode>=12,collision_so_far=bool(env.bad)))
                    offset=np.zeros(arm.dof) if step<32 else probe_offset(step-32,arm.dof) if step<64 else .08*np.sin((step-64)/12)*direction
                    action=np.clip(anchor+offset-obs['q']+c.bias,-.04,.04)
                    after,command,info=env.step(action,0.);c.update(obs,after,command);obs=after;contact_steps+=bool(info['bad'])
            print('data body',arm.seed,'complete',flush=True)
    g,c,mask=[np.stack([r[i] for r in samples]) for i in range(3)];validation=np.array([r['validation'] for r in provenance])
    assert len(samples)==16384 and validation.sum()==4096
    np.savez_compressed(ROOT/'contexts.npz',g=g,c=c,mask=mask,validation=validation)
    (ROOT/'provenance.json').write_text(json.dumps(provenance));(ROOT/'data_audit.json').write_text(json.dumps(dict(rows=len(samples),train=int((~validation).sum()),validation=int(validation.sum()),contact_steps=contact_steps,contexts_sha256=fingerprint(ROOT/'contexts.npz'),provenance_sha256=fingerprint(ROOT/'provenance.json')),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='data_ready')))


def train():
    torch.set_num_threads(4);spec=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text());assert fingerprint(ROOT/'contexts.npz')==audit['contexts_sha256']
    with np.load(ROOT/'contexts.npz') as f:values={k:torch.tensor(f[k],device='cuda') for k in f.files}
    train_idx=torch.where(~values['validation'])[0];val_idx=torch.where(values['validation'])[0]
    target=analytical_targets(values['g'],values['c'],values['mask'])
    # Check every target independently in NumPy on the active joint block.
    for i in range(len(target)):
        n=int(values['mask'][i].sum());g=values['g'][i,:n,:n].cpu().numpy().astype(float);c=values['c'][i,:n].cpu().numpy().astype(float)
        np.testing.assert_allclose(target[i,:n].cpu().numpy(),np.linalg.solve(g,c),rtol=1e-5,atol=1e-6)
    vg=torch.Generator(device='cuda').manual_seed(98999);held=val_idx[torch.randperm(len(val_idx),device='cuda',generator=vg)[:512]];synthetic=batch(512,'cuda',vg)
    for seed in range(3):
        torch.manual_seed(98000+seed);teacher=MotionTransformer().cuda();teacher.load_state_dict(torch.load(PARENTS/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));teacher.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        model=MotionTransformer().cuda();model.load_state_dict(teacher.state_dict());optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        def metrics(m):
            was=m.training;m.eval()
            with torch.no_grad():
                pred=predict(m,values['g'][held],values['c'][held],values['mask'][held]);mask=values['mask'][held]
                measured=float(((pred-target[held]).square()*mask[:,:,None]).sum()/(6*mask.sum()))
                g,c,mask,t,_=synthetic;replay=float(((predict(m,g,c,mask)-t).square()*mask[:,:,None]).sum()/(6*mask.sum()))
            m.train(was);return dict(held_trajectory_mse=measured,synthetic_mse=replay)
        initial=metrics(teacher);gen=torch.Generator(device='cuda').manual_seed(98100+seed);losses=[]
        for step in range(2000):
            idx=train_idx[torch.randint(len(train_idx),(64,),device='cuda',generator=gen)];sg,sc,sm,st,_=batch(64,'cuda',gen)
            g=torch.cat([sg,values['g'][idx]]);c=torch.cat([sc,values['c'][idx]]);mask=torch.cat([sm,values['mask'][idx]]);wanted=torch.cat([st,target[idx]])
            with torch.no_grad():old=predict(teacher,sg,sc,sm)
            pred=predict(model,g,c,mask);fit=((pred-wanted).square()*mask[:,:,None]).sum()/(6*mask.sum())
            weight=torch.where(sm.sum(-1)==6,2.,.5)[:,None,None];anchor=((pred[:64]-old).square()*sm[:,:,None]*weight).sum()/(6*sm.sum());loss=fit+anchor
            assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append([float(fit.detach()),float(anchor.detach())])
            if (step+1)%500==0:print('train',seed,step+1,losses[-1],flush=True)
        folder=ROOT/f'seed_{seed}';folder.mkdir();model.eval();torch.save(model.state_dict(),folder/'policy.pt');meta=dict(seed=seed,steps=2000,initial=initial,final=metrics(model),losses=losses,policy_sha256=fingerprint(folder/'policy.pt'),parent_sha256=fingerprint(PARENTS/f'seed_{seed}/policy.pt'))
        (folder/'training.json').write_text(json.dumps(meta,indent=2));print('trained',seed,meta['final'],flush=True)
    for p,h in spec['parents'].items():assert fingerprint(p)==h
    (ROOT/'status.json').write_text(json.dumps(dict(status='trained')))


def evaluate(case,task,method,parent,student,mode,record_step=None):
    original=previous.continuation_prediction
    def prediction(c,env,branch,active,jac):
        if not active or mode=='parent':return original(c,env,'learned',active,jac)
        saved=c.model;c.model=student
        try:return LocalPose.predict(c)
        finally:c.model=saved
    with patch.object(previous,'continuation_prediction',prediction):r=previous.evaluate(case,task,method,parent,'analytical',record_step)
    if mode=='parent':
        old=json.loads((previous.ROOT/'trials'/f'{case}_{method}_learned.json').read_text());old['mode']='analytical';assert json.loads(json.dumps(r))==old
    r['mode']=mode;return r


def evaluation():
    torch.set_num_threads(4);assert json.loads((ROOT/'status.json').read_text())['status']=='trained';(ROOT/'trials').mkdir()
    tasks=json.loads((previous.PRIOR/'cases.json').read_text());AXES.update(json.loads((previous.PRIOR/'axis_catalog.json').read_text()));parents,parent_hashes=previous.models();students={};hashes={}
    for seed in range(3):
        folder=ROOT/f'seed_{seed}';meta=json.loads((folder/'training.json').read_text());assert fingerprint(folder/'policy.pt')==meta['policy_sha256'];m=MotionTransformer().cuda();m.load_state_dict(torch.load(folder/'policy.pt',map_location='cuda',weights_only=True));students[seed]=m.eval();hashes[str(folder/'policy.pt')]=meta['policy_sha256']
    bodies=set(json.loads((ROOT/'protocol.json').read_text())['training_bodies']);assert bodies.isdisjoint(t['arm']['seed'] for t in tasks)
    (ROOT/'evaluation_protocol.json').write_text(json.dumps(dict(checkpoints=hashes,parent_checkpoints=parent_hashes,cases_sha256=fingerprint(previous.PRIOR/'cases.json'),training_body_disjoint=True,scope='Fixed final checkpoints, post-probe substitution only;96 full tasks; no selection; original E97 prefixes.',test_evaluated=False),indent=2))
    rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            method=f'candidate_{seed}';handoff=None
            for mode in ('parent','student'):
                r=evaluate(case,t,method,parents[method],students[seed],mode)
                if handoff is None:handoff=r['continuation_handoff']
                assert handoff==r['continuation_handoff'];(ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','handoff','continuation_handoff')})
        print('evaluation case',case,'complete',flush=True)
    for p,h in {**hashes,**parent_hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','train','evaluate']);args=parser.parse_args();{'prepare':prepare,'train':train,'evaluate':evaluation}[args.stage]()

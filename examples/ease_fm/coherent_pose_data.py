"""E57: fixed measured procedural contexts, exact E55 driver replays."""
import argparse
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
from .real_arm_reach import load_models
from .retention_fresh import GuardedPose
from .procedural_pose_audit import observe,vector,reference_step,execute,direction_jacobian


def pack(controller,obs,error,truth):
    n=controller.n;g=np.zeros((7,7));c=np.zeros((7,6));j=np.zeros((7,6));b=np.zeros(7);q=np.zeros(7)
    g[:n,:n]=controller.gram;c[:n]=controller.cross;j[:n]=truth.T;b[:n]=controller.bias;q[:n]=obs['q']
    return dict(g=g.astype('float32'),c=c.astype('float32'),mask=np.arange(7)<n,j=j.astype('float32'),
        error=error.astype('float32'),bias=b.astype('float32'),q=q.astype('float32'))


def run(data,reference,parents,training,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    ref=Path(reference);p=json.loads((ref/'protocol.json').read_text());old=json.loads((ref/'drivers.json').read_text())
    manifest=register(data);arms={a['seed']:Arm(**a) for a in manifest['arms'] if a['split']=='train'}
    assert fingerprint(Path(data)/'manifest.json')==p['manifest_sha256']
    models={'analytical':None};hashes={}
    for condition,folder in [('original',parents),('prediction',str(Path(training)/'prediction')),('control',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{condition}_{seed}']=mm[f'neural_{seed}']
    assert hashes==p['checkpoints']
    protocol=dict(experiment='E57 data',source=reference,source_drivers_sha256=fingerprint(ref/'drivers.json'),
        manifest_sha256=p['manifest_sha256'],checkpoints=hashes,test_evaluated=False,
        split='All E55 drivers on eight training bodies. Entire trajectories from original_2/prediction_2/control_2 are validation; all other drivers train. Same bodies/goals, so this is trajectory holdout, not embodiment generalization.',
        labels='Current geometric XYZ/scaled-tool-Z Jacobian from procedural simulator, training labels only. Measured history statistics, error, bias and joint position come from the same actual state. No native data. This changes label semantics from synthetic clean-ridge targets as well as context distribution; cross-E56 comparisons cannot isolate coherence alone.',
        selection='Every pre-step state, including unsuccessful prefixes; exact source replay required. No success filtering or target-arm adaptation.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['coherent_pose_data.py','procedural_pose_audit.py','pick_place_control.py','mixed_sim.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    samples=[];splits=[];provenance=[];audits=[];real_jac=mujoco.mj_jacSite
    for task in p['tasks']:
        arm=arms[task['body']]
        for name,model in models.items():
            expected=next(r for r in old if r['body']==arm.seed and r['driver']==name)
            env=MixedReach(arm,horizon=1000,synchronize=True);env.reset(0,task['goals'][0][:3],spread=0)
            ref_pose=vector(observe(env));controller=GuardedPose(arm.dof,model)
            validation=name.endswith('_2')
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged labels only')):
                for step,wanted in enumerate(expected['trace']):
                    obs=observe(env);error=reference_step(ref_pose,task['goals'],step)-vector(obs)
                    jp=np.zeros((3,arm.dof));jr=np.zeros_like(jp);real_jac(env.model,env.data,jp,jr,0)
                    truth=np.concatenate([jp,.15*direction_jacobian(obs['rotation'],jr)])
                    samples.append(pack(controller,obs,error,truth));splits.append(validation)
                    provenance.append(dict(body=arm.seed,driver=name,step=step,validation=validation))
                    ref_pose,actual=execute(env,controller,ref_pose,task['goals'],step);assert actual==wanted
            audits.append(dict(body=arm.seed,driver=name,steps=len(expected['trace']),exact=True,validation=validation))
            print(arm.seed,name,'exact',len(expected['trace']),flush=True)
    arrays={key:np.stack([r[key] for r in samples]) for key in samples[0]};arrays['validation']=np.array(splits)
    np.savez_compressed(root/'contexts.npz',**arrays)
    (root/'provenance.json').write_text(json.dumps(provenance))
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'audit.json').write_text(json.dumps(dict(rows=len(samples),train=sum(not s for s in splits),validation=sum(splits),exact_replays=audits,
        contexts_sha256=fingerprint(root/'contexts.npz'),provenance_sha256=fingerprint(root/'provenance.json'),test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','reference','parents','training','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

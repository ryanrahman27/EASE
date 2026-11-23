"""E46: prespecified fresh native placements, old/new matched checkpoints."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .real_arms import verify_assets
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace,endpoint_ik
from .pick_place_control import PoseController,calibrate
from .native_reliability import sequence,rollout
from .benchmark import fingerprint


class GuardedPose(PoseController):
    def predict(self):
        if self.model is None:return super().predict()
        with patch.object(np.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
            return super().predict()


def prepare(assets,old,new,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);manifest=verify_assets(assets)
    candidates={name:[(x,y) for x in xs for y in [-.02,.05]] for name,xs in [('piper',[.21,.24,.27]),('panda',[.32,.37,.42])]}
    protocol=dict(experiment='E46',old=old,new=new,candidates=candidates,asset_commit=manifest['commit'],
        ik_seed=45000,probe_seed=41000,transfer='Y +.04 for initial Y=-.02; -.04 for Y=.05',test_evaluated=False,
        selection='Prespecified before candidate evaluation. All 12 candidates receive nine nominal endpoint IK checks; retain all accepted cases and record rejections. No success filtering.',
        control='Same E41 PiPER object-offset placement and E42 Panda object-height release. Native mechanics unchanged. All old/new seeds and shared privileged/analytical references. Same home and 32 probe steps; no reset after probe.',
        scope='Fresh native development placements; not the reserved final test. Evaluate fixed E45 checkpoints regardless of retention outcome, and report failed gates rather than promoting them.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['retention_fresh.py','native_reliability.py','real_arm_pick_place.py','real_arms.py','pick_place_control.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rng=np.random.default_rng(45000);tasks=[];screen=[]
    for name,points in candidates.items():
        for xy in points:
            target=[xy[0],xy[1]+(.04 if xy[1]<0 else -.04)]
            env=NativePickPlace(assets,name,xy,target,.2,True);q=env.observe()['q'];poses=[]
            for _,goal,_,_ in sequence(env):
                q=endpoint_ik(env,goal,rng,q)
                if q is None:break
                poses.append(q.tolist())
            screen.append(dict(robot=name,xy=xy,target=target,endpoints=len(poses),accepted=len(poses)==9))
            if len(poses)==9:tasks.append(dict(robot=name,xy=xy,target=target,home=poses[0]))
            print('screen',name,xy,len(poses),flush=True)
    (root/'cases.json').write_text(json.dumps(tasks,indent=2));(root/'screening.json').write_text(json.dumps(screen,indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='prepared',test_evaluated=False)))


def evaluate(assets,output):
    torch.set_num_threads(4);root=Path(output);p=json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='prepared'
    assert not (root/'records.json').exists();verify_assets(assets)
    old,oh=load_models(p['old']);new,nh=load_models(p['new']);hashes={**oh,**nh}
    models={'privileged':None,'analytical':None,**{f'old_{s}':old[f'neural_{s}'] for s in range(3)},**{f'new_{s}':new[f'neural_{s}'] for s in range(3)}}
    (root/'evaluation.json').write_text(json.dumps(dict(checkpoints=hashes,cases_sha256=fingerprint(root/'cases.json')),indent=2))
    tasks=json.loads((root/'cases.json').read_text());rows=[]
    for method,model in models.items():
        for i,t in enumerate(tasks):
            env=NativePickPlace(assets,t['robot'],t['xy'],t['target'],.2,True);controller=GuardedPose(env.n,model)
            guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
            with guard:
                probe=calibrate(env,t['home'],controller,p['probe_seed']+i)
                result=rollout(env,None if method=='privileged' else controller,object_release=t['robot']=='panda')
            rows.append(dict(robot=t['robot'],method=method,case=i,probe=probe,**result))
            (root/'records.json').write_text(json.dumps(rows,indent=2))
            print('fresh',method,t['robot'],i,result['success'],result['failure'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','evaluate']);p.add_argument('--assets',required=True);p.add_argument('--output',required=True)
    p.add_argument('--old');p.add_argument('--new');args=vars(p.parse_args());mode=args.pop('mode')
    if mode=='prepare':prepare(**args)
    else:args.pop('old');args.pop('new');evaluate(**args)

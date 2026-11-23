"""Matched observation-only position/direction controllers on physical manipulation."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .arm_pick_place import PickPlace,rollout,ROT_SCALE
from .mixed_sim import Arm,AXES
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint


class PoseController:
    """Same motion-estimator weights independently predict XYZ and scaled tool-axis motion."""
    def __init__(self,n,model=None):
        self.n=n;self.model=model;self.gram=1e-6*np.eye(n);self.cross=np.zeros((n,6));self.bias=np.zeros(n)

    @torch.no_grad()
    def predict(self):
        if self.model is None:return np.linalg.solve(self.gram,self.cross).T,self.bias
        device=next(self.model.parameters()).device
        gram=torch.zeros((2,7,7),device=device);cross=torch.zeros((2,7,3),device=device)
        gram[:,:self.n,:self.n]=torch.as_tensor(self.gram,device=device,dtype=torch.float32)
        for i in range(2):cross[i,:self.n]=torch.as_tensor(self.cross[:,i*3:(i+1)*3],device=device,dtype=torch.float32)
        mask=torch.arange(7,device=device)[None].expand(2,-1)<self.n
        prediction=self.model(gram,cross,mask).cpu().numpy()[:,:self.n]
        return np.concatenate([prediction[0].T,prediction[1].T]),self.bias

    def update(self,before,after,command):
        dq=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
        self.gram=.98*self.gram+np.outer(dq,dq)+2e-8*np.eye(self.n)
        self.cross=.98*self.cross+np.outer(dq,dx);self.bias=.8*self.bias+.2*(command-dq)


def calibrate(env,home,controller,seed):
    rng=np.random.default_rng(seed);obs=env.reset(home);records=[]
    for step in range(32):
        if step%4==0:offset=rng.uniform(-.012,.012,env.n)
        action=np.clip(np.asarray(home)+offset-obs['q'],-.04,.04)
        after,command,info=env.step(action,0.);controller.update(obs,after,command);obs=after
        records.append(dict(command=command.tolist(),position=obs['position'].tolist(),bad=info['bad']))
    return records


def run(cases,models,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    source=Path(cases);AXES.update(json.loads((source/'axis_catalog.json').read_text()))
    tasks=json.loads((source/'cases.json').read_text());loaded={'analytical':None};hashes={}
    for seed in range(3):
        folder=Path(models)/f'seed_{seed}';meta=json.loads((folder/'training.json').read_text());sha=fingerprint(folder/'policy.pt');assert sha==meta['policy_sha256']
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(folder/'policy.pt',map_location='cuda',weights_only=True));loaded[f'neural_{seed}']=m.eval();hashes[str(folder/'policy.pt')]=sha
    (root/'protocol.json').write_text(json.dumps(dict(experiment='E38',cases_sha256=fingerprint(source/'cases.json'),checkpoints=hashes,
        scope='Training-family manipulation pilot; same scripted phase sequencer, same 32 measured probe steps, no inference weight updates. Same frozen estimator applied separately to position and tool-Z-direction statistics. Direction scale .15; yaw unconstrained. No learned grasp/phase decisions.',
        calibration='V2 correction: 32 steps of home-relative joint targets, offsets uniform +/-.012 rad held for four steps; aperture open, no reset, measured q feedback. Counts in task budget; all calibration contact retained. V2 grasp/lower waypoints raised 1 cm for physical finger clearance; same correction for all methods. Observed development tasks, not independent evaluation.',test_evaluated=False),indent=2))
    (root/'source').mkdir()
    for name in ['arm_pick_place.py','pick_place_control.py','pick_place_fixture.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    reference=[]
    for i,task in enumerate(tasks):
        env=PickPlace(Arm(**task['arm']),task['xy'],task['target'])
        probe=calibrate(env,task['home'],PoseController(env.n),38000+i)
        result=rollout(env,task['home'],reset=False)
        reference.append(dict(method='privileged',case=i,dof=env.n,body=env.arm.seed,probe=probe,**result))
    (root/'privileged.json').write_text(json.dumps(reference,indent=2))
    print('privileged',sum(r['success'] for r in reference),'/',len(reference),flush=True)
    rows=[]
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        for name,model in loaded.items():
            for i,task in enumerate(tasks):
                env=PickPlace(Arm(**task['arm']),task['xy'],task['target']);controller=PoseController(env.n,model)
                probe=calibrate(env,task['home'],controller,38000+i)
                # The 6x6 shared task solve remains allowed. Predict is separately inverse-guarded in tests.
                result=rollout(env,task['home'],controller,reset=False)
                rows.append(dict(method=name,case=i,dof=env.n,body=env.arm.seed,probe=probe,**result))
                (root/'records.json').write_text(json.dumps(rows,indent=2))
                print(name,i,'success',result['success'],'grasp',result['grasped'],'lift',result['lifted'],'collision',result['collision'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['cases','models','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

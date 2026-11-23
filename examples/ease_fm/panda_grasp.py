"""E42: test measured full-orientation control for native Panda grasp stability."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .arm_pick_place import ROT_SCALE,direction_jacobian
from .native_reliability import rollout
from .pick_place_control import PoseController,calibrate
from .real_arm_pick_place import NativePickPlace
from .real_arm_reach import load_models
from .real_arms import verify_assets
from .benchmark import fingerprint


class OrientationController(PoseController):
    def __init__(self,n,model=None):
        super().__init__(n,model);self.cross=np.zeros((n,9))

    @torch.no_grad()
    def predict(self):
        if self.model is None:return np.linalg.solve(self.gram,self.cross).T,self.bias
        device=next(self.model.parameters()).device
        gram=torch.zeros((3,7,7),device=device);cross=torch.zeros((3,7,3),device=device)
        gram[:,:self.n,:self.n]=torch.as_tensor(self.gram,device=device,dtype=torch.float32)
        for i in range(3):cross[i,:self.n]=torch.as_tensor(self.cross[:,3*i:3*i+3],device=device,dtype=torch.float32)
        mask=torch.arange(7,device=device)[None].expand(3,-1)<self.n
        result=self.model(gram,cross,mask).cpu().numpy()[:,:self.n]
        return np.concatenate([r.T for r in result]),self.bias

    def update(self,before,after,command):
        dq=after['q']-before['q']
        dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2]),
            ROT_SCALE*(after['rotation'][:,0]-before['rotation'][:,0])]
        self.gram=.98*self.gram+np.outer(dq,dq)+2e-8*np.eye(self.n)
        self.cross=.98*self.cross+np.outer(dq,dx);self.bias=.8*self.bias+.2*(command-dq)


class GraspArm(NativePickPlace):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.telemetry=[]

    def step(self,*args):
        obs,command,info=super().step(*args)
        self.telemetry.append(dict(tool_rotation=obs['rotation'].tolist(),object_rotation=self.data.xmat[self.obj].reshape(3,3).tolist(),
            opening=self.opening(),gripper_force=float(self.data.actuator_force[self.gripper]),
            object_velocity=self.data.qvel[self.model.jnt_dofadr[self.oj]:self.model.jnt_dofadr[self.oj]+6].tolist()))
        return obs,command,info

    def full_jacobian(self):
        jp=np.zeros((3,self.model.nv));jr=np.zeros_like(jp)
        mujoco.mj_jacSite(self.model,self.data,jp,jr,self.site)
        rotation=self.observe()['rotation'];xframe=rotation[:,[1,2,0]]
        return np.concatenate([jp[:,self.vids],ROT_SCALE*direction_jacobian(rotation,jr[:,self.vids]),
            ROT_SCALE*direction_jacobian(xframe,jr[:,self.vids])])


def nearest_cube_axis(rotation):
    axes=np.array([[1.,0,0],[0,1.,0],[-1.,0,0],[0,-1.,0]])
    return axes[np.argmax(axes@rotation[:,0])]


def run(assets,models,cases,output,variant='yaw'):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    source=Path(cases);tasks=json.loads((source/'cases.json').read_text())
    manifest=verify_assets(assets);loaded,hashes=load_models(models)
    protocol=dict(experiment='E42',variant=variant,cases_sha256=fingerprint(source/'cases.json'),
        source=str(source),checkpoints=hashes,commit=manifest['commit'],test_evaluated=False,
        intervention='Yaw variant adds measured tool-X displacement statistics and same frozen Transformer third forward prediction; nearest world cardinal tool-X target at reset, aligned to initial cube faces. Baseline is exact E41 revision 3. No force, friction, scoring or model-weight change.',
        scope='Matched observed nine Panda placements, same homes/probes/budgets; no policy filtering or final-test evaluation.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['panda_grasp.py','native_reliability.py','pick_place_control.py','real_arm_pick_place.py','real_arms.py','motion_transformer.py']:
        shutil.copyfile(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for method,model in loaded.items():
        for i,task in enumerate(tasks):
            if task['robot']!='panda':continue
            env=GraspArm(assets,'panda',task['xy'],task['target'],.2,True)
            controller=(OrientationController if variant=='yaw' else PoseController)(env.n,model)
            obs=env.reset(task['home']);axis=nearest_cube_axis(obs['rotation']) if variant=='yaw' else None
            guard=nullcontext() if method=='privileged' else patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden'))
            with guard:
                probe=calibrate(env,task['home'],controller,41000+i)
                result=rollout(env,None if method=='privileged' else controller,target_x=axis)
            if variant=='baseline':
                previous=next(r for r in json.loads((source/'records.json').read_text()) if r['method']==method and r['case']==i)
                for key in ['trace','phases','steps','success','final_object']:assert result[key]==previous[key],key
            rows.append(dict(method=method,case=i,robot='panda',probe=probe,telemetry=env.telemetry,**result))
            (root/'records.json').write_text(json.dumps(rows,indent=2))
            print(variant,method,i,result['success'],result['failure'],result['steps'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','models','cases','output']:p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=['baseline','yaw'],default='yaw');run(**vars(p.parse_args()))

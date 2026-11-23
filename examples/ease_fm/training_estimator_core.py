"""E114: observation-only history attribution on reused unseen development bodies."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .arm_pick_place import PickPlace,targets,ROT_SCALE
from .local_probe import probe_rollout,probe_offset
from .local_history import LocalPose
from .pick_place_control import calibrate
from .mixed_sim import Arm,AXES
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint

ROOT=Path('outputs/ease_estimator_repair_v1')
TASKS=Path('outputs/ease_unseen_progress_v1')
MODES=('online','frozen_response','frozen_all','frozen_bias')

def load():
    AXES.update(json.loads((TASKS/'axis_catalog.json').read_text()))
    tasks=json.loads((TASKS/'cases.json').read_text());models={'analytical':None};hashes={}
    for seed in range(3):
        p=Path(f'outputs/ease_task_retention_training_v1/retained/seed_{seed}/policy.pt')
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));models[f'learned_{seed}']=m.eval();hashes[str(p)]=fingerprint(p)
    return tasks,models,hashes

def evaluate(case,t,method,model,mode,capture=None,save_states=False,contexts=None):
    telemetry=[];states=[];updates=0;active=False
    class Controller(LocalPose):
        def update(self,before,after,command):
            nonlocal updates
            if not active:return super().update(before,after,command)
            j,_=self.predict();dq=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
            if contexts is not None:
                g=np.zeros((7,7),np.float32);cross=np.zeros((7,6),np.float32);mask=np.arange(7)<self.n;b=np.zeros(7,np.float32);delta=np.zeros(7,np.float32)
                g[:self.n,:self.n]=self.gram;cross[:self.n]=self.cross;b[:self.n]=self.bias;delta[:self.n]=dq
                error=np.r_[env.goal-before['position'],ROT_SCALE*(np.array([0.,0.,1.])-before['rotation'][:,2])]
                contexts.append(dict(g=g,c=cross,mask=mask,error=error.astype(np.float32),bias=b,dq=delta,dx=dx.astype(np.float32),case=case,phase=env.phase,clean=not env.bad,validation=case%3==2))
            telemetry.append(dict(phase=env.phase,position_prediction_error=float(np.linalg.norm(dx[:3]-(j@dq)[:3])),command_norm=float(np.linalg.norm(command))))
            if mode in ('online','frozen_bias'):
                saved_bias=self.bias.copy();super().update(before,after,command);updates+=1
                if mode=='frozen_bias':
                    self.bias=saved_bias
                    np.testing.assert_array_equal(self.bias,initial['bias'])
            elif mode=='frozen_response':self.bias=.8*self.bias+.2*(command-dq)
    class ObservedArm(PickPlace):
        def step(self,action,aperture):
            if getattr(self,'phase',None)=='probe':
                action=np.clip(self.anchor+probe_offset(self.probe_step,self.n)-self.observe()['q']+c.bias,-.04,.04);self.probe_step+=1
            r=super().step(action,aperture)
            if active:
                if save_states:states.append(np.r_[self.data.qpos.copy(),self.data.qvel.copy()])
                if capture:capture(self)
            return r
    env=ObservedArm(Arm(**t['arm']),t['xy'],t['target']);c=Controller(env.n,model)
    def state():return dict(gram=c.gram.tolist(),cross=c.cross.tolist(),bias=c.bias.tolist(),history_count=len(c.history))
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibration=calibrate(env,t['home'],c,38000+case);initial=state();active=True
        def phases():
            for name,goal,aperture,budget in targets(env.xy,env.target):
                env.phase=name;env.goal=goal
                if name=='close':
                    if mode in ('online','frozen_bias'):c.local=True;c.rebuild()
                    env.anchor=env.observe()['q'].copy();env.probe_step=0;env.phase='probe'
                    env.goal=env.observe()['position'].copy()
                    yield 'probe',env.goal,0.,32
                    assert env.probe_step==32;env.phase=name;env.goal=goal
                yield name,goal,aperture,budget
        r=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
    final=state()
    if mode in ('frozen_response','frozen_all'):
        for key in ('gram','cross','history_count'):assert initial[key]==final[key]
    if mode in ('frozen_all','frozen_bias'):assert initial['bias']==final['bias']
    assert updates==(len(r['trace']) if mode in ('online','frozen_bias') else 0)
    result=dict(case=case,body=t['arm']['seed'],family=t['arm']['family'],dof=env.n,method=method,mode=mode,calibration=calibration,initial_estimator=initial,final_estimator=final,response_updates=updates,telemetry=telemetry,**r)
    if save_states:
        stem=f'{case}_{method}_{mode}';np.savez_compressed(ROOT/'states'/f'{stem}.npz',states=np.array(states));(ROOT/'states'/f'{stem}.xml').write_text(env.xml)
    return result

"""E109 rollout: identical physical loop with approach-only prefix invariant."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import gentle_transition as previous
from .gentle_transition import prior,ProbeArm,probe_rollout,LocalPose,targets,calibrate,Arm,AXES,fingerprint,ramp_action,MOTION
ROOT=Path('outputs/ease_single_phase_v1')

def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None):
    class GentleArm(ProbeArm):
        def step(self,action,aperture):
            if not hasattr(self,'phase_index') or getattr(self,'phase',None)!=getattr(self,'last_phase',None):self.phase_index=0;self.last_phase=getattr(self,'phase',None)
            if mode=='gentle' and getattr(self,'phase',None)==affected_phase and affected_phase is not None:action=ramp_action(action,self.controller.bias,self.phase_index)
            self.phase_index+=1
            r=super().step(action,aperture)
            if getattr(self,'phase',None) in ('close',*MOTION,'release','settle'):
                obs=self.observe();pads,bad=self.contacts();obj=self.data.xpos[self.obj].copy()
                contacts.append(dict(step=self.steps,phase=self.phase,pads=sorted(pads),object=obj.tolist(),tool=obs['position'].tolist(),relative=(obs['rotation'].T@(obj-obs['position'])).tolist()))
            if record_step is not None and getattr(self,'filter_enabled',False):record_step(self)
            return r
    env=GentleArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.route=None;env.path_log=[];env.waypoint=0;env.probe_step=0;contacts=[]
    class Controller(LocalPose):
        def predict(self):
            saved=self.model
            if self.local and env.phase!='probe':self.model=students[('retained',seed)]
            try:return super().predict()
            finally:self.model=saved
    c=Controller(env.n,parents[f'candidate_{seed}']);calibration=calibrate(env,t['home'],c,38000+case);env.controller=c;env.filter_enabled=True;env.filter_log=[]
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name;env.goal=goal
            if name=='close':
                c.local=True;c.rebuild();env.anchor=env.observe()['q'].copy();env.phase='probe';env.goal=env.observe()['position'].copy()
                yield 'probe',env.goal,0.,32
                assert env.probe_step==32;env.phase=name;env.goal=goal
            if mode!='baseline' and name==affected_phase:budget+=20
            yield name,goal,aperture,budget
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobians forbidden')):r=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
    original=json.loads((prior.ROOT/'training_trials'/f'{case}_{seed}.json').read_text())
    assert calibration==original['calibration']
    prefix=lambda x:[r for r in x['trace'] if r['phase'] in ('approach',)]
    assert prefix(r)==prefix(original)
    if mode=='baseline':assert json.loads(json.dumps(r))=={k:original[k] for k in r}
    return dict(affected_phase=affected_phase,case=case,seed=seed,mode=mode,dof=env.n,validation=case%3==2,calibration=calibration,contacts=contacts,**r)


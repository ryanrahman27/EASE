"""E103 fixed training-arm screen: smooth task-command onset versus budget control."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from . import onpolicy_history as prior
from .local_probe import ProbeArm,probe_rollout
from .local_history import LocalPose
from .arm_pick_place import targets
from .pick_place_control import calibrate
from .mixed_sim import Arm,AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_gentle_transition_v1')
MODES=('baseline','budget','gentle')
MOTION=('lift','transport','lower','retreat')


def ramp_action(action,bias,index):
    u=min((index+1)/20.,1.);scale=u*u*(3-2*u)
    if scale==1.:return action.copy()
    return np.clip(bias+scale*(action-bias),-.04,.04)


def evaluate(case,t,seed,mode,parents,students,record_step=None):
    class GentleArm(ProbeArm):
        def step(self,action,aperture):
            if not hasattr(self,'phase_index') or getattr(self,'phase',None)!=getattr(self,'last_phase',None):self.phase_index=0;self.last_phase=getattr(self,'phase',None)
            if mode=='gentle' and getattr(self,'phase',None) in MOTION:action=ramp_action(action,self.controller.bias,self.phase_index)
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
            if mode!='baseline' and name in MOTION:budget+=20
            yield name,goal,aperture,budget
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobians forbidden')):r=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
    original=json.loads((prior.ROOT/'training_trials'/f'{case}_{seed}.json').read_text())
    assert calibration==original['calibration']
    prefix=lambda x:[r for r in x['trace'] if r['phase'] in ('approach','descend','probe','close')]
    assert prefix(r)==prefix(original)
    if mode=='baseline':assert json.loads(json.dumps(r))=={k:original[k] for k in r}
    return dict(case=case,seed=seed,mode=mode,dof=env.n,validation=case%3==2,calibration=calibration,contacts=contacts,**r)


def run():
    torch.set_num_threads(4);assert not (ROOT/'records.json').exists() and not list((ROOT/'trials').glob('*.json'))
    ROOT.mkdir(exist_ok=True);(ROOT/'trials').mkdir(exist_ok=True);(ROOT/'source').mkdir(exist_ok=True)
    tasks=json.loads((prior.TASKS/'cases.json').read_text());AXES.update(json.loads((prior.TASKS/'axis_catalog.json').read_text()));parents,hashes=prior.previous.models();students,hh=prior.e99.load_students();hashes.update(hh)
    arms=json.loads(Path('outputs/ease_paired_factorial_v2/manifest.json').read_text())['arms']
    assert all(t['arm']==next(a for a in arms if a['seed']==t['arm']['seed'] and a['split']=='train') for t in tasks)
    paths=[prior.TASKS/'cases.json',prior.TASKS/'axis_catalog.json',*sorted((prior.ROOT/'training_trials').glob('*.json'))];inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E103',checkpoints=hashes,inputs=inputs,test_evaluated=False,
        design='All12 original training placements x3 fixed E99 retained checkpoints x baseline/budget/gentle=108 tasks. Original E66 approach and32-step probe; E99 retained model thereafter. No route planner on training bodies. Third placement per body remains held-placement validation,24 training/12 validation pairs. No new weights.',
        intervention='At each lift/transport/lower/retreat onset, scale pre-geometry action minus measured bias with cubic smoothstep u^2(3-2u), u=min((step+1)/20,1), add bias and clip .04. Exact original command from step20 onward. Other phases unchanged. Both budget and gentle add20 maximum control steps to each of these four phases; baseline uses original budgets. Existing target, aperture, early-exit and success criteria unchanged. Budget control isolates extra available time. No contact feedback used for actions;40ms contact/object diagnostics only.',
        gate='Proceed to full development transfer only if gentle strictly improves successes over BOTH baseline and budget, does not increase collisions, preserves every success and each seed count on the24 training pairs, and preserves every reference success with nonincreasing collisions on the12 held-placement pairs. Otherwise record failure, do not retune the fixed20-step ramp or evaluate transfer.',
        videos='Cases0 and2 seed0 allthree modes; append first gentle loss against each control with both modes; add missing success/failure category. Exact full-result replay, frame counts and hashes.',
        limits='Four original training bodies. Fixed hypothesis motivated by outcome-selected development diagnostics, not independent test. Ramp acts on joint-command correction rather than directly limiting Cartesian velocity; geometry filter can change the executed command. Existing privileged geometry remains. No active grasp stabilization or retreat-path redesign.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for mode in MODES:
                r=evaluate(case,t,seed,mode,parents,students);(ROOT/'trials'/f'{case}_{seed}_{mode}.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','contacts','calibration')})
        print('training-arm case',case,'complete',flush=True)
    for p,h in {**hashes,**inputs}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':run()

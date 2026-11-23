"""E104: single-phase ramps with phase-matched time controls."""
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
    prefix=lambda x:[r for r in x['trace'] if r['phase'] in ('approach','descend','probe','close')]
    assert prefix(r)==prefix(original)
    if mode=='baseline':assert json.loads(json.dumps(r))=={k:original[k] for k in r}
    return dict(affected_phase=affected_phase,case=case,seed=seed,mode=mode,dof=env.n,validation=case%3==2,calibration=calibration,contacts=contacts,**r)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((prior.TASKS/'cases.json').read_text());AXES.update(json.loads((prior.TASKS/'axis_catalog.json').read_text()));parents,hashes=prior.previous.models();students,hh=prior.e99.load_students();hashes.update(hh)
    old=json.loads((previous.ROOT/'protocol.json').read_text());inputs=dict(old['inputs']);inputs[str(previous.ROOT/'records.json')]=fingerprint(previous.ROOT/'records.json')
    spec=dict(experiment='E104',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All36 original training-body task/seed pairs x baseline plus four individual phases x budget/gentle=324 full rollouts. Fixed E99 retained models after original E66 prefix/probe. Third placement per body is held validation,24/12 pairs. No route planner, unchanged privileged geometry. Every baseline equals E103.',
        intervention='Exactly one of lift/transport/lower/retreat receives20 extra maximum steps in budget and gentle. Only gentle applies unchanged E10320-step smoothstep to action minus measured bias in that phase; other phases unchanged. Same weights, targets, fingers, early exits and success criteria. No contact feedback. All four phase conditions reported; no duration or seed selection.',
        gate='For each phase separately, same E103 gate versus baseline AND its phase-matched budget: strict training success improvement, nonincreasing collisions, every success retained and no seed count decline on24 training pairs; retain every success and nonincreasing collisions on12 held placements. Only passing phases eligible for full development transfer; no best-phase promotion when all fail.',
        videos='Case2 seed0 baseline and all eight phase controls; append first gentle regression versus baseline and phase budget for each phase, with reference. Missing success/failure category appended if needed. Every replay must match full serialized result.',
        limits='Phase ablation after failed combined ramp. Training-body development screen, not independent generalization or a new learned policy. Phase budgets can alter time-outs and interactions. No final test.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);shutil.copy2(Path(previous.__file__),ROOT/'source'/'gentle_transition.py');rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for phase,mode in [(None,'baseline')]+[(p,m) for p in MOTION for m in ('budget','gentle')]:
                r=evaluate(case,t,seed,mode,parents,students,affected_phase=phase)
                baseline=json.loads((previous.ROOT/'trials'/f'{case}_{seed}_baseline.json').read_text())
                if phase is None:assert {k:v for k,v in r.items() if k!='affected_phase'}==baseline
                else:
                    order=['approach','descend','probe','close','lift','transport','lower','release','retreat','settle'];before=set(order[:order.index(phase)])
                    assert [x for x in r['trace'] if x['phase'] in before]==[x for x in baseline['trace'] if x['phase'] in before]
                (ROOT/'trials'/f'{case}_{seed}_{phase}_{mode}.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','contacts','calibration')})
        print('case',case,'complete;',len(rows),'rollouts',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))

if __name__=='__main__':run()

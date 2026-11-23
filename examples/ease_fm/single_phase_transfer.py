"""Conditional E104 full development check; only training-gate passing phases."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import torch
from .single_phase import ROOT,prior,MOTION,ramp_action,AXES,fingerprint
from . import local_probe


def evaluate(case,t,seed,phase,mode,parents,students,record_step=None):
    original_step=local_probe.ProbeArm.step;original_targets=local_probe.targets;active={}
    def step(env,action,aperture):
        active['env']=env
        if not hasattr(env,'e104_index') or getattr(env,'phase',None)!=getattr(env,'e104_phase',None):env.e104_index=0;env.e104_phase=getattr(env,'phase',None)
        if mode=='gentle' and getattr(env,'phase',None)==phase and getattr(getattr(env,'controller',None),'switched',False):action=ramp_action(action,env.controller.bias,env.e104_index)
        env.e104_index+=1;return original_step(env,action,aperture)
    def targets(xy,target):
        for name,goal,aperture,budget in original_targets(xy,target):
            if mode!='baseline' and name==phase and active['env'].route is not None:budget+=20
            yield name,goal,aperture,budget
    with patch.object(local_probe.ProbeArm,'step',step),patch.object(local_probe,'targets',targets):
        r=prior.e99.evaluate(case,t,f'candidate_{seed}',parents[f'candidate_{seed}'],students[('retained',seed)],'retained',record_step)
    old=json.loads((prior.e99.ROOT/'trials'/f'{case}_candidate_{seed}_retained.json').read_text())
    if mode=='baseline' or not r['route_used']:assert json.loads(json.dumps(r))==old
    if phase is not None:
        order=['approach','descend','probe','close','lift','transport','lower','release','retreat','settle'];before=set(order[:order.index(phase)])
        assert [x for x in r['trace'] if x['phase'] in before]==[x for x in old['trace'] if x['phase'] in before]
    r.update(mode=mode,affected_phase=phase,seed=seed)
    return r


def run():
    summary=json.loads((ROOT/'summary.json').read_text());passing=summary['passing']
    assert passing,'No phase passed the prespecified training gate'
    torch.set_num_threads(4);out=ROOT/'transfer';out.mkdir(exist_ok=False);(out/'trials').mkdir();(out/'source').mkdir();shutil.copy2(__file__,out/'source'/Path(__file__).name)
    tasks=json.loads((prior.previous.PRIOR/'cases.json').read_text());AXES.update(json.loads((prior.previous.PRIOR/'axis_catalog.json').read_text()));parents,hashes=prior.previous.models();students,hh=prior.e99.load_students();hashes.update(hh)
    (out/'protocol.json').write_text(json.dumps(dict(phases=passing,checkpoints=hashes,design='All16 development tasks x3 seeds x baseline plus phase-matched budget/gentle for every training-gate passing phase. Fixed E99 retained weights after identical original-parent/planner/probe prefix. Only admitted routes receive extra phase budget or ramp; no-route behavior unchanged. No selection using transfer outcomes.',gate='Each phase must improve success versus original E99 baseline and its budget control, preserve every reference success and per-seed count, with nonincreasing collisions. No final test.'),indent=2))
    rows=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for phase,mode in [(None,'baseline')]+[(p,m) for p in passing for m in ('budget','gentle')]:
                r=evaluate(case,t,seed,phase,mode,parents,students)
                (out/'trials'/f'{case}_{seed}_{phase}_{mode}.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','handoff','continuation_handoff')})
        print('transfer case',case,'complete',flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    (out/'records.json').write_text(json.dumps(rows,indent=2))


if __name__=='__main__':run()

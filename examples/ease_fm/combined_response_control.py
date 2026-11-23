"""Freeze and execute E136 conjunction without simulator-only gate inputs."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .combined_response_value import ROOT,MOTION,VALUE
from .compact_response_value import features as compact
from .structured_response import preaction,fit
from .abstaining_task_value_gate import read,save,fingerprint,DATA,OLD,PHASES
from . import broad_task_value_gate as broad
from .bounded_history_correction import ROT_SCALE,rotation_error

def freeze():
    chosen=read(ROOT/'selected.json');assert chosen['gains']>0 and chosen['losses']==0
    for f in range(4):
        shutil.copy2(MOTION/'models'/f'scalar_{f}.npz',ROOT/'models'/f'motion_{f}.npz')
        shutil.copy2(VALUE/'models'/f'lambda_1.0_fold_{f}.npz',ROOT/'models'/f'value_{f}.npz')
    shutil.copy2(MOTION/'folds.json',ROOT/'folds.json')
    d=np.load(MOTION/'dataset.npz');np.savez(ROOT/'models'/'motion_final.npz',weights=fit('scalar',d['effect'],d['target'],d['phase']))
    d=np.load(VALUE/'dataset.npz');x,y=d['x'],d['y'];mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);z=np.c_[np.ones(len(x)),np.clip((x-mean)/scale,-10,10)];reg=np.eye(z.shape[1]);reg[0,0]=0
    w=np.linalg.solve(z.T@z/len(x)+reg,z.T@y/len(x));np.savez(ROOT/'models'/'value_final.npz',weights=w,mean=mean,scale=scale)
    save(ROOT/'frozen_rule.json',chosen['rule']);_,_,hashes=broad.setup();save(ROOT/'baseline_checkpoints.json',hashes)
    for name in ('combined_response_control.py','combined_response_value.py','compact_response_value.py','structured_response.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'frozen_rule.json',ROOT/'protocol.json',ROOT/'selected.json',*(ROOT/'models').glob('*.npz'),*(ROOT/'source').glob('*.py')]})
    print('Frozen E136 models/rule before any fresh rollout or held evaluation',flush=True)

def evaluate(case,t,seed,models,motion,value,rule):
    previous=broad.previous;method=f'candidate_{seed}';original=previous.bounded_action;rollout=previous.core.probe_rollout;chosen=None;off=False;counts={};decisions=[]
    def bounded(a,b):return a.copy() if off else original(a,b)
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            nonlocal chosen,off
            phase=env.phase;k=counts.get(phase,0);counts[phase]=k+1
            if phase in PHASES and k==0 and chosen is None and not env.bad:
                obs=env.observe();old=c.model
                try:c.model=models[f'correction_{seed}'];j,bias=c.predict()
                finally:c.model=old
                rot=rotation_error(np.eye(3),obs['rotation']);error=np.r_[env.goal-obs['position'],ROT_SCALE*rot];proposal=np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
                base=broad.features(c.n,c.gram,c.cross,c.bias,obs['q'],obs['position'],env.goal,phase,c.local,action,proposal)
                x=np.r_[base,obs['position'],rot].astype(np.float32);effect,common,e=preaction(x.astype(np.float64));direction=e-motion[0]*common;direction/=max(np.linalg.norm(direction),1e-12);progress=float(direction@(motion[0]*effect));z=compact(x);advantage=float(np.r_[1.,np.clip((z-value['mean'])/value['scale'],-10,10)]@value['weights'])
                selected=progress>rule['motion_margin'] and advantage>rule['value_margin'];decisions.append(dict(phase=phase,features=x.tolist(),compact=z.tolist(),progress=progress,advantage=advantage,selected=selected))
                if selected:chosen=phase
            off=phase==chosen and k<8
            return original_step(action,aperture)
        env.step=step
        return rollout(env,home,c,**kwargs)
    with patch.object(previous,'bounded_action',bounded),patch.object(previous.core,'probe_rollout',observer):r,_=previous.evaluate(case,t,method,models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded',save=True)
    save(ROOT/'actions'/f'{case}_{method}_decisions.json',dict(chosen=chosen,decisions=decisions))
    return {k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(chosen=chosen,seed=seed)

def run(seed,held=False):
    if held:
        s=read(ROOT/'rollout_summary.json');assert s['fitting']['gains']>0 and s['fitting']['losses']==0 and s['fitting']['exact_counterfactuals']==144
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    tasks,models,hashes=broad.setup();broad.previous.ROOT=ROOT;broad.previous.core.ROOT=ROOT;folds=read(ROOT/'folds.json');rule=read(ROOT/'frozen_rule.json');path=ROOT/f'records_{"held" if held else "fit"}_{seed}.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if bool(t['validation'])!=held or case in done:continue
        suffix='final' if held else str(folds[str(case)]);motion=np.load(ROOT/'models'/f'motion_{suffix}.npz')['weights'];value=np.load(ROOT/'models'/f'value_{suffix}.npz')
        r=evaluate(case,t,seed,models,motion,value,rule);rows.append(r);save(path,rows);print('E136','held' if held else 'fit',seed,case,r['success'],r['chosen'],flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h

def summarize():
    rows=sum([read(p) for p in sorted(ROOT.glob('records_fit_*.json'))],[]);assert len(rows)==144;expected={(r['case'],r['seed']):r for r in read(ROOT/'selected.json')['episodes']};changes=[]
    for r in rows:
        case,seed=r['case'],r['seed'];e=expected[case,seed];assert r['chosen']==e['phase'] and r['success']==e['success'];method=f'on_{seed}' if r['chosen'] is None else f'off_{r["chosen"]}_{seed}';stem=f'{case}_{method}_frozen_bias';actual=f'{case}_candidate_{seed}_frozen_bias'
        a=read(DATA/'collection/trials'/f'{stem}.json');b=read(ROOT/'trials'/f'{actual}.json');a['method']=b['method'];assert a==b
        np.testing.assert_array_equal(np.load(DATA/'collection/states'/f'{stem}.npz')['states'],np.load(ROOT/'states'/f'{actual}.npz')['states'])
        if r['success']!=e['baseline_success']:changes.append(dict(case=case,seed=seed,phase=r['chosen'],before=e['baseline_success'],after=r['success']))
    fitting=dict(n=144,success=sum(r['success'] for r in rows),baseline_success=97,gains=sum(r['after'] for r in changes),losses=sum(r['before'] for r in changes),changes=changes,exact_counterfactuals=144)
    summary=dict(fitting=fitting,held_evaluated=False,E120_unchanged=True)
    held=sum([read(p) for p in sorted(ROOT.glob('records_held_*.json'))],[])
    if held:
        assert len(held)==24;baseline={(r['case'],int(r['method'].split('_')[-1])):r for r in read(OLD/'records.json') if r['method'].startswith('candidate_')};changes=[]
        for r in held:
            a=baseline[r['case'],r['seed']]
            if a['success']!=r['success']:changes.append(dict(case=r['case'],seed=r['seed'],phase=r['chosen'],before=a['success'],after=r['success']))
        summary.update(held_evaluated=True,held=dict(n=24,success=sum(r['success'] for r in held),baseline_success=sum(baseline[r['case'],r['seed']]['success'] for r in held),gains=sum(r['after'] for r in changes),losses=sum(r['before'] for r in changes),changes=changes))
    save(ROOT/'rollout_summary.json',summary);save(ROOT/'records.json',rows+held);print(summary,flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(int(sys.argv[2]),'held' in sys.argv)
    else:{'freeze':freeze,'summarize':summarize}[sys.argv[1]]()

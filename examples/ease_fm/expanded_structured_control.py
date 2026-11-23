"""E138: frozen fresh-placement evaluation, only after fitting qualification."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .expanded_structured_pairs import ROOT
from .expanded_structured_training import dependencies
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .compact_response_value import features
from .structured_response import preaction
from . import broad_task_value_gate as broad
from .bounded_history_correction import ROT_SCALE,rotation_error

EVAL=ROOT/'fresh_evaluation'

class Gate:
    def __init__(self,kind,suffix='final'):
        self.kind=kind;self.tree=None
        if kind=='consensus':self.models=[dict(np.load(p)) for p in sorted((ROOT/'models').glob(f'consensus_{suffix}_exclude_*.npz'))]
        else:
            self.models=[dict(np.load(ROOT/'models'/f'ridge_{suffix}.npz'))]
            if kind=='trees':_,joblib,_=dependencies();self.tree=joblib.load(ROOT/'models'/f'trees_{suffix}.joblib')
        assert self.models
    def scores(self,raw):
        x=features(raw);effect,common,error=preaction(raw.astype(np.float64));values=[];progress=[]
        for m in self.models:
            z=np.clip((x-m['mean'])/m['scale'],-10,10);v=float(self.tree.predict(z[None])[0]) if self.tree is not None else float(np.r_[1.,z]@m['weights']);s=m['scalar'][0];direction=error-s*common;direction/=max(np.linalg.norm(direction),1e-12);values.append(v);progress.append(float(direction@(s*effect)))
        return min(values),min(progress)

def prepare():
    selected=read(ROOT/'selected.json');assert selected['gains']>0 and selected['losses']==0 and selected['kind']!='abstain'
    for p,h in read(ROOT/'frozen_training.json').items():assert fingerprint(p)==h
    EVAL.mkdir(exist_ok=False)
    for folder in ('trials','states','actions','source'):(EVAL/folder).mkdir()
    tasks=read(ROOT/'cases.json');assert len(tasks)==104;fresh=read(ROOT/'fresh_cases.json');assert len(fresh)==48;save(EVAL/'cases.json',tasks+fresh)
    save(EVAL/'protocol.json',dict(kind=selected['kind'],value_margin=.05,progress_margin=1e-5,pulse_steps=8,fresh_case_ids=list(range(104,152)),calibration='Original38000+case index; new fresh IDs104..151. Shared exactly across candidate/E120.',design='All48 predeclared fresh placements times3 frozen motion seeds, E120 and frozen candidate:288 trajectories. No success/reachability filtering, threshold changes, retraining or test-driven selection. Known four training bodies and fitting-source homes; this is new placement generalization, not held ancestry or unseen morphology.',secondary='Also run original eight development-held placements paired with E120 after freeze (48 trajectories). Report separately because their outcomes were seen in earlier studies. No candidate tuning on either test. This secondary check was added while collection was ongoing, before expanded model results, to avoid hiding known harder placements.',baseline='Unchanged E120 bounded action residual, identical initial calibration and motion checkpoints.'))
    for name in ('expanded_structured_control.py','expanded_structured_training.py','compact_response_value.py','structured_response.py'):shutil.copy2(Path(__file__).with_name(name),EVAL/'source'/name)
    save(EVAL/'frozen.json',{str(p):fingerprint(p) for p in [EVAL/'protocol.json',EVAL/'cases.json',*(EVAL/'source').glob('*.py')]});print('Prepared frozen fresh test',selected['kind'],'288 full episodes',flush=True)

def evaluate(case,t,seed,models,gate):
    previous=broad.previous;original=previous.bounded_action;rollout=previous.core.probe_rollout;off=False;chosen=None;counts={};decisions=[]
    def bounded(a,b):return a.copy() if off else original(a,b)
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            nonlocal off,chosen
            phase=env.phase;k=counts.get(phase,0);counts[phase]=k+1
            if phase in PHASES and k==0 and chosen is None and not env.bad:
                obs=env.observe();old=c.model
                try:c.model=models[f'correction_{seed}'];j,bias=c.predict()
                finally:c.model=old
                rot=rotation_error(np.eye(3),obs['rotation']);error=np.r_[env.goal-obs['position'],ROT_SCALE*rot];proposal=np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
                base=broad.features(c.n,c.gram,c.cross,c.bias,obs['q'],obs['position'],env.goal,phase,c.local,action,proposal);raw=np.r_[base,obs['position'],rot].astype(np.float32);value,progress=gate.scores(raw);accept=value>.05 and progress>1e-5
                decisions.append(dict(phase=phase,raw=raw.tolist(),value=value,progress=progress,selected=accept))
                if accept:chosen=phase
            off=phase==chosen and k<8
            return original_step(action,aperture)
        env.step=step
        return rollout(env,home,c,**kwargs)
    with patch.object(previous,'bounded_action',bounded),patch.object(previous.core,'probe_rollout',observer):r,_=previous.evaluate(case,t,f'candidate_{seed}',models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded',save=True)
    save(EVAL/'actions'/f'{case}_candidate_{seed}_decisions.json',dict(chosen=chosen,decisions=decisions));return r,chosen

def run(seed,oldheld=False):
    for p,h in {**read(ROOT/'frozen_training.json'),**read(EVAL/'frozen.json')}.items():assert fingerprint(p)==h
    _,models,hashes=broad.setup();broad.previous.ROOT=EVAL;broad.previous.core.ROOT=EVAL;tasks=read(EVAL/'cases.json');gate=Gate(read(EVAL/'protocol.json')['kind']);path=EVAL/f'records_{"oldheld_" if oldheld else ""}seed_{seed}.json';rows=read(path) if path.exists() else [];done={(r['case'],r['method']) for r in rows};original=broad.previous.core.LocalPose.predict;cache={};case_ids=[i for i,t in enumerate(tasks[:104]) if t['validation']] if oldheld else list(range(104,152))
    def cached(c):
        if c.model is None:return original(c)
        key=id(c.model),c.n,c.gram.tobytes(),c.cross.tobytes()
        if key not in cache:cache[key]=original(c)[0].copy()
        return cache[key].copy(),c.bias
    with patch.object(broad.previous.core.LocalPose,'predict',cached):
        for case in case_ids:
            cache.clear()
            for kind in ('e120','candidate'):
                method=f'{kind}_{seed}'
                if (case,method) in done:continue
                if kind=='e120':r,_=broad.previous.evaluate(case,tasks[case],method,models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded',save=True);chosen=None
                else:
                    r,chosen=evaluate(case,tasks[case],seed,models,gate);b=read(EVAL/'trials'/f'{case}_e120_{seed}_frozen_bias.json');assert r['calibration']==b['calibration'] and r['initial_estimator']==b['initial_estimator']
                    if chosen is None:
                        b['method']=r['method'];assert r==b;np.testing.assert_array_equal(np.load(EVAL/'states'/f'{case}_{method}_frozen_bias.npz')['states'],np.load(EVAL/'states'/f'{case}_e120_{seed}_frozen_bias.npz')['states'])
                rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(seed=seed,chosen=chosen));temp=path.with_suffix('.tmp');save(temp,rows);temp.replace(path)
            print('E138', 'oldheld' if oldheld else 'fresh',seed,case,len(rows),'of',2*len(case_ids),'episodes',flush=True)
    assert len(rows)==2*len(case_ids)
    for p,h in {**hashes,**read(ROOT/'frozen_training.json'),**read(EVAL/'frozen.json')}.items():assert fingerprint(p)==h

def summarize():
    rows=sum([read(EVAL/f'records_seed_{s}.json') for s in range(3)],[]);assert len(rows)==288;lookup={(r['case'],r['method']):r for r in rows};pairs=[]
    for case in range(104,152):
        for seed in range(3):
            a,b=lookup[case,f'e120_{seed}'],lookup[case,f'candidate_{seed}'];pairs.append(dict(case=case,seed=seed,dof=a['dof'],before=a['success'],after=b['success'],baseline_collision=a['collision'],collision=b['collision'],phase=b['chosen']))
    def stats(rr):return dict(n=len(rr),baseline_success=sum(r['before'] for r in rr),success=sum(r['after'] for r in rr),gains=sum(r['after'] and not r['before'] for r in rr),losses=sum(r['before'] and not r['after'] for r in rr),baseline_collisions=sum(r['baseline_collision'] for r in rr),collisions=sum(r['collision'] for r in rr))
    summary=dict(all=stats(pairs),by_dof={str(n):stats([r for r in pairs if r['dof']==n]) for n in (6,7)},changes=[r for r in pairs if r['before']!=r['after']],fresh_test=True,known_training_bodies=True,E120_unchanged=True)
    secondary=sum([read(p) for p in sorted(EVAL.glob('records_oldheld_seed_*.json'))],[])
    if secondary:
        assert len(secondary)==48;look={(r['case'],r['method']):r for r in secondary};pp=[]
        for case in sorted({r['case'] for r in secondary}):
            for seed in range(3):
                a,b=look[case,f'e120_{seed}'],look[case,f'candidate_{seed}'];pp.append(dict(case=case,seed=seed,dof=a['dof'],before=a['success'],after=b['success'],baseline_collision=a['collision'],collision=b['collision'],phase=b['chosen']))
        summary['oldheld']=stats(pp);summary['oldheld_changes']=[r for r in pp if r['before']!=r['after']];save(EVAL/'oldheld_paired.json',pp)
    save(EVAL/'records.json',rows+secondary);save(EVAL/'paired.json',pairs);save(EVAL/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(int(sys.argv[2]),'oldheld' in sys.argv)
    else:{'prepare':prepare,'summarize':summarize}[sys.argv[1]]()

"""E134: execute one-step off pulse selected by E133 OOF scalar predictor."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .structured_response import ROOT as PRIOR, preaction, fit
from .abstaining_task_value_gate import read,save,fingerprint,DATA,OLD,PHASES
from . import broad_task_value_gate as broad
from .bounded_history_correction import ROT_SCALE,rotation_error

ROOT=Path('outputs/ease_one_step_response_control_v1')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','actions'):(ROOT/folder).mkdir()
    shutil.copy2(PRIOR/'cases.json',ROOT/'cases.json');shutil.copy2(PRIOR/'folds.json',ROOT/'folds.json')
    for f in range(4):shutil.copy2(PRIOR/'models'/f'scalar_{f}.npz',ROOT/'models'/f'scalar_{f}.npz')
    d=np.load(PRIOR/'dataset.npz');w=fit('scalar',d['effect'],d['target'],d['phase']);np.savez(ROOT/'models'/'scalar_final.npz',weights=w)
    _,_,hashes=broad.setup()
    save(ROOT/'protocol.json',dict(experiment='E134',checkpoints=hashes,inputs={str(p):fingerprint(p) for p in [PRIOR/'summary.json',PRIOR/'dataset.npz',DATA/'collection/records.json',OLD/'records.json']},
        design='Fixed E133 scalar response,10um predicted one-step progress benefit, first clean phase only. Turn correction off for exactly ONE action, then resume E120. No repeated decisions after selection, physics selector or privileged state. Existing phase progress goal and measured history only. Clean flag is an evaluation/task terminal-failure rule; no contact geometry enters score.',
        evaluation='144 fitting episodes with family-excluded scalar weights; compare complete rollouts to E120. No hyperparameter selection or fitting on final task labels. If positive gain and zero E120 success losses, freeze and evaluate original24 held-placement episodes with final scalar fit to all fitting families. Previously used held placements are development validation, not a fresh independent test.',
        margin_m=1e-5,pulse_steps=1,E120_unchanged=True,held_evaluated=False))
    for name in ('one_step_response_control.py','structured_response.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in (ROOT/'models').glob('*.npz')})


def evaluate(case,t,seed,models,w):
    previous=broad.previous;method=f'candidate_{seed}';original=previous.bounded_action;rollout=previous.core.probe_rollout
    chosen=None;off=False;counts={};decisions=[]
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
                error=np.r_[env.goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
                proposal=np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
                x=broad.features(c.n,c.gram,c.cross,c.bias,obs['q'],obs['position'],env.goal,phase,c.local,action,proposal).astype(np.float64)
                effect,common,e=preaction(x);direction=e-w[0]*common;direction/=max(np.linalg.norm(direction),1e-12);pred=float(direction@(w[0]*effect));select=pred>1e-5
                decisions.append(dict(phase=phase,features=x.tolist(),predicted_progress_m=pred,selected=select))
                if select:chosen=phase
            off=phase==chosen and k==0
            return original_step(action,aperture)
        env.step=step
        return rollout(env,home,c,**kwargs)
    with patch.object(previous,'bounded_action',bounded),patch.object(previous.core,'probe_rollout',observer):r,logs=previous.evaluate(case,t,method,models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded',save=True)
    save(ROOT/'actions'/f'{case}_{method}_decisions.json',dict(chosen=chosen,decisions=decisions))
    return {k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(chosen=chosen,seed=seed)


def run(seed,held=False):
    tasks,models,hashes=broad.setup();broad.previous.ROOT=ROOT;broad.previous.core.ROOT=ROOT
    if held:
        summary=read(ROOT/'summary.json');assert summary['fitting']['gains']>0 and summary['fitting']['losses']==0
    folds=read(ROOT/'folds.json');path=ROOT/f'records_{"held" if held else "fit"}_{seed}.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if t['validation']!=held or case in done:continue
        w=np.load(ROOT/'models'/('scalar_final.npz' if held else f'scalar_{folds[str(case)]}.npz'))['weights']
        r=evaluate(case,t,seed,models,w);rows.append(r);save(path,rows);print('E134',seed,case,r['success'],r['chosen'],flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h


def summarize():
    ref={(r['case'],r['seed']):r for r in read(DATA/'collection/records.json') if r['off_phase'] is None};rows=sum([read(p) for p in sorted(ROOT.glob('records_fit_*.json'))],[])
    assert len(rows)==144
    changes=[dict(case=r['case'],seed=r['seed'],phase=r['chosen'],before=ref[r['case'],r['seed']]['success'],after=r['success']) for r in rows if ref[r['case'],r['seed']]['success']!=r['success']]
    summary=dict(fitting=dict(n=len(rows),baseline_success=sum(ref[r['case'],r['seed']]['success'] for r in rows),success=sum(r['success'] for r in rows),gains=sum(r['after'] for r in changes),losses=sum(r['before'] for r in changes),changes=changes),held_evaluated=False,E120_unchanged=True)
    save(ROOT/'summary.json',summary);save(ROOT/'records.json',rows);print(summary,flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(int(sys.argv[2]),'held' in sys.argv)
    else:{'prepare':prepare,'summarize':summarize}[sys.argv[1]]()

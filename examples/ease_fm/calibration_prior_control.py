"""E145: preserve measured calibration in locally unexcited motion directions."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import history_estimator_control as old
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_calibration_prior_control_v1')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('source','trials','states','actions'):(ROOT/folder).mkdir()
    shutil.copy2(old.ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E145',motivation='E144 constant-history matched diagnosis: preserving measured calibration lowers clean next-motion error. Test this as an analytical engineering control, not a neural gain.',controller='Initial J0 is solved from actual32-step calibration statistics at first task prediction. Every subsequent prediction uses last32 measured transitions with .98 recency and ridge1e-6, solving (Q^T W Q + lambda I) J = Q^T W X + lambda J0. Fixed J0, current measured data, frozen measured bias. Same DLS/task/probe/controller parameters. No privileged Jacobian or physics selector. Float64 solve with history castfloat32 thenfloat64, matching E144 uniform32.',evaluation='All96 fitting placements; no training, outcome-selected subset or parameter sweep. Compare E144 uniform32 to isolate prior term, and other fixed references once complete. This is development on known bodies, no fresh/held tests. E120 unchanged.',inputs={str(old.ROOT/'cases.json'):fingerprint(old.ROOT/'cases.json'),str(old.ROOT/'diagnosis_constant/summary.json'):fingerprint(old.ROOT/'diagnosis_constant/summary.json')}))
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in (ROOT/'protocol.json',ROOT/'cases.json',ROOT/'source'/Path(__file__).name)})

def run():
    _,_,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=ROOT;original_rollout=core.probe_rollout;rows=[];path=ROOT/'records.json'
    if path.exists():rows=read(path)
    done={r['case'] for r in rows}
    for case,t in enumerate(read(ROOT/'cases.json')):
        if t['validation'] or case in done:continue
        cache={};logs=[]
        def predict(c):
            if 'prior' not in cache:cache['prior']=np.linalg.solve(c.gram,c.cross)
            q,x=old.history(c);key=(q.tobytes(),x.tobytes())
            if cache.get('key')!=key:
                q=q[:,:c.n].astype(float);x=x.astype(float);w=.98**np.arange(31,-1,-1);g=q.T@(q*w[:,None])+1e-6*np.eye(c.n);cross=q.T@(x*w[:,None])+1e-6*cache['prior'];cache['j']=np.linalg.solve(g,cross).T;cache['key']=key
            return cache['j'].copy(),c.bias
        def observer(env,home,c,**kwargs):
            original_step=env.step
            def step(action,aperture):
                before=env.observe();after,command,info=original_step(action,aperture);logs.append(dict(phase=env.phase,q_before=before['q'].tolist(),nominal_action=np.asarray(action).tolist(),executed_command=command.tolist(),bad=info['bad']));return after,command,info
            env.step=step;return original_rollout(env,home,c,**kwargs)
        with patch.object(core.LocalPose,'predict',predict),patch.object(core,'probe_rollout',observer):r=core.evaluate(case,t,'calibration_prior',None,'frozen_bias',save_states=True)
        reference=read(old.ROOT/'trials'/f'{case}_constant_frozen_bias.json');assert reference['calibration']==r['calibration'] and reference['initial_estimator']==r['initial_estimator'];assert r['initial_estimator']['bias']==r['final_estimator']['bias'];np.testing.assert_array_equal(cache['prior'],np.linalg.solve(r['initial_estimator']['gram'],r['initial_estimator']['cross']))
        stem=f'{case}_calibration_prior_frozen_bias';save(ROOT/'trials'/f'{stem}.json',r);save(ROOT/'actions'/f'{stem}.json',logs);rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')});save(path,rows);print('E145',case,r['success'],flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'summary.json',{str(n):dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)) for n in ('all',6,7) if (rr:=[r for r in rows if n=='all' or r['dof']==n])});save(ROOT/'verification.json',dict(calibration_and_initial_statistics_identical=True,prior_is_actual_initial_measured_solve=True,frozen_bias=True,source_and_E120_hashes_unchanged=True));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()

"""E160: fitting-only measured-motion rate limits around frozen controllers."""
import json
import shutil
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import history_estimator_control as old
from .abstaining_task_value_gate import read, save, fingerprint
from .arm_pick_place import PickPlace

ROOT = Path('outputs/ease_task_contact_control_v1')
REFERENCE = old.ROOT

def diagnose():
    rows=read(REFERENCE/'records.json'); output={}
    for method in ('e120_ensemble','uniform32'):
        phases=Counter(); pairs=Counter(); failures=[]
        for r in rows:
            if r['method']!=method: continue
            t=read(REFERENCE/'trials'/f"{r['case']}_{method}_frozen_bias.json")
            first=next((v for v in t['trace'] if v['bad']),None)
            if first:
                phases[f"{r['dof']}:{first['phase']}"]+=1
                pairs.update('/'.join(sorted(x)) for x in first['bad'])
            if not r['success']: failures.append(dict(case=r['case'],dof=r['dof'],grasped=r['grasped'],lifted=r['lifted'],first_contact=first))
        output[method]=dict(first_contact_phases=dict(phases),first_contact_pairs=dict(pairs),failures=failures)
    ROOT.mkdir(exist_ok=True);save(ROOT/'contact_diagnosis.json',output)
    print({m:{k:v for k,v in d.items() if k!='failures'} for m,d in output.items()},flush=True)

def prepare():
    assert not (ROOT/'protocol.json').exists()
    ROOT.mkdir(exist_ok=True)
    for name in ('trials','states','actions','histories','source','limits'):(ROOT/name).mkdir()
    for name in ('cases.json','folds.json','reference_choices.json'):shutil.copy2(REFERENCE/name,ROOT/name)
    save(ROOT/'protocol.json',dict(experiment='E160',scope='All96 fitting placements on four known bodies; no E159/fresh or other held execution. Keep E120 weights and original results immutable.',
      design='Two matched controller interventions: E120 ensemble and uniform32. After the original complete action (including bounded E120 residual), subtract frozen calibrated bias and scale this joint-motion vector uniformly by min(1, .004/norm(Jv position), .0045/norm(Jv scaled-axis), .02/maxabs(v)). Add the same bias and retain the original final +/- .04 clip. J is the current measured-history estimate; no simulator Jacobian, geometry, future physics, object pose or contact input. This is a rate limiter, not a safety guarantee.',
      isolation='Preserve all task waypoints, phase budgets/termination, success and contact criteria, initial32 calibration, close32 probe, history updates and frozen bias. Apply only after calibration and never to probe. No new checkpoint training or sweep. Existing E144 reference episodes reused, plus exact replay of eight prespecified ancestry-root cases for both baselines before interventions.',
      verification_cases=[0,1,3,4,6,7,9,10],
      evaluation='Report all96 success, six/seven success, collision count, first-contact phases, paired gains/losses for each estimator and against E120. Positive fitting screen requires improvement over own baseline, neither DOF worse, no higher collisions; separately report zero-loss retention. No automatic held testing or promotion. If limiting slows progress, report phase-budget failures rather than extending budgets after outcomes.',
      parameters=dict(position_step_m=.004,scaled_axis_step=.0045,joint_step_rad=.02),
      inputs={str(REFERENCE/n):fingerprint(REFERENCE/n) for n in ('records.json','cases.json','frozen.json')}))
    for name in ('task_contact_control.py','history_estimator_control.py','training_estimator_core.py','local_probe.py','bounded_history_correction.py','local_history.py','arm_pick_place.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def evaluate(case,task,method,models,choice,enabled):
    core=old.broad.previous.core; original_rollout=core.probe_rollout; logs=[]
    class LimitedArm(PickPlace):
        def step(self,action,aperture):
            c=getattr(self,'limit_controller',None)
            if c is not None:
                original=np.asarray(action).copy(); final=original.copy(); scale=1.; bias=c.bias.copy(); motion=original-bias; predicted=None
                if self.phase!='probe':
                    j,_=c.predict();predicted=j@motion
                    if enabled:
                        scale=min(1.,.004/max(np.linalg.norm(predicted[:3]),1e-12),.0045/max(np.linalg.norm(predicted[3:]),1e-12),.02/max(np.max(np.abs(motion)),1e-12))
                        final=np.clip(bias+scale*motion,-.04,.04)
                logs.append(dict(phase=self.phase,base_action=original.tolist(),final_action=final.tolist(),bias=bias.tolist(),scale=float(scale),predicted_motion=None if predicted is None else predicted.tolist(),q_before=self.observe()['q'].tolist()))
                action=final
            return super().step(action,aperture)
    def rollout(env,home,c,**kwargs):
        env.limit_controller=c
        return original_rollout(env,home,c,**kwargs)
    with patch.object(old,'ROOT',ROOT),patch.object(core,'PickPlace',LimitedArm),patch.object(core,'probe_rollout',rollout):
        result=old.evaluate(case,task,method,models,None,choice)
    stem=f'{case}_{method}_frozen_bias';save(ROOT/'limits'/f'{stem}.json',logs)
    trial=read(ROOT/'trials'/f'{stem}.json');assert len(logs)==len(trial['trace'])
    for l in logs:
        if l['phase']=='probe' or not enabled:np.testing.assert_array_equal(l['base_action'],l['final_action'])
        else:
            predicted=np.array(l['predicted_motion']);s=min(1.,.004/max(np.linalg.norm(predicted[:3]),1e-12),.0045/max(np.linalg.norm(predicted[3:]),1e-12),.02/max(np.max(np.abs(np.array(l['base_action'])-l['bias'])),1e-12))
            assert s==l['scale'];np.testing.assert_array_equal(np.clip(np.array(l['bias'])+s*(np.array(l['base_action'])-l['bias']),-.04,.04),l['final_action'])
    states=np.load(ROOT/'states'/f'{stem}.npz')['states'];assert len(states)==len(logs)
    if not enabled:
        baseline=read(REFERENCE/'trials'/f'{stem}.json')
        for k in ('trace','calibration','initial_estimator','final_estimator','success','collision','phases'):assert trial[k]==baseline[k],k
        np.testing.assert_array_equal(states,np.load(REFERENCE/'states'/f'{stem}.npz')['states'])
    return result

def run(method,verify=False):
    _,models,hashes=old.broad.setup();old.broad.previous.ROOT=ROOT;old.broad.previous.core.ROOT=ROOT
    tasks=read(ROOT/'cases.json');choices=read(ROOT/'reference_choices.json');folds=read(ROOT/'folds.json')
    path=ROOT/f"{'replay' if verify else 'records'}_{method}.json";rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    cases=read(ROOT/'protocol.json')['verification_cases'] if verify else [i for i,t in enumerate(tasks) if not t['validation']]
    if not verify:assert len(read(ROOT/f'replay_{method}.json'))==8
    for case in cases:
        assert not tasks[case]['validation']
        if case in done:continue
        r=evaluate(case,tasks[case],method,models,choices[folds[str(case)]],not verify);rows.append(r);save(path,rows);print('E160',method,'replay' if verify else 'limited',case,r['success'],flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    if sys.argv[1] in ('run','replay'):run(sys.argv[2],sys.argv[1]=='replay')
    else:{'diagnose':diagnose,'prepare':prepare}[sys.argv[1]]()

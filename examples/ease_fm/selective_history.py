"""E122: causal measured-prediction gate on the fixed E120 residual."""
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import bounded_history_correction as previous
from .bounded_history_correction import fingerprint,ROT_SCALE
ROOT=Path('outputs/ease_selective_history_v1')
OLD=Path('outputs/ease_bounded_history_v1')

def setup():
    tasks,models,hashes=previous.setup();previous.ROOT=ROOT;previous.core.ROOT=ROOT
    return tasks,models,hashes

def evaluate(case,task,method,base,correction,selective=True,save=True):
    original_rollout=previous.core.probe_rollout;original_action=previous.bounded_action
    scores=np.zeros(2);count=0;allow=True;ledger=[]
    def bounded(a,b):return original_action(a,b) if allow or not selective else a.copy()
    def observer(env,home,controller,**kwargs):
        original_step=env.step
        def step(action,aperture):
            nonlocal scores,count,allow
            before=env.observe();jb,_=controller.predict();oldmodel=controller.model
            try:
                controller.model=correction;jc,_=controller.predict()
            finally:controller.model=oldmodel
            # Only completed preceding transitions can influence this decision.
            allow=bool(count<8 or scores[1]<=.95*scores[0]);pre=scores.copy()
            after,command,info=original_step(action,aperture)
            dq=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
            errors=np.array([np.sum((j@dq-dx)**2) for j in (jb,jc)])
            ledger.append(dict(step=count,phase=env.phase,previous_scores=pre.tolist(),allow=allow,active=env.phase!='probe',errors=errors.tolist(),dq=dq.tolist(),dx=dx.tolist(),base_prediction=(jb@dq).tolist(),correction_prediction=(jc@dq).tolist(),clean_before=count==0 or not ledger[-1]['collision_seen'],collision_seen=bool(env.bad)))
            scores=.9*scores+.1*errors;count+=1
            return after,command,info
        env.step=step
        return original_rollout(env,home,controller,**kwargs)
    with patch.object(previous.core,'probe_rollout',observer),patch.object(previous,'bounded_action',bounded):
        result,logs=previous.evaluate(case,task,method,base,correction,'bounded',save=save)
    if save:(ROOT/'actions'/f'{case}_{method}_gate.json').write_text(json.dumps(ledger))
    return result,logs,ledger

def prepare():
    tasks,models,hashes=setup();ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','actions','source','diagnosis'):(ROOT/d).mkdir()
    shutil.copy2(OLD/'cases.json',ROOT/'cases.json');rows=[]
    for r in json.loads((OLD/'records.json').read_text()):
        if r['method'].startswith('unbounded_'):continue
        original=r['method'];r['method']=original.replace('candidate_','unbounded_');a=f"{r['case']}_{original}_frozen_bias";b=f"{r['case']}_{r['method']}_frozen_bias"
        trial=json.loads((OLD/'trials'/f'{a}.json').read_text());trial['method']=r['method'];(ROOT/'trials'/f'{b}.json').write_text(json.dumps(trial))
        for ext in ('xml','npz'):shutil.copy2(OLD/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    assert len(rows)==168;(ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'protocol.json').write_text(json.dumps(dict(experiment='E122',checkpoints=hashes,inputs={str(OLD/n):fingerprint(OLD/n) for n in ('records.json','summary.json','cases.json')},
        gate='Retain E120 for first8 task transitions. Thereafter enable its unchanged25% residual capped0.001 rad/joint only if E119 preceding EWMA squared6D motion prediction error <=0.95 times E118 error. EWMA0.9 decay, zero initialization, no phase reset. Predictions made before observing each transition; errors from actual measured joint/task motion. Probe unchanged; its measured transitions contribute evidence. Gate uses no outcomes, identities, contacts, physics simulation or future transitions. No cumulative budget.',
        diagnosis='Exactly replay six E120 gains and four E120 losses versus E118 with shadow causal gate, including held gains only as diagnostic; threshold fixed before inspection. Compare clean pre-collision phase error evidence. Association is not causal action value.',
        evaluation='One fixed gate, all24 training-body placements x3 seeds. No fitting or parameter sweep; compare all E118/E120 successes and six specific gains. No unseen-body evaluation or physics selector.'),indent=2))
    for n in ('selective_history.py','bounded_history_correction.py','training_estimator_core.py','local_probe.py','local_history.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)

def diagnose():
    tasks,models,hashes=setup();summary=json.loads((OLD/'summary.json').read_text());rows=[]
    for p in summary['paired']:
        seed=p['seed']
        for change in p['changes']:
            case=change['case'];method=f'unbounded_{seed}';stem=f'{case}_{method}_frozen_bias';old=json.loads((ROOT/'trials'/f'{stem}.json').read_text());states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy()
            r,logs,ledger=evaluate(case,tasks[case],method,models[f'reference_{seed}'],models[f'correction_{seed}'],selective=False)
            assert r==old;np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states'])
            clean=[x for x in ledger if x['active'] and x['clean_before']]
            row=dict(case=case,seed=seed,gain=change['gain'],group=change['group'],steps=len(clean),gate_on_fraction=float(np.mean([x['allow'] for x in clean])),base_error=float(np.mean([x['errors'][0] for x in clean])),correction_error=float(np.mean([x['errors'][1] for x in clean])))
            row['phases']={phase:dict(n=len(pp),gate_on_fraction=float(np.mean([x['allow'] for x in pp])),base_error=float(np.mean([x['errors'][0] for x in pp])),correction_error=float(np.mean([x['errors'][1] for x in pp]))) for phase in sorted(set(x['phase'] for x in clean)) if (pp:=[x for x in clean if x['phase']==phase])}
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='phases'}),flush=True)
    (ROOT/'diagnosis/summary.json').write_text(json.dumps(rows,indent=2))

def run():
    tasks,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==168
    for case,t in enumerate(tasks):
        for seed in range(3):
            r,_,_=evaluate(case,t,f'candidate_{seed}',models[f'reference_{seed}'],models[f'correction_{seed}'])
            old=json.loads((ROOT/'trials'/f'{case}_reference_{seed}_frozen_bias.json').read_text());assert r['calibration']==old['calibration'] and r['initial_estimator']==old['initial_estimator']
            rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('evaluated',case,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    {'prepare':prepare,'diagnose':diagnose,'evaluate':run}[sys.argv[1]]()

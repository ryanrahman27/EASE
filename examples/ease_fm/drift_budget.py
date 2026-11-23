"""E121: diagnose cumulative residual drift; fixed episode residual budget."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import bounded_history_correction as previous
from .bounded_history_correction import fingerprint
ROOT=Path('outputs/ease_drift_budget_v1')
OLD=previous.ROOT
BUDGET=.004


def setup():
    tasks,models,hashes=previous.setup();previous.ROOT=ROOT;previous.core.ROOT=ROOT
    return tasks,models,hashes


def evaluate(case,task,method,base,correction,save=True):
    cumulative=np.zeros(task['arm']['dof']) if 'dof' in task['arm'] else None
    ledger=[];original=previous.bounded_action
    def bounded(a,b):
        nonlocal cumulative
        if cumulative is None:cumulative=np.zeros_like(a)
        before=cumulative.copy();raw=original(a,b)-a
        delta=np.clip(raw,-BUDGET-cumulative,BUDGET-cumulative)
        result=np.clip(a+delta,-.04,.04);cumulative+=result-a
        assert np.max(np.abs(cumulative))<=BUDGET+1e-12
        ledger.append(dict(before=before.tolist(),requested=raw.tolist(),applied=(result-a).tolist(),after=cumulative.tolist()))
        return result
    with patch.object(previous,'bounded_action',bounded):
        result,logs=previous.evaluate(case,task,method,base,correction,'bounded',save=save)
    if save:(ROOT/'actions'/f'{case}_{method}_budget.json').write_text(json.dumps(ledger))
    return result,logs,ledger


def prepare():
    tasks,models,hashes=setup();ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','actions','source','diagnosis'):(ROOT/d).mkdir()
    shutil.copy2(OLD/'cases.json',ROOT/'cases.json');rows=[]
    for r in json.loads((OLD/'records.json').read_text()):
        if r['method'].startswith('unbounded_'):continue
        source=r['method'];r['method']=source.replace('candidate_','unbounded_')
        a=f"{r['case']}_{source}_frozen_bias";b=f"{r['case']}_{r['method']}_frozen_bias"
        trial=json.loads((OLD/'trials'/f'{a}.json').read_text());trial['method']=r['method'];(ROOT/'trials'/f'{b}.json').write_text(json.dumps(trial))
        for ext in ('xml','npz'):shutil.copy2(OLD/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    assert len(rows)==168;(ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'protocol.json').write_text(json.dumps(dict(experiment='E121',checkpoints=hashes,inputs={str(OLD/n):fingerprint(OLD/n) for n in ('records.json','summary.json','cases.json')},budget_rad=BUDGET,
        intervention='Same frozen E118/E119 models and E120 25% residual capped0.001 rad per joint per step. Additionally bound signed accumulated applied residual to +/-0.004 rad/joint across the entire task, never reset at phase boundaries. Opposite corrections can unwind the ledger. Calibration and explicit probe unchanged. This is a command-residual budget, NOT a bound on actual joint or Cartesian drift, contact or safety.',
        evaluation='All24 training-body placements x3 seeds; fixed single budget, no fitting or sweep. Require recovery of all four E120 regressions while preserving all52 E120 successes and original E118 successes. No physics selector or unseen-body evaluation.'),indent=2))
    for name in ('drift_budget.py','bounded_history_correction.py','training_estimator_core.py','local_probe.py','local_history.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)


def diagnose():
    tasks,models,hashes=setup();findings=[]
    summary=json.loads((OLD/'summary.json').read_text())
    for p in summary['paired']:
        seed=p['seed']
        for change in p['changes']:
            if change['gain']:continue
            case=change['case'];method=f'reference_{seed}';stem=f'{case}_{method}_frozen_bias'
            # Recollect exact baseline actions, without changing any source artifact.
            old=json.loads((ROOT/'trials'/f'{stem}.json').read_text())
            result,baseline=previous.evaluate(case,tasks[case],method,models[method],models[f'correction_{seed}'],'baseline',save=False)
            assert result==old
            (ROOT/'diagnosis'/f'{case}_{seed}_baseline_actions.json').write_text(json.dumps(baseline))
            corrected=json.loads((OLD/'actions'/f'{case}_candidate_{seed}_frozen_bias.json').read_text())
            trial=json.loads((OLD/'trials'/f'{case}_candidate_{seed}_frozen_bias.json').read_text())
            # Match phase and within-phase step, not global time after timing diverges.
            index={};counts={}
            for r in baseline:
                phase=r['phase'];k=counts.get(phase,0);counts[phase]=k+1;index[phase,k]=r
            counts={};cum=np.zeros(7);series=[];first_collision=None
            for i,r in enumerate(corrected):
                phase=r['phase'];k=counts.get(phase,0);counts[phase]=k+1
                if r['active']:cum+=np.array(r['nominal_action'])-r['base_action']
                ref=index.get((phase,k));entry=dict(step=i,phase=phase,phase_step=k,signed_residual_max_rad=float(np.max(np.abs(cum))),signed_residual=cum.tolist(),collision=bool(trial['trace'][i]['bad']))
                if ref is not None:
                    entry.update(q_drift_max_rad=float(np.max(np.abs(np.array(r['q_before'])-ref['q_before']))),xyz_drift_mm=float(np.linalg.norm(np.array(r['position_before'])-ref['position_before'])*1000),base_action_drift_max_rad=float(np.max(np.abs(np.array(r['base_action'])-ref['base_action']))),gram_drift_norm=float(np.linalg.norm(np.array(r['gram'])-ref['gram'])),cross_drift_norm=float(np.linalg.norm(np.array(r['cross'])-ref['cross'])))
                series.append(entry)
                if entry['collision'] and first_collision is None:first_collision=entry
            (ROOT/'diagnosis'/f'{case}_{seed}_drift.json').write_text(json.dumps(series,indent=2))
            finding=dict(case=case,seed=seed,first_collision=first_collision,max_signed_residual_rad=max(r['signed_residual_max_rad'] for r in series),baseline_exact=True)
            findings.append(finding);print(json.dumps(finding),flush=True)
    (ROOT/'diagnosis/summary.json').write_text(json.dumps(findings,indent=2))


def run():
    tasks,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==168
    for case,t in enumerate(tasks):
        for seed in range(3):
            r,logs,ledger=evaluate(case,t,f'candidate_{seed}',models[f'reference_{seed}'],models[f'correction_{seed}'])
            ref=json.loads((ROOT/'trials'/f'{case}_reference_{seed}_frozen_bias.json').read_text());assert r['calibration']==ref['calibration'] and r['initial_estimator']==ref['initial_estimator']
            rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('evaluated',case,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h


if __name__=='__main__':
    import sys
    {'prepare':prepare,'diagnose':diagnose,'evaluate':run}[sys.argv[1]]()

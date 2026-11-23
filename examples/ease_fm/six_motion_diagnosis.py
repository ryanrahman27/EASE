"""E116 same-history response audit plus whole-task estimator substitution."""
import json,shutil
from pathlib import Path
import torch
import numpy as np
from .six_motion_core import ROOT,load,evaluate
from .benchmark import fingerprint
OLD=Path('outputs/ease_frozen_bias_v1')

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for name in ('trials','states','source'):(ROOT/name).mkdir()
    tasks,models,hashes=load();cases=[i for i,t in enumerate(tasks) if len(t['arm']['lengths'])==6];assert len(cases)==8
    inputs={str(OLD/'records.json'):fingerprint(OLD/'records.json')}
    for case in cases:
        for method in models:
            for suffix in ('json',):
                p=OLD/'trials'/f'{case}_{method}_frozen_bias.{suffix}';inputs[str(p)]=fingerprint(p)
    spec=dict(experiment='E116',checkpoints=hashes,inputs=inputs,
        design='All8 six-joint E115 tasks,3 learned checkpoints plus analytical reference:32 exact baseline replays. Both estimators predict the same next measured transition from identical pre-update gram/cross statistics. Report clean pre-first-collision and phase-specific residuals, response/action disagreement, command saturation and history conditioning. No exact response Jacobians.',
        interventions='For all24 learned checkpoint/task pairs including successes, replace only predicted response matrix with analytical ridge from identical current history starting either at local probe or at lift (48 full episodes). Continue same response/history updates, frozen calibrated bias, task budgets and physical scene; no reset at handoff. This is causal estimator substitution, not training or a promoted hybrid. No geometry filter, physics selector or privileged Jacobian.',
        verification='Baselines match E115 full records and qpos/qvel arrays exactly; switched runs share exact baseline trace prefix before intervention. Same initial statistics/calibration, unchanged hashes. Prediction residual comparisons are same-transition diagnostics; trajectory correlations alone do not prove causes.',
        videos='First baseline success/failure per learned seed, first substitution recovery and regression per condition; include baseline and both interventions for those pairs. All outcomes retained. Reused development tasks; no final test.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for n in ('build_six_diagnosis.py','six_motion_core.py','six_motion_diagnosis.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)
    rows=[]
    for case in cases:
        for method,model in models.items():
            old=json.loads((OLD/'trials'/f'{case}_{method}_frozen_bias.json').read_text())
            baseline=None
            for condition in (('baseline',) if model is None else ('baseline','probe','lift')):
                r=evaluate(case,tasks[case],method,model,'frozen_bias',save_states=True,condition=condition)
                if condition=='baseline':
                    assert {k:v for k,v in r.items() if k not in ('condition','diagnostics')}==old
                    np.testing.assert_array_equal(np.load(ROOT/'states'/f'{case}_{method}_baseline.npz')['states'],np.load(OLD/'states'/f'{case}_{method}_frozen_bias.npz')['states']);baseline=r
                else:
                    idx=next(i for i,v in enumerate(baseline['trace']) if v['phase']==condition)
                    assert r['trace'][:idx]==baseline['trace'][:idx]
                    assert r['initial_estimator']==baseline['initial_estimator'] and r['calibration']==baseline['calibration']
                (ROOT/'trials'/f'{case}_{method}_{condition}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','diagnostics','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('case',case,'complete',len(rows),flush=True)
    for p,h in {**hashes,**inputs}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_baselines=32,exact_prefixes=48,unchanged_inputs_weights=True,privileged_jacobian_blocked=True),indent=2))

if __name__=='__main__':run()

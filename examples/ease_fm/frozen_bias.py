"""E115: complete motion/bias update factorial with 64 new episodes."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from .frozen_bias_core import ROOT,load,evaluate
from .benchmark import fingerprint
OLD=Path('outputs/ease_adaptation_ablation_v1')

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','source'):(ROOT/d).mkdir()
    tasks,models,hashes=load();oldrows=json.loads((OLD/'records.json').read_text());assert len(oldrows)==192
    inputs={str(p):fingerprint(p) for p in OLD.rglob('*') if p.is_file() and p.parent.name in ('trials','states')}
    for name in ('protocol.json','records.json','verification.json'):inputs[str(OLD/name)]=fingerprint(OLD/name)
    spec=dict(experiment='E115',new_episodes=64,reused_episodes=192,checkpoints=hashes,inputs=inputs,
        design='Complete E114 2x2 response/bias update factorial. Add frozen_bias for all16 tasks x analytical and3 learned checkpoints. Bias fixed exactly at post-calibration value on every task transition; response gram/cross/history continue updating, with same recent32 switch and local probe. Same tasks, calibration, checkpoint throughout, phase sequence, gains, physics and success criteria. No selector, geometry filter, task Jacobian or model-derived prior.',
        comparison='Reuse all192 E114 controls without counting them as new trials. Exactly reexecute case0 and8 analytical/learned_0 across old3 modes using extended evaluator, including full states. Assert initial estimator and calibration identical to E114 for all64 new episodes. Check bias invariance every update. No tuning or checkpoint promotion. Reused unseen development bodies, reserved final test untouched.',
        videos='First new-condition success/failure per checkpoint. Paired new/online clips for first gain and loss per DOF for learned_0. Include additional new-condition exact replays for outcome categories. Outcome-selected diagnostics.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for folder in ('trials','states'):
        for p in (OLD/folder).iterdir():shutil.copy2(p,ROOT/folder/p.name)
    for name in ('build_frozen_bias.py','frozen_bias_core.py','frozen_bias.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    checks=[]
    for case in (0,8):
        for method in ('analytical','learned_0'):
            for mode in ('online','frozen_response','frozen_all'):
                states=[]
                def capture(env):states.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
                r=evaluate(case,tasks[case],method,models[method],mode,capture=capture)
                stem=f'{case}_{method}_{mode}';assert json.loads(json.dumps(r))==json.loads((OLD/'trials'/f'{stem}.json').read_text())
                np.testing.assert_array_equal(states,np.load(OLD/'states'/f'{stem}.npz')['states']);checks.append(stem)
        print('control replay',case,'exact',flush=True)
    rows=oldrows.copy()
    for case,t in enumerate(tasks):
        for method,model in models.items():
            r=evaluate(case,t,method,model,'frozen_bias',save_states=True)
            old=json.loads((OLD/'trials'/f'{case}_{method}_online.json').read_text())
            assert r['calibration']==old['calibration'] and r['initial_estimator']==old['initial_estimator']
            (ROOT/'trials'/f'{case}_{method}_frozen_bias.json').write_text(json.dumps(r))
            rows.append({k:v for k,v in r.items() if k not in ('trace','calibration','initial_estimator','final_estimator','telemetry')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('case',case,'new episodes',len(rows)-192,flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(new_episodes=64,reused_episodes=192,exact_old_controls=checks,all_initial_states_matched=True,bias_invariant_every_update=True,inputs_weights_unchanged=True),indent=2))

if __name__=='__main__':run()

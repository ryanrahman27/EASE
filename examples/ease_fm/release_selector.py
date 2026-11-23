"""E113: release-only extension of E112, six-joint development diagnosis."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import unseen_progress_selector as frozen
from .unseen_progress import models
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_release_selector_v1')
OLD=Path('outputs/ease_unseen_progress_v1')

def evaluate(case,t,seed,mode,parents,students,record_step=None,affected_phase=None,timings=None):
    phases=frozen.PHASES+('release',) if mode=='physics' else frozen.PHASES
    with patch.object(frozen,'PHASES',phases):
        r=frozen.evaluate(case,t,seed,'physics',parents,students,record_step,timings=timings)
    r.update(body=t['body'],candidate=t['candidate'],validation=True)
    ref=json.loads((OLD/'trials'/f'{case}_{seed}_None_physics.json').read_text())
    if mode=='baseline':assert json.loads(json.dumps(r))==ref
    else:
        before=lambda x:[v for v in x['trace'] if v['phase'] not in ('release','retreat','settle')]
        assert before(r)==before(ref) and r['calibration']==ref['calibration']
    r['mode']=mode
    return r

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for name in ('trials','source'):(ROOT/name).mkdir()
    tasks=json.loads((OLD/'cases.json').read_text())[:8];assert all(len(t['arm']['lengths'])==6 for t in tasks)
    AXES.update(json.loads((OLD/'axis_catalog.json').read_text()));parents,students,hashes=models()
    base=Path(__file__).parent
    inputs={str(p):fingerprint(p) for p in [OLD/'cases.json',OLD/'axis_catalog.json',OLD/'summary.json',base/'unseen_progress_selector.py',base/'progress_action_bank.py',base/'unseen_progress_rollout.py']}
    spec=dict(experiment='E113',inputs=inputs,checkpoints=hashes,
        design='All24 six-joint E112 task/seed pairs, original frozen selector versus same selector extended to release:48 full episodes. Reused development cohort, not independent transfer evidence. No checkpoint/threshold/bank/ranking/aperture/budget changes. Release retains original1% pose-error progress criterion, even near its target. Before first forbidden contact only; no rescue after prior collision.',
        gate='Strict task-success improvement, retain every original-selector success, nonincreasing collisions and no per-seed decline. All failures included. No tuning after outcomes.',
        verification='Every original-selector full serialized replay equals E112. All extended runs equal original before release. Every selected prediction matches execution. Diagnose candidate constraint intersections at the first collision and log release decisions.',
        videos='Case0 seed0 both modes, every gain/loss both modes, first remaining collision and contact-free failure both modes. Also first release-warning pair, even if outcome unchanged. Outcome-selected illustrations, full serialized replay.',
        limits='Exact simulator state/model access remains privileged. Fixed40ms horizon; unchanged progress gate may preclude release safety actions. This experiment isolates extending phase eligibility, not optimizing release.',reserved_test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));(ROOT/'cases.json').write_text(json.dumps(tasks,indent=2));shutil.copy2(OLD/'axis_catalog.json',ROOT/'axis_catalog.json')
    for name in ('release_selector.py','unseen_progress_selector.py','unseen_progress_rollout.py','progress_action_bank.py'):shutil.copy2(base/name,ROOT/'source'/name)
    rows=[];timings=[]
    for case,t in enumerate(tasks):
        for seed in range(3):
            for mode in ('baseline','physics'):
                tt=[];r=evaluate(case,t,seed,mode,parents,students,timings=tt)
                for x in tt:x['mode']=mode
                timings.extend(tt);(ROOT/'trials'/f'{case}_{seed}_None_{mode}.json').write_text(json.dumps(r));a=r['physics_audit']
                rows.append(dict(**{k:v for k,v in r.items() if k not in ('trace','contacts','calibration','physics_audit')},changed_actions=sum(x['changed'] for x in a),release_changes=sum(x['changed'] and x['phase']=='release' for x in a),release_warnings=sum(x['phase']=='release' and x['unsafe_original'] and x['clean'] and not x['prior_episode_collision'] for x in a),executed_predictions=len(a)))
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'timings.json').write_text(json.dumps(timings));print('case',case,'complete',len(rows),flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_e112_replays=24,exact_pre_release_prefixes=24,exact_predictions=sum(r['executed_predictions'] for r in rows),hashes_unchanged=True),indent=2))

if __name__=='__main__':run()

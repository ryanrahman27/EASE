"""E161: fixed elevated release waypoint, otherwise unchanged E120 ensemble."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import history_estimator_control as old
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_release_clearance_control_v1')
REFERENCE=old.ROOT

def prepare():
    ROOT.mkdir(exist_ok=False)
    for f in ('trials','states','actions','histories','source'):(ROOT/f).mkdir()
    for f in ('cases.json','folds.json','reference_choices.json'):shutil.copy2(REFERENCE/f,ROOT/f)
    save(ROOT/'protocol.json',dict(experiment='E161',scope='All96 fitting placements, frozen E120 ensemble. No held outcomes or execution. This intervention is separate from E160 and uses the original unscaled controller.',
      motivation='Fitting audit shows pad/table contacts during lowering and retreat. The rigid fixture pad bottom is25mm below the grasp-center site when upright. Original lower/release site height245mm above200mm tabletop leaves20mm nominal pad clearance, vulnerable to tilt/tracking error. Add a fixed20mm margin rather than querying simulator collision distances.',
      design='Only lower and release grasp-center target heights change from245 to265mm. This is elevated release with a short free fall, not gentle contact placement. Original grasp, lift, transport, retreat, calibration, local probe, E120 bounded residual, controller gain/damping, phase budgets and strict task/contact criteria unchanged. No object pose/contact signal, exact Jacobian, physics selector or adaptive height search.',
      verification='Every complete state and trace before first lower action must exactly equal the original E144 E120 ensemble. Verify identical calibration and frozen bias. All96 placements retained, including pre-lower failures.',
      screen='Success strictly above original E12072/96, neither DOF worse than36/48, no more than21 collision episodes; report all rescues/regressions and lost successes separately. Fitting development only; no automatic held test or promotion.',
      lower_release_height_m=.265,inputs={str(REFERENCE/p):fingerprint(REFERENCE/p) for p in ('cases.json','records.json','frozen.json')}))
    for f in ('release_clearance_control.py','history_estimator_control.py','training_estimator_core.py','local_probe.py','bounded_history_correction.py','arm_pick_place.py'):shutil.copy2(Path(__file__).with_name(f),ROOT/'source'/f)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def run():
    _,models,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=ROOT;old.broad.previous.ROOT=ROOT;original=core.targets
    tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');choices=read(ROOT/'reference_choices.json');path=ROOT/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    def targets(xy,target):
        for name,goal,aperture,budget in original(xy,target):
            goal=goal.copy()
            if name in ('lower','release'):goal[2]=.265
            yield name,goal,aperture,budget
    for case,t in enumerate(tasks):
        if t['validation'] or case in done:continue
        with patch.object(old,'ROOT',ROOT),patch.object(core,'targets',targets):r=old.evaluate(case,t,'e120_ensemble',models,None,choices[folds[str(case)]])
        stem=f'{case}_e120_ensemble_frozen_bias';trial=read(ROOT/'trials'/f'{stem}.json');base=read(REFERENCE/'trials'/f'{stem}.json');start=next(i for i,v in enumerate(trial['trace']) if v['phase']=='lower');oldstart=next(i for i,v in enumerate(base['trace']) if v['phase']=='lower');assert start==oldstart;assert trial['trace'][:start]==base['trace'][:start]
        np.testing.assert_array_equal(np.load(ROOT/'states'/f'{stem}.npz')['states'][:start],np.load(REFERENCE/'states'/f'{stem}.npz')['states'][:start]);assert trial['initial_estimator']['bias']==trial['final_estimator']['bias'];r['exact_prefix_steps']=start;rows.append(r);save(path,rows);print('E161',case,r['success'],'prefix',start,flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(episodes=96,all_pre_lower_states_exact=True,frozen_bias=True,checkpoints_and_source_unchanged=True,no_held_execution=True))

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()

"""E162: matched release-margin intervention with analytical and learned maps."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import history_estimator_control as old
from .release_clearance_control import ROOT as E120_ROOT
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_release_clearance_transfer_v1')
UNIFORM=old.ROOT
LEARNED=Path('outputs/ease_fixed_uniform_control_v1/standard')

def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir()
    save(ROOT/'protocol.json',dict(experiment='E162',scope='Matched fitting-only controller comparison on96 placements. Added after partial E161 fitting outcomes showed rescues, before any E162 outcome. No parameter change or optimization; adaptive development, not independent validation.',
      intervention='Exactly the E161 lower/release target height265mm, original245mm for the reference. No rate limiter. E120 comparison reused from E161; additionally test uniform32 and frozen E153/E154 three-model learned weighted-map ensemble with original four family folds. No retraining, seed selection, contact/geometry input or physics selector.',
      comparisons='For each estimator, compare original versus elevated release, both DOFs, collisions and paired rescues/regressions. Then compare learned versus analytical and E120 under the same elevated-release controller. All three references stay immutable. All96 placements per estimator, no success filtering or held execution.',
      checks='All states and traces before first lower action must match the corresponding original controller exactly. Preserve calibration, fixed probe, frozen bias, histories, original task criteria and budgets.',
      interpretation='Elevated release allows a short drop; not gentle contact placement. A task-controller gain does not establish learned-estimator superiority. No held test is opened or selected by this study.'))
    for worker,source in (('uniform32',UNIFORM),('learned',LEARNED)):
        root=ROOT/worker;root.mkdir()
        for f in ('trials','states','actions','histories','source'):(root/f).mkdir()
        for f in ('cases.json','folds.json','reference_choices.json'):shutil.copy2(source/f,root/f)
        if worker=='learned':shutil.copytree(source/'models',root/'models')
        shutil.copy2(ROOT/'protocol.json',root/'protocol.json');shutil.copy2(__file__,root/'source'/Path(__file__).name)
        save(root/'frozen.json',{str(p):fingerprint(p) for p in root.rglob('*') if p.is_file() and p.name!='frozen.json'})
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def run(worker):
    root=ROOT/worker;source=LEARNED if worker=='learned' else UNIFORM;core=old.broad.previous.core;original_targets=core.targets
    def targets(xy,target):
        for name,goal,aperture,budget in original_targets(xy,target):
            goal=goal.copy()
            if name in ('lower','release'):goal[2]=.265
            yield name,goal,aperture,budget
    with patch.object(old,'ROOT',root),patch.object(core,'targets',targets):old.run(worker)
    rows=read(root/f'records_{worker}.json');assert len(rows)==96
    for r in rows:
        stem=f"{r['case']}_{r['method']}_frozen_bias";a=read(root/'trials'/f'{stem}.json');b=read(source/'trials'/f'{stem}.json');i=next(i for i,v in enumerate(a['trace']) if v['phase']=='lower');j=next(i for i,v in enumerate(b['trace']) if v['phase']=='lower');assert i==j and a['trace'][:i]==b['trace'][:i];np.testing.assert_array_equal(np.load(root/'states'/f'{stem}.npz')['states'][:i],np.load(source/'states'/f'{stem}.npz')['states'][:i]);assert a['calibration']==b['calibration'];assert a['initial_estimator']==b['initial_estimator'];assert a['initial_estimator']['bias']==a['final_estimator']['bias'];r['exact_prefix_steps']=i
    save(root/'records.json',rows);save(root/'verification.json',dict(episodes=96,all_pre_lower_states_exact=True,calibration_and_initial_estimator_identical=True,frozen_bias=True,no_held_execution=True));print('E162',worker,'verified',sum(r['success'] for r in rows),'/96',flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()

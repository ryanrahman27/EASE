"""E167: PRIVILEGED exact-motion matched route diagnostic, fitting only."""
from pathlib import Path
from unittest.mock import patch
import shutil
import numpy as np
from . import exact_map_control as exact
from .abstaining_task_value_gate import read,save,fingerprint
from .release_clearance_control import ROOT as BASE
ROOT=Path('outputs/ease_early_exact_diagnostic_v1')
CONDITIONS=('release_only','conditional_high')

def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E167',scope='PRIVILEGED DIAGNOSTIC ONLY.48 six-joint fitting placements x2 matched controllers=96 new episodes. Registered after E164/E165 outcomes, while E166 finishes. No held evaluation, no learned training or labels, E120 unchanged.',
      design='Exact current MuJoCo tool-position/tool-axis Jacobian in the original E149 DLS controller, frozen measured calibration bias and original close probe. Compare release-only265mm lower/release targets versus identical controller plus E165 once-per-episode lateral-distance>8mm conditional400mm approach target. All other goals, budgets, success and contact criteria unchanged.',
      interpretation='Matched comparison isolates route change with exact local kinematics. This is not E120: no learned reference/correction model or bounded E120 residual is used. It is not a mathematical task-success upper bound or collision-free planner. No lookahead, collision clearance query or route selection from simulator geometry. Used only to diagnose whether accurate current motion maps remove the earlier route/contact problem.',
      verification='Release-only pre-lower states and traces must exactly reproduce E149; both variants must match original calibration and frozen bias. Inactive conditional routes must reproduce the complete matched release-only trajectory. All48 placements retained.',inputs={str(BASE/p):fingerprint(BASE/p) for p in ('cases.json','records.json')}))
    for condition in CONDITIONS:
        root=ROOT/condition;root.mkdir()
        for folder in ('trials','states','actions','decisions'):(root/folder).mkdir()
    for f in ('early_exact_diagnostic.py','exact_map_control.py','arm_pick_place.py','local_probe.py'):shutil.copy2(Path(__file__).with_name(f),ROOT/'source'/f)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def run():
    _,_,hashes=exact.old.broad.setup();original_targets=exact.targets;original_rollout=exact.probe_rollout;tasks=read(ROOT/'cases.json');rows=[]
    for case,t in enumerate(tasks):
        if t['validation'] or len(t['home'])!=6:continue
        for condition in CONDITIONS:
            root=ROOT/condition;decision={}
            def rollout(env,home,c,**kwargs):
                p=env.observe()['position'];d=float(np.linalg.norm(p[:2]-env.xy));decision.update(start_position=p.tolist(),lateral_distance=d,active=bool(condition=='conditional_high' and d>.008));return original_rollout(env,home,c,**kwargs)
            def targets(xy,target):
                assert decision
                for phase,goal,aperture,budget in original_targets(xy,target):
                    goal=goal.copy()
                    if phase in ('lower','release'):goal[2]=.265
                    if phase=='approach' and decision['active']:goal[2]=.4
                    yield phase,goal,aperture,budget
            with patch.object(exact,'ROOT',root),patch.object(exact,'targets',targets),patch.object(exact,'probe_rollout',rollout):r=exact.evaluate(case,t)
            stem=f'{case}_privileged_exact_map_frozen_bias';a=read(root/'trials'/f'{stem}.json');states=np.load(root/'states'/f'{stem}.npz')['states']
            if condition=='release_only':
                b=read(exact.ROOT/'trials'/f'{stem}.json');end=next(i for i,x in enumerate(a['trace']) if x['phase']=='lower');assert end==next(i for i,x in enumerate(b['trace']) if x['phase']=='lower');assert a['trace'][:end]==b['trace'][:end];np.testing.assert_array_equal(states[:end],np.load(exact.ROOT/'states'/f'{stem}.npz')['states'][:end])
            elif not decision['active']:
                reference=ROOT/'release_only';b=read(reference/'trials'/f'{stem}.json');assert a['trace']==b['trace'];np.testing.assert_array_equal(states,np.load(reference/'states'/f'{stem}.npz')['states'])
            assert a['initial_estimator']['bias']==a['final_estimator']['bias'];save(root/'decisions'/f'{case}.json',decision);r['condition']=condition;r['active']=decision['active'];rows.append(r);save(ROOT/'records.json',rows);print('E167 privileged',condition,case,r['success'],flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(episodes=96,all_pre_lower_release_reference_states_exact=True,inactive_conditional_states_exact=True,calibration_identical=True,frozen_bias=True,no_held_execution=True,privileged_diagnostic_only=True,E120_unchanged=True))

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()

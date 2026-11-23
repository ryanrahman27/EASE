"""E165: raise the approach only when measured lateral travel is required."""
from pathlib import Path
from unittest.mock import patch
import shutil
import numpy as np
from . import history_estimator_control as old
from .release_clearance_control import ROOT as BASE
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_conditional_approach_control_v1')
REPLAYS=(6,7,9,10)

def prepare():
    ROOT.mkdir(exist_ok=False)
    for f in ('trials','states','actions','histories','source','decisions'):(ROOT/f).mkdir()
    for f in ('cases.json','folds.json','reference_choices.json'):shutil.copy2(BASE/f,ROOT/f)
    save(ROOT/'protocol.json',dict(experiment='E165',scope='All48 six-joint fitting placements plus4 seven-joint exact no-op replays;44 seven-joint results reused unchanged. No held execution. Added after partial E164 outcomes, so this is adaptive fitting development.',
      motivation='Unconditional higher approach rescues case13 but creates an unnecessary vertical excursion and descent collision at originally aligned case1. High approach also moves some approach contacts into descent, showing that extra height alone is not full collision avoidance.',
      rule='At task entry after unchanged calibration, measure lateral distance from current tool position to object XY target. For six-joint bodies only, raise approach330 to400mm only if lateral distance>8mm, the original task position tolerance. No threshold sweep, body ID, pose lookup, geometry, simulator Jacobian or physics selector. Other cases retain the E161 baseline exactly. Decision frozen once at entry; no later re-evaluation.',
      preserved='E120 ensemble weights/residual,265mm lower/release fix,240mm descent, all other targets, phase budgets, probe, frozen measured bias and strict success/contact criteria unchanged. Short-drop release remains disclosed.',
      checks='All inactive episodes must reproduce original complete states/traces exactly. Identical calibration and frozen bias for all new runs. Compare against E161 and E164 high approach; preserve all outcomes.',
      screen='Above37/48 six-joint success, fewer than10 approach/descent first-contact episodes, no more than11 collision episodes, no lost E161 six-joint successes. No automatic held evaluation or E120 replacement.',
      threshold_m=.008,raised_height_m=.400,inputs={str(BASE/p):fingerprint(BASE/p) for p in ('records.json','cases.json','verification.json')}))
    for f in ('conditional_approach_control.py','history_estimator_control.py','training_estimator_core.py','bounded_history_correction.py','local_probe.py','arm_pick_place.py'):shutil.copy2(Path(__file__).with_name(f),ROOT/'source'/f)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})

def run():
    _,models,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=ROOT;old.broad.previous.ROOT=ROOT;original_targets=core.targets;original_rollout=core.probe_rollout;tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');choices=read(ROOT/'reference_choices.json');base_rows=read(BASE/'records.json');path=ROOT/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if t['validation'] or case in done:continue
        n=len(t['home']);stem=f'{case}_e120_ensemble_frozen_bias'
        if n==7 and case not in REPLAYS:
            r=next(dict(r) for r in base_rows if r['case']==case);r['provenance']='reused_unchanged_seven_joint_baseline';rows.append(r);save(path,rows);continue
        decision={};goals=[]
        def rollout(env,home,c,**kwargs):
            position=env.observe()['position'];distance=float(np.linalg.norm(position[:2]-env.xy));decision.update(start_position=position.tolist(),xy=env.xy.tolist(),lateral_distance=distance,active=bool(n==6 and distance>.008))
            return original_rollout(env,home,c,**kwargs)
        def targets(xy,target):
            assert decision
            for name,goal,aperture,budget in original_targets(xy,target):
                goal=goal.copy()
                if name in ('lower','release'):goal[2]=.265
                if name=='approach' and decision['active']:goal[2]=.4
                goals.append(dict(phase=name,goal=goal.tolist(),budget=budget));yield name,goal,aperture,budget
        with patch.object(old,'ROOT',ROOT),patch.object(core,'targets',targets),patch.object(core,'probe_rollout',rollout):r=old.evaluate(case,t,'e120_ensemble',models,None,choices[folds[str(case)]])
        decision['goals']=goals;save(ROOT/'decisions'/f'{case}.json',decision);a=read(ROOT/'trials'/f'{stem}.json');b=read(BASE/'trials'/f'{stem}.json');assert a['calibration']==b['calibration'] and a['initial_estimator']==b['initial_estimator'];assert a['initial_estimator']['bias']==a['final_estimator']['bias']
        if not decision['active']:
            for k in ('trace','phases','success','collision','final_estimator'):assert a[k]==b[k],k
            np.testing.assert_array_equal(np.load(ROOT/'states'/f'{stem}.npz')['states'],np.load(BASE/'states'/f'{stem}.npz')['states'])
        r['active']=decision['active'];r['provenance']='new_six_joint_conditional_episode' if n==6 else 'exact_seven_joint_noop_replay';rows.append(r);save(path,rows);print('E165',case,n,'active',decision['active'],'success',r['success'],flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(new_six_joint_episodes=48,seven_joint_exact_replays=4,reused_seven_joint_episodes=44,all_inactive_states_exact=True,calibration_identical=True,frozen_bias=True,checkpoints_unchanged=True,release_fix_preserved=True,no_held_execution=True))

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()

"""E164: fixed early six-joint controller changes, E161 release fix retained."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from . import history_estimator_control as old
from .release_clearance_control import ROOT as BASE
from .abstaining_task_value_gate import read,save,fingerprint
from .arm_pick_place import PickPlace

ROOT=Path('outputs/ease_early_clearance_control_v1')
VARIANTS=('high_approach','centered_early')
SEVEN_REPLAYS=(6,7,9,10)

def prepare():
    ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir()
    save(ROOT/'protocol.json',dict(experiment='E164',scope='All48 six-joint fitting placements for each of two fixed variants. All48 seven-joint E161 episodes reused unchanged per variant; four seven-joint ancestry-root episodes replayed exactly per variant to verify no-op implementation.96 new intervention episodes plus8 exact replays, not192 independent new trials. No held evaluation.',
      baseline='Unchanged E120 three-model ensemble plus the fixed E161265mm lower/release target. Original E120 checkpoints and E161 results immutable. Preserve calibration, close32 probe, frozen measured bias, success/contact definitions and phase budgets.',
      high_approach='Only on six-joint bodies, raise approach target from330 to400mm. Original descent target240mm and all other task targets, budgets and phase rules unchanged. This may lengthen descent; do not extend its100-step budget after outcomes.',
      centered_early='Only on six-joint bodies during approach/descent, after original complete E120 action, compute comfort gradient g=-sign(q)*max(abs(q)-2.0,0). Project .05*g using N=I-J.T@solve(J@J.T+.015**2 I,J), with the live E120 reference measured-history motion estimate. Uniformly cap projected delta at.005rad/joint, then retain original +/- .04 action clip. No home attractor, fixed joint index, body lookup, extra probe or condition search. Log projection residual and actual action.',
      diagnosis='E163 privileged OFFLINE geometry audit found all11 six-joint collision episodes on body84000; logged contact labels3/4 are geometry IDs attached to link1/link2, not link3/link4. Inspected starting postures had6–9mm elbow margin and joints near limits. Geometry is never provided to either controller or estimator.',
      screen='Compare all48 six-joint success/collision/first-contact phases and every paired rescue/regression to E16137/48 successes,11 collision episodes. A qualifying variant must improve six-joint success, reduce early approach/descent first contacts, not increase collisions, and retain every E161 six-joint success. If both qualify, prefer high_approach as the simpler controller. No automatic held test or baseline replacement.',
      inputs={str(BASE/p):fingerprint(BASE/p) for p in ('records.json','cases.json','protocol.json','verification.json')},seven_replay_cases=list(SEVEN_REPLAYS)))
    for variant in VARIANTS:
        root=ROOT/variant;root.mkdir()
        for f in ('trials','states','actions','histories','control','source'):(root/f).mkdir()
        for f in ('cases.json','folds.json','reference_choices.json'):shutil.copy2(BASE/f,root/f)
        shutil.copy2(ROOT/'protocol.json',root/'protocol.json')
    for f in ('early_clearance_control.py','history_estimator_control.py','training_estimator_core.py','bounded_history_correction.py','local_probe.py','arm_pick_place.py'):shutil.copy2(Path(__file__).with_name(f),ROOT/'source'/f)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',*(ROOT/'source').glob('*.py')]})

def run(variant):
    assert variant in VARIANTS;root=ROOT/variant;_,models,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=root;old.broad.previous.ROOT=root;tasks=read(root/'cases.json');folds=read(root/'folds.json');choices=read(root/'reference_choices.json');original_targets=core.targets;original_rollout=core.probe_rollout
    rows=read(root/'records.json') if (root/'records.json').exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(tasks):
        if t['validation'] or case in done:continue
        n=len(t['home']);stem=f'{case}_e120_ensemble_frozen_bias'
        if n==7 and case not in SEVEN_REPLAYS:
            r=next(dict(r) for r in read(BASE/'records.json') if r['case']==case);r['provenance']='reused_unchanged_seven_joint_baseline';rows.append(r);save(root/'records.json',rows);continue
        logs=[]
        def targets(xy,target):
            for name,goal,aperture,budget in original_targets(xy,target):
                goal=goal.copy()
                if name in ('lower','release'):goal[2]=.265
                if n==6 and variant=='high_approach' and name=='approach':goal[2]=.400
                yield name,goal,aperture,budget
        class CenteredArm(PickPlace):
            def step(self,action,aperture):
                c=getattr(self,'early_controller',None)
                if c is not None:
                    original=np.asarray(action).copy();delta=np.zeros(self.n);residual=None;gradient=None;q=self.observe()['q'];active=self.n==6 and variant=='centered_early' and self.phase in ('approach','descend')
                    if active:
                        j,_=c.predict();gradient=-np.sign(q)*np.maximum(np.abs(q)-2.,0.);projector=np.eye(self.n)-j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),j);delta=projector@(.05*gradient);delta*=min(1.,.005/max(np.max(np.abs(delta)),1e-12));residual=j@delta;action=np.clip(original+delta,-.04,.04)
                    logs.append(dict(phase=self.phase,goal=self.goal.tolist(),q_before=q.tolist(),base_action=original.tolist(),delta=delta.tolist(),final_action=np.asarray(action).tolist(),active=active,gradient=None if gradient is None else gradient.tolist(),task_space_delta=None if residual is None else residual.tolist()))
                return super().step(action,aperture)
        def rollout(env,home,c,**kwargs):
            env.early_controller=c
            return original_rollout(env,home,c,**kwargs)
        with patch.object(old,'ROOT',root),patch.object(core,'targets',targets),patch.object(core,'PickPlace',CenteredArm),patch.object(core,'probe_rollout',rollout):r=old.evaluate(case,t,'e120_ensemble',models,None,choices[folds[str(case)]])
        save(root/'control'/f'{stem}.json',logs);trial=read(root/'trials'/f'{stem}.json');baseline=read(BASE/'trials'/f'{stem}.json');assert trial['calibration']==baseline['calibration'] and trial['initial_estimator']==baseline['initial_estimator'];assert trial['initial_estimator']['bias']==trial['final_estimator']['bias'];assert len(logs)==len(trial['trace'])
        for row in logs:
            assert row['goal'][2]==.265 if row['phase'] in ('lower','release') else True
            if not row['active']:np.testing.assert_array_equal(row['base_action'],row['final_action'])
            else:assert np.max(np.abs(np.array(row['final_action'])-row['base_action']))<=.005000000001
        if n==7:
            for k in ('trace','phases','success','collision','final_estimator'):assert trial[k]==baseline[k],k
            np.testing.assert_array_equal(np.load(root/'states'/f'{stem}.npz')['states'],np.load(BASE/'states'/f'{stem}.npz')['states']);r['provenance']='exact_seven_joint_noop_replay'
        else:r['provenance']='new_six_joint_intervention'
        rows.append(r);save(root/'records.json',rows);print('E164',variant,case,n,r['success'],flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(root/'verification.json',dict(new_six_joint_episodes=48,exact_seven_joint_replays=4,reused_seven_joint_episodes=44,release_fix_preserved=True,calibration_identical=True,frozen_bias=True,checkpoints_unchanged=True,no_held_execution=True))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:prepare()

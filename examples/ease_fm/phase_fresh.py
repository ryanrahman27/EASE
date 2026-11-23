"""E71: prospective placement validation of the frozen E70 phase budget."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .phase_retention import ContactArm,trial
from .arm_pick_place import ik,targets
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .action_objective_audit import cloned_state,copy_controller
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_phase_fresh_v1')


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    parents=json.loads(Path('outputs/ease_phase_retention_v1/protocol.json').read_text())
    inputs=parents['inputs'];bodies={};old=[]
    for split,folder in [('familiar','outputs/ease_arm_pick_place_v2'),('unseen','outputs/ease_arm_pick_place_unseen_v1')]:
        p=Path(folder);AXES.update(json.loads((p/'axis_catalog.json').read_text()))
        for t in json.loads((p/'cases.json').read_text()):
            key=(t['arm']['family'],t['arm']['seed']);bodies.setdefault(key,dict(split=split,**t));old.append(t)
    candidates=[]
    for body,t in enumerate(bodies.values()):
        for xy in ([.325,.025],[.375,-.025],[.325,-.065]):
            assert all(tuple(xy)!=tuple(r['xy']) for r in old)
            candidates.append(dict(candidate=len(candidates),body=body,arm=t['arm'],split=t['split'],xy=xy,target=[xy[0],xy[1]+.06],ik_start=t['home']))
    source_hashes={}
    for name in ('phase_retention.py','arm_pick_place.py','pick_place_control.py','retention_fresh.py','action_objective_audit.py'):
        p=Path(__file__).with_name(name);assert fingerprint(p)==fingerprint(Path('outputs/ease_phase_retention_v1/source')/name)
        source_hashes[str(p)]=fingerprint(p);shutil.copy2(p,ROOT/'source'/name)
    shutil.copy2(Path(__file__),ROOT/'source'/'phase_fresh.py')
    protocol=dict(experiment='E71',inputs=inputs,checkpoints=parents['checkpoints'],source_hashes=source_hashes,candidates=candidates,test_evaluated=False,
        design='Three fixed new XY placements per distinct previously screened body; same bodies and axis families, not new embodiments. Y transfer +6cm. All candidates fixed before IK or rollout. No replacement of rejected cases.',
        screening='Nine nominal endpoints; seed71000+candidate, ten attempts per endpoint, previous accepted endpoint as warm start, old home only as initial IK start. Existing1mm position/.003 scaled-direction tolerances and no forbidden endpoint contact. Structural endpoint filtering only, all failures retained. No policy-success screening.',
        evaluation='Frozen E70 baseline versus extended: +40 maximum steps per moving phase and +20 close, no gate. Seven frozen estimators. One32-step measured probe seed71100+candidate cloned across methods/conditions, no reset. Probe contact retained. No training or hyperparameter search.',
        decision='Require greater pooled learned success, no pooled collision increase and no reduction in per-checkpoint aggregate six-joint successes to replicate E70. Also report every casewise regression, time cost, robot-family breakdown and screening coverage. A pass supports further development validation only, not deployment, novelty or publication readiness.')
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2));screen=[];tasks=[]
    for t in candidates:
        env=ContactArm(Arm(**t['arm']),t['xy'],t['target']);q=t['ik_start'];poses=[];rng=np.random.default_rng(71000+t['candidate'])
        for phase,goal,_,_ in targets(env.xy,env.target):
            q=ik(env,goal,rng,start=q,attempts=10)
            if q is None:break
            poses.append(q.tolist())
        accepted=len(poses)==9
        screen.append(dict(candidate=t['candidate'],body=t['body'],dof=env.n,accepted=accepted,endpoints=len(poses),failed_phase=None if accepted else phase))
        if accepted:tasks.append(dict(**t,home=poses[0]))
        (ROOT/'screening.json').write_text(json.dumps(screen,indent=2))
        print('screen',t['candidate'],env.n,len(poses),flush=True)
    (ROOT/'cases.json').write_text(json.dumps(tasks,indent=2));models={'analytical':None};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==parents['checkpoints'];rows=[]
    for case,t in enumerate(tasks):
        env=ContactArm(Arm(**t['arm']),t['xy'],t['target']);initial=GuardedPose(env.n,None)
        probe=calibrate(env,t['home'],initial,71100+t['candidate'])
        (ROOT/f'probe_{case}.json').write_text(json.dumps(probe))
        for method,model in models.items():
            for mode in ('baseline','extended'):
                with cloned_state(env),patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
                    result=trial(env,t['home'],copy_controller(initial,model),mode)
                assert env.steps==32 and result['abort'] is None
                row=dict(case=case,candidate=t['candidate'],body=t['body'],family=t['arm']['family'],split=t['split'],dof=env.n,method=method,mode=mode,**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(row));rows.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,'candidate',t['candidate'],'baseline/extended',[sum(r['success'] for r in rows if r['case']==case and r['mode']==m) for m in ('baseline','extended')],flush=True)
    assert len(rows)==14*len(tasks)
    for p,h in {**inputs,**hashes,**source_hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(candidates=len(candidates),accepted=len(tasks),trials=len(rows),hashes_unchanged=True,cloned_calibration=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()

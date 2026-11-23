"""E112 prospectively frozen unseen-body evaluation of the E111 selector."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from .unseen_progress_selector import ROOT,evaluate,prior
from .mixed_sim import AXES,Arm
from .geometry_filter import GeometryArm
from .arm_pick_place import targets,ik
from .benchmark import fingerprint

OLD=Path('outputs/ease_direction_novelaxis_v1')
FROZEN=Path('outputs/ease_progress_selector_v1')

def models():
    parents,hashes=prior.prior.previous.models();students,hh=prior.prior.e99.load_students();hashes.update(hh)
    return parents,students,hashes

def prepare():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for name in ('source','trials'):(ROOT/name).mkdir()
    tasks=json.loads((OLD/'cases.json').read_text());catalog=json.loads((OLD/'axis_catalog.json').read_text());AXES.update(catalog)
    parents,students,hashes=models();candidates=[]
    for body in range(8):
        original=tasks[2*body];arm=dict(original['arm']);rng=np.random.default_rng(1120100+body)
        arm.update(seed=1120100+body,split='prospective_e112')
        for name in ('lengths','masses'):arm[name]=(np.asarray(arm[name])*rng.uniform(.9,1.1,len(arm[name]))).tolist()
        for name in ('strength','damping'):arm[name]=float(arm[name]*rng.uniform(.9,1.1))
        for xy in ([.34,-.015],[.315,.04]):
            i=len(candidates);candidates.append(dict(candidate=i,body=body,arm=arm,xy=xy,target=[xy[0],xy[1]+.06],ik_start=original['home']))
    base=Path(__file__).parent
    inputs={str(p):fingerprint(p) for p in [OLD/'cases.json',OLD/'axis_catalog.json',FROZEN/'protocol.json',FROZEN/'summary.json',base/'progress_selector.py',base/'progress_action_bank.py',base/'early_physics_rollout.py']}
    for name in ('progress_selector.py','progress_action_bank.py','early_physics_rollout.py'):
        assert fingerprint(base/name)==fingerprint(FROZEN/'source'/name)
    spec=dict(experiment='E112',inputs=inputs,checkpoints=hashes,candidates=candidates,
        design='Eight prospectively sampled bodies, two per E87 novel axis order, two fixed placements each, all three frozen checkpoint seeds, baseline versus E111 selector. Body parameter perturbations +/-10%, seeds1120100..1120107. No replacement, checkpoint selection or policy-success screening. These axis orders were used in earlier development, but are outside the original training-body layouts; new body parameters are prospective.',
        screening='All nine task endpoints must pass existing collision-free IK with10 attempts, seed1121000+candidate. Save every rejection; accepted home is first endpoint. Screening checks feasibility, not policy success. Report accepted-task and all16-candidate denominators.',
        control='Exact E111 selector/bank unchanged,40ms, progress and nonpenetration checks. No route planner, no added budgets, no retraining, no rescue after first contact. Same parent-to-retained32-step probe transition. Calibration seed38000+accepted-case and bank seed11000000+case*100000+checkpoint-seed*10000+step unchanged from E111.',
        gate='Strict overall success increase, retain every baseline success, nonincreasing collisions, no per-seed or per-DOF success decline. Report all losses even if aggregate improves. No tuning after evaluation. Body-cluster bootstrap10000 resamples seed1125000, descriptive only.',
        verification='Adapter compatibility: six prespecified E111 pairs (cases1/8/10, seeds0/1) both modes exact full serialization. All selected next states/contact unions exact; baseline and selector calibration/approach prefixes identical. Frozen source/checkpoint hashes verified before and after.',
        videos='First accepted case seed0 both modes; every paired gain/loss both modes; first remaining selector collision and first contact-free selector failure both modes; append missing outcome category. Outcome-selected illustrations, full serialized replay/frame/hash verification.',
        scope='Prospective replication cohort, not reserved final test. Exact state, physics, geometry and IK initialization are privileged. Procedural arms, not commercial MJCF/hardware. Axis families previously inspected in development; no claim of untouched axis-family evaluation.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));(ROOT/'axis_catalog.json').write_text(json.dumps(catalog,indent=2))
    for p in base.glob('*.py'):shutil.copy2(p,ROOT/'source'/p.name)
    oldtasks=json.loads((prior.prior.TASKS/'cases.json').read_text());checks=[]
    for case in (1,8,10):
        for seed in (0,1):
            for mode in ('baseline','physics'):
                r=evaluate(case,oldtasks[case],seed,mode,parents,students)
                old=json.loads((FROZEN/'trials'/f'{case}_{seed}_None_{mode}.json').read_text())
                assert json.loads(json.dumps(r))==old,(case,seed,mode)
                checks.append(dict(case=case,seed=seed,mode=mode,exact=True))
                print('compatibility',case,seed,mode,flush=True)
    (ROOT/'compatibility.json').write_text(json.dumps(checks,indent=2))
    accepted=[];screen=[]
    for t in candidates:
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);q=t['ik_start'];poses=[];rng=np.random.default_rng(1121000+t['candidate'])
        for phase,goal,_,_ in targets(env.xy,env.target):
            q=ik(env,goal,rng,start=q,attempts=10)
            if q is None:break
            poses.append(q.tolist())
        ok=len(poses)==9;screen.append(dict(candidate=t['candidate'],body=t['body'],accepted=ok,endpoints=len(poses),failed_phase=None if ok else phase))
        if ok:accepted.append(dict(**t,home=poses[0]))
        print('screen',t['candidate'],ok,flush=True)
    (ROOT/'screening.json').write_text(json.dumps(screen,indent=2));(ROOT/'cases.json').write_text(json.dumps(accepted,indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='prepared')))

def run():
    torch.set_num_threads(4);assert json.loads((ROOT/'status.json').read_text())['status']=='prepared'
    spec=json.loads((ROOT/'protocol.json').read_text());AXES.update(json.loads((ROOT/'axis_catalog.json').read_text()));tasks=json.loads((ROOT/'cases.json').read_text());parents,students,_=models();rows=[];timings=[]
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for case,t in enumerate(tasks):
        for seed in range(3):
            baseline=None
            for mode in ('baseline','physics'):
                r=evaluate(case,t,seed,mode,parents,students,timings=timings)
                if baseline is None:baseline=r
                else:
                    assert r['calibration']==baseline['calibration']
                    assert [v for v in r['trace'] if v['phase']=='approach']==[v for v in baseline['trace'] if v['phase']=='approach']
                r.update(body=t['body'],candidate=t['candidate'],validation=True)
                (ROOT/'trials'/f'{case}_{seed}_None_{mode}.json').write_text(json.dumps(r));a=r['physics_audit']
                rows.append(dict(**{k:v for k,v in r.items() if k not in ('trace','contacts','calibration','physics_audit')},changed_actions=sum(x['changed'] for x in a),executed_predictions=len(a),warnings=sum(x['clean'] and not x['prior_episode_collision'] and x['unsafe_original'] for x in a)))
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'timings.json').write_text(json.dumps(timings));print('case',case,'complete',len(rows),'rollouts',flush=True)
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_compatibility_replays=12,exact_paired_prefixes=len(rows)//2,exact_selected_predictions=sum(r['executed_predictions'] for r in rows),hashes_unchanged=True,reserved_test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);a=p.parse_args();globals()[a.stage]()

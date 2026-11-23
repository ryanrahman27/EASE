"""Prespecify new bodies before initial-posture screening or policy execution."""
from pathlib import Path
import json,shutil
import numpy as np
from .abstaining_task_value_gate import read,save,fingerprint
from .unseen_estimator import ROOT,REFERENCE,CONDITIONS,old
from .mixed_sim import AXES,Arm
from .arm_pick_place import PickPlace,ik

def signature(arm,catalog):
    return json.dumps([catalog[arm['family']],*[arm[k] for k in ('lengths','masses','strength','damping')]],sort_keys=True)

def run():
    assert not ROOT.exists();ROOT.mkdir();(ROOT/'source').mkdir()
    for p,h in read(REFERENCE/'frozen.json').items():assert fingerprint(p)==h
    _,_,hashes=old.broad.setup();catalog=read('outputs/ease_unseen_progress_v1/axis_catalog.json');AXES.update(catalog);save(ROOT/'axis_catalog.json',catalog)
    families=('six_amb_train_0','six_amb_train_1','axis_dev87_6_ZYXYXY','axis_dev87_6_ZYXYZY');parent_tasks=read(REFERENCE/'cases.json')+read('outputs/ease_direction_novelaxis_v1/cases.json');parents={}
    for f in families:
        group={t['arm']['seed']:t for t in parent_tasks if t['arm']['family']==f};parents[f]=[group[k] for k in sorted(group)]
    bodies=[];placements=([.30,-.06],[.34,-.015],[.38,.04],[.32,.07])
    for f in families:
        for variant in range(3):
            i=len(bodies);parent=parents[f][variant%len(parents[f])];arm=dict(parent['arm']);rng=np.random.default_rng(1770100+i);arm.update(seed=1770100+i,split='prospective_e177')
            for k in ('lengths','masses'):arm[k]=(np.array(arm[k])*rng.uniform(.9,1.1,6)).tolist()
            for k in ('strength','damping'):arm[k]=float(arm[k]*rng.uniform(.9,1.1))
            bodies.append(dict(body=i,arm=arm,parent_seed=parent['arm']['seed'],ik_start=parent['home']))
    # Inspect only pre-existing case manifests, not outcome files, for exact body reuse.
    manifests=list(Path('outputs').rglob('*cases*.json'));seen_seeds=set();seen_signatures=set();scanned=[]
    for p in manifests:
        if ROOT in p.parents:continue
        try:tasks=read(p)
        except (ValueError,OSError):continue
        if not isinstance(tasks,list):continue
        found=False
        for t in tasks:
            if not isinstance(t,dict) or not isinstance(t.get('arm'),dict):continue
            a=t['arm'];seen_seeds.add(a['seed']);found=True
            if a['family'] in catalog:seen_signatures.add(signature(a,catalog))
        if found:scanned.append(dict(path=str(p),sha256=fingerprint(p)))
    assert not {b['arm']['seed'] for b in bodies}&seen_seeds
    assert not {signature(b['arm'],catalog) for b in bodies}&seen_signatures
    save(ROOT/'novelty_audit.json',dict(prior_manifests=scanned,new_body_seeds=[b['arm']['seed'] for b in bodies],no_seed_overlap=True,no_physical_parameter_signature_overlap=True,axis_families_previously_used=True))
    save(ROOT/'bodies.json',bodies)
    protocol=dict(experiment='E177',scope='Prospective new procedural six-joint bodies, not untouched axis families or commercial robots.12 bodies x4 placements=48 candidate tasks; both frozen E120 and analytical. No retraining, retuning, replacement or policy-success screening.',cohort='Seeds1770100..1770111;3 bodies per four previously used axis families. Independent +/-10% length,mass,strength,damping perturbations around archived parent bodies. Both familiar and development-axis families are labeled separately. Exact body signatures and seeds checked against all pre-existing case manifests.',initialization='Only common initial approach posture is screened: target[.34,0,.33], original collision-free IK,10 starts, seed1772000+body. No descend/grasp/transport endpoint screening. All4 placements retained for each accepted body. Reject only uninitializable bodies or initial contacts; no replacements. Report all48-candidate and accepted-task denominators.',bodies=bodies,placements=placements,target_rule='Same X, Y+0.06',controller='Frozen E176 controller and estimator recipes, including early/late gates, static object-safe planning, protections, guarded handoff, release fix, history rules and success scoring. New data adapter removes only assertions against unrelated historical fitting-case files.',budget=read(REFERENCE/'protocol.json')['budget'],checkpoints=hashes,reference_protocol_sha256=fingerprint(REFERENCE/'protocol.json'),replay='Before any prospective episode, both estimators exactly replay fitting cases0,26,43,64,67:10 full compatibility episodes.',analysis='Paired outcomes, collisions, interactions, all failures, per body/family and body-cluster descriptive uncertainty. Single deterministic rollout per case/method. No post-outcome tuning. Failure videos are retained.',videos='First accepted case of each body and all success/collision-discordant cases for both estimators; append any missing outcome category. Every case has complete raw state/action/history records.',reserved_legacy_tests_evaluated=False)
    save(ROOT/'protocol.json',protocol)
    names=('unseen_estimator.py','unseen_estimator_core.py','unseen_estimator_prepare.py','build_unseen_estimator.py',*[p.name for p in (REFERENCE/'source').glob('*.py') if p.name not in ('frozen_estimator_audit.py','frozen_estimator_report.py','frozen_estimator_videos.py','frozen_estimator_failure_review.py','build_frozen_estimator_audit.py')])
    for name in set(names):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    # Protocol/source lock precedes initialization checks and prospective outcomes.
    save(ROOT/'design_frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'bodies.json',ROOT/'axis_catalog.json',*(ROOT/'source').glob('*.py')]})
    tasks=[];screen=[]
    for b in bodies:
        env=PickPlace(Arm(**b['arm']),[.34,0],[.34,.06]);home=ik(env,np.array([.34,0,.33]),np.random.default_rng(1772000+b['body']),start=b['ik_start'],attempts=10)
        body_ok=home is not None
        for variant,xy in enumerate(placements):
            candidate=b['body']*4+variant;target=[xy[0],xy[1]+.06];ok=body_ok
            if ok:
                env=PickPlace(Arm(**b['arm']),xy,target);env.reset(home);ok=not env.contacts()[1]
            screen.append(dict(candidate=candidate,body=b['body'],family=b['arm']['family'],accepted=bool(ok),reason=None if ok else ('initial_contact' if body_ok else 'no_initial_posture')))
            if ok:tasks.append(dict(candidate=candidate,body_index=b['body'],arm=b['arm'],xy=xy,target=target,home=home.tolist(),validation=True,group='prospective_e177'))
        print('body',b['body'],b['arm']['family'],'initializable',body_ok,flush=True)
    save(ROOT/'screening.json',screen);save(ROOT/'cases.json',tasks)
    for name in (*CONDITIONS,*(f'replay_{m}' for m in CONDITIONS)):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    frozen=read(ROOT/'design_frozen.json')
    for p,h in frozen.items():assert fingerprint(p)==h
    frozen.update({str(ROOT/'cases.json'):fingerprint(ROOT/'cases.json'),str(ROOT/'screening.json'):fingerprint(ROOT/'screening.json')});save(ROOT/'frozen.json',frozen);save(ROOT/'status.json',dict(status='prepared',candidate_tasks=48,accepted_tasks=len(tasks)));print('accepted',len(tasks),'of48 candidates',flush=True)

if __name__=='__main__':run()

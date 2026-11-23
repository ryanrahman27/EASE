"""E87: prospective new-morphology pilot and matched response ablation."""
import json
import itertools
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .direction_fulltask import FullTaskArm,sequence
from .geometry_filter import GeometryArm
from .arm_pick_place import rollout,ik,targets
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_direction_novelaxis_v1')


from .direction_newbody import AblationArm


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={}
    def read(p):
        inputs[str(p)]=fingerprint(p);return json.loads(p.read_text())
    prior=read(Path('outputs/ease_direction_fulltask_v1/protocol.json'));bodies={};models={'analytical':None};hashes={}
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;AXES.update(read(p/'axis_catalog.json'))
        for t in read(p/'cases.json'):bodies.setdefault((t['arm']['family'],t['arm']['seed']),t)
    manifest_path=Path('outputs/ease_paired_factorial_v2/manifest.json');manifest=read(manifest_path)
    protocols={name:read(Path('outputs')/name/'protocol.json') for name in ('ease_motion_transformer_v1','ease_motion_replicate_v1','ease_robust_training_v1','ease_coherent_pose_data_v1','ease_estimator_distillation_v1')}
    for name in ('ease_motion_transformer_v1','ease_motion_replicate_v1','ease_coherent_pose_data_v1'):
        assert protocols[name]['manifest_sha256']==fingerprint(manifest_path)
    assert protocols['ease_robust_training_v1']['train_sha256']==fingerprint(manifest_path)
    provenance_path=Path('outputs/ease_coherent_pose_data_v1/provenance.json');provenance=read(provenance_path)
    assert protocols['ease_estimator_distillation_v1']['provenance_sha256']==fingerprint(provenance_path)
    train_bodies=[a for a in manifest['arms'] if a['split']=='train']
    train_ids={a['seed'] for a in train_bodies}
    measured_ids={r['body'] for r in provenance if not r['validation'] and r['driver'] in ('analytical','original_0','original_1')}
    assert measured_ids<=train_ids
    def norm(sequence):return tuple(tuple(float(x) for x in axis.split()) for axis in sequence)
    train_sequences={norm(manifest['axis_catalog'][a['family']]) for a in train_bodies}
    excluded={norm(v) for v in manifest['axis_catalog'].values()}|{norm(v) for v in AXES.values()}
    # Development catalog exclusion only; no reserved goals, trajectories or outcomes are used.
    development=read(Path('outputs/ease_broad_development_v2/data/manifest.json'))
    excluded.update(norm(v) for v in development['axis_catalog'].values())
    lookup={'X':'1 0 0','Y':'0 1 0','Z':'0 0 1'};chosen=[]
    rng=np.random.default_rng(870000)
    for n in (6,7):
        pool=[]
        for tail in itertools.product('XYZ',repeat=n-2):
            word='ZY'+''.join(tail);axes=[lookup[c] for c in word]
            if any(a==b for a,b in zip(word,word[1:])):continue
            if norm(axes) in excluded:continue
            pool.append((word,axes))
        for index in rng.permutation(len(pool))[:2]:
            word,axes=pool[index];family=f'axis_dev87_{n}_{word}';AXES[family]=axes
            chosen.append(dict(family=family,dof=n,word=word,axes=axes))
    assert len(chosen)==4
    candidates=[]
    for family_index,family in enumerate(chosen):
        t=next(t for t in bodies.values() if len(t['arm']['lengths'])==family['dof'])
        for replicate in range(2):
            body=2*family_index+replicate;seed=870100+body;rng=np.random.default_rng(seed);arm=dict(t['arm']);arm.update(seed=seed,split='development_unseen_axis',family=family['family'])
            for name in ('lengths','masses'):arm[name]=(np.array(arm[name])*rng.uniform(.9,1.1,len(arm[name]))).tolist()
            for name in ('strength','damping'):arm[name]=float(arm[name]*rng.uniform(.9,1.1))
            for xy in ([.35,0.],[.325,.025]):
                candidates.append(dict(candidate=len(candidates),body=body,family_index=family_index,replicate=replicate,source_body_seed=t['arm']['seed'],arm=arm,xy=xy,target=[xy[0],xy[1]+.06],ik_start=t['home']))
    assert len(candidates)==16 and all(norm(r['axes']) not in excluded for r in chosen)
    axis_audit=dict(train_body_ids=sorted(train_ids),measured_distillation_body_ids=sorted(measured_ids),train_sequence_count=len(train_sequences),excluded_sequence_count=len(excluded),chosen=chosen,
        training_disjoint=True,catalog_disjoint=True,scope='Exact ordered axis sequences. All catalog metadata excluded to avoid reserved geometry reuse; no reserved goal arrays, trajectory files or outcomes used. Synthetic random linear-system training remains part of checkpoint lineage; no claim of disjoint dynamics functions.')
    (ROOT/'axis_audit.json').write_text(json.dumps(axis_audit,indent=2))
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E87',inputs=inputs,checkpoints=hashes,candidates=candidates,test_evaluated=False,
        design='Four prospective axis arrangements, two per DOF, uniformly selected without replacement from lexicographic ZY-prefixed XYZ sequences with no adjacent equal axes, excluding training and all known metadata catalogs. Fixed RNG870000; two parameterizations per arrangement seeds870100+body, same +/-10% range and two fixed placements. All16 candidates and axis exclusion audit fixed before IK/outcomes. No replacement. Novel sequences do not prove inequivalent kinematic functions.',
        screening='Same nine nominal endpoint IK rule, ten attempts each, previous endpoint warm start, old body home only initial IK start, RNG871000+candidate. Report all rejections. No policy-success screening. Measured32-step calibration RNG872000+candidate; no training or reserved-test access.',
        comparison='Every accepted case x seven frozen estimators x baseline geometry filter / sustained measured-response correction / sustained commanded-displacement correction. Same E82 candidate set, geometry, path sampling, progress constraint and thresholds. Only predictor changes in command ablation; output clipping remains +/-0.08. Both corrected modes update measured response history identically but command mode does not use it for scoring.',
        gate='Response must improve pooled learned task success over baseline without more collisions or ANY lost baseline-successful six-joint case. Attribution to measured response additionally requires higher learned success than the matched command ablation without more collisions. Report all changes, seed and DOF breakdowns, screening coverage and query costs. Pilot evidence only; do not tune on outcomes.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));(ROOT/'axis_catalog.json').write_text(json.dumps(AXES,indent=2))
    for name in ('direction_novelaxis.py','direction_newbody.py','direction_fulltask.py','direction_correction.py','response_prediction.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        p=Path(__file__).with_name(name);frozen=Path('outputs/ease_direction_newbody_v1/source')/name
        if frozen.exists():assert fingerprint(p)==fingerprint(frozen)
        shutil.copy2(p,ROOT/'source'/name)
    tasks=[];screen=[]
    for t in candidates:
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);q=t['ik_start'];poses=[];rng=np.random.default_rng(871000+t['candidate'])
        for phase,goal,_,_ in targets(env.xy,env.target):
            q=ik(env,goal,rng,start=q,attempts=10)
            if q is None:break
            poses.append(q.tolist())
        accepted=len(poses)==9
        screen.append(dict(candidate=t['candidate'],body=t['body'],dof=env.n,accepted=accepted,endpoints=len(poses),failed_phase=None if accepted else phase))
        if accepted:tasks.append(dict(**t,home=poses[0]))
        (ROOT/'screening.json').write_text(json.dumps(screen,indent=2));print('screen',t['candidate'],accepted,len(poses),flush=True)
    (ROOT/'cases.json').write_text(json.dumps(tasks,indent=2));rows=[];exact=0
    for case,t in enumerate(tasks):
        probe_ref=None
        for method,model in models.items():
            for mode in ('baseline','response','command'):
                env=AblationArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.initialize_response();c=GuardedPose(env.n,model)
                probe=calibrate(env,t['home'],c,872000+t['candidate'])
                if probe_ref is None:
                    probe_ref=probe;(ROOT/f'probe_{case}.json').write_text(json.dumps(probe))
                assert probe==probe_ref
                env.controller=c;env.filter_enabled=True;env.filter_log=[];env.correction_log=[];env.mode='baseline' if mode=='baseline' else 'sustained';env.command_ablation=mode=='command';env.first_warning=None;env.pulse_used=0
                with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
                    result=rollout(env,t['home'],c,reset=False,sequence=sequence(env))
                assert env.response.updates==env.steps
                if mode=='baseline':
                    reference=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);reference.prepare_geometry();cc=GuardedPose(env.n,model)
                    assert calibrate(reference,t['home'],cc,872000+t['candidate'])==probe_ref
                    reference.controller=cc;reference.filter_enabled=True;reference.filter_log=[]
                    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):check=rollout(reference,t['home'],cc,reset=False)
                    assert result==check and env.filter_log==reference.filter_log;exact+=1
                logs=[r['correction'] for r in env.correction_log if r['correction'] is not None]
                row=dict(case=case,body=t['body'],candidate=t['candidate'],method=method,mode=mode,dof=env.n,changed_actions=sum(r['changed'] for r in logs),correction_queries=sum(r['queries'] for r in logs),filter_queries=sum(r['queries'] for r in env.filter_log),**result)
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(dict(**row,correction_log=env.correction_log)))
                rows.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,'of',len(tasks),'complete',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,exact_baselines=exact,hashes_unchanged=True,causal_updates_verified=True,test_evaluated=False),indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()

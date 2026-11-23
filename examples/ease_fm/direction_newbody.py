"""E85: prospective new-morphology pilot and matched response ablation."""
import json
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

ROOT=Path('outputs/ease_direction_newbody_v1')


class CommandResponse:
    def predict(self,command,velocity):
        return np.clip(command,-.08,.08),int(np.sum(np.abs(command)>.08))


class AblationArm(FullTaskArm):
    def corrected_action(self,selected,aperture,jac,error,response):
        return super().corrected_action(selected,aperture,jac,error,CommandResponse() if self.command_ablation else response)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={}
    def read(p):
        inputs[str(p)]=fingerprint(p);return json.loads(p.read_text())
    prior=read(Path('outputs/ease_direction_fulltask_v1/protocol.json'));bodies={};models={'analytical':None};hashes={}
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;AXES.update(read(p/'axis_catalog.json'))
        for t in read(p/'cases.json'):bodies.setdefault((t['arm']['family'],t['arm']['seed']),t)
    candidates=[]
    for body,t in enumerate(bodies.values()):
        seed=850000+body;rng=np.random.default_rng(seed);arm=dict(t['arm']);arm['seed']=seed;arm['split']='development_new_morphology'
        for name in ('lengths','masses'):arm[name]=(np.array(arm[name])*rng.uniform(.9,1.1,len(arm[name]))).tolist()
        for name in ('strength','damping'):arm[name]=float(arm[name]*rng.uniform(.9,1.1))
        for xy in ([.35,0.],[.325,.025]):
            candidates.append(dict(candidate=len(candidates),body=body,source_body_seed=t['arm']['seed'],arm=arm,xy=xy,target=[xy[0],xy[1]+.06],ik_start=t['home']))
    assert len(bodies)==6 and len(candidates)==12
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E85',inputs=inputs,checkpoints=hashes,candidates=candidates,test_evaluated=False,
        design='Prospective development pilot: six new parameterizations, one per existing body/family, seeds850000+body; independently multiply lengths, masses, strength and damping by Uniform(.9,1.1). Same axes and topology; not new axis families or commercial robots. Two fixed placements per body, all12 candidates written before IK or rollouts. No replacement.',
        screening='Same nine nominal endpoint IK rule, ten attempts each, previous endpoint warm start, old body home only initial IK start, RNG851000+candidate. Report all rejections. No policy-success screening. Measured32-step calibration RNG852000+candidate; no training or reserved-test access.',
        comparison='Every accepted case x seven frozen estimators x baseline geometry filter / sustained measured-response correction / sustained commanded-displacement correction. Same E82 candidate set, geometry, path sampling, progress constraint and thresholds. Only predictor changes in command ablation; output clipping remains +/-0.08. Both corrected modes update measured response history identically but command mode does not use it for scoring.',
        gate='Response must improve pooled learned task success over baseline without more collisions or ANY lost baseline-successful six-joint case. Attribution to measured response additionally requires higher learned success than the matched command ablation without more collisions. Report all changes, seed and DOF breakdowns, screening coverage and query costs. Pilot evidence only; do not tune on outcomes.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));(ROOT/'axis_catalog.json').write_text(json.dumps(AXES,indent=2))
    for name in ('direction_newbody.py','direction_fulltask.py','direction_correction.py','response_prediction.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        p=Path(__file__).with_name(name);frozen=Path('outputs/ease_direction_fulltask_v1/source')/name
        if frozen.exists():assert fingerprint(p)==fingerprint(frozen)
        shutil.copy2(p,ROOT/'source'/name)
    tasks=[];screen=[]
    for t in candidates:
        env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);q=t['ik_start'];poses=[];rng=np.random.default_rng(851000+t['candidate'])
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
                probe=calibrate(env,t['home'],c,852000+t['candidate'])
                if probe_ref is None:
                    probe_ref=probe;(ROOT/f'probe_{case}.json').write_text(json.dumps(probe))
                assert probe==probe_ref
                env.controller=c;env.filter_enabled=True;env.filter_log=[];env.correction_log=[];env.mode='baseline' if mode=='baseline' else 'sustained';env.command_ablation=mode=='command';env.first_warning=None;env.pulse_used=0
                with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
                    result=rollout(env,t['home'],c,reset=False,sequence=sequence(env))
                assert env.response.updates==env.steps
                if mode=='baseline':
                    reference=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);reference.prepare_geometry();cc=GuardedPose(env.n,model)
                    assert calibrate(reference,t['home'],cc,852000+t['candidate'])==probe_ref
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

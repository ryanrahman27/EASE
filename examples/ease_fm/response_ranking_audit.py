"""E60: frozen E57 action predictions versus copied procedural dynamics."""
import argparse
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .paired_data import register
from .mixed_sim import Arm, MixedReach
from .real_arm_reach import load_models
from .retention_fresh import GuardedPose
from .procedural_pose_audit import observe, vector, reference_step, execute, direction_jacobian
from .procedural_action_audit import clone
from .action_objective_audit import operator, copy_controller
from .action_objective_report import agreement
from .joint_response_audit import features, predict


def decomposition(jac, truth, predicted_dq, actual_dq, actual_dx):
    estimator=(jac-truth)@predicted_dq
    actuator=truth@(predicted_dq-actual_dq)
    nonlinear=truth@actual_dq-actual_dx
    total=jac@predicted_dq-actual_dx
    assert np.allclose(estimator+actuator+nonlinear,total,atol=1e-12,rtol=1e-10)
    return {k:v.tolist() for k,v in dict(estimator=estimator,actuator=actuator,nonlinear=nonlinear,total=total).items()}


def run(data, parents, training, reference, response, replay, output):
    torch.set_num_threads(4)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    ref=Path(reference);source=json.loads((ref/'protocol.json').read_text())
    old=json.loads((ref/'drivers.json').read_text())
    manifest=register(data);arms={a['seed']:Arm(**a) for a in manifest['arms'] if a['split']=='train'}
    assert fingerprint(Path(data)/'manifest.json')==source['manifest_sha256']
    models={'analytical':None};hashes={}
    for condition,folder in [('original',parents),('prediction',str(Path(training)/'prediction')),('control',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{condition}_{seed}']=mm[f'neural_{seed}']
    response=Path(response);response_hashes={str(f):fingerprint(f) for f in response.glob('fold_*.npz')}
    assert len(response_hashes)==16
    previous=json.loads((Path(replay)/'records.json').read_text())
    drivers=['analytical','original_0','original_1','original_2']
    protocol=dict(experiment='E60',response_hashes=response_hashes,replay=replay,replay_sha256=fingerprint(Path(replay)/'records.json'),checkpoints=hashes,reference=reference,source_sha256=fingerprint(ref/'drivers.json'),
        manifest_sha256=source['manifest_sha256'],tasks=source['tasks'],drivers=drivers,sample_steps=source['sample_steps'],
        scope='Fixed analytical/E35 E55 histories on eight training bodies, all nine original sample times. Ten frozen candidates, one/five steps, identical history and full simulator state. No native data, training, selection or final-test evaluation. These states overlap E57 data and are mechanism diagnostics, not independent validation.',
        decomposition='predicted_dx-actual_dx = (estimated_J-true_J)*predicted_dq + true_J*(predicted_dq-actual_dq) + (true_J*actual_dq-actual_dx). predicted_dq is clipped/limited command minus measured bias. Terms are vectors and can cancel; their squared magnitudes do not add.',
        proxies='estimated: estimated_J and predicted_dq; geometric: true_J and predicted_dq (training surrogate); realized_joint: true_J and actual_dq (post-hoc actuator oracle). Residuals share first-step reference. Five-step target continues scheduled ramp, so first-step score is not an exact five-step objective.',test_evaluated=False)
    protocol['response_design']='Use immutable E59 leave-one-body-out folds: no response fit or normalization includes this body. Score original candidate commands without changing actions. Full history and command/velocity residual predictions combine with geometric or estimated Jacobians. Geometry is diagnostic only. No native data, tuning or policy training. Drivers/estimators previously saw these bodies; holdout is for response fitting only.'
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['response_ranking_audit.py','joint_response_audit.py','motion_gap_audit.py','procedural_pose_audit.py','procedural_action_audit.py','action_objective_audit.py','action_objective_report.py','mixed_sim.py','retention_fresh.py','pick_place_control.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    rows=[];replays=[];exact=0;real_jac=mujoco.mj_jacSite
    for task in source['tasks']:
        arm=arms[task['body']]
        folds={}
        for kind in ['history','command_velocity']:
            f=np.load(response/f'fold_{arm.seed}_{kind}.npz')
            folds[kind]=((f['mean'],f['scale'],f['weight']),f['columns'])
        for driver in drivers:
            expected=next(r for r in old if r['body']==arm.seed and r['driver']==driver)
            env=MixedReach(arm,horizon=1000,synchronize=True);env.reset(0,task['goals'][0][:3],spread=0)
            c=GuardedPose(arm.dof,models[driver]);reference_pose=vector(observe(env));prev_command=np.zeros(arm.dof);prev_motion=np.zeros(arm.dof)
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden in control')):
                for step,wanted in enumerate(expected['trace']):
                    matched=None;before_q=env.data.qpos.copy()
                    if step in protocol['sample_steps']:
                        obs=observe(env);start=vector(obs);error=reference_step(reference_pose,task['goals'],step)-start
                        jp=np.zeros((3,arm.dof));jr=np.zeros_like(jp);real_jac(env.model,env.data,jp,jr,0)
                        truth=np.concatenate([jp,.15*direction_jacobian(obs['rotation'],jr)])
                        candidates=[]
                        for name,model in models.items():
                            candidate=copy_controller(c,model);jac,bias=candidate.predict()
                            raw=operator(jac)@(.35*error)+bias
                            command=np.clip(obs['q']+np.clip(raw,-.04,.04),-2.5,2.5)-obs['q']
                            predicted_dq=command-bias;branch={};r=reference_pose.copy()
                            x=features(command.astype(np.float32).astype(float),obs['q'],env.data.qvel.copy(),bias,prev_command,prev_motion)
                            response_predictions={kind:command.astype(np.float32).astype(float)-bias+predict(fold,x[:,columns]) for kind,(fold,columns) in folds.items()}
                            with clone(env):
                                for offset in range(5):
                                    r,info=execute(env,candidate,r,task['goals'],step+offset)
                                    if offset==0:
                                        after=observe(env);actual_dq=after['q']-obs['q'];actual_dx=vector(after)-start
                                        np.testing.assert_array_equal(info['command'],command.astype(np.float32).astype(float))
                                        terms=decomposition(jac,truth,predicted_dq,actual_dq,actual_dx)
                                        scores={k:float(np.sum((error-dx)**2)) for k,dx in [('estimated',jac@predicted_dq),('geometric',truth@predicted_dq),('realized_joint',truth@actual_dq)]}
                                        for kind,dq in response_predictions.items():
                                            scores[kind+'_geometric']=float(np.sum((error-truth@dq)**2))
                                            scores[kind+'_estimated']=float(np.sum((error-jac@dq)**2))
                                        response_errors={kind:float(np.mean((dq-actual_dq)**2)) for kind,dq in response_predictions.items()}
                                        response_errors['baseline']=float(np.mean((predicted_dq-actual_dq)**2))
                                        assert np.isclose(info['error_squared'],np.sum((error-actual_dx)**2),atol=1e-14)
                                    if offset in [0,4]:branch[str(offset+1)]=info
                            if name==driver:matched=branch['1']
                            candidates.append(dict(method=name,scores=scores,branch=branch,decomposition=terms,response_errors=response_errors,clipped_joints=int(np.sum(np.abs(raw)>.04))))
                        rows.append(dict(case=arm.seed,robot=f'{arm.dof}dof',driver=driver,index=step,candidates=candidates))
                    reference_pose,actual=execute(env,c,reference_pose,task['goals'],step)
                    assert actual==wanted
                    prev_command=np.asarray(actual['command']);prev_motion=env.data.qpos-before_q
                    if matched is not None:assert matched==actual;exact+=1
            replays.append(dict(body=arm.seed,driver=driver,steps=len(expected['trace']),exact=True))
            (root/'trials'/f'{arm.seed}_{driver}.json').write_text(json.dumps([r for r in rows if r['case']==arm.seed and r['driver']==driver]))
            print(arm.seed,driver,'exact',flush=True)
    for path,sha in {**hashes,**response_hashes}.items():assert fingerprint(path)==sha
    for row in rows:
        old_row=next(r for r in previous if (r['case'],r['driver'],r['index'])==(row['case'],row['driver'],row['index']))
        for candidate,old_candidate in zip(row['candidates'],old_row['candidates']):
            assert candidate['method']==old_candidate['method'] and candidate['branch']==old_candidate['branch']
            for key,value in old_candidate['scores'].items():assert candidate['scores'][key]==value
    assert len(rows)==exact==288 and len(replays)==32
    (root/'records.json').write_text(json.dumps(rows))
    (root/'audit.json').write_text(json.dumps(dict(exact_replays=replays,exact_first_steps=exact,states=len(rows),branches=len(rows)*10,records_sha256=fingerprint(root/'records.json'),unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    report(root)


def report(root):
    root=Path(root);rows=json.loads((root/'records.json').read_text());p=json.loads((root/'protocol.json').read_text())
    audit=json.loads((root/'audit.json').read_text());assert fingerprint(root/'records.json')==audit['records_sha256']
    for path,sha in {**p['checkpoints'],**p['response_hashes']}.items():assert fingerprint(path)==sha
    lines=['# E60: held-out response models rank counterfactual actions','',p['response_design'],'',
        '288 fixed states, 2,880 five-step branches; all branch records and original scores exactly reproduce E58. All 32 full driver trajectories and 288 matching-driver first steps replay exactly. No actions change; new predictors score the same alternatives.', '',
        '## Body-macro pairwise agreement','','| DoF | Horizon | Bias + true J | Command/velocity + true J | History + true J | Bias + estimated J | History + estimated J | Realized motion + true J |','|---|---:|---:|---:|---:|---:|---:|---:|']
    scores=['geometric','command_velocity_geometric','history_geometric','estimated','history_estimated','realized_joint'];rankings=[]
    for robot in ['6dof','7dof']:
        rr=[r for r in rows if r['robot']==robot]
        for h in [1,5]:
            metrics={s:agreement(rr,s,h) for s in scores};rankings.append(dict(robot=robot,horizon=h,metrics=metrics))
            lines.append(f'| {robot} | {h} | '+' | '.join(f"{metrics[s]['case_mean']:.2%}" for s in scores)+' |')
    lines+=['','True J and realized motion are diagnostic oracles. Estimated J is each candidate estimator; therefore its scores include estimator error as well as response error. Ties at 1e-12 are excluded. Five-step outcomes use continuing reference ramps and online updates; a first-step prediction does not model this entire rollout.', '',
        '## Counterfactual joint-response RMSE (rad, body macro)','','| DoF | Bias baseline | Command/velocity | History |','|---|---:|---:|---:|']
    errors=[]
    for robot in ['6dof','7dof']:
        rr=[r for r in rows if r['robot']==robot];metrics={}
        for kind in ['baseline','command_velocity','history']:
            per_body={str(body):float(np.sqrt(np.mean([c['response_errors'][kind] for r in rr if r['case']==body for c in r['candidates']]))) for body in sorted({r['case'] for r in rr})}
            metrics[kind]=dict(mean=float(np.mean(list(per_body.values()))),per_body=per_body)
        errors.append(dict(robot=robot,metrics=metrics))
        lines.append(f'| {robot} | '+' | '.join(f"{metrics[k]['mean']:.6f}" for k in metrics)+' |')
    lines+=['','## Per-body first-step geometric ranking','','| Body | Bias baseline | History predictor |','|---|---:|---:|']
    for body in sorted({r['case'] for r in rows}):
        rr=[r for r in rows if r['case']==body]
        lines.append(f"| {body} | {agreement(rr,'geometric',1)['case_mean']:.2%} | {agreement(rr,'history_geometric',1)['case_mean']:.2%} |")
    contacts=sum(c['branch']['5']['collision'] for r in rows for c in r['candidates'])
    lines+=['',f'Contacts: {contacts}/2880 branches. Bodies are omitted from response fitting, but were used upstream by the motion estimators/drivers; this is not a full unseen-embodiment policy test. Histories overlap earlier diagnostics. Eight dependent body clusters, no significance or native manipulation claim. Preserve E35; no policy training or promotion.']
    (root/'summary.json').write_text(json.dumps(dict(rankings=rankings,response_errors=errors,contacts=contacts),indent=2))
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'report_audit.json').write_text(json.dumps(dict(records_sha256=fingerprint(root/'records.json'),exact_E58_branches=2880,unchanged_response_folds=16,unchanged_policy_checkpoints=9,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','parents','training','reference','response','replay','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))


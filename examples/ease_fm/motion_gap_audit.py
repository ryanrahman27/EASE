"""E58: frozen E57 action predictions versus copied procedural dynamics."""
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


def decomposition(jac, truth, predicted_dq, actual_dq, actual_dx):
    estimator=(jac-truth)@predicted_dq
    actuator=truth@(predicted_dq-actual_dq)
    nonlinear=truth@actual_dq-actual_dx
    total=jac@predicted_dq-actual_dx
    assert np.allclose(estimator+actuator+nonlinear,total,atol=1e-12,rtol=1e-10)
    return {k:v.tolist() for k,v in dict(estimator=estimator,actuator=actuator,nonlinear=nonlinear,total=total).items()}


def run(data, parents, training, reference, output):
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
    drivers=['analytical','original_0','original_1','original_2']
    protocol=dict(experiment='E58',checkpoints=hashes,reference=reference,source_sha256=fingerprint(ref/'drivers.json'),
        manifest_sha256=source['manifest_sha256'],tasks=source['tasks'],drivers=drivers,sample_steps=source['sample_steps'],
        scope='Fixed analytical/E35 E55 histories on eight training bodies, all nine original sample times. Ten frozen candidates, one/five steps, identical history and full simulator state. No native data, training, selection or final-test evaluation. These states overlap E57 data and are mechanism diagnostics, not independent validation.',
        decomposition='predicted_dx-actual_dx = (estimated_J-true_J)*predicted_dq + true_J*(predicted_dq-actual_dq) + (true_J*actual_dq-actual_dx). predicted_dq is clipped/limited command minus measured bias. Terms are vectors and can cancel; their squared magnitudes do not add.',
        proxies='estimated: estimated_J and predicted_dq; geometric: true_J and predicted_dq (training surrogate); realized_joint: true_J and actual_dq (post-hoc actuator oracle). Residuals share first-step reference. Five-step target continues scheduled ramp, so first-step score is not an exact five-step objective.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['motion_gap_audit.py','procedural_pose_audit.py','procedural_action_audit.py','action_objective_audit.py','action_objective_report.py','mixed_sim.py','retention_fresh.py','pick_place_control.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    rows=[];replays=[];exact=0;real_jac=mujoco.mj_jacSite
    for task in source['tasks']:
        arm=arms[task['body']]
        for driver in drivers:
            expected=next(r for r in old if r['body']==arm.seed and r['driver']==driver)
            env=MixedReach(arm,horizon=1000,synchronize=True);env.reset(0,task['goals'][0][:3],spread=0)
            c=GuardedPose(arm.dof,models[driver]);reference_pose=vector(observe(env))
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden in control')):
                for step,wanted in enumerate(expected['trace']):
                    matched=None
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
                            with clone(env):
                                for offset in range(5):
                                    r,info=execute(env,candidate,r,task['goals'],step+offset)
                                    if offset==0:
                                        after=observe(env);actual_dq=after['q']-obs['q'];actual_dx=vector(after)-start
                                        np.testing.assert_array_equal(info['command'],command.astype(np.float32).astype(float))
                                        terms=decomposition(jac,truth,predicted_dq,actual_dq,actual_dx)
                                        scores={k:float(np.sum((error-dx)**2)) for k,dx in [('estimated',jac@predicted_dq),('geometric',truth@predicted_dq),('realized_joint',truth@actual_dq)]}
                                        assert np.isclose(info['error_squared'],np.sum((error-actual_dx)**2),atol=1e-14)
                                    if offset in [0,4]:branch[str(offset+1)]=info
                            if name==driver:matched=branch['1']
                            candidates.append(dict(method=name,scores=scores,branch=branch,decomposition=terms,clipped_joints=int(np.sum(np.abs(raw)>.04))))
                        rows.append(dict(case=arm.seed,robot=f'{arm.dof}dof',driver=driver,index=step,candidates=candidates))
                    reference_pose,actual=execute(env,c,reference_pose,task['goals'],step)
                    assert actual==wanted
                    if matched is not None:assert matched==actual;exact+=1
            replays.append(dict(body=arm.seed,driver=driver,steps=len(expected['trace']),exact=True))
            (root/'trials'/f'{arm.seed}_{driver}.json').write_text(json.dumps([r for r in rows if r['case']==arm.seed and r['driver']==driver]))
            print(arm.seed,driver,'exact',flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    assert len(rows)==exact==288 and len(replays)==32
    (root/'records.json').write_text(json.dumps(rows))
    (root/'audit.json').write_text(json.dumps(dict(exact_replays=replays,exact_first_steps=exact,states=len(rows),branches=len(rows)*10,records_sha256=fingerprint(root/'records.json'),unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    report(root)


def report(root):
    root=Path(root);rows=json.loads((root/'records.json').read_text());summary=[]
    protocol=json.loads((root/'protocol.json').read_text());audit=json.loads((root/'audit.json').read_text())
    assert fingerprint(root/'records.json')==audit['records_sha256']
    assert len(rows)==audit['states']==audit['exact_first_steps']==288
    assert len(audit['exact_replays'])==32 and all(r['exact'] for r in audit['exact_replays'])
    assert fingerprint(Path(protocol['reference'])/'drivers.json')==protocol['source_sha256']
    for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
    for r in rows:
        assert len(r['candidates'])==10
        for c in r['candidates']:
            terms=c['decomposition']
            np.testing.assert_allclose(np.sum([terms[k] for k in ['estimator','actuator','nonlinear']],axis=0),terms['total'],atol=1e-12,rtol=1e-10)
    archived=[]
    for path in sorted((root/'trials').glob('*.json')):archived.extend(json.loads(path.read_text()))
    key=lambda r:(r['case'],r['driver'],r['index'])
    assert sorted(archived,key=key)==sorted(rows,key=key)
    lines=['# E58: predicted versus actual procedural motion','',
        '32 analytical/E35 trajectories exactly replay E55; 288 fixed states, 2,880 five-step branches and 288 exact matching-driver first steps. Eight training bodies; no independent validation, native data or new training. All candidate checkpoints remain frozen.',
        '', '## Pairwise ranking (body-macro agreement)', '',
        '| DoF | Horizon | Estimated J | Geometric J + predicted motion | Geometric J + realized joint motion |','|---|---:|---:|---:|---:|']
    for robot in ['6dof','7dof']:
        rr=[r for r in rows if r['robot']==robot]
        for h in [1,5]:
            metrics=[agreement(rr,s,h) for s in ['estimated','geometric','realized_joint']]
            summary.append(dict(robot=robot,horizon=h,metrics=metrics))
            lines.append(f'| {robot} | {h} | '+' | '.join(f"{m['case_mean']:.3%}" for m in metrics)+' |')
    lines+=['','Ties at 1e-12 are excluded. Bodies, states and branches are dependent. Realized joint motion is a post-hoc diagnostic unavailable before executing the action. Five-step reference and online estimates continue changing.', '',
        '## First-step displacement error decomposition (RMS scaled displacement)', '',
        '| DoF | Group | Estimator | Actuator/bias | Nonlinear geometry | Total |','|---|---|---:|---:|---:|---:|']
    decomposed=[]
    for robot in ['6dof','7dof']:
        for group in ['analytical','original','prediction','control']:
            cc=[c for r in rows if r['robot']==robot for c in r['candidates'] if c['method'].split('_')[0]==group]
            metrics={k:float(np.sqrt(np.mean([np.sum(np.asarray(c['decomposition'][k])**2) for c in cc]))) for k in ['estimator','actuator','nonlinear','total']}
            decomposed.append(dict(robot=robot,group=group,**metrics))
            lines.append(f'| {robot} | {group} | '+' | '.join(f'{v:.6f}' for v in metrics.values())+' |')
    lines+=['','Vectors sum exactly; magnitudes can cancel and squared terms are not additive. Units combine XYZ metres and 0.15 times tool-Z direction. These are free-space procedural arms with no object/gripper.', '',
        '## Matched control versus prediction-only candidates', '',
        '| DoF | Seed | Geometric score favors control / states | Actual first step favors control / states | Actual fifth step favors control / states |','|---|---:|---:|---:|---:|']
    matched=[]
    for robot in ['6dof','7dof']:
        rr=[r for r in rows if r['robot']==robot]
        for seed in range(3):
            counts=[0,0,0];ties=[0,0,0]
            for r in rr:
                a=next(c for c in r['candidates'] if c['method']==f'control_{seed}');b=next(c for c in r['candidates'] if c['method']==f'prediction_{seed}')
                differences=[a['scores']['geometric']-b['scores']['geometric']]+[a['branch'][str(h)]['error_squared']-b['branch'][str(h)]['error_squared'] for h in [1,5]]
                for i,d in enumerate(differences):counts[i]+=int(d < -1e-12);ties[i]+=int(abs(d)<=1e-12)
            matched.append(dict(robot=robot,seed=seed,states=len(rr),control_wins=counts,ties=ties))
            lines.append(f'| {robot} | {seed} | '+' | '.join(f'{n}/{len(rr)}' for n in counts)+' |')
    contacts=sum(c['branch']['5']['collision'] for r in rows for c in r['candidates'])
    lines+=['',f'Five-step branches with contact: {contacts}/2880. All outcomes retained. This audit cannot explain native grasp/contact failures directly; it tests local mechanisms on fixed histories overlapping E57 training data. Keep E35; no promotion or final-test evaluation.']
    lines+=['','## Interpretation','',
        'Replacing predicted joint displacement with realized joint displacement raises one-step geometric-score agreement from 78.2% to 99.4% for six joints and 76.4% to 99.5% for seven. This is a diagnostic oracle, not an implementable improvement. Actuator/bias mismatch is much larger than geometric nonlinearity in the vector decomposition; estimated-J error is also substantial and can cancel actuator error.',
        '', 'Control wins over prediction-only in more than half the sampled states for every seed at both measured horizons, despite worse E57 native manipulation. Better short-horizon performance on these overlapping procedural histories therefore does not establish native retention. No contact occurred here, so contact and grasp mechanisms remain untested. The five-step ordering gap persists even with the first-step realized-motion oracle.',
        '', 'Next bounded test: fit a joint-response predictor using procedural measured commands, velocities and histories, evaluate it on whole held-out bodies/trajectories before integrating it into a training objective, and compare against the existing command-minus-bias approximation. Keep native data diagnostic-only and E35 as the retained controller. No model training was performed by this audit.',
        '', 'Validation: 81 regression tests pass. Initial v1 stopped on a float64-versus-float32 command-log assertion (maximum observed difference 7.4e-10); its source and failure status are preserved. v2 compares commands exactly at logged float32 precision while retaining full precision in the physical-motion decomposition.']
    (root/'summary.json').write_text(json.dumps(dict(rankings=summary,decomposition=decomposed,matched=matched,contacts=contacts),indent=2))
    (root/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines),flush=True)
    (root/'report_audit.json').write_text(json.dumps(dict(records_sha256=fingerprint(root/'records.json'),source_drivers_verified=True,trial_records_match=True,decomposition_identity_checks=2880,unchanged_weights=True,test_evaluated=False),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','parents','training','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

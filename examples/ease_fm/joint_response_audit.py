"""E59: leave-one-body-out measured joint-response regression."""
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
from .procedural_pose_audit import observe, vector, execute


def features(command,q,velocity,bias,previous_command,previous_motion):
    """Per-joint measured inputs only; no IDs, simulator parameters or future data."""
    return np.column_stack([command,velocity,q,np.sin(q),np.cos(q),bias,previous_command,previous_motion])


def fit(x,y):
    mean=x.mean(0);scale=x.std(0);scale=np.maximum(scale,1e-8)
    z=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty=np.eye(z.shape[1])*1e-3;penalty[0,0]=0
    weight=np.linalg.solve(z.T@z/len(z)+penalty,z.T@y/len(z))
    return mean,scale,weight


def predict(model,x):
    mean,scale,weight=model
    return np.column_stack([np.ones(len(x)),(x-mean)/scale])@weight


def run(data,parents,reference,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);ref=Path(reference);p=json.loads((ref/'protocol.json').read_text())
    old=json.loads((ref/'drivers.json').read_text());models,hashes=load_models(parents)
    assert fingerprint(Path(data)/'manifest.json')==p['manifest_sha256']
    for path,sha in hashes.items():assert p['checkpoints'][path]==sha
    models={'analytical':None,**{f'original_{s}':models[f'neural_{s}'] for s in range(3)}}
    protocol=dict(experiment='E59',checkpoints=hashes,reference=reference,source_sha256=fingerprint(ref/'drivers.json'),
        scope='32 exact E55 analytical/E35 trajectories, all 150 transitions, eight procedural training bodies. No native data or benchmark final-test use. Body holdout applies to response fitting only: upstream E35 drivers previously trained on these bodies.',
        design='Eight leave-one-body-out folds, no tuning. Fit on analytical/original_0/original_1 from seven other bodies; original_2 trajectories from those bodies form separate trajectory validation. All four drivers of the omitted body are body validation. Training normalization only. No selected checkpoint or refit on evaluation rows.',
        model='Shared per-joint linear ridge residual correction to command-minus-bias, fixed lambda .001 in mean-squared objective. Inputs: current limited command, pre-step qvel, q, sin(q), cos(q), measured bias, previous command and previous realized dq. No body/joint ID, geometry, gains, Jacobian or future state. First history inputs zero. Baselines: command-minus-bias and a command/velocity-only residual ridge.',
        limits='One-step regression on driver trajectories, no counterfactual actions, no task-success or closed-loop stability claim. All transitions retained including contacts. Fold estimates share training bodies and are dependent.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for name in ['joint_response_audit.py','procedural_pose_audit.py','mixed_sim.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    arrays={k:[] for k in ['x','y','baseline','body','driver','step','joint','dof']};replays=[]
    for task in p['tasks']:
        arm=Arm(**next(a for a in manifest['arms'] if a['seed']==task['body']));assert arm.split=='train'
        for driver,model in models.items():
            env=MixedReach(arm,horizon=1000,synchronize=True);env.reset(0,task['goals'][0][:3],spread=0)
            c=GuardedPose(arm.dof,model);r=vector(observe(env));prev_command=np.zeros(arm.dof);prev_motion=np.zeros(arm.dof)
            trace=next(d['trace'] for d in old if d['body']==arm.seed and d['driver']==driver)
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('No privileged response inputs')):
                for step,wanted in enumerate(trace):
                    q=env.data.qpos.copy();velocity=env.data.qvel.copy();bias=c.bias.copy()
                    r,actual=execute(env,c,r,task['goals'],step);assert actual==wanted
                    command=np.asarray(actual['command']);motion=env.data.qpos-q
                    x=features(command,q,velocity,bias,prev_command,prev_motion)
                    for key,value in dict(x=x,y=motion,baseline=command-bias,body=np.full(arm.dof,arm.seed),driver=np.full(arm.dof,driver),step=np.full(arm.dof,step),joint=np.arange(arm.dof),dof=np.full(arm.dof,arm.dof)).items():arrays[key].extend(value)
                    prev_command=command.copy();prev_motion=motion.copy()
            replays.append(dict(body=arm.seed,driver=driver,steps=len(trace),exact=True))
            print(arm.seed,driver,'exact',flush=True)
    a={k:np.asarray(v) for k,v in arrays.items()};np.savez_compressed(root/'transitions.npz',**a)
    scores=[];predictions=[]
    for body in sorted(set(a['body'])):
        train=(a['body']!=body)&(a['driver']!='original_2')
        trajectory=(a['body']!=body)&(a['driver']=='original_2');held=a['body']==body
        assert not np.any(train&(trajectory|held))
        for name,columns in [('command_velocity',[0,1]),('history',list(range(8)))]:
            x=a['x'][:,columns];model=fit(x[train],(a['y']-a['baseline'])[train])
            np.savez(root/f'fold_{body}_{name}.npz',mean=model[0],scale=model[1],weight=model[2],columns=columns)
            for split,mask in [('body',held),('trajectory',trajectory)]:
                pred=a['baseline'][mask]+predict(model,x[mask]);truth=a['y'][mask];baseline=a['baseline'][mask]
                scores.append(dict(body=int(body),dof=int(a['dof'][held][0]),model=name,split=split,count=int(mask.sum()),rmse=float(np.sqrt(np.mean((pred-truth)**2))),baseline_rmse=float(np.sqrt(np.mean((baseline-truth)**2)))))
                if split=='body':predictions.append(dict(body=int(body),model=name,indices=np.flatnonzero(mask).tolist(),prediction=pred.tolist()))
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'scores.json').write_text(json.dumps(scores,indent=2));(root/'predictions.json').write_text(json.dumps(predictions))
    (root/'audit.json').write_text(json.dumps(dict(exact_replays=replays,transitions=sum(r['steps'] for r in replays),joint_rows=len(a['y']),data_sha256=fingerprint(root/'transitions.npz'),unchanged_weights=True,test_evaluated=False),indent=2))
    lines=['# E59: held-out-body joint-response prediction','',protocol['design'],'',protocol['model'],'',protocol['scope'],'',
        '| Held-out body | DoF | Joint rows | Bias baseline RMSE (rad) | Command/velocity RMSE | History RMSE |','|---|---:|---:|---:|---:|---:|']
    for body in sorted(set(a['body'])):
        s=[r for r in scores if r['body']==body and r['split']=='body'];v,h=s
        lines.append(f"| {body} | {h['dof']} | {h['count']} | {h['baseline_rmse']:.6f} | {v['rmse']:.6f} | {h['rmse']:.6f} |")
    lines+=['',protocol['limits'],'','The held-out body excludes all its trajectories from fitting. Other-body original_2 trajectories are also excluded; their metrics are retained in scores.json. Eight bodies provide few independent units, and no novel axis-family holdout is claimed. Logged executed commands are float32; velocities, positions and realized displacement are measured in float64. No policy/controller changes or promotion.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');(root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['data','parents','reference','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

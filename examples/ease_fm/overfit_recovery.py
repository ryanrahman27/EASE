"""Collect shared expert recovery chunks from all tiny-subset policies' visited states."""
import argparse
import json
from pathlib import Path
import shutil
import mujoco
import numpy as np
import torch

from .benchmark import fingerprint
from .mixed_data import load
from .mixed_sim import Arm, MixedReach
from .joint_policy import JointFlowPolicy
from .overfit import DirectMLP, predict


def collect(data,run,output,device='cuda'):
    torch.set_num_threads(4)
    run,output=Path(run),Path(output); output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((run/'protocol.json').read_text()); manifest=load(data)
    assert protocol['manifest_sha256']==fingerprint(Path(data)/'manifest.json')
    arm=next(Arm(**a) for a in manifest['arms'] if a['seed']==protocol['body_seed'])
    episodes=[e for e in manifest['episodes'] if e['file'] in protocol['episode_files']]
    goals={g['id']:g for g in manifest['goals']['train']}
    plan={'scope':'Expert recovery chunks from every fifth non-colliding policy-visited state, original training starts only',
          'source_run':str(run),'source_checkpoints':{},'all_variants_share_union':True,'stride':5,'horizon':16,
          'body_seed':arm.seed,'episode_files':protocol['episode_files'],'manifest_sha256':protocol['manifest_sha256'],
          'test_evaluated':False,'branch_state':'MuJoCo mjSTATE_INTEGRATION plus task and success/contact bookkeeping'}
    for kind in protocol['variants']:
        plan['source_checkpoints'][kind]=fingerprint(run/kind/f"policy_{protocol['steps']}.pt")
    (output/'protocol.json').write_text(json.dumps(plan,indent=2))
    shutil.copyfile(__file__,output/'collector_source.py')
    rows=[]; records=[]
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    for kind in protocol['variants']:
        model=(DirectMLP(arm.dof) if kind=='direct_mlp' else JointFlowPolicy()).to(device).eval()
        model.load_state_dict(torch.load(run/kind/f"policy_{protocol['steps']}.pt",map_location=device,weights_only=True))
        for e in episodes:
            goal=goals[e['goal_id']]; env=MixedReach(arm)
            obs=env.reset(goal['seed']+(400000 if e['variant']=='broad_start' else 0),goal['xyz'],spread=.35 if e['variant']=='broad_start' else .05)
            noise=torch.randn(150,1,16,arm.dof,generator=torch.Generator().manual_seed(goal['seed']+600000)).to(device)
            mask=torch.ones(1,arm.dof,dtype=torch.bool,device=device)
            for step in range(150):
                if step%5==0 and not env.collision:
                    branch=MixedReach(arm); branch.reset(0,goal['xyz'])
                    state=np.empty(mujoco.mj_stateSize(env.model,spec))
                    mujoco.mj_getState(env.model,env.data,state,spec)
                    mujoco.mj_setState(branch.model,branch.data,state,spec); mujoco.mj_forward(branch.model,branch.data)
                    branch.steps,branch.streak,branch.collision=env.steps,env.streak,env.collision
                    np.testing.assert_allclose(branch.observe()['joints'],obs['joints'],atol=1e-6)
                    commands=[]; collided=False
                    for k in range(16):
                        _,done,info=branch.step(branch.expert()); commands.append(info['executed_action'])
                        collided |= info['collision']
                        if done:
                            break
                    record={'source':kind,'episode':e['file'],'step':step,'collision':collided,'label_steps':len(commands)}
                    records.append(record)
                    if not collided:
                        target=np.zeros((16,arm.dof),np.float32); valid=np.zeros_like(target)
                        target[:len(commands)]=np.asarray(commands)/.04; valid[:len(commands)]=1
                        rows.append((obs['joints'],obs['task'],target,valid))
                with torch.no_grad():
                    command=predict(model,kind,torch.tensor(obs['joints'],device=device)[None],torch.tensor(obs['task'],device=device)[None],mask,noise[step])[0].cpu().numpy()
                obs,done,info=env.step(command)
                if done:
                    break
        print(kind,'cumulative recovery queries',len(rows),flush=True)
    np.savez_compressed(output/'recovery.npz',**{key:np.stack([r[i] for r in rows]) for i,key in enumerate(['joints','task','action','valid'])})
    (output/'collection.json').write_text(json.dumps({'records':records,'accepted':len(rows),'sha256':fingerprint(output/'recovery.npz')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--run',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda')
    collect(**vars(p.parse_args()))

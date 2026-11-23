"""E34: fixed-checkpoint, matched-budget observation robustness experiment."""
import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .mixed_sim import Arm, MixedReach
from .paired_data import register
from .motion_transformer import MotionTransformer

CONDITIONS = [dict(name='clean'), dict(name='noise_1mm', noise=.001),
              dict(name='noise_5mm', noise=.005), dict(name='delay_1', delay=1),
              dict(name='delay_3', delay=3), dict(name='history_4', window=4),
              dict(name='history_8', window=8)]


def measure(obs, rng, noise):
    result={k: np.array(v,copy=True) for k,v in obs.items()}
    # Position-only additive noise; velocity is unused by both estimators.
    if noise:
        result['joints'][:,0]+=rng.normal(0,noise,len(result['joints']))
        result['task'][:3]+=rng.normal(0,noise,3)
    return result


def window_statistics(transitions, n):
    gram=1e-6*np.eye(n);cross=np.zeros((n,3));bias=np.zeros(n)
    for dq,dx,command in transitions:
        gram=.98*gram+np.outer(dq,dq)+2e-8*np.eye(n)
        cross=.98*cross+np.outer(dq,dx)
        bias=.8*bias+.2*(command-dq)
    return gram,cross,bias


@torch.no_grad()
def evaluate(manifest, model, condition, device):
    arms={a['seed']:Arm(**a) for a in manifest['arms']}; records=[]
    goals=manifest['goals']['validation'];delay=condition.get('delay',0)
    for offset in range(0,len(goals),48):
        batch=goals[offset:offset+48];size=len(batch)
        gram=np.zeros((size,7,7));cross=np.zeros((size,7,3));bias=np.zeros((size,7))
        masks=np.zeros((size,7),bool);envs=[];streams=[];commands=[];rngs=[];histories=[]
        for i,g in enumerate(batch):
            arm=replace(arms[g['pair']],home=tuple(g['home']));n=arm.dof
            env=MixedReach(arm,synchronize=True);envs.append(env)
            rng=np.random.default_rng(340000+g['id']);rngs.append(rng)
            streams.append([measure(env.reset(0,g['xyz'],spread=0),rng,condition.get('noise',0))])
            commands.append([]);histories.append(deque(maxlen=condition.get('window')))
            gram[i,:n,:n]=1e-6*np.eye(n);masks[i,:n]=True
        mask=torch.tensor(masks,device=device);active=np.ones(size,bool)
        for step in range(150):
            predicted=None
            if model is not None:
                predicted=model(torch.tensor(gram,device=device,dtype=torch.float32),
                    torch.tensor(cross,device=device,dtype=torch.float32),mask).cpu().numpy()
            for i in np.flatnonzero(active):
                env=envs[i];n=env.arm.dof if hasattr(env,'arm') else int(masks[i].sum())
                index=max(0,step-delay);before=streams[i][index]
                jac=(np.linalg.solve(gram[i,:n,:n],cross[i,:n]).T if model is None else predicted[i,:n].T)
                error=before['task'][3:]-before['task'][:3]
                action=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(3),.5*error)+bias[i,:n],-.04,.04)
                if model is None and step==0:action=np.full(n,.01)
                after,done,info=env.step(action);commands[i].append(info['executed_action'].copy())
                streams[i].append(measure(after,rngs[i],condition.get('noise',0)))
                next_index=max(0,step+1-delay)
                if next_index>index:
                    observed=streams[i][next_index]
                    dq=(observed['joints'][:,0]-before['joints'][:,0]).astype(float)
                    dx=(observed['task'][:3]-before['task'][:3]).astype(float)
                    command=commands[i][index] # timestamp-aligned with the delayed transition
                    if condition.get('window'):
                        histories[i].append((dq,dx,command))
                        gram[i,:n,:n],cross[i,:n],bias[i,:n]=window_statistics(histories[i],n)
                    else:
                        gram[i,:n,:n]=.98*gram[i,:n,:n]+np.outer(dq,dq)+2e-8*np.eye(n)
                        cross[i,:n]=.98*cross[i,:n]+np.outer(dq,dx)
                        bias[i,:n]=.8*bias[i,:n]+.2*(command-dq)
                if done:
                    records.append(dict(goal_id=batch[i]['id'],body_seed=batch[i]['pair'],dof=n,
                        family_id=batch[i]['family_id'],condition=condition['name'],steps=env.steps,
                        **{k:info[k] for k in ('success','error','collision')}))
                    active[i]=False
            if not active.any():break
    return records


def run(data,replication,output,device='cuda'):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    manifest=register(data);entries=json.loads((Path(replication)/'index.json').read_text())
    for e in entries:assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
    protocol=dict(experiment='E34',conditions=CONDITIONS,checkpoints=entries,
        manifest_sha256=fingerprint(Path(data)/'manifest.json'),test_evaluated=False,
        design='All 1024 E33 development tasks, 3 frozen neural seeds and fixed-probe analytical. No tuning. 150 task steps, zero separate calibration, same success/contact criteria.',
        noise='Independent Gaussian joint-position radians and EE-position meters with equal numeric sigma. Goal exact. One deterministic noise realization per task, shared across methods at equal time; true simulator state determines success.',
        delay='One or three 25Hz steps (40/120ms), joint and EE observations delayed together. Reuse initial observation during startup, no fictitious transition updates; commands aligned to delayed transition timestamps.',
        history='Only last 4 or 8 observed transitions retained; reconstruct identical exponentially weighted statistics and bias from prior each time. This tests restricted memory, not independently controlled physical excitation.',
        analysis='Report every condition and family, paired outcomes and interaction costs. Clean replay must reproduce E33 success/contact/step counts. No best-condition selection or final-test evaluation.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,root/'source'/p.name)
    models={'analytical':None}
    for e in entries:
        m=MotionTransformer().to(device);m.load_state_dict(torch.load(Path(e['folder'])/'policy.pt',map_location=device,weights_only=True));models[f'neural_{e["seed"]}']=m.eval()
    original_solve=np.linalg.solve
    def control_solve(a,b):
        assert a.shape[-2:]==(3,3),'Analytical joint-model inverse forbidden'
        return original_solve(a,b)
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        for condition in CONDITIONS:
            for name,model in models.items():
                if model is None:rows=evaluate(manifest,model,condition,device)
                else:
                    with patch.object(np.linalg,'solve',side_effect=control_solve),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
                        rows=evaluate(manifest,model,condition,device)
                assert len(rows)==1024 and len({r['goal_id'] for r in rows})==1024
                if condition['name']=='clean':
                    old=json.loads((Path(data).parent/('fixed_probe_analytical.json' if model is None else name+'.json')).read_text())
                    old={r['goal_id']:r for r in old}
                    assert all(all(r[k]==old[r['goal_id']][k] for k in ('success','collision','steps')) for r in rows),'Clean replay mismatch'
                (root/f'{condition["name"]}__{name}.json').write_text(json.dumps(rows,indent=2))
                print(condition['name'],name,sum(r['success'] for r in rows),'/1024',flush=True)
    for e in entries:assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('data','replication','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda');run(**vars(parser.parse_args()))

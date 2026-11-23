"""Training-episode closed-loop audit; no optimization or held-out test use."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .benchmark import fingerprint
from .joint_policy import JointFlowPolicy
from .mixed_data import load
from .mixed_sim import Arm, MixedReach, explore


def audit(data, run, output, device='cuda'):
    torch.set_num_threads(4)
    root, run, output = Path(data), Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = load(root)
    arms = {a['seed']: Arm(**a) for a in manifest['arms'] if a['split']=='train'}
    goals = {g['id']: g for g in manifest['goals']['train']}
    episodes = [e for e in manifest['episodes'] if e['split']=='train' and e['success'] and not e['collision']]
    protocol = {'scope': 'All accepted training episodes, exact recorded resets; post-hoc diagnostic',
        'episodes': len(episodes), 'variants': ['flow_only', 'flow_response'], 'context': 'correct32, separate fresh calibration, frozen during task',
        'calibration_seed': 'goal seed + 900000', 'flow_seed': 'goal seed + 600000 + (400000 for broad_start)',
        'solver_steps': 10, 'test_evaluated': False, 'weights_updated': False,
        'manifest_sha256': fingerprint(root/'manifest.json'),
        'teacher_probe': 'Every fifth recorded transition; first-action RMSE, one deterministic noise draw per query'}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    shutil.copyfile(__file__, output/'training_audit_source.py')
    contexts, replay = {}, []
    for entry in episodes:
        arm, goal = arms[entry['body_seed']], goals[entry['goal_id']]
        key = (arm.seed, goal['id'])
        if key not in contexts:
            contexts[key] = explore(MixedReach(arm), goal['seed']+900000, 32)
        env = MixedReach(arm)
        broad = entry['variant']=='broad_start'
        obs = env.reset(goal['seed']+(400000 if broad else 0), goal['xyz'], spread=.35 if broad else .05)
        with np.load(root/entry['file']) as trajectory:
            np.testing.assert_allclose(obs['joints'], trajectory['joints'][0], atol=1e-6)
            np.testing.assert_allclose(obs['task'], trajectory['task'][0], atol=1e-6)
            for command in trajectory['action']:
                _, done, info = env.step(command)
                if done:
                    break
        replay.append({'file': entry['file'], **{k: info[k] for k in ['success','error','collision']}})
    (output/'expert_replay.json').write_text(json.dumps(replay, indent=2))
    print('Expert recorded-action replay:', sum(r['success'] for r in replay), '/', len(replay), flush=True)
    for variant in protocol['variants']:
        folder = run/variant
        meta = json.loads((folder/'training.json').read_text())
        assert meta['manifest_sha256']==protocol['manifest_sha256']
        assert meta['policy_sha256']==fingerprint(folder/'policy.pt')
        model = JointFlowPolicy(history=True, auxiliary=True).to(device).eval()
        model.load_state_dict(torch.load(folder/'policy.pt', map_location=device, weights_only=True))
        records = []
        for offset in range(0, len(episodes), 48):
            batch = episodes[offset:offset+48]
            batch_arms = [arms[e['body_seed']] for e in batch]
            batch_goals = [goals[e['goal_id']] for e in batch]
            masks = torch.tensor([[j < a.dof for j in range(7)] for a in batch_arms], device=device)
            h = np.zeros((len(batch),32,7,11),np.float32)
            for i,(e,a) in enumerate(zip(batch,batch_arms)):
                h[i,:,:a.dof] = contexts[(a.seed,e['goal_id'])]
            with torch.no_grad():
                latent = model.encode(torch.tensor(h,device=device),torch.full((len(batch),),32,device=device),masks)
            envs = [MixedReach(a) for a in batch_arms]
            obs = [env.reset(g['seed']+(400000 if e['variant']=='broad_start' else 0),g['xyz'],
                             spread=.35 if e['variant']=='broad_start' else .05) for env,g,e in zip(envs,batch_goals,batch)]
            noise = torch.stack([torch.randn(150,16,7,generator=torch.Generator().manual_seed(g['seed']+600000+
                         (400000 if e['variant']=='broad_start' else 0))) for g,e in zip(batch_goals,batch)],dim=1).to(device)
            active = np.ones(len(batch),bool)
            for step in range(150):
                q = np.zeros((len(batch),7,2),np.float32)
                for i,o in enumerate(obs):
                    q[i,:len(o['joints'])]=o['joints']
                task = np.stack([o['task'] for o in obs])
                actions = model.sample(torch.tensor(q,device=device),torch.tensor(task,device=device),masks,noise[step],latent).cpu().numpy()[:,0]
                for i in np.flatnonzero(active):
                    obs[i],done,info=envs[i].step(actions[i,:batch_arms[i].dof])
                    if done:
                        records.append({'file':batch[i]['file'],'body_seed':batch_arms[i].seed,'family':batch_arms[i].family,
                            'start':batch[i]['variant'],'goal_id':batch[i]['goal_id'],'steps':envs[i].steps,
                            **{k:info[k] for k in ['success','error','collision']}})
                        active[i]=False
                if not active.any():
                    break
            print(variant,'completed',offset+len(batch),'/',len(episodes),'successes',sum(r['success'] for r in records),flush=True)
        # Teacher states isolate action prediction from accumulated rollout drift.
        teacher = []
        for entry in episodes:
            arm=arms[entry['body_seed']]
            with np.load(root/entry['file']) as trajectory:
                indices=np.arange(0,len(trajectory['action']),5)
                q=torch.tensor(trajectory['joints'][indices],device=device)
                task=torch.tensor(trajectory['task'][indices],device=device)
                mask=torch.ones(len(indices),arm.dof,dtype=torch.bool,device=device)
                h=torch.tensor(contexts[(arm.seed,entry['goal_id'])],device=device)[None].expand(len(indices),-1,-1,-1)
                with torch.no_grad():
                    z=model.encode(h,torch.full((len(indices),),32,device=device),mask)
                    noise=torch.randn(len(indices),16,arm.dof,generator=torch.Generator().manual_seed(goals[entry['goal_id']]['seed']+700000)).to(device)
                    action=model.sample(q,task,mask,noise,z)[:,0].cpu().numpy()
                target=trajectory['action'][indices]
                teacher.append({'file':entry['file'],'family':arm.family,'squared_error_sum':float(np.square(action-target).sum()),
                                'target_squared_sum':float(np.square(target).sum()),'scalar_count':target.size})
        (output/(variant+'.json')).write_text(json.dumps({'checkpoint_sha256':meta['policy_sha256'],'records':records,'teacher_probe':teacher},indent=2))
    summary=[]
    for variant in protocol['variants']:
        result=json.loads((output/(variant+'.json')).read_text())
        for family in ['six','seven']:
            for start in ['nominal','broad_start']:
                selected=[r for r in result['records'] if r['family']==family and r['start']==start]
                summary.append({'variant':variant,'family':family,'start':start,'successes':sum(r['success'] for r in selected),
                    'trials':len(selected),'mean_final_error_m':float(np.mean([r['error'] for r in selected]))})
        for family in ['six','seven']:
            selected=[r for r in result['teacher_probe'] if r['family']==family]
            print(variant,family,'teacher-state action RMSE',np.sqrt(sum(r['squared_error_sum'] for r in selected)/sum(r['scalar_count'] for r in selected)),flush=True)
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--run',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda')
    audit(**vars(parser.parse_args()))

"""Full training-body control gate with capacity-matched objectives and shared recovery."""
import argparse
import json
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
import torch

from .benchmark import fingerprint
from .joint_policy import JointFlowPolicy
from .mixed_data import MixedData
from .mixed_sim import Arm, MixedReach, explore


def commands(model, kind, q, task, mask, noise, latent):
    if kind=='flow':
        return model.sample(q,task,mask,noise,latent)[:,0]
    return model(torch.zeros_like(noise),torch.zeros(len(q),device=q.device),q,task,mask,latent)[:,0].clamp(-1,1)*.04


def fit(dataset,folder,kind,steps,device,manifest_hash,recovery_hash=None,checkpoint_steps=(),on_checkpoint=None):
    folder.mkdir()
    q,task,mask,action,valid=[t.to(device) for t in dataset.tensors]
    torch.manual_seed(0)
    model=JointFlowPolicy(history=True).to(device)
    model.joint_mean.copy_(q[mask].mean(0)); model.joint_std.copy_(q[mask].std(0,unbiased=False).clamp_min(.01))
    model.task_mean.copy_(task.mean(0)); model.task_std.copy_(task.std(0,unbiased=False).clamp_min(.01))
    h=torch.tensor(np.concatenate([c.reshape(-1,11) for c in dataset.contexts]),device=device)
    model.history_mean.copy_(h.mean(0)); model.history_std.copy_(h.std(0,unbiased=False).clamp_min(.01))
    opt=torch.optim.AdamW(model.parameters(),lr=.0003)
    batch_rng=torch.Generator(device=device).manual_seed(1000)
    noise_rng=torch.Generator(device=device).manual_seed(2000)
    context_rng=np.random.default_rng(3000)
    losses=[]; start=time.perf_counter()
    for step in range(steps):
        idx=torch.randint(len(q),(32,),device=device,generator=batch_rng)
        history,lengths=dataset.histories(idx.cpu().numpy(),context_rng)
        z=model.encode(history.to(device),lengths.to(device),mask[idx])
        if kind=='flow':
            noise=torch.randn(action[idx].shape,device=device,generator=noise_rng)
            tau=torch.rand(len(idx),device=device,generator=noise_rng)
            prediction=model(noise.lerp(action[idx],tau[:,None,None]),tau,q[idx],task[idx],mask[idx],z)
            target=action[idx]-noise
        else:
            prediction=model(torch.zeros_like(action[idx]),torch.zeros(len(idx),device=device),q[idx],task[idx],mask[idx],z)
            target=action[idx]
        loss=((prediction-target).square()*valid[idx]).sum()/valid[idx].sum()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
        losses.append(loss.item())
        if (step+1)%1000==0:
            print(folder.name,step+1,'loss',losses[-1],flush=True)
        if step+1 in checkpoint_steps and on_checkpoint is not None:
            on_checkpoint(step+1,model,opt,losses,{
                'batch':batch_rng.get_state(),'noise':noise_rng.get_state(),
                'context':context_rng.bit_generator.state,
                'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []})
    model.eval()
    torch.save(model.state_dict(),folder/'policy.pt')
    (folder/'training.json').write_text(json.dumps({'kind':kind,'seed':0,'steps':steps,'losses':losses,
        'transitions':len(q),'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,
        'manifest_sha256':manifest_hash,'recovery_sha256':recovery_hash,'policy_sha256':fingerprint(folder/'policy.pt'),
        'torch':torch.__version__,'device':device},indent=2))
    return model


def recovery_chunk(env,observation):
    """Clone full integration state; expert-only branch never affects the policy env."""
    branch=MixedReach(env.arm, synchronize=env.synchronize)
    branch.reset(0,env.goal)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.model,spec))
    mujoco.mj_getState(env.model,env.data,state,spec)
    mujoco.mj_setState(branch.model,branch.data,state,spec); mujoco.mj_forward(branch.model,branch.data)
    branch.steps,branch.streak,branch.collision=env.steps,env.streak,env.collision
    np.testing.assert_allclose(branch.observe()['joints'],observation['joints'],atol=1e-6)
    np.testing.assert_allclose(branch.observe()['task'],observation['task'],atol=1e-6)
    actions=[]; collision=False
    for _ in range(16):
        _,done,info=branch.step(branch.expert())
        actions.append(info['executed_action']); collision |= info['collision']
        if done:
            break
    target=np.zeros((16,7),np.float32); valid=np.zeros_like(target)
    target[:len(actions),:env.arm.dof]=np.asarray(actions)/.04
    valid[:len(actions),:env.arm.dof]=1
    return target,valid,collision


@torch.no_grad()
def rollout(dataset,model,kind,folder,device,collect=False,mode='correct32'):
    manifest=dataset.manifest
    arms={a['seed']:Arm(**a) for a in manifest['arms'] if a['split']=='train'}
    goals={g['id']:g for g in manifest['goals']['train']}
    episodes=[e for e in manifest['episodes'] if e['split']=='train' and e['success'] and not e['collision']]
    records=[]; recovery=[]; queries=[]
    cache={}
    for offset in range(0,len(episodes),48):
        batch=episodes[offset:offset+48]
        envs=[MixedReach(arms[e['body_seed']], synchronize=manifest.get('synchronized_observations',False)) for e in batch]
        mask=torch.tensor([[j<env.arm.dof for j in range(7)] for env in envs],device=device)
        histories=np.zeros((len(batch),32,7,11),np.float32)
        length=0 if mode=='empty' else 32
        for i,(env,e) in enumerate(zip(envs,batch)):
            donor=env.arm
            if mode in ('wrong32','wrong_axes32'):
                donor=next(a for a in arms.values() if a.dof==env.arm.dof and a.seed!=env.arm.seed
                           and (mode!='wrong_axes32' or a.family!=env.arm.family))
            key=(donor.seed,e['goal_id'])
            if length and key not in cache:
                cache[key]=explore(MixedReach(donor, synchronize=manifest.get('synchronized_observations',False)),goals[e['goal_id']]['seed']+900000,32)
            if length:
                histories[i,:,:env.arm.dof]=cache[key]
        latent=model.encode(torch.tensor(histories,device=device),torch.full((len(batch),),length,device=device),mask)
        obs=[env.reset(goals[e['goal_id']]['seed']+(400000 if e['variant']=='broad_start' else 0),goals[e['goal_id']]['xyz'],
                       spread=.35 if e['variant']=='broad_start' else .05) for env,e in zip(envs,batch)]
        noise=torch.stack([torch.randn(150,16,7,generator=torch.Generator().manual_seed(goals[e['goal_id']]['seed']+600000+
                     (400000 if e['variant']=='broad_start' else 0))) for e in batch],dim=1).to(device)
        active=np.ones(len(batch),bool)
        for step in range(150):
            q=np.zeros((len(batch),7,2),np.float32)
            for i,o in enumerate(obs):
                q[i,:len(o['joints'])]=o['joints']
            task=np.stack([o['task'] for o in obs])
            action=commands(model,kind,torch.tensor(q,device=device),torch.tensor(task,device=device),mask,noise[step],latent).cpu().numpy()
            for i in np.flatnonzero(active):
                if collect and step%10==0 and not envs[i].collision:
                    target,valid,collision=recovery_chunk(envs[i],obs[i])
                    queries.append({'source':kind,'file':batch[i]['file'],'step':step,'collision':collision})
                    if not collision:
                        recovery.append((q[i].copy(),task[i].copy(),mask[i].cpu().numpy(),target,valid,offset+i))
                obs[i],done,info=envs[i].step(action[i,:envs[i].arm.dof])
                if done:
                    records.append({'file':batch[i]['file'],'body_seed':envs[i].arm.seed,'family':envs[i].arm.family,
                        'goal_id':batch[i]['goal_id'],'start':batch[i]['variant'],'steps':envs[i].steps,
                        **{k:info[k] for k in ['success','error','collision']}})
                    active[i]=False
            if not active.any():
                break
        print(folder.name,mode,'rollouts',offset+len(batch),'/',len(episodes),'success',sum(r['success'] for r in records),flush=True)
    result={'mode':mode,'records':records,'checkpoint_sha256':fingerprint(folder/'policy.pt'),'test_evaluated':False}
    (folder/(mode+'.json')).write_text(json.dumps(result,indent=2))
    if collect:
        np.savez_compressed(folder/'recovery.npz',**{key:np.stack([r[i] for r in recovery]) for i,key in enumerate(['joints','task','mask','action','valid','episode'])})
        (folder/'queries.json').write_text(json.dumps({'queries':queries,'accepted':len(recovery),'sha256':fingerprint(folder/'recovery.npz')},indent=2))
    return records


def add_recovery(dataset,paths):
    arrays=[]; ids=[]
    for path in paths:
        record=json.loads((path.parent/'queries.json').read_text())
        if fingerprint(path)!=record['sha256']:
            raise ValueError('Recovery hash mismatch')
        with np.load(path) as data:
            arrays.append(tuple(torch.from_numpy(data[k]) for k in ['joints','task','mask','action','valid']))
            ids.append(data['episode'].astype(np.int64))
    dataset.tensors=tuple(torch.cat([dataset.tensors[i]]+[a[i] for a in arrays]) for i in range(5))
    dataset.row_episode=np.concatenate([dataset.row_episode]+ids)
    return dataset


def summarize(root):
    rows=[]
    for phase in ['original','recovery']:
        for kind in ['flow','direct']:
            folder=root/(phase+'_'+kind)
            result=json.loads((folder/'correct32.json').read_text())
            for family in sorted({r['family'] for r in result['records']}):
                for start in ['nominal','broad_start']:
                    cases=[r for r in result['records'] if r['family']==family and r['start']==start]
                    rows.append({'phase':phase,'model':kind,'family':family,'start':start,
                        'successes':sum(r['success'] for r in cases),'trials':len(cases),
                        'mean_error_m':float(np.mean([r['error'] for r in cases]))})
    (root/'summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2),flush=True)


def run(data,output,steps=10000,device='cuda',experiment='E12'):
    torch.set_num_threads(4)
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    dataset=MixedData(data)
    if not dataset.manifest.get('synchronized_observations',False):
        raise ValueError('Full-control recovery requires synchronized Cartesian observations; regenerate with --synchronize')
    manifest_hash=fingerprint(Path(data)/'manifest.json')
    protocol={'experiment':experiment,'scope':'Full training-body control, one training seed, no held-out evaluation',
        'seed':0,'steps_per_model':steps,'batch_size':32,'lr':.0003,'models':['flow','direct'],
        'capacity':'Identical history-conditioned joint architecture and initial weights, no auxiliary loss',
        'training_context':'Original same-body other-episode contexts; recovered rows exclude their source episode',
        'sampling':'Uniform rows; paired batch/context streams within data condition; all phases train from scratch',
        'recovery':'Shared union of every tenth non-colliding state from BOTH original-data policies on all accepted training episodes; expert 16-step branches',
        'evaluation':'All accepted training episodes, correct32 separate calibration frozen during task, fixed paired noise',
        'gate':'At least 95 percent overall success and at least 90 percent for every training body; exploratory single-seed gate',
        'original_transitions':len(dataset.row_episode),'manifest_sha256':manifest_hash,'test_evaluated':False}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p,source/p.name)
    for kind in protocol['models']:
        folder=root/('original_'+kind)
        model=fit(dataset,folder,kind,steps,device,manifest_hash)
        rollout(dataset,model,kind,folder,device,collect=True)
        del model
    paths=[root/('original_'+k)/'recovery.npz' for k in protocol['models']]
    dataset=add_recovery(dataset,paths)
    hashes={str(p.relative_to(root)):fingerprint(p) for p in paths}
    (root/'shared_recovery.json').write_text(json.dumps({'hashes':hashes,'combined_transitions':len(dataset.row_episode)},indent=2))
    recovery_hash=fingerprint(root/'shared_recovery.json')
    for kind in protocol['models']:
        folder=root/('recovery_'+kind)
        model=fit(dataset,folder,kind,steps,device,manifest_hash,recovery_hash)
        records=rollout(dataset,model,kind,folder,device)
        by_body={b:[r for r in records if r['body_seed']==b] for b in sorted({r['body_seed'] for r in records})}
        gate=sum(r['success'] for r in records)/len(records)>=.95 and all(sum(r['success'] for r in rs)/len(rs)>=.9 for rs in by_body.values())
        (folder/'gate.json').write_text(json.dumps({'passed':gate,'overall':[sum(r['success'] for r in records),len(records)],
            'by_body':{b:[sum(r['success'] for r in rs),len(rs)] for b,rs in by_body.items()}},indent=2))
        del model
    summarize(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=10000); p.add_argument('--device',default='cuda')
    p.add_argument('--experiment',default='E12')
    run(**vars(p.parse_args()))

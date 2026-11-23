"""Single-body overfit gate: flow, matched direct chunk regression, and simple MLP."""
import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from .benchmark import fingerprint
from .joint_policy import JointFlowPolicy
from .mixed_data import MixedData
from .mixed_sim import Arm, MixedReach


class DirectMLP(nn.Module):
    def __init__(self, count):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(count*2+6,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,count))
        self.register_buffer('mean',torch.zeros(count*2+6))
        self.register_buffer('std',torch.ones(count*2+6))

    def forward(self,q,task):
        return self.net((torch.cat([q.flatten(1),task],1)-self.mean)/self.std)


def predict(model,kind,q,task,mask,noise,solver=10):
    if kind=='flow':
        return model.sample(q,task,mask,noise,steps=solver)[:,0]
    if kind=='direct_chunk':
        return model(torch.zeros_like(noise),torch.zeros(len(q),device=q.device),q,task,mask)[:,0].clamp(-1,1)*.04
    return model(q,task).clamp(-1,1)*.04


@torch.no_grad()
def evaluate(model,kind,q,task,action,arm,episodes,goals,device):
    count=arm.dof
    mask=torch.ones(len(q),count,dtype=torch.bool,device=device)
    noise=torch.randn(len(q),16,count,generator=torch.Generator().manual_seed(1901)).to(device)
    predictions=[]
    for offset in range(0,len(q),128):
        predictions.append(predict(model,kind,q[offset:offset+128],task[offset:offset+128],mask[offset:offset+128],noise[offset:offset+128]))
    error=torch.cat(predictions)-action[:,0]*.04
    teacher={'rmse_per_joint_rad':error.square().mean(0).sqrt().tolist(),'rmse_rad':error.square().mean().sqrt().item()}
    repeats=3 if kind=='flow' else 1
    cases=[(e,r) for e in episodes for r in range(repeats)]
    envs=[MixedReach(arm) for _ in cases]
    obs=[env.reset(goals[e['goal_id']]['seed']+(400000 if e['variant']=='broad_start' else 0),goals[e['goal_id']]['xyz'],
                   spread=.35 if e['variant']=='broad_start' else .05) for env,(e,r) in zip(envs,cases)]
    noise=torch.stack([torch.randn(150,16,count,generator=torch.Generator().manual_seed(goals[e['goal_id']]['seed']+600000+r*10000)) for e,r in cases],dim=1).to(device)
    mask=torch.ones(len(cases),count,dtype=torch.bool,device=device)
    active=np.ones(len(cases),bool)
    records=[]
    traces=[[] for _ in cases]
    qmin=q[:,:,0].min(0).values.cpu().numpy()
    qmax=q[:,:,0].max(0).values.cpu().numpy()
    for step in range(150):
        jq=np.stack([o['joints'] for o in obs]); tk=np.stack([o['task'] for o in obs])
        commands=predict(model,kind,torch.tensor(jq,device=device),torch.tensor(tk,device=device),mask,noise[step]).cpu().numpy()
        for i in np.flatnonzero(active):
            expert=envs[i].expert()  # Diagnostic only; never changes the policy command.
            traces[i].append({'step':step,'error_m':float(np.linalg.norm(obs[i]['task'][:3]-obs[i]['task'][3:])),
                'q':jq[i,:,0].tolist(),'command':commands[i].tolist(),'expert_command':expert.tolist(),
                'outside_training_q_range':((jq[i,:,0]<qmin)|(jq[i,:,0]>qmax)).tolist()})
            obs[i],done,info=envs[i].step(commands[i])
            if done:
                e,r=cases[i]
                records.append({'file':e['file'],'goal_id':e['goal_id'],'start':e['variant'],'noise_repeat':r,
                    'steps':envs[i].steps,**{k:info[k] for k in ['success','error','collision']},'trace':traces[i]})
                active[i]=False
        if not active.any():
            break
    return {'teacher':teacher,'records':records}


def run(data,output,steps=5000,device='cuda',recovery=None):
    torch.set_num_threads(4)
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    dataset=MixedData(data)
    manifest=dataset.manifest
    arm=next(Arm(**a) for a in manifest['arms'] if a['split']=='train' and a['family']=='six')
    accepted=[e for e in manifest['episodes'] if e['split']=='train' and e['success'] and not e['collision']]
    chosen=[i for i,e in enumerate(accepted) if e['body_seed']==arm.seed and e['goal_id']<4]
    episodes=[accepted[i] for i in chosen]
    rows=np.flatnonzero(np.isin(dataset.row_episode,chosen))
    q,task,mask,action,valid=[x[rows].to(device) for x in dataset.tensors]
    q,mask,action,valid=q[:,:arm.dof],mask[:,:arm.dof],action[:,:,:arm.dof],valid[:,:,:arm.dof]
    original_rows=len(q)
    if recovery:
        recovery_path=Path(recovery)
        recovery_meta=json.loads((recovery_path.parent/'protocol.json').read_text())
        recovery_record=json.loads((recovery_path.parent/'collection.json').read_text())
        if recovery_meta['body_seed']!=arm.seed or recovery_meta['episode_files']!=[e['file'] for e in episodes]:
            raise ValueError('Recovery subset mismatch')
        if recovery_meta['manifest_sha256']!=fingerprint(Path(data)/'manifest.json') or recovery_record['sha256']!=fingerprint(recovery_path):
            raise ValueError('Recovery provenance mismatch')
        with np.load(recovery) as extra:
            q=torch.cat([q,torch.tensor(extra['joints'],device=device)])
            task=torch.cat([task,torch.tensor(extra['task'],device=device)])
            action=torch.cat([action,torch.tensor(extra['action'],device=device)])
            valid=torch.cat([valid,torch.tensor(extra['valid'],device=device)])
            mask=torch.ones(q.shape[:2],dtype=torch.bool,device=device)
    goals={g['id']:g for g in manifest['goals']['train']}
    protocol={'experiment':'E10' if recovery else 'E09','scope':'Training overfit gate, not transfer evaluation','body_seed':arm.seed,
        'goals':[0,1,2,3],'episode_files':[e['file'] for e in episodes],'transitions':len(q),'original_transitions':original_rows,
        'recovery_sha256':fingerprint(Path(recovery)) if recovery else None,'steps':steps,'seed':0,
        'batch_size':32,'learning_rate':.0003,'history':False,'weights_updated_only_on_selected_training_rows':True,
        'variants':['flow','direct_chunk','direct_mlp'],'capacity_control':'flow/direct_chunk use identical architecture and initialization; MLP is an intentionally simpler diagnostic',
        'evaluation_updates':sorted(set([min(1000,steps),steps])),'flow_rollout_noise_repeats':3,
        'gate':'All selected episodes succeed; flow must succeed for all three noise repeats',
        'test_evaluated':False,'manifest_sha256':fingerprint(Path(data)/'manifest.json')}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p,source/p.name)
    print('selected',len(episodes),'episodes',len(q),'transitions',flush=True)
    for kind in protocol['variants']:
        folder=root/kind; folder.mkdir()
        torch.manual_seed(0)
        model=(DirectMLP(arm.dof) if kind=='direct_mlp' else JointFlowPolicy()).to(device)
        if kind=='direct_mlp':
            inputs=torch.cat([q.flatten(1),task],1)
            model.mean.copy_(inputs.mean(0)); model.std.copy_(inputs.std(0,unbiased=False).clamp_min(.01))
        else:
            model.joint_mean.copy_(q[mask].mean(0)); model.joint_std.copy_(q[mask].std(0,unbiased=False).clamp_min(.01))
            model.task_mean.copy_(task.mean(0)); model.task_std.copy_(task.std(0,unbiased=False).clamp_min(.01))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
        generator=torch.Generator(device=device).manual_seed(1000)
        noise_generator=torch.Generator(device=device).manual_seed(2000)
        losses=[]; started=time.perf_counter()
        for step in range(1,steps+1):
            model.train()
            idx=torch.randint(len(q),(32,),device=device,generator=generator)
            if kind=='direct_mlp':
                loss=(model(q[idx],task[idx])-action[idx,0]).square().mean()
            else:
                if kind=='flow':
                    noise=torch.randn(action[idx].shape,device=device,generator=noise_generator)
                    tau=torch.rand(len(idx),device=device,generator=noise_generator)
                    prediction=model(noise.lerp(action[idx],tau[:,None,None]),tau,q[idx],task[idx],mask[idx])
                    target=action[idx]-noise
                else:
                    prediction=model(torch.zeros_like(action[idx]),torch.zeros(len(idx),device=device),q[idx],task[idx],mask[idx])
                    target=action[idx]
                loss=((prediction-target).square()*valid[idx]).sum()/valid[idx].sum()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step()
            losses.append(loss.item())
            if step%500==0:
                print(kind,step,losses[-1],flush=True)
            if step in protocol['evaluation_updates']:
                model.eval()
                result=evaluate(model,kind,q[:original_rows],task[:original_rows],action[:original_rows],arm,episodes,goals,device)
                (folder/f'evaluation_{step}.json').write_text(json.dumps(result,indent=2))
                torch.save(model.state_dict(),folder/f'policy_{step}.pt')
                print(kind,step,'success',sum(r['success'] for r in result['records']),'/',len(result['records']),'teacher RMSE',result['teacher']['rmse_rad'],flush=True)
        metadata={'parameters':sum(p.numel() for p in model.parameters()),'losses':losses,'seconds_including_evaluation':time.perf_counter()-started,
            'manifest_sha256':protocol['manifest_sha256'],'checkpoint_hashes':{p.name:fingerprint(p) for p in folder.glob('*.pt')},
            'source_hashes':{p.name:fingerprint(p) for p in source.glob('*.py')},'torch':torch.__version__}
        (folder/'training.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=5000); p.add_argument('--device',default='cuda')
    p.add_argument('--recovery')
    run(**vars(p.parse_args()))

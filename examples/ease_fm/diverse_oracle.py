"""Privileged physical-parameter reference for an unresolved training-control gate."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from .mixed_data import MixedData
from .mixed_sim import Arm, AXES, MixedReach
from .benchmark import fingerprint
from .full_control import fit, add_recovery, commands, recovery_chunk


def oracle_tokens(arm, padded=False):
    count=7 if padded else arm.dof
    values=np.zeros((count,11),np.float32)
    for j,axis in enumerate(AXES[arm.family]):
        values[j,:3]=[float(x) for x in axis.split()]
        values[j,3:7]=[arm.lengths[j],arm.masses[j],arm.strength,arm.damping]
    return np.broadcast_to(values,(32,count,11)).copy()


class OracleData(MixedData):
    def __init__(self,directory):
        super().__init__(directory)
        self.arms={a['seed']:Arm(**a) for a in self.manifest['arms'] if a['split']=='train'}
        self.contexts=[oracle_tokens(a) for a in self.arms.values()]

    def histories(self,indices,rng):
        histories=np.stack([oracle_tokens(self.arms[self.episode_body[self.row_episode[i]]],padded=True) for i in indices])
        return torch.from_numpy(histories),torch.full((len(indices),),32,dtype=torch.long)

@torch.no_grad()
def oracle_rollout(dataset,model,kind,folder,device,collect=False,mode='correct32'):
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
        for i,env in enumerate(envs):
            histories[i] = oracle_tokens(env.arm, padded=True)
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


def run(data,source,output,steps=10000,device='cuda'):
    torch.set_num_threads(4)
    root=Path(output); source=Path(source); root.mkdir(parents=True,exist_ok=False)
    dataset=OracleData(data)
    gate=json.loads((source/'recovery_direct/gate.json').read_text())
    if gate['passed']: raise ValueError('Conditional oracle diagnostic is intended for a failed training gate')
    paths=[source/('original_'+kind)/'recovery.npz' for kind in ['flow','direct']]
    dataset=add_recovery(dataset,paths)
    hashes={str(p):fingerprint(p) for p in paths}
    protocol={'experiment':'E15-oracle','scope':'Privileged training-body reference; not a deployable history policy',
        'seed':0,'steps':steps,'batch_size':32,'parameters':'Same 244289-parameter architecture as history policy',
        'inputs':'Local joint-axis XYZ, link length, link mass, actuator strength, damping; four zero columns; repeated across 32 token positions',
        'real_calibration_actions':0,'privileged_inputs':True,'body_ids_or_goals_in_context':False,
        'normalization':'Oracle token statistics from training bodies only',
        'comparison':'Same corrected dataset, recovery union, row sampling, initialization seed, objective and update budget; context information and distribution change',
        'interpretation':'Success would suggest an information/extraction bottleneck; failure would leave controller fitting unresolved. Not a formal achievable upper bound.',
        'manifest_sha256':fingerprint(Path(data)/'manifest.json'),'recovery_hashes':hashes,'test_evaluated':False}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    src=root/'source'; src.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p,src/p.name)
    folder=root/'direct'
    model=fit(dataset,folder,'direct',steps,device,protocol['manifest_sha256'],fingerprint(source/'shared_recovery.json'))
    records=oracle_rollout(dataset,model,'direct',folder,device,mode='oracle')
    bodies={b:[r for r in records if r['body_seed']==b] for b in sorted({r['body_seed'] for r in records})}
    passed=sum(r['success'] for r in records)/len(records)>=.95 and all(sum(r['success'] for r in rows)/len(rows)>=.9 for rows in bodies.values())
    result={'passed':passed,'overall':[sum(r['success'] for r in records),len(records)],
        'by_body':{b:[sum(r['success'] for r in rows),len(rows)] for b,rows in bodies.items()}}
    (root/'gate.json').write_text(json.dumps(result,indent=2)); print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--source',required=True); p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=10000); p.add_argument('--device',default='cuda')
    run(**vars(p.parse_args()))

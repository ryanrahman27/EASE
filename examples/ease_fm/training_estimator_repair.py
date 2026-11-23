"""E117 matched training-body diagnosis and inverse-free estimator fine tuning."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from .training_estimator_core import ROOT,evaluate,load
from .mixed_sim import AXES
from .benchmark import fingerprint
from .motion_transformer import MotionTransformer
from .control_training import predict,batch
from .estimator_distillation import analytical_targets
from .task_retention_training import action_tensor
TASKS=Path('outputs/ease_arm_pick_place_v2')

def setup():
    tasks,models,hashes=load();AXES.update(json.loads((TASKS/'axis_catalog.json').read_text()));tasks=json.loads((TASKS/'cases.json').read_text())
    assert len(tasks)==12 and all(t['arm']['split']=='train' for t in tasks)
    return tasks,models,hashes

def prepare():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','source'):(ROOT/d).mkdir()
    tasks,models,hashes=setup();inputs={str(TASKS/n):fingerprint(TASKS/n) for n in ('cases.json','axis_catalog.json')}
    spec=dict(experiment='E117',parents=hashes,inputs=inputs,
        data='All12 original training-body tasks x analytical/3 E99 checkpoints; frozen calibrated bias, response updates and32 local probe, no selector or geometry filter. Save identical pre-update histories, measured next transition, nominal task error and analytical targets. All outcomes retained. First2 placements/body train, third held out across every driver; no unseen-body data used.',
        training='Three fixed final1500-update runs from E99 retained seeds, AdamW2e-5, batch128 (96 clean training-task contexts +32 synthetic), gradient clip1. Equal sampling of6/7DOF task contexts. Loss: ridge-matrix MSE +0.05 normalized analytical-action MSE on non-probe task rows +0.02 measured response MSE normalized by0.003^2. No parent-action anchor on task rows; targets are analytical actions from same history. No matrix solve in network inference. All targets/calibration/controller/scales unchanged. No validation-based checkpoint or seed selection.',
        evaluation='Reuse48 baseline episodes; run36 candidate full training-body tasks (including held placements). Report matched prediction/action/response errors separately from task success. Gate: pooled success improves, no old success lost, no seed or DOF declines, no collision increase. No automatic unseen evaluation or promotion on failure.',
        videos='First success/failure for each parent and candidate seed, analytical reference; paired first gain/loss for each seed, both methods. Outcome-selected diagnostic videos. All traces,states,models and hashes saved.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for n in ('build_training_estimator.py','training_estimator_core.py','training_estimator_repair.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)
    samples=[];rows=[]
    for case,t in enumerate(tasks):
        initial=None
        for method,model in models.items():
            r=evaluate(case,t,method,model,'frozen_bias',save_states=True,contexts=samples)
            if initial is None:initial=r['initial_estimator']
            assert r['initial_estimator']==initial
            (ROOT/'trials'/f'{case}_{method}_frozen_bias.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('data case',case,len(rows),flush=True)
    keys=('g','c','mask','error','bias','dq','dx','case','phase','clean','validation');arrays={k:np.stack([r[k] for r in samples]) for k in keys}
    np.savez_compressed(ROOT/'contexts.npz',**arrays)
    (ROOT/'data_audit.json').write_text(json.dumps(dict(rows=len(samples),clean_training=int(np.sum(arrays['clean']&~arrays['validation'])),clean_held=int(np.sum(arrays['clean']&arrays['validation'])),training_bodies=sorted({t['arm']['seed'] for t in tasks}),contexts_sha256=fingerprint(ROOT/'contexts.npz')),indent=2))
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h

def train():
    torch.set_num_threads(4);_,parents,hashes=setup();f=np.load(ROOT/'contexts.npz');v={k:torch.tensor(f[k],device='cuda') for k in f.files if k!='phase'}
    v['action_eligible']=torch.tensor(f['phase']!='probe',device='cuda');v['target']=analytical_targets(v['g'],v['c'],v['mask']);v['action_target']=action_tensor(v['target'],v['error'],v['bias'],v['mask'])
    train_ids={n:torch.where(v['clean']&~v['validation']&(v['mask'].sum(-1)==n))[0] for n in (6,7)}
    val=torch.where(v['clean']&v['validation'])[0]
    def metrics(model):
        model.eval();totals=[]
        with torch.no_grad():
            for ids in val.split(256):
                p=predict(model,v['g'][ids],v['c'][ids],v['mask'][ids]);m=v['mask'][ids];a=action_tensor(p,v['error'][ids],v['bias'][ids],m)
                pe=torch.einsum('bnd,bn->bd',p,v['dq'][ids]);ae=torch.einsum('bnd,bn->bd',v['target'][ids],v['dq'][ids]);eligible=v['action_eligible'][ids]
                for i,idx in enumerate(ids):totals.append(dict(dof=int(m[i].sum()),matrix_mse=float((p[i]-v['target'][idx]).square().sum()/(6*m[i].sum())),action_rmse=float(torch.sqrt(((a[i]-v['action_target'][idx]).square()*m[i]).sum()/m[i].sum())) if eligible[i] else None,xyz_error=float(torch.linalg.vector_norm(pe[i,:3]-v['dx'][idx,:3])),analytical_xyz_error=float(torch.linalg.vector_norm(ae[i,:3]-v['dx'][idx,:3]))))
        return {str(n):{k:float(np.mean([r[k] for r in totals if (n=='all' or r['dof']==n) and r[k] is not None])) for k in ('matrix_mse','action_rmse','xyz_error','analytical_xyz_error')} for n in ('all',6,7)}
    for seed in range(3):
        torch.manual_seed(117000+seed);model=MotionTransformer().cuda();model.load_state_dict(parents[f'learned_{seed}'].state_dict());initial=metrics(model);model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5);gen=torch.Generator(device='cuda').manual_seed(117100+seed);history=[];digest=hashlib.sha256()
        for step in range(1500):
            ids=torch.cat([pool[torch.randint(len(pool),(48,),device='cuda',generator=gen)] for pool in train_ids.values()]);sg,sc,sm,st,_=batch(32,'cuda',gen)
            g=torch.cat([v['g'][ids],sg]);c=torch.cat([v['c'][ids],sc]);m=torch.cat([v['mask'][ids],sm]);target=torch.cat([v['target'][ids],st]);p=predict(model,g,c,m)
            fit=((p-target).square()*m[:,:,None]).sum()/(6*m.sum())
            action=action_tensor(p[:96],v['error'][ids],v['bias'][ids],m[:96]);eligible=v['action_eligible'][ids,None]*m[:96]
            action_loss=((action-v['action_target'][ids]).square()*eligible).sum()/(eligible.sum()*.04**2)
            response=torch.einsum('bnd,bn->bd',p[:96],v['dq'][ids]);response_loss=(response-v['dx'][ids]).square().mean()/.003**2
            loss=fit+.05*action_loss+.02*response_loss;assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
            digest.update(ids.cpu().numpy().tobytes());history.append([float(fit.detach()),float(action_loss.detach()),float(response_loss.detach())])
            if (step+1)%500==0:print('train',seed,step+1,history[-1],flush=True)
        folder=ROOT/f'seed_{seed}';folder.mkdir();model.eval();torch.save(model.state_dict(),folder/'policy.pt')
        (folder/'training.json').write_text(json.dumps(dict(seed=seed,steps=1500,initial=initial,final=metrics(model),losses=history,batch_hash=digest.hexdigest(),policy_sha256=fingerprint(folder/'policy.pt')),indent=2));print('trained',seed,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

def evaluation():
    torch.set_num_threads(4);tasks,parents,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==48
    models={}
    for seed in range(3):
        p=ROOT/f'seed_{seed}/policy.pt';m=MotionTransformer().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));models[f'candidate_{seed}']=m.eval()
    for case,t in enumerate(tasks):
        for method,model in models.items():
            r=evaluate(case,t,method,model,'frozen_bias',save_states=True);old=json.loads((ROOT/'trials'/f'{case}_learned_{method[-1]}_frozen_bias.json').read_text());assert r['initial_estimator']==old['initial_estimator'] and r['calibration']==old['calibration']
            (ROOT/'trials'/f'{case}_{method}_frozen_bias.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('eval case',case,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'evaluate':evaluation}[sys.argv[1]]()

"""Matched-history audit of E144; no new test outcomes or privileged Jacobians."""
import shutil
import numpy as np
import torch
from . import history_estimator_control as control
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .learned_history_weights import token_features
ROOT=control.ROOT/'diagnosis'

def run(worker=None):
    global ROOT
    if worker:ROOT=control.ROOT/('diagnosis_'+worker)
    record_path=control.ROOT/(f'records_{worker}.json' if worker else 'records.json')
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);models=control.load_neural();folds=read(control.ROOT/'folds.json');choices=read(control.ROOT/'reference_choices.json');rows=read(record_path);out=[];chunks=[]
    for row in rows:
        case,method=row['case'],row['method'];stem=f'{case}_{method}_frozen_bias';p=control.ROOT/'histories'/f'{stem}.npz';d=np.load(p);ids=np.flatnonzero(d['clean'] & (np.arange(len(d['clean']))%4==0));f=folds[str(case)];ch=choices[f]
        if not len(ids):continue
        trial=read(control.ROOT/'trials'/f'{stem}.json');initial=trial['initial_estimator'];prior=np.zeros((7,6));prior[:row['dof']]=np.linalg.solve(initial['gram'],initial['cross'])
        q=torch.tensor(d['dq_history'][ids],device='cuda');x=torch.tensor(d['dx_history'][ids],device='cuda');mask=torch.tensor(np.tile(d['mask'],(len(ids),1)),device='cuda');query=torch.tensor(d['query'][ids],device='cuda',dtype=torch.float64);target=d['target'][ids];preds={};norms={};conditions={}
        with torch.no_grad():
            feature=token_features(q,x,mask);neural=torch.stack([m(q,x,mask,feature)[0] for m in models[f]]).mean(0);js={'learned':neural};qq,xx=q.double(),x.double();eye=torch.eye(7,device='cuda',dtype=torch.float64)
            for name,tau,ridge in [('constant',ch['forgetting'],ch['ridge']),('uniform',.98,1e-6),('prior_anchored',.98,1e-6)]:
                w=tau**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);g=qq.transpose(1,2)@(qq*w[None,:,None]);cross=qq.transpose(1,2)@(xx*w[None,:,None]);rhs=cross+ridge*torch.tensor(prior,device='cuda')[None] if name=='prior_anchored' else cross;js[name]=torch.linalg.solve(g+ridge*eye[None],rhs)
            for name,j in js.items():preds[name]=(query[:,None]@j).squeeze(1).cpu().numpy();norms[name]=torch.linalg.vector_norm(j,dim=(1,2)).cpu().numpy()
        for phase in np.unique(d['phase'][ids]):
            keep=d['phase'][ids]==phase
            out.append(dict(case=case,controller=method,dof=row['dof'],phase=int(phase),n=int(keep.sum()),metrics={k:dict(position_mse=float(np.mean((v[keep,:3]-target[keep,:3])**2)),rotation_mse=float(np.mean((v[keep,3:]-target[keep,3:])**2)),jacobian_norm_mean=float(norms[k][keep].mean()),jacobian_norm_max=float(norms[k][keep].max())) for k,v in preds.items()}))
        chunks.append(dict(case=np.full(len(ids),case),fold=np.full(len(ids),f),controller=np.full(len(ids),control.METHODS.index(method)),step=ids,dq=d['dq_history'][ids],dx=d['dx_history'][ids],mask=np.tile(d['mask'],(len(ids),1)),query=d['query'][ids],target=target,prior=np.tile(prior[None],(len(ids),1,1)),phase=d['phase'][ids]))
    np.savez_compressed(ROOT/'dataset.npz',**{k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]});save(ROOT/'rows.json',out);summary={}
    for controller in control.METHODS:
        summary[controller]={}
        for dof in ('all',6,7):
            rr=[r for r in out if r['controller']==controller and (dof=='all' or r['dof']==dof)];n=sum(r['n'] for r in rr)
            if not n:continue
            summary[controller][str(dof)]={k:dict(position_rmse_mm=float(np.sqrt(sum(r['n']*r['metrics'][k]['position_mse'] for r in rr)/n)*1000),rotation_rmse_mm=float(np.sqrt(sum(r['n']*r['metrics'][k]['rotation_mse'] for r in rr)/n)*1000),n=n) for k in ('learned','constant','uniform','prior_anchored')}
    save(ROOT/'summary.json',summary);save(ROOT/'protocol.json',dict(scope='Post-E144 fitting diagnosis. Every fourth transition whose pre-action history has no previous collision. Actual next joint displacement is the supervised query. Same family-excluded learned/constant estimates on identical measured histories. Initial calibration prior is measured, never privileged. Prior-anchored analytical estimator is diagnostic and not a new learned result.',episodes=len(rows),worker=worker,inputs={str(record_path):fingerprint(record_path)}));shutil.copy2(__file__,ROOT/'source.py');print(summary,flush=True)

if __name__=='__main__':
    import sys
    run(sys.argv[1] if len(sys.argv)>1 else None)

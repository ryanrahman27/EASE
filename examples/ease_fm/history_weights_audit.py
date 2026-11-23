"""E141 independent prediction audit and stronger fitting-only reference controls."""
import shutil
from pathlib import Path
import numpy as np
import torch
from .learned_history_weights import ROOT,HistoryWeights,SEEDS,uniform
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    torch.set_num_threads(4);d=np.load(ROOT/'dataset.npz');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query','target')};o=np.load(ROOT/'oof.npz')['prediction'];ref=np.load(ROOT/'references.npz');fold=d['fold'];target=d['target'];n=len(fold);maximum=0.
    with torch.no_grad():
        initial=HistoryWeights().cuda().eval();j,_,_=initial(v['dq'][:64],v['dx'][:64],v['mask'][:64]);u=uniform(v['dq'][:64],v['dx'][:64])[0];torch.testing.assert_close(j,u,rtol=1e-6,atol=1e-7)
        for f in range(4):
            ids,va=np.flatnonzero(fold!=f),np.flatnonzero(fold==f);assert not set(d['family'][ids])&set(d['family'][va]);norm=np.load(ROOT/'models'/f'fold_{f}'/'normalization.npz');np.testing.assert_array_equal(norm['training'],ids);t=target[ids];sc=np.r_[np.full(3,max(float(np.sqrt(np.mean(t[:,:,:3]**2))),1e-5)),np.full(3,max(float(np.sqrt(np.mean(t[:,:,3:]**2))),1e-5))].astype(np.float32);np.testing.assert_array_equal(sc,norm['scale'])
            for si,seed in enumerate(SEEDS):
                m=HistoryWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{f}'/f'{seed}.pt',weights_only=True))
                for start in range(0,len(va),256):
                    ii=va[start:start+256];j,_,_=m(v['dq'][ii],v['dx'][ii],v['mask'][ii]);p=(v['query'][ii]@j).cpu().numpy();maximum=max(maximum,float(np.max(np.abs(p-o[si,ii]))));np.testing.assert_allclose(p,o[si,ii],atol=1e-7,rtol=1e-4)
    # Diagnostic controls added after the primary result; each parameter uses training folds only.
    p=ref['uniform'];scaled=np.zeros_like(p);coefficients=[]
    for f in range(4):
        tr,va=fold!=f,fold==f;a=np.sum(p[tr]*target[tr],axis=(0,1))/np.maximum(np.sum(p[tr]**2,axis=(0,1)),1e-20);scaled[va]=p[va]*a;coefficients.append(a.tolist())
    candidates=[];predictions=[]
    with torch.no_grad():
        for forgetting in (.8,.9,.95,.98,1.):
            w=forgetting**torch.arange(31,-1,-1,device='cuda',dtype=torch.float32);g=v['dq'].transpose(1,2)@(v['dq']*w[None,:,None]);c=v['dq'].transpose(1,2)@(v['dx']*w[None,:,None])
            for ridge in (1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4):
                j=torch.linalg.solve(g+ridge*torch.eye(7,device='cuda')[None],c);predictions.append((v['query']@j).cpu().numpy());candidates.append(dict(forgetting=forgetting,ridge=ridge))
    pp=np.stack(predictions);tuned=np.zeros_like(p);chosen=[]
    for f in range(4):
        tr,va=fold!=f,fold==f;scale=np.load(ROOT/'models'/f'fold_{f}'/'normalization.npz')['scale'];loss=np.mean(((pp[:,tr]-target[None,tr])/scale)**2,axis=(1,2,3));best=int(np.argmin(loss));tuned[va]=pp[best,va];chosen.append(dict(fold=f,choice=candidates[best],training_loss=float(loss[best])))
    extra={'zero':np.zeros_like(p),'axis_scaled_uniform':scaled,'tuned_constant_kernel':tuned};metrics={};dof=d['mask'].sum(1)
    for name,pred in extra.items():
        metrics[name]={}
        for count in ('all',6,7):
            ids=np.ones(n,bool) if count=='all' else dof==count;e=pred[ids]-target[ids];metrics[name][str(count)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,:,3:]**2))*1000))
    np.savez_compressed(ROOT/'extra_references.npz',**extra);save(ROOT/'reference_fit.json',dict(note='Post-result diagnostic reference controls; no neural-model tuning. All reference parameters selected within each fitting fold only.',axis_coefficients=coefficients,kernel_grid=candidates,kernel_choices=chosen));summary=read(ROOT/'summary.json');ratio=(summary['metrics']['learned']['all']['position_rmse_mm']/metrics['tuned_constant_kernel']['all']['position_rmse_mm'])**2;save(ROOT/'audit_summary.json',dict(extra_reference_metrics=metrics,learned_vs_tuned_kernel_position_mse_ratio=ratio,max_prediction_reproduction_error_m=maximum))
    for p,h in {**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'input_hashes.json'),**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(initial_model_matches_uniform=True,history_only_features=True,predictions_reproduced=True,max_error_m=maximum,folds_disjoint=True,normalization_training_only=True,reference_tuning_training_only=True,inputs_unchanged=True,E120_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(read(ROOT/'audit_summary.json'),flush=True)

if __name__=='__main__':run()

"""E150: matched observed-only versus joint-balanced fine-tuning, fixed folds."""
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import calibration_prior_learning as prior
from .dense_joint_supervision import ROOT as DATA
from .history_estimator_control import ROOT as CONTROL
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_dense_joint_training_v1')
CONDITIONS=('observed','mixed')

def prepare():
    assert read(DATA/'verification.json')['all_labels_reconstructed_from_saved_states']==9614;ROOT.mkdir(exist_ok=False)
    for folder in ('models','source'):(ROOT/folder).mkdir()
    d=dict(np.load(DATA/'dataset.npz'));rows=read(DATA/'rows.json');queries=[];targets=[];cached={}
    for r in rows:
        if r['case'] not in cached:cached[r['case']]=dict(np.load(CONTROL/'histories'/f"{r['case']}_constant_frozen_bias.npz"))
        h=cached[r['case']];queries.append(h['query'][r['step']]);targets.append(h['target'][r['step']])
    d['query']=np.concatenate([np.asarray(queries)[:,None],d['query']],1);d['target']=np.concatenate([np.asarray(targets)[:,None],d['target']],1);d['query_mask']=np.concatenate([np.ones((len(rows),1),bool),d['query_mask']],1);np.savez_compressed(ROOT/'dataset.npz',**d);shutil.copy2(DATA/'cases.json',ROOT/'cases.json');shutil.copy2(DATA/'rows.json',ROOT/'rows.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);shutil.copy2(Path(prior.__file__),ROOT/'source'/Path(prior.__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E150',inputs={str(p):fingerprint(p) for p in [DATA/'dataset.npz',DATA/'verification.json',prior.ROOT/'frozen.json',*(prior.ROOT/'models').rglob('*')] if p.is_file()},design='Same738 phase-entry histories, same96 fitting placements, same four ancestry-family folds, same three fixed E146 starting checkpoints. Compare continued observed-next-motion supervision against equal-weight observed plus all signed joint-probe targets. This controls for extra fitting updates and context selection. Probes are measured simulated transitions; no exact Jacobian labels or future input.',training='Each condition/fold/seed:1000 AdamW steps,lr1e-4,wd1e-4,batch128 histories,clip1,.001 log-weight regularization. Sample placements uniformly and histories uniformly within placement. Keep E146 training-only normalization, architecture, initial measured calibration prior, .98 recency and1e-6 ridge unchanged. Observed objective uses original next joint/tool motion. Mixed objective averages observed normalizedMSE and mean active-probe normalizedMSE equally. All three final Jacobians averaged; no best checkpoint/seed selection.',screen='Primary mixed ensemble: at least5% lower directional position MSE than fixed E146 on these histories; neither joint count directional error worse; directional rotation MSE<=1.05x; observed-next-motion position MSE<=1.05x. Also report observed-only fine-tuning to isolate dense-target effect. Passing permits complete fitting control only. E120 unchanged; held/fresh sets remain closed.'))

def train():
    torch.set_num_threads(4);d=dict(np.load(ROOT/'dataset.npz'));v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','prior','query','target','query_mask')};n=len(d['case']);out=np.empty((2,3,n,15,6),float);reference=np.empty((3,n,15,6),float);logs=[]
    with torch.no_grad():f=prior.features(v['dq'],v['dx'],v['mask'],v['prior'])
    for split in range(4):
        ids=np.flatnonzero(d['fold']!=split);va=np.flatnonzero(d['fold']==split);cases,counts=np.unique(d['case'][ids],return_counts=True);freq=dict(zip(cases,counts));prob=np.array([1/freq[c] for c in d['case'][ids]]);prob/=prob.sum();scale=torch.tensor(np.load(prior.ROOT/'models'/f'fold_{split}'/'normalization.npz')['scale'],device='cuda')
        for si,seed in enumerate(prior.SEEDS):
            initial=torch.load(prior.ROOT/'models'/f'fold_{split}'/f'{seed}.pt',weights_only=True);base=prior.PriorWeights().cuda().eval();base.load_state_dict(initial)
            with torch.no_grad():
                j,_=base(v['dq'][va],v['dx'][va],v['mask'][va],v['prior'][va],f[va]);reference[si,va]=(v['query'][va].double()@j).cpu().numpy()
            for ci,condition in enumerate(CONDITIONS):
                torch.manual_seed(seed);rng=np.random.default_rng(seed);m=prior.PriorWeights().cuda();m.load_state_dict(initial);opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);history=[]
                for step in range(1000):
                    ii=rng.choice(ids,128,replace=True,p=prob);j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],v['prior'][ii],f[ii]);p=v['query'][ii].double()@j;error=((p-v['target'][ii])/scale)**2;observed=error[:,0].mean();mask=v['query_mask'][ii,1:];dense=((error[:,1:].mean(-1)*mask).sum(1)/mask.sum(1)).mean();mse=observed if condition=='observed' else .5*(observed+dense);loss=mse+.001*torch.mean(lw**2);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                    if step%100==0:history.append(dict(step=step,loss=float(loss.detach()),observed=float(observed.detach()),dense=float(dense.detach())))
                directory=ROOT/'models'/condition/f'fold_{split}';directory.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),directory/f'{seed}.pt');m.eval()
                with torch.no_grad():
                    j,_=m(v['dq'][va],v['dx'][va],v['mask'][va],v['prior'][va],f[va]);out[ci,si,va]=(v['query'][va].double()@j).cpu().numpy()
                logs.append(dict(condition=condition,fold=split,seed=seed,train_indices=ids.tolist(),validation_indices=va.tolist(),history=history));print('E150',condition,split,seed,flush=True)
    np.savez_compressed(ROOT/'predictions.npz',trained=out,reference=reference);save(ROOT/'training.json',logs);metrics={}
    for name,p in [('E146',reference.mean(0)),*[(condition,out[ci].mean(0)) for ci,condition in enumerate(CONDITIONS)]]:
        metrics[name]={}
        for domain in ('observed','directional'):
            metrics[name][domain]={}
            for n in ('all',6,7):
                keep=d['mask'].sum(1)==n if n!='all' else np.ones(len(d['case']),bool);e=p[keep]-d['target'][keep];e=e[:,0] if domain=='observed' else e[:,1:][d['query_mask'][keep,1:]];metrics[name][domain][str(n)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,3:]**2))*1000))
    a,b=metrics['mixed'],metrics['E146'];qualified=(a['directional']['all']['position_rmse_mm']/b['directional']['all']['position_rmse_mm'])**2<=.95 and all(a['directional'][str(n)]['position_rmse_mm']<=b['directional'][str(n)]['position_rmse_mm'] for n in (6,7)) and a['directional']['all']['rotation_rmse_mm']**2<=1.05*b['directional']['all']['rotation_rmse_mm']**2 and a['observed']['all']['position_rmse_mm']**2<=1.05*b['observed']['all']['position_rmse_mm']**2;save(ROOT/'summary.json',dict(metrics=metrics,qualifies_for_control=qualified,fresh_evaluated=False));save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'dataset.npz',ROOT/'predictions.npz',ROOT/'training.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*')] if p.is_file()});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train}[sys.argv[1]]()

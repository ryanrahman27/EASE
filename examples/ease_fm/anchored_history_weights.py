"""E142: stable kernel reference and mass-normalized learned history weighting."""
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .learned_history_weights import ROOT as PRIOR,HistoryWeights,token_features,SEEDS
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_anchored_history_weights_v1')

class AnchoredWeights(HistoryWeights):
    def __init__(self,forgetting=1.,ridge=1e-7):
        super().__init__();del self.ridge_log;self.register_buffer('forgetting',torch.tensor(forgetting,dtype=torch.float64));self.register_buffer('ridge',torch.tensor(ridge,dtype=torch.float64))
    def forward(self,dq,dx,mask,features=None):
        f=token_features(dq,dx,mask) if features is None else features;logw=np.log(5)*torch.tanh(self.output(self.encoder(self.input(f))).squeeze(-1));base=self.forgetting**torch.arange(dq.shape[1]-1,-1,-1,device=dq.device,dtype=torch.float64);w=logw.double().exp();w=w/(torch.sum(w*base[None],1,keepdim=True)/base.sum());weighted=w*base[None];q,z=dq.double(),dx.double();g=q.transpose(1,2)@(q*weighted[:,:,None])+self.ridge*torch.eye(7,device=dq.device,dtype=torch.float64)[None];c=q.transpose(1,2)@(z*weighted[:,:,None]);j=torch.linalg.solve(g,c)*mask[:,:,None];return j,torch.log(w)

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source'):(ROOT/folder).mkdir()
    inputs=[PRIOR/'dataset.npz',PRIOR/'oof.npz',PRIOR/'audit_summary.json',PRIOR/'reference_fit.json',PRIOR/'verification.json']
    save(ROOT/'protocol.json',dict(experiment='E142',inputs={str(p):fingerprint(p) for p in inputs},
        motivation='E141 neural weights improve uniform history but not the training-tuned constant kernel; all folds chose the low-ridge grid boundary. Check that boundary before attributing gains to attention.',
        reference='Within each fitting fold only, choose normalized next-motion MSE over forgetting .8,.9,.95,.98,1 and ridge1e-10,1e-9,1e-8,3e-8,1e-7,3e-7,1e-6,1e-5. Compute Gram/cross and solve in float64 for numerical stability. Compare float64 and original float32 uniform controls. No held or fresh task evaluation.',
        learned='Initialize same8-feature32-wide2-layer Transformer weight model at that fold-selected kernel. Freeze forgetting and ridge. Normalize positive transition weights to preserve total recency-weighted mass, preventing a global weight scale from merely changing effective ridge. Solve in float64, network float32. Only relative data-dependent transition weighting is learned.',
        training='Same four family folds, same3 seeds141100..141102,1500 steps,AdamW3e-4 weight_decay1e-4,batch128 starts/both queries,clip1. Training-group motion scales copied from E141; loss normalized forward MSE +.001 mean normalized log-weight squared. No task success labels, seed/checkpoint selection or learned global ridge.',
        advancement='For neural control, pooled position MSE at least5% below selected constant kernel; every seed no worse; neither joint-count pooled error worse; rotation MSE<=1.05x constant. Independently assess the fitted constant kernel versus uniform32 as an engineering control candidate; never claim its gain as a Transformer gain. E120 unchanged.'))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def references():
    torch.set_num_threads(4);d=np.load(PRIOR/'dataset.npz');q=torch.tensor(d['dq'],device='cuda',dtype=torch.float64);x=torch.tensor(d['dx'],device='cuda',dtype=torch.float64);query=torch.tensor(d['query'],device='cuda',dtype=torch.float64);target=d['target'];fold=d['fold'];grid=[];pred=[]
    with torch.no_grad():
        for forgetting in (.8,.9,.95,.98,1.):
            w=forgetting**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);g=q.transpose(1,2)@(q*w[None,:,None]);c=q.transpose(1,2)@(x*w[None,:,None])
            for ridge in (1e-10,1e-9,1e-8,3e-8,1e-7,3e-7,1e-6,1e-5):
                j=torch.linalg.solve(g+ridge*torch.eye(7,device='cuda',dtype=torch.float64)[None],c);p=(query@j).cpu().numpy();assert np.isfinite(p).all();grid.append(dict(forgetting=forgetting,ridge=ridge));pred.append(p)
    pp=np.stack(pred);out=np.zeros_like(target,dtype=np.float64);choices=[]
    for f in range(4):
        tr,va=fold!=f,fold==f;scale=np.load(PRIOR/'models'/f'fold_{f}'/'normalization.npz')['scale'];loss=np.mean(((pp[:,tr]-target[None,tr])/scale)**2,axis=(1,2,3));best=int(np.argmin(loss));out[va]=pp[best,va];choices.append(dict(fold=f,index=best,**grid[best],training_loss=float(loss[best])));print('E142 reference',choices[-1],flush=True)
    uniform_index=next(i for i,r in enumerate(grid) if r==dict(forgetting=.98,ridge=1e-6));np.savez_compressed(ROOT/'references.npz',selected=out,uniform64=pp[uniform_index]);save(ROOT/'reference_grid.json',grid);save(ROOT/'reference_choices.json',choices)

def train():
    torch.set_num_threads(4);d=np.load(PRIOR/'dataset.npz');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query','target')};fold=d['fold'];choices=read(ROOT/'reference_choices.json');n=len(fold)
    with torch.no_grad():f=token_features(v['dq'],v['dx'],v['mask'])
    out=np.empty((3,n,2,6),np.float64);logs=[]
    for split in range(4):
        ids,va=np.flatnonzero(fold!=split),np.flatnonzero(fold==split);assert not set(d['family'][ids])&set(d['family'][va]);choice=choices[split];directory=ROOT/'models'/f'fold_{split}';directory.mkdir();shutil.copy2(PRIOR/'models'/f'fold_{split}'/'normalization.npz',directory/'normalization.npz');scale=torch.tensor(np.load(directory/'normalization.npz')['scale'],device='cuda',dtype=torch.float64)
        for si,seed in enumerate(SEEDS):
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=AnchoredWeights(choice['forgetting'],choice['ridge']).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);history=[]
            for step in range(1500):
                idx=rng.choice(ids,128,replace=True);j,lw=m(v['dq'][idx],v['dx'][idx],v['mask'][idx],f[idx]);p=v['query'][idx].double()@j;mse=torch.mean(((p-v['target'][idx])/scale)**2);loss=mse+.001*torch.mean(lw**2);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                if step%100==0:history.append(dict(step=step,loss=float(loss.detach()),mse=float(mse.detach())))
            m.eval();torch.save(m.state_dict(),directory/f'{seed}.pt')
            with torch.no_grad():
                for start in range(0,len(va),256):
                    ii=va[start:start+256];j,_=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);out[si,ii]=(v['query'][ii].double()@j).cpu().numpy()
            logs.append(dict(fold=split,seed=seed,losses=history));print('E142 trained',split,seed,flush=True)
    np.savez_compressed(ROOT/'oof.npz',prediction=out,fold=fold);save(ROOT/'training.json',logs);summarize();shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'references.npz',ROOT/'reference_choices.json',ROOT/'reference_grid.json',ROOT/'oof.npz',ROOT/'training.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*.py')] if p.is_file()})

def summarize():
    d=np.load(PRIOR/'dataset.npz');truth=d['target'];dof=d['mask'].sum(1);ref=np.load(ROOT/'references.npz');o=np.load(ROOT/'oof.npz')['prediction'];metrics={}
    for name,p in [('constant',ref['selected'][None]),('uniform64',ref['uniform64'][None]),('learned',o)]:
        metrics[name]={}
        for count in ('all',6,7):
            ids=np.ones(len(dof),bool) if count=='all' else dof==count;e=p[:,ids]-truth[None,ids];metrics[name][str(count)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:,:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,:,:,3:]**2))*1000),position_mse_by_seed=np.mean(e[:,:,:,:3]**2,axis=(1,2,3)).tolist())
    a,b=metrics['learned'],metrics['constant'];ratio=(a['all']['position_rmse_mm']/b['all']['position_rmse_mm'])**2;advance=ratio<=.95 and all(v<=b['all']['position_mse_by_seed'][0] for v in a['all']['position_mse_by_seed']) and all(a[str(n)]['position_rmse_mm']<=b[str(n)]['position_rmse_mm'] for n in (6,7)) and a['all']['rotation_rmse_mm']**2<=1.05*b['all']['rotation_rmse_mm']**2
    save(ROOT/'summary.json',dict(metrics=metrics,position_mse_ratio=ratio,neural_qualifies_for_control=bool(advance),fresh_evaluated=False,task_rollouts=0,E120_unchanged=True));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'references':references,'train':train,'summarize':summarize}[sys.argv[1]]()

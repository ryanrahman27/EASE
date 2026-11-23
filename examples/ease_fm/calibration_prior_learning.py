"""E146: learn history weights around a measured calibration prior."""
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .learned_history_weights import HistoryWeights,token_features,SEEDS
from .history_estimator_control import ROOT as CONTROL
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_calibration_prior_learning_v1')
SOURCE=CONTROL/'diagnosis_constant'

def features(q,x,mask,prior):
    f=token_features(q,x,mask);res=x-q@prior.float();extra=torch.stack((torch.log1p(torch.linalg.vector_norm(res[:,:,:3],dim=-1)/.0001),torch.log1p(torch.linalg.vector_norm(res[:,:,3:],dim=-1)/.0001)),-1);return torch.cat((f,extra),-1)

def solve(q,x,mask,prior,logw):
    base=.98**torch.arange(31,-1,-1,device=q.device,dtype=torch.float64);w=logw.double().exp();w=w/(torch.sum(w*base[None],1,keepdim=True)/base.sum());weighted=w*base[None];q,x=q.double(),x.double();g=q.transpose(1,2)@(q*weighted[:,:,None])+1e-6*torch.eye(7,device=q.device,dtype=torch.float64)[None];cross=q.transpose(1,2)@(x*weighted[:,:,None])+1e-6*prior;return torch.linalg.solve(g,cross)*mask[:,:,None],torch.log(w)

class PriorWeights(HistoryWeights):
    def __init__(self):
        super().__init__();self.input=nn.Linear(10,32);del self.ridge_log
    def forward(self,q,x,mask,prior,f=None):
        if f is None:f=features(q,x,mask,prior)
        lw=np.log(5)*torch.tanh(self.output(self.encoder(self.input(f))).squeeze(-1));return solve(q,x,mask,prior,lw)

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source'):(ROOT/folder).mkdir()
    shutil.copy2(SOURCE/'dataset.npz',ROOT/'dataset.npz');shutil.copy2(CONTROL/'cases.json',ROOT/'cases.json');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E146',inputs={str(SOURCE/'dataset.npz'):fingerprint(SOURCE/'dataset.npz')},data='6706 measured transitions sampled every4 steps before any prior collision, across all96 constant-kernel fitting episodes. No success filtering. Temporally correlated samples; effective placement count96. Four unchanged ancestry-family folds. Fresh and old held placements excluded. Training sampling balances placements so long failed episodes cannot dominate.',model='Same two-layer32-wide4-head history Transformer, now10 scalar features: prior8 measured features plus position/orientation residual norms against initial measured calibration Jacobian. Positive relative transition weights, normalized total recency mass. Fixed forgetting.98 and ridge1e-6 toward actual calibration J0; solve in float64. Model sees past history and measured J0 residuals only. Actual future joint increment is a supervised query, never a network feature. No phase/body/contact/object truth.',training='Three fixed seeds141100..141102,1500 AdamW updates each fold,lr3e-4,wd1e-4,batch128,gradientclip1. Loss normalized measured next-motion MSE plus.001 mean log-weight squared; training-only position/orientation RMS scales, placement-balanced. Fixed all-three Jacobian mean is primary; no best seed/checkpoint selection.',screen='Primary all-three ensemble position MSE at least5% below same fixed analytical calibration prior, neither joint count worse, rotation MSE no more than1.05x. Individual seed results and all regressions disclosed. Passing justifies complete fitting episodes, not fresh test. E120 remains unchanged.'))

def train():
    torch.set_num_threads(4);d=np.load(ROOT/'dataset.npz');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','prior','query','target')};n=len(d['case']);out=np.empty((3,n,6),float);reference=np.empty((n,6),float);logs=[]
    with torch.no_grad():
        f=features(v['dq'],v['dx'],v['mask'],v['prior']);j,_=solve(v['dq'],v['dx'],v['mask'],v['prior'],torch.zeros((n,32),device='cuda'));reference[:]=(v['query'].double()[:,None]@j).squeeze(1).cpu().numpy()
    for split in range(4):
        ids=np.flatnonzero(d['fold']!=split);va=np.flatnonzero(d['fold']==split);cases,counts=np.unique(d['case'][ids],return_counts=True);freq=dict(zip(cases,counts));prob=np.array([1/freq[c] for c in d['case'][ids]]);prob/=prob.sum();squared=np.sum(d['target'][ids]**2*prob[:,None],axis=0);scale=np.r_[np.repeat(np.sqrt(squared[:3].mean()),3),np.repeat(np.sqrt(squared[3:].mean()),3)].clip(1e-5);scale_t=torch.tensor(scale,device='cuda');directory=ROOT/'models'/f'fold_{split}';directory.mkdir();np.savez(directory/'normalization.npz',scale=scale,train=ids,validation=va,probability=prob)
        for si,seed in enumerate(SEEDS):
            torch.manual_seed(seed);rng=np.random.default_rng(seed);m=PriorWeights().cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);history=[]
            for step in range(1500):
                ii=rng.choice(ids,128,replace=True,p=prob);j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],v['prior'][ii],f[ii]);p=(v['query'][ii].double()[:,None]@j).squeeze(1);mse=torch.mean(((p-v['target'][ii])/scale_t)**2);loss=mse+.001*torch.mean(lw**2);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                if step%100==0:history.append(dict(step=step,loss=float(loss.detach()),mse=float(mse.detach())))
            m.eval();torch.save(m.state_dict(),directory/f'{seed}.pt')
            with torch.no_grad():
                for start in range(0,len(va),256):
                    ii=va[start:start+256];j,_=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],v['prior'][ii],f[ii]);out[si,ii]=(v['query'][ii].double()[:,None]@j).squeeze(1).cpu().numpy()
            logs.append(dict(fold=split,seed=seed,history=history));print('E146 trained',split,seed,flush=True)
    np.savez_compressed(ROOT/'predictions.npz',learned=out,reference=reference);save(ROOT/'training.json',logs);metrics={}
    for name,p in [('reference',reference[None]),('ensemble',out.mean(0)[None]),('individual',out)]:
        metrics[name]={}
        for dof in ('all',6,7):
            keep=np.ones(n,bool) if dof=='all' else d['mask'].sum(1)==dof;e=p[:,keep]-d['target'][None,keep];metrics[name][str(dof)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,:,3:]**2))*1000),position_mse_by_seed=np.mean(e[:,:,:3]**2,axis=(1,2)).tolist())
    l,b=metrics['ensemble'],metrics['reference'];qualified=(l['all']['position_rmse_mm']/b['all']['position_rmse_mm'])**2<=.95 and all(l[str(n)]['position_rmse_mm']<=b[str(n)]['position_rmse_mm'] for n in (6,7)) and l['all']['rotation_rmse_mm']**2<=1.05*b['all']['rotation_rmse_mm']**2;save(ROOT/'summary.json',dict(metrics=metrics,qualifies_for_control=qualified,fresh_evaluated=False));save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'dataset.npz',ROOT/'predictions.npz',ROOT/'training.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*')] if p.is_file()});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train}[sys.argv[1]]()

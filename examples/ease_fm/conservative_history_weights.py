"""E157: one fixed conservative modulation around uniform history weights."""
import shutil
from pathlib import Path
import numpy as np
import torch
from .anchored_history_weights import AnchoredWeights
from .fixed_uniform_learning import ROOT as SOURCE
from .learned_history_weights import SEEDS,token_features
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_conservative_history_weights_v1')

class ConservativeWeights(AnchoredWeights):
    def forward(self,dq,dx,mask,features=None):
        _,lw=super().forward(dq,dx,mask,features);base=self.forgetting**torch.arange(31,-1,-1,device=dq.device,dtype=torch.float64);w=(.25*lw).exp();w=w/(torch.sum(w*base[None],1,keepdim=True)/base.sum());weights=w*base[None];q,x=dq.double(),dx.double();g=q.transpose(1,2)@(q*weights[:,:,None])+self.ridge*torch.eye(7,device=dq.device,dtype=torch.float64)[None];cross=q.transpose(1,2)@(x*weights[:,:,None]);return torch.linalg.solve(g,cross)*mask[:,:,None],torch.log(w)

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'protocol.json',dict(experiment='E157',rule='One fixed attenuation: raise E153 learned normalized weights to power.25, then renormalize recency mass. Kernel.98/ridge1e-6, no prior term. Preserve frozen network/checkpoints; no retraining or temperature sweep. Maximum transition-weight ratio shrinks from25 to sqrt(5). This trades some prediction gain for a smaller perturbation of the stronger uniform controller.',motivation='E154 standard improves task success77/96 vs uniform74, but has regressions. E155/E156 posture fixes interact badly with large learned map changes. Check a conservative modulation before another full task experiment.',screen='Require ensemble measured position MSE below uniform on both joint counts and overall, scaled rotation MSE<=1.05x. This explicitly tests a stability tradeoff rather than requiring the earlier5% accuracy margin. If it passes, use the already fixed lift-only posture rule with matched analytical/E120 references; predeclare a small diagnostic retention gate. No fresh tests or E120 changes here.',inputs={str(SOURCE/p):fingerprint(SOURCE/p) for p in ('dataset.npz','predictions.npz','frozen.json','verification.json')}))
    d=dict(np.load(SOURCE/'dataset.npz'));v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query')};out=np.empty((3,len(d['case']),6),float)
    with torch.no_grad():
        f=token_features(v['dq'],v['dx'],v['mask'])
        for split in range(4):
            ids=np.flatnonzero(d['fold']==split)
            for si,seed in enumerate(SEEDS):
                m=ConservativeWeights().cuda().eval();m.load_state_dict(torch.load(SOURCE/'models'/f'fold_{split}'/f'{seed}.pt',weights_only=True));assert float(m.forgetting)==.98 and float(m.ridge)==1e-6
                for start in range(0,len(ids),256):
                    ii=ids[start:start+256];j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);out[si,ii]=(v['query'][ii].double()[:,None]@j).squeeze(1).cpu().numpy();assert float((lw.max(1).values-lw.min(1).values).max())<=.5*np.log(5)+1e-6;w=.98**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);torch.testing.assert_close((lw.exp()*w).sum(1),w.sum().expand(len(ii)),atol=1e-12,rtol=1e-12)
    reference=np.load(SOURCE/'predictions.npz')['reference'];metrics={}
    for name,p in [('uniform',reference),('conservative',out.mean(0))]:
        metrics[name]={}
        for n in ('all',6,7):
            keep=d['mask'].sum(1)==n if n!='all' else np.ones(len(d['case']),bool);e=p[keep]-d['target'][keep];metrics[name][str(n)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,3:]**2))*1000))
    a,b=metrics['conservative'],metrics['uniform'];qualified=all(a[str(n)]['position_rmse_mm']<b[str(n)]['position_rmse_mm'] for n in ('all',6,7)) and a['all']['rotation_rmse_mm']**2<=1.05*b['all']['rotation_rmse_mm']**2;np.savez_compressed(ROOT/'predictions.npz',prediction=out,reference=reference);save(ROOT/'summary.json',dict(metrics=metrics,qualifies_for_diagnostic=qualified,fresh_evaluated=False));save(ROOT/'verification.json',dict(weight_ratio_bound=True,weight_mass_preserved=True,fixed_kernel=True,source_weights_unchanged=all(fingerprint(p)==h for p,h in read(SOURCE/'frozen.json').items())));save(ROOT/'frozen.json',{str(ROOT/p):fingerprint(ROOT/p) for p in ('protocol.json','predictions.npz','summary.json','source/conservative_history_weights.py')});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':run()

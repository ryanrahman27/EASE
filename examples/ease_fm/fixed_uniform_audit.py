"""E153 independent zero-prior reference and prediction verification."""
import shutil
import numpy as np
import torch
from .fixed_uniform_learning import ROOT
from .anchored_history_weights import AnchoredWeights
from .learned_history_weights import token_features,SEEDS
from .abstaining_task_value_gate import read,save,fingerprint,family

def run():
    torch.set_num_threads(4);d=dict(np.load(ROOT/'dataset.npz'));p=dict(np.load(ROOT/'predictions.npz'));v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query')};tasks=read(ROOT/'cases.json');q,x=d['dq'].astype(float),d['dx'].astype(float);w=.98**np.arange(31,-1,-1);g=q.transpose(0,2,1)@(q*w[None,:,None])+1e-6*np.eye(7)[None];cross=q.transpose(0,2,1)@(x*w[None,:,None]);j=np.linalg.solve(g,cross);reference=(d['query'][:,None]@j).squeeze(1);np.testing.assert_allclose(reference,p['reference'],atol=1e-10,rtol=1e-7)
    with torch.no_grad():
        f=token_features(v['dq'],v['dx'],v['mask']);initial=AnchoredWeights(.98,1e-6).cuda().eval();j,lw=initial(v['dq'][:128],v['dx'][:128],v['mask'][:128],f[:128]);np.testing.assert_allclose((v['query'][:128].double()[:,None]@j).squeeze(1).cpu().numpy(),reference[:128],rtol=1e-7,atol=1e-10)
        for split in range(4):
            norm=dict(np.load(ROOT/'models'/f'fold_{split}'/'normalization.npz'));ids,va=norm['train'],norm['validation'];np.testing.assert_array_equal(ids,np.flatnonzero(d['fold']!=split));np.testing.assert_array_equal(va,np.flatnonzero(d['fold']==split));assert not {family(tasks,int(i)) for i in d['case'][ids]}&{family(tasks,int(i)) for i in d['case'][va]}
            for si,seed in enumerate(SEEDS):
                m=AnchoredWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{split}'/f'{seed}.pt',weights_only=True));assert float(m.forgetting)==.98 and float(m.ridge)==1e-6;pred=[]
                for start in range(0,len(va),256):
                    ii=va[start:start+256];j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);pred.append((v['query'][ii].double()[:,None]@j).squeeze(1).cpu().numpy());wt=.98**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);torch.testing.assert_close((lw.exp()*wt).sum(1),wt.sum().expand(len(ii)),atol=1e-12,rtol=1e-12)
                np.testing.assert_array_equal(np.concatenate(pred),p['learned'][si,va])
    for path,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['inputs']}.items():assert fingerprint(path)==h
    save(ROOT/'verification.json',dict(independent_uniform_reference=True,initial_model_matches_uniform=True,all_predictions_exact=True,calibration_prior_unused=True,family_exclusion=True,fixed_kernel_preserved=True,weight_mass_preserved=True,hashes_unchanged=True));shutil.copy2(__file__,ROOT/'source'/__file__.split('/')[-1]);print('E153 verified',flush=True)

if __name__=='__main__':run()

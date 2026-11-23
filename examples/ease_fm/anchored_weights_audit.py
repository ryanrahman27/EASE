"""Independent E142/E143 reproduction before any controller deployment."""
import shutil
from pathlib import Path
import numpy as np
import torch
from .anchored_history_weights import ROOT,PRIOR,AnchoredWeights
from .history_weight_ensemble import ROOT as ENSEMBLE
from .learned_history_weights import SEEDS
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    torch.set_num_threads(4);d=np.load(PRIOR/'dataset.npz');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query','target')};fold=d['fold'];choices=read(ROOT/'reference_choices.json');grid=read(ROOT/'reference_grid.json');o=np.load(ROOT/'oof.npz')['prediction'];ref=np.load(ROOT/'references.npz')['selected'];reproduced=np.empty_like(o);allpred=[]
    with torch.no_grad():
        q,x=v['dq'].double(),v['dx'].double()
        for r in grid:
            w=r['forgetting']**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);g=q.transpose(1,2)@(q*w[None,:,None])+r['ridge']*torch.eye(7,device='cuda',dtype=torch.float64)[None];c=q.transpose(1,2)@(x*w[None,:,None]);allpred.append((v['query'].double()@torch.linalg.solve(g,c)).cpu().numpy())
        pp=np.stack(allpred)
        for f in range(4):
            tr,va=fold!=f,fold==f;ids=np.flatnonzero(va);n=np.load(ROOT/'models'/f'fold_{f}'/'normalization.npz');np.testing.assert_array_equal(n['training'],np.flatnonzero(tr));assert not set(d['family'][tr])&set(d['family'][va]);loss=np.mean(((pp[:,tr]-d['target'][None,tr])/n['scale'])**2,axis=(1,2,3));assert int(np.argmin(loss))==choices[f]['index'];np.testing.assert_allclose(pp[choices[f]['index'],va],ref[va],rtol=1e-12,atol=1e-12)
            init=AnchoredWeights(choices[f]['forgetting'],choices[f]['ridge']).cuda().eval();ii=ids[:32];j,lw=init(v['dq'][ii],v['dx'][ii],v['mask'][ii]);np.testing.assert_allclose((v['query'][ii].double()@j).cpu().numpy(),ref[ii],rtol=1e-10,atol=1e-12)
            for si,seed in enumerate(SEEDS):
                m=AnchoredWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{f}'/f'{seed}.pt',weights_only=True));assert float(m.forgetting)==choices[f]['forgetting'] and float(m.ridge)==choices[f]['ridge']
                for start in range(0,len(ids),256):
                    ii=ids[start:start+256];j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii]);p=(v['query'][ii].double()@j).cpu().numpy();reproduced[si,ii]=p;np.testing.assert_allclose(p,o[si,ii],rtol=1e-6,atol=1e-9);base=m.forgetting**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);mass=(lw.exp()*base).sum(1);torch.testing.assert_close(mass,base.sum().expand_as(mass),atol=1e-10,rtol=1e-10)
    eo=np.load(ENSEMBLE/'oof.npz');np.testing.assert_allclose(reproduced.mean(0),eo['all_three'],rtol=1e-6,atol=1e-9)
    for i in range(3):np.testing.assert_allclose(np.delete(reproduced,i,axis=0).mean(0),eo[f'without_{i}'],rtol=1e-6,atol=1e-9)
    for root in (ROOT,ENSEMBLE):
        for p,h in {**read(root/'protocol.json')['inputs'],**read(root/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(reference_grid_choices_reproduced=True,folds_disjoint=True,initial_models_equal_selected_kernel=True,kernel_mass_preserved=True,neural_predictions_reproduced=True,max_prediction_error=float(np.max(abs(reproduced-o))),inputs_unchanged=True));save(ENSEMBLE/'verification.json',dict(all_seed_average_reproduced=True,all_leave_one_out_pairs_reproduced=True,no_best_seed_selection=True,inputs_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('Verified E142 kernels, mass normalization, all models and E143 ensembles',flush=True)

if __name__=='__main__':run()

"""Independently verify E146 measured priors, splits, initialization and predictions."""
import numpy as np
import torch
from .calibration_prior_learning import ROOT,SOURCE,PriorWeights,features,SEEDS,CONTROL
from .abstaining_task_value_gate import read,save,fingerprint,family

def run():
    torch.set_num_threads(4);d=np.load(ROOT/'dataset.npz');tasks=read(ROOT/'cases.json');recorded=np.load(ROOT/'predictions.npz');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','prior','query','target')};n=len(d['case']);assert n==6706
    for case in np.unique(d['case']):
        assert not tasks[case]['validation'];t=read(CONTROL/'trials'/f'{case}_constant_frozen_bias.json');expected=np.zeros((7,6));expected[:len(tasks[case]['home'])]=np.linalg.solve(t['initial_estimator']['gram'],t['initial_estimator']['cross']);np.testing.assert_array_equal(d['prior'][d['case']==case],np.tile(expected[None],(sum(d['case']==case),1,1)))
    independent=[]
    for i in range(n):
        q,x=d['dq'][i].astype(float),d['dx'][i].astype(float);w=.98**np.arange(31,-1,-1);j=np.linalg.solve(q.T@(q*w[:,None])+1e-6*np.eye(7),q.T@(x*w[:,None])+1e-6*d['prior'][i]);independent.append(d['query'][i]@j)
    np.testing.assert_allclose(independent,recorded['reference'],rtol=1e-7,atol=1e-10)
    with torch.no_grad():
        f=features(v['dq'],v['dx'],v['mask'],v['prior']);initial=PriorWeights().cuda().eval();j,lw=initial(v['dq'][:128],v['dx'][:128],v['mask'][:128],v['prior'][:128],f[:128]);p=(v['query'][:128].double()[:,None]@j).squeeze(1).cpu().numpy();np.testing.assert_allclose(p,recorded['reference'][:128],atol=1e-12,rtol=1e-10);assert torch.max(torch.abs(lw))<1e-12
        for split in range(4):
            norm=np.load(ROOT/'models'/f'fold_{split}'/'normalization.npz');ids,va=norm['train'],norm['validation'];np.testing.assert_array_equal(ids,np.flatnonzero(d['fold']!=split));np.testing.assert_array_equal(va,np.flatnonzero(d['fold']==split));assert not {family(tasks,int(i)) for i in d['case'][ids]}&{family(tasks,int(i)) for i in d['case'][va]}
            cases,counts=np.unique(d['case'][ids],return_counts=True);freq=dict(zip(cases,counts));prob=np.array([1/freq[c] for c in d['case'][ids]]);prob/=prob.sum();np.testing.assert_array_equal(prob,norm['probability']);squared=np.sum(d['target'][ids]**2*prob[:,None],axis=0);scale=np.r_[np.repeat(np.sqrt(squared[:3].mean()),3),np.repeat(np.sqrt(squared[3:].mean()),3)].clip(1e-5);np.testing.assert_array_equal(scale,norm['scale'])
            for si,seed in enumerate(SEEDS):
                m=PriorWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{split}'/f'{seed}.pt',weights_only=True));pp=[]
                for start in range(0,len(va),256):
                    ii=va[start:start+256];j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],v['prior'][ii],f[ii]);pp.append((v['query'][ii].double()[:,None]@j).squeeze(1).cpu().numpy());w=.98**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);torch.testing.assert_close((lw.exp()*w).sum(1),w.sum().expand(len(ii)),rtol=1e-12,atol=1e-12)
                np.testing.assert_array_equal(np.concatenate(pp),recorded['learned'][si,va])
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['inputs']}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(actual_calibration_priors=True,independent_numpy_reference=True,initial_model_matches_reference=True,all_predictions_exact=True,family_exclusion=True,training_only_normalization=True,balanced_placement_sampling=True,weight_mass_preserved=True,hashes_unchanged=True,transitions=n,placements=len(np.unique(d['case']))));print('E146 verified',flush=True)

if __name__=='__main__':run()

"""Verify final fitting-only refit, then audit all prospective evaluation trajectories."""
import shutil
from pathlib import Path
import numpy as np
import torch
from .fresh_fixed_weights import ROOT,SOURCE,FRESH,SEEDS
from .fresh_fixed_weights_control import METHODS,verify_frozen
from .anchored_history_weights import AnchoredWeights
from .learned_history_weights import token_features
from .abstaining_task_value_gate import read,save,fingerprint

def model():
    torch.set_num_threads(4);assert not (ROOT/'cases.json').exists();d=dict(np.load(SOURCE/'dataset.npz'));tasks=read(SOURCE/'cases.json');assert all(not tasks[int(c)]['validation'] for c in d['case']);assert len(set(d['case']))==96;norm=dict(np.load(ROOT/'normalization.npz'));cases,counts=np.unique(d['case'],return_counts=True);freq=dict(zip(cases,counts));prob=np.array([1/freq[c] for c in d['case']]);prob/=prob.sum();np.testing.assert_array_equal(prob,norm['probability']);squared=np.sum(d['target']**2*prob[:,None],axis=0);scale=np.r_[np.repeat(np.sqrt(squared[:3].mean()),3),np.repeat(np.sqrt(squared[3:].mean()),3)].clip(1e-5);np.testing.assert_array_equal(scale,norm['scale']);assert all(abs(prob[d['case']==c].sum()-1/96)<1e-12 for c in cases)
    v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query')};recorded=np.load(ROOT/'fitting_predictions.npz')['prediction'];reference=np.load(SOURCE/'predictions.npz')['reference']
    with torch.no_grad():
        f=token_features(v['dq'],v['dx'],v['mask']);initial=AnchoredWeights(.98,1e-6).cuda().eval();j,_=initial(v['dq'][:128],v['dx'][:128],v['mask'][:128],f[:128]);np.testing.assert_allclose((v['query'][:128].double()[:,None]@j).squeeze(1).cpu().numpy(),reference[:128],atol=1e-10,rtol=1e-7)
        for si,seed in enumerate(SEEDS):
            m=AnchoredWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'{seed}.pt',weights_only=True));assert float(m.forgetting)==.98 and float(m.ridge)==1e-6;pred=[]
            for start in range(0,len(prob),256):
                ii=np.arange(start,min(start+256,len(prob)));j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);pred.append((v['query'][ii].double()[:,None]@j).squeeze(1).cpu().numpy());w=.98**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);torch.testing.assert_close((lw.exp()*w).sum(1),w.sum().expand(len(ii)),atol=1e-12,rtol=1e-12)
            np.testing.assert_array_equal(np.concatenate(pred),recorded[si])
    for p,h in {**read(ROOT/'model_freeze.json'),**read(ROOT/'protocol.json')['inputs']}.items():assert fingerprint(p)==h
    assert fingerprint(FRESH)==read(ROOT/'protocol.json')['fresh_geometry_sha256'];save(ROOT/'model_verification.json',dict(fitting_only_96_placements=True,training_only_normalization=True,placement_balanced_sampling=True,all_seed_predictions_exact=True,initial_model_uniform=True,mass_preserved=True,fixed_kernel=True,fresh_not_executed=True,checkpoint_source_hashes_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('E159 final models verified before fresh execution',flush=True)

def outcomes():
    verify_frozen();rows=read(ROOT/'records.json');assert len(rows)==192
    for index in range(48):
        case=10000+index;reference=read(ROOT/'trials'/f'{case}_uniform32_frozen_bias.json')
        for method in METHODS:
            stem=f'{case}_{method}_frozen_bias';t=read(ROOT/'trials'/f'{stem}.json');assert t['calibration']==reference['calibration'] and t['initial_estimator']==reference['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias'];h=dict(np.load(ROOT/'histories'/f'{stem}.npz'));a=read(ROOT/'actions'/f'{stem}.json');states=np.load(ROOT/'states'/f'{stem}.npz')['states'];assert len(h['query'])==len(a)==len(states)==len(t['trace']);np.testing.assert_array_equal(h['dq_history'][1:,-1],h['query'][:-1]);np.testing.assert_array_equal(h['dx_history'][1:,-1],h['target'][:-1]);np.testing.assert_allclose(np.diff(np.array([v['q_before'] for v in a]),axis=0),h['query'][:-1,:t['dof']],atol=1e-8,rtol=1e-5)
    save(ROOT/'verification.json',dict(prospective_episodes=192,all48_placements_included=True,identical_calibration_across_methods=True,initial_statistics_identical=True,frozen_measured_bias=True,actual_joint_queries_and_history_order=True,all_state_trace_lengths=True,weights_and_sources_unchanged=True,prior_retention_gate_was_not_passed=True,amendment_registered_before_outcomes=True,no_fresh_training_or_retuning=True));print('E159 all prospective trajectories verified',flush=True)

if __name__=='__main__':
    import sys
    {'model':model,'outcomes':outcomes}[sys.argv[1]]()

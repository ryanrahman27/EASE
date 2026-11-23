"""E159: explicitly amended prospective net-improvement validation, E120 immutable."""
import shutil
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch import nn
from .fixed_uniform_learning import ROOT as SOURCE
from .anchored_history_weights import AnchoredWeights
from .learned_history_weights import SEEDS,token_features
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_fresh_fixed_weights_v1')
FRESH=Path('outputs/ease_family_consensus_gate_v1/fresh_cases.json')

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','actions','histories'):(ROOT/folder).mkdir()
    assert fingerprint(FRESH)==fingerprint('outputs/ease_expanded_structured_pairs_v1/fresh_cases.json')
    inputs=[SOURCE/'dataset.npz',SOURCE/'summary.json',SOURCE/'verification.json',Path('outputs/ease_fixed_uniform_control_v1/summary.json'),Path('outputs/ease_family_consensus_gate_v1/protocol.json')]
    save(ROOT/'protocol.json',dict(experiment='E159',registered_utc=datetime.now(timezone.utc).isoformat(),
        amendment='Explicit amendment before any reserved fresh outcome. E154 standard yields77/96 vs uniform74 and E120 ensemble72, with better both-DOF counts and fewer collisions, but fails the legacy individual-seed case13 retention rule and has other regressions. E137/E154 no-regression advancement requirements were NOT passed and are not rewritten. After failed targeted repairs, freeze the simplest aggregate-improving candidate and ask a different prospective question: does net improvement generalize? This is research validation, not E120 replacement or a claim of loss-free improvement. All prior failed gates/results remain saved.',
        selection='Choose original standard controller with E153 fixed-uniform learned weighting, not posture variants, calibration priors or conservative temperatures. Selection uses fitting/development results only. Refit the identical architecture/training recipe once on all fitting histories, all three fixed seeds retained and averaged. No fresh outcome or geometry enters fitting, normalization, checkpoint or hyperparameter choice.',
        training='Exactly6706 fitting transitions,96 known-body fitting placements. Fixed.98forgetting, zero-centered1e-6ridge, normalized relative history weights,8-feature32-wide2-layer Transformer. Three seeds141100..141102,1500 AdamW3e-4 updates,wd1e-4,batch128,clip1, normalized next-motionMSE+.001 logweight-square. Sample placements uniformly, then histories within placement. Position/orientation RMS normalization computed from all fitting rows only. Save final checkpoints; no checkpoint selection.',
        fresh='The48 E137 placements were generated before E137 outcomes, seed137000, six per eight fitting ancestral homes, same four known bodies. Exact original geometry file remains unchanged. This is new-placement validation, NOT unseen-body transfer. No reachability or success filtering. All48 cases and outcomes included. Evaluation case10000+i gives calibration seed48000+i, identical across methods.',
        controllers='Four fixed methods: learned all-three final Jacobian ensemble; matched uniform32(.98,1e-6); original full-stat analytical until close then local32; E120 all-three reference/correction ensemble with original bounded residual. Same32 initial calibration,32close probe, frozen measured bias, DLS/task waypoints/budgets/action bounds and success/contact criteria. No posture correction, physics selector, exact Jacobian, object/contact feature, body ID or online gradient update.',
        analysis='Primary: paired learned-minus-uniform task success across48 placements. Secondary: E120 ensemble and full-stat analytical, joint counts, all four bodies, collisions, paired rescues/losses. Positive net pattern requires higher overall success than both uniform/E120, no lower per-DOF count, and no higher collision count. Report regressions regardless; no-loss status is a separate field. Four known bodies limit independence; descriptive placement counts and body-cluster uncertainty do not establish unseen-embodiment or conference-ready claims.',
        stopping='Run this fixed prospective comparison once. No retuning, new checkpoint selection, filtering or repeated attempts using fresh outcomes. Keep the reserved set held out from future fitting. A negative result remains negative. E120 is immutable and not automatically promoted/replaced.',
        fresh_geometry_sha256=fingerprint(FRESH),inputs={str(p):fingerprint(p) for p in inputs}))
    for name in ('fresh_fixed_weights.py','fresh_fixed_weights_control.py','anchored_history_weights.py','learned_history_weights.py','training_estimator_core.py','bounded_history_correction.py','local_history.py','pick_place_control.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print('E159 amendment registered before fresh outcomes; original gates remain failed',flush=True)

def train():
    torch.set_num_threads(4);d=dict(np.load(SOURCE/'dataset.npz'));v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','query','target')};cases,counts=np.unique(d['case'],return_counts=True);freq=dict(zip(cases,counts));prob=np.array([1/freq[c] for c in d['case']]);prob/=prob.sum();squared=np.sum(d['target']**2*prob[:,None],axis=0);scale=np.r_[np.repeat(np.sqrt(squared[:3].mean()),3),np.repeat(np.sqrt(squared[3:].mean()),3)].clip(1e-5);np.savez(ROOT/'normalization.npz',scale=scale,probability=prob,cases=d['case']);scale_t=torch.tensor(scale,device='cuda');logs=[];out=[]
    with torch.no_grad():f=token_features(v['dq'],v['dx'],v['mask'])
    for seed in SEEDS:
        torch.manual_seed(seed);rng=np.random.default_rng(seed);m=AnchoredWeights(.98,1e-6).cuda();opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-4);history=[]
        for step in range(1500):
            ii=rng.choice(len(prob),128,replace=True,p=prob);j,lw=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);p=(v['query'][ii].double()[:,None]@j).squeeze(1);mse=torch.mean(((p-v['target'][ii])/scale_t)**2);loss=mse+.001*torch.mean(lw**2);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
            if step%100==0:history.append(dict(step=step,loss=float(loss.detach()),mse=float(mse.detach())))
        m.eval();torch.save(m.state_dict(),ROOT/'models'/f'{seed}.pt');pred=[]
        with torch.no_grad():
            for start in range(0,len(prob),256):
                ii=np.arange(start,min(start+256,len(prob)));j,_=m(v['dq'][ii],v['dx'][ii],v['mask'][ii],f[ii]);pred.append((v['query'][ii].double()[:,None]@j).squeeze(1).cpu().numpy())
        out.append(np.concatenate(pred));logs.append(dict(seed=seed,history=history));print('E159 final fitting seed',seed,'complete',flush=True)
    out=np.stack(out);np.savez_compressed(ROOT/'fitting_predictions.npz',prediction=out);save(ROOT/'training.json',logs);reference=np.load(SOURCE/'predictions.npz')['reference'];metrics={}
    for name,p in [('uniform',reference),('final_ensemble',out.mean(0))]:
        metrics[name]={}
        for n in ('all',6,7):
            keep=d['mask'].sum(1)==n if n!='all' else np.ones(len(prob),bool);e=p[keep]-d['target'][keep];metrics[name][str(n)]=dict(position_rmse_mm=float(np.sqrt(np.mean(e[:,:3]**2))*1000),rotation_rmse_mm=float(np.sqrt(np.mean(e[:,3:]**2))*1000))
    assert all(metrics['final_ensemble'][str(n)]['position_rmse_mm']<metrics['uniform'][str(n)]['position_rmse_mm'] for n in ('all',6,7));save(ROOT/'fitting_check.json',dict(metrics=metrics,training_data_only=True,all_seeds_retained=True));save(ROOT/'model_freeze.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'normalization.npz',ROOT/'training.json',ROOT/'fitting_predictions.npz',*(ROOT/'models').glob('*'),*(ROOT/'source').glob('*')]});print(metrics,flush=True)

def evaluation_prepare():
    assert (ROOT/'model_verification.json').exists()
    for p,h in read(ROOT/'model_freeze.json').items():assert fingerprint(p)==h
    assert fingerprint(FRESH)==read(ROOT/'protocol.json')['fresh_geometry_sha256'];tasks=read(FRESH);assert len(tasks)==48 and all(t['validation'] for t in tasks);shutil.copy2(FRESH,ROOT/'cases.json');assert len(set(t['arm']['seed'] for t in tasks))==4
    from . import broad_task_value_gate as broad
    _,_,hashes=broad.setup();save(ROOT/'reference_checkpoints.json',hashes);save(ROOT/'evaluation_freeze.json',{str(ROOT/p):fingerprint(ROOT/p) for p in ('cases.json','reference_checkpoints.json','model_freeze.json','protocol.json')});save(ROOT/'status.json',dict(status='frozen_before_fresh_execution',fresh_episodes=0,prior_retention_gate_passed=False,amended_net_improvement_question=True));print('Frozen192 prospective episodes, all48 placements and four controllers',flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'evaluation_prepare':evaluation_prepare}[sys.argv[1]]()

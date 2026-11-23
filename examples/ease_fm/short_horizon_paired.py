"""E132: dense short-horizon paired supervision, explicitly privileged inputs."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .abstaining_task_value_gate import DATA,COLLECT,OLD,SEEDS,PHASES,Gate,read,save,family,fingerprint

ROOT=Path('outputs/ease_short_horizon_paired_v1')
PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')
LONG=Path('outputs/ease_privileged_task_state_v1')
HORIZONS=(1,2,4,8)


class ShortModel(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(182,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,12))
    def forward(self,x):return self.net(x)


def make_targets(meta):
    goals={};on_cache={};rows=[];progress=[];collision=[];hashes={}
    for r in meta:
        case,seed,phase=r['case'],r['seed'],r['phase'];key=(case,seed)
        if key not in on_cache:
            bp=COLLECT/'branches'/f'{case}_on_{seed}.json';tp=COLLECT/'trials'/f'{case}_on_{seed}_frozen_bias.json'
            on_cache[key]=(read(bp),read(tp));hashes[str(bp)]=fingerprint(bp);hashes[str(tp)]=fingerprint(tp)
        onb,ont=on_cache[key];op=COLLECT/'branches'/f'{case}_off_{phase}_{seed}.json';tp=COLLECT/'trials'/f'{case}_off_{phase}_{seed}_frozen_bias.json'
        offb,offt=read(op),read(tp);hashes[str(op)]=fingerprint(op);hashes[str(tp)]=fingerprint(tp)
        step=offb['intervention'];assert step==onb['entries'][phase]['step'];assert onb['signatures'][:step+1]==offb['signatures'][:step+1]
        assert onb['metrics'][step]['clean_before'] and offb['metrics'][step]['clean_before']
        sp=COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz';sq=COLLECT/'snapshots'/f'{case}_off_{phase}_{seed}_{phase}.npz'
        a,b=np.load(sp),np.load(sq)
        for name in a.files:np.testing.assert_array_equal(a[name],b[name])
        hashes[str(sp)]=fingerprint(sp);hashes[str(sq)]=fingerprint(sq);goal=a['goal']
        assert step+8<=min(len(ont['trace']),len(offt['trace']))
        diff=[];bad=[];crossed=[]
        for h in HORIZONS:
            j=step+h-1
            diff.append(float(np.linalg.norm(goal-ont['trace'][j]['position'])-np.linalg.norm(goal-offt['trace'][j]['position'])))
            bad.append([float(any(t['bad'] for t in trial['trace'][step:j+1])) for trial in (ont,offt)])
            crossed.append(bool(any(t['phase']!=phase for trial in (ont,offt) for t in trial['trace'][step:j+1])))
        progress.append(diff);collision.append(np.array(bad).T.reshape(-1));rows.append(dict(case=case,seed=seed,phase=phase,entry_step=step,phase_changed_within_horizon=crossed,on_success=ont['success'],off_success=offt['success']))
    return np.array(progress,np.float32),np.array(collision,np.float32),rows,hashes


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json'):shutil.copy2(LONG/name,ROOT/name)
    inputs=[LONG/n for n in ('features.npz','primary_outcomes.json','selected_rules.json','verification.json','protocol.json')]
    inputs += [DATA/'gate_rows.json',COLLECT/'records.json',PRIOR/'oof_predictions.npz',OLD/'records.json',OLD/'summary.json']
    inputs += [p for p in (LONG/'models').rglob('*') if p.is_file()]
    grid=[dict(progress_margin=m,risk_cap=r) for m in (0.,.0001,.00025,.0005,.001) for r in (.05,.1,.25)]
    grid.append(dict(progress_margin=1e9,risk_cap=0.))
    save(ROOT/'protocol.json',dict(experiment='E132',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},horizons=list(HORIZONS),grid=grid,
        inputs_label='Same182 privileged E131 inputs, explicitly simulator-only upper-information reference; not measured/deployable sensing.',
        target='At1,2,4,8 steps after identical phase-entry states: (entry-goal distance on minus off), meters, and any forbidden collision during that prefix separately for on/off. Entry goal frozen for all horizons even if the scripted phase advances. At most8-step off pulse then E120 resumes. No final success labels in training.',
        dense='Four continuous paired progress labels plus eight prefix-collision labels per original phase entry. Same1108 fitting rows; no new independent starts or simulator rollouts. Phase transitions within the window are recorded rather than filtered using future information.',
        training='Same four ancestral-family splits,3 seeds126100..126102,1500 full-batch AdamW steps, lr.001 weight_decay.001.182-64-32-12 tanh-hidden network:4 linear standardized progress outputs,8 collision logits. Sum of mean normalized-progress MSE and mean collision BCE, equal task weights. Input normalization fixed to E131 fold; progress mean/std training-fold only, floor1e-6m. No resampling, early stopping or checkpoint selection.',
        initialization='Original139-column and hidden32-layer weights/biases match same-seed baseline; extra43 input columns zero. Output head changes to12 targets; objective and output parameterization change together.',
        primary='Off selected at first clean phase iff all3 predictions show8-step progress benefit >0.00025m and every predicted off collision probability at1/2/4/8 steps is <0.1. At most one pulse. Oracle diagnostic uses actual short progress/collision under the same thresholds; it uses future data and is NOT an inference policy.',
        secondary='Same fixed progress-margin/risk-cap grid for learned short targets. Choose zero E120 final-success losses, most gains, fewer switches, higher progress margin, lower risk cap. Final outcomes used only for tuning-validation rule selection, never model training. Full abstention included.',
        evaluation='144 fitting placement/estimator full-task successes reconstructed from matched continuations. Compare E120, E131 long-target primary, learned short primary and true-short-target diagnostic. Original held-placement benchmark untouched. No fresh online-policy rollout or post-collision trajectory equivalence claim.',
        baseline='E120 immutable. Short-horizon prediction improvement alone cannot justify promotion.'))
    meta=read(DATA/'gate_rows.json');tasks=read(ROOT/'cases.json');assert all(not tasks[r['case']]['validation'] for r in meta)
    progress,collision,rows,hashes=make_targets(meta)
    np.savez_compressed(ROOT/'targets.npz',progress=progress,collision=collision)
    save(ROOT/'target_rows.json',rows);save(ROOT/'target_input_hashes.json',hashes)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print('Prepared',progress.size,'continuous and',collision.size,'collision targets',flush=True)


def data():
    x=np.load(LONG/'features.npz')['x'];t=np.load(ROOT/'targets.npz');folds=np.load(PRIOR/'oof_predictions.npz')['row_fold']
    return x,t['progress'],t['collision'],folds,read(DATA/'gate_rows.json'),read(ROOT/'cases.json')


def predict(models,n,x):
    with torch.no_grad():
        xx=torch.tensor(np.clip((x-n['mean'])/n['scale'],-10,10));raw=np.stack([m(xx).numpy() for m in models])
    progress=raw[:,:,:4]*n['target_scale']+n['target_mean'];collision=1/(1+np.exp(-np.clip(raw[:,:,4:],-80,80)))
    return progress,collision


def train():
    torch.set_num_threads(4);x,y,c,folds,meta,tasks=data()
    pp=np.empty((3,len(x),4),np.float32);pc=np.empty((3,len(x),8),np.float32);audit=[]
    for fold in range(4):
        ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not {family(tasks,meta[i]['case']) for i in ids}&{family(tasks,meta[i]['case']) for i in val}
        old=np.load(LONG/'models'/f'fold_{fold}'/'normalization.npz');np.testing.assert_array_equal(ids,old['training_indices'])
        n=dict(mean=old['mean'],scale=old['scale'],target_mean=y[ids].mean(0),target_scale=np.maximum(y[ids].std(0),1e-6),training_indices=ids)
        directory=ROOT/'models'/f'fold_{fold}';directory.mkdir();np.savez_compressed(directory/'normalization.npz',**n)
        xx=torch.tensor(np.clip((x[ids]-n['mean'])/n['scale'],-10,10));yy=torch.tensor((y[ids]-n['target_mean'])/n['target_scale']);cc=torch.tensor(c[ids]);models=[];losses=[]
        for seed in SEEDS:
            torch.manual_seed(seed);base=Gate();torch.manual_seed(seed);m=ShortModel()
            with torch.no_grad():
                m.net[0].weight[:,:139].copy_(base.net[0].weight);m.net[0].weight[:,139:].zero_();m.net[0].bias.copy_(base.net[0].bias);m.net[2].load_state_dict(base.net[2].state_dict())
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
            for _ in range(1500):
                out=m(xx);mse=nn.functional.mse_loss(out[:,:4],yy);bce=nn.functional.binary_cross_entropy_with_logits(out[:,4:],cc);loss=mse+bce;opt.zero_grad();loss.backward();opt.step()
            m.eval();torch.save(m.state_dict(),directory/f'{seed}.pt');models.append(m);losses.append(dict(progress_mse=float(mse.detach()),collision_bce=float(bce.detach())))
        pp[:,val],pc[:,val]=predict(models,n,x[val]);audit.append(dict(fold=fold,training_rows=len(ids),validation_rows=len(val),losses=losses));print('trained short fold',fold,flush=True)
    np.savez_compressed(ROOT/'oof.npz',progress=pp,collision=pc,row_fold=folds);save(ROOT/'training.json',audit)
    frozen={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    for name in ('protocol.json','targets.npz','target_rows.json','target_input_hashes.json','oof.npz','training.json'):frozen[name]=fingerprint(ROOT/name)
    save(ROOT/'frozen_training.json',frozen)


def score(accepted,rule,meta,tasks):
    lookup={(r['case'],r['seed'],r['off_phase']):r for r in read(COLLECT/'records.json')};index={(r['case'],r['seed'],r['phase']):i for i,r in enumerate(meta)};episodes=[]
    for case,t in enumerate(tasks):
        if t['validation']:continue
        for seed in range(3):
            phase=next((p for p in PHASES if (case,seed,p) in index and accepted[index[case,seed,p]]),None)
            on,off=lookup[case,seed,None],lookup[case,seed,phase];episodes.append(dict(case=case,seed=seed,phase=phase,baseline_success=on['success'],success=off['success']))
    return dict(rule=rule,n=len(episodes),baseline_success=sum(r['baseline_success'] for r in episodes),success=sum(r['success'] for r in episodes),gains=sum(r['success'] and not r['baseline_success'] for r in episodes),losses=sum(r['baseline_success'] and not r['success'] for r in episodes),switches=sum(r['phase'] is not None for r in episodes),episodes=episodes)


def accepted(progress,collision,rule):
    return (progress[:,:,3].min(0)>rule['progress_margin'])&(collision[:,:,4:].max(axis=(0,2))<rule['risk_cap'])


def select(grid):
    return max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],r['rule']['progress_margin'],-r['rule']['risk_cap']))


def compare():
    x,y,c,folds,meta,tasks=data();o=np.load(ROOT/'oof.npz');pp,pc=o['progress'],o['collision'];mean=pp.mean(0);risk=pc.mean(0)
    prior=np.zeros_like(c);meanref=np.zeros_like(y)
    for fold in range(4):prior[folds==fold]=c[folds!=fold].mean(0);meanref[folds==fold]=y[folds!=fold].mean(0)
    metric=[]
    for j,h in enumerate(HORIZONS):
        nz=np.abs(y[:,j])>1e-6
        metric.append(dict(horizon=h,rows=len(y),progress_nonzero_above_1um=int(nz.sum()),progress_rmse_mm=float(np.sqrt(np.mean((mean[:,j]-y[:,j])**2))*1000),zero_rmse_mm=float(np.sqrt(np.mean(y[:,j]**2))*1000),training_mean_rmse_mm=float(np.sqrt(np.mean((meanref[:,j]-y[:,j])**2))*1000),sign_accuracy_nonzero=float(np.mean(np.sign(mean[nz,j])==np.sign(y[nz,j]))),on_collision_count=int(c[:,j].sum()),off_collision_count=int(c[:,4+j].sum()),on_brier=float(np.mean((risk[:,j]-c[:,j])**2)),off_brier=float(np.mean((risk[:,4+j]-c[:,4+j])**2)),on_frequency_brier=float(np.mean((prior[:,j]-c[:,j])**2)),off_frequency_brier=float(np.mean((prior[:,4+j]-c[:,4+j])**2))))
    rule=dict(progress_margin=.00025,risk_cap=.1)
    primary=dict(long_target=read(LONG/'primary_outcomes.json')['privileged'],learned_short=score(accepted(pp,pc,rule),rule,meta,tasks),true_short_diagnostic=score(accepted(y[None],c[None],rule),rule,meta,tasks))
    grid=[score(accepted(pp,pc,r),r,meta,tasks) for r in read(ROOT/'protocol.json')['grid']];selected=select(grid)
    rows=read(ROOT/'target_rows.json');transitions=np.array([r['phase_changed_within_horizon'] for r in rows]).sum(0).tolist()
    summary=dict(horizon_metrics=metric,phase_transition_rows=transitions,primary={k:{a:b for a,b in v.items() if a!='episodes'} for k,v in primary.items()},selected={k:v for k,v in selected.items() if k!='episodes'},held_evaluated=False,new_rollouts=0)
    save(ROOT/'summary.json',summary);save(ROOT/'primary_outcomes.json',primary);save(ROOT/'validation_grid.json',grid);save(ROOT/'selected_rule.json',selected);print(json.dumps(summary,indent=2),flush=True)


def media():
    _,_,_,_,meta,tasks=data();primary=read(ROOT/'primary_outcomes.json');selected=read(ROOT/'selected_rule.json');wanted={}
    def add(r,why):
        assert not tasks[r['case']]['validation'];wanted.setdefault((r['case'],r['seed'],r['phase']),[]).append(why);wanted.setdefault((r['case'],r['seed'],None),[]).append('E120 reference')
    for name in ('long_target','learned_short'):
        for r in primary[name]['episodes']:
            if r['success']!=r['baseline_success']:add(r,name)
    # Representative true-short-target gains/losses per joint count, plus all learned changes.
    for dof in (6,7):
        for gain in (True,False):
            candidates=[r for r in primary['true_short_diagnostic']['episodes'] if r['success']==gain and r['success']!=r['baseline_success'] and len(tasks[r['case']]['home'])==dof]
            if candidates:add(candidates[0],'true-short diagnostic example')
    for r in selected['episodes']:
        if r['success']!=r['baseline_success']:add(r,'selected-rule change')
    for d in read(Path('outputs/ease_task_value_ranking_diagnosis_v1/diagnostics.json')):add(dict(case=d['case'],seed=d['seed'],phase=d['phase']),'E128 diagnostic pair')
    records=[];hashes={};pairs=[]
    for (case,seed,phase),reasons in wanted.items():
        method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias';trial=read(COLLECT/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:
                p=COLLECT/folder/f'{stem}.{ext}';shutil.copy2(p,ROOT/folder/p.name);hashes[str(p)]=fingerprint(p)
        records.append(dict(case=case,method=method,mode='frozen_bias',success=trial['success'],collision=trial['collision'],selection_reasons=reasons))
    save(ROOT/'records.json',records);save(ROOT/'media_input_hashes.json',hashes)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','short_horizon_paired import ROOT').replace('E128','E132').replace('Task-value ranking diagnosis','PRIVILEGED short-horizon target study').replace('Diagnosis only; no new policy.','Outcome-selected fitting-family counterfactuals; short-target training uses PRIVILEGED inputs.').replace('Measured-history control | No physics selector or geometry filter','PRIVILEGED input study | Executed counterfactual replay')
    Path(__file__).with_name('short_horizon_videos.py').write_text(s);print('Prepared',len(records),'videos',flush=True)


def verify():
    torch.set_num_threads(4);x,y,c,folds,meta,tasks=data();yy,cc,rows,hashes=make_targets(meta)
    np.testing.assert_array_equal(y,yy);np.testing.assert_array_equal(c,cc);assert rows==read(ROOT/'target_rows.json') and hashes==read(ROOT/'target_input_hashes.json')
    o=np.load(ROOT/'oof.npz')
    for fold in range(4):
        ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold);n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz');old=np.load(LONG/'models'/f'fold_{fold}'/'normalization.npz')
        assert not {family(tasks,meta[i]['case']) for i in ids}&{family(tasks,meta[i]['case']) for i in val}
        for key in ('mean','scale','training_indices'):np.testing.assert_array_equal(n[key],old[key])
        np.testing.assert_array_equal(n['target_mean'],y[ids].mean(0));np.testing.assert_array_equal(n['target_scale'],np.maximum(y[ids].std(0),1e-6))
        models=[]
        for seed in SEEDS:
            m=ShortModel();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));m.eval();models.append(m)
        pp,pc=predict(models,n,x[val]);np.testing.assert_array_equal(pp,o['progress'][:,val]);np.testing.assert_array_equal(pc,o['collision'][:,val])
    grid=[score(accepted(o['progress'],o['collision'],r),r,meta,tasks) for r in read(ROOT/'protocol.json')['grid']]
    assert grid==read(ROOT/'validation_grid.json') and select(grid)==read(ROOT/'selected_rule.json')
    for p,h in read(ROOT/'protocol.json')['inputs'].items():assert fingerprint(p)==h
    for p,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/p)==h
    save(ROOT/'verification.json',dict(all_dense_targets_reconstructed=True,prefix_snapshots_exact=len(meta),folds_disjoint=True,input_and_target_normalization_training_only=True,oof_predictions_exact=True,outcome_reconstruction_exact=True,held_rows=0,E120_unchanged=True))
    print('Verified dense labels, all folds and outcome reconstructions',flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'compare':compare,'media':media,'verify':verify}[sys.argv[1]]()

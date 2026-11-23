"""E129: direct paired advantage versus two success heads, fitting families only."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .abstaining_task_value_gate import DATA, COLLECT, OLD, SEEDS, PHASES, Gate, read, save, family, probabilities, fingerprint

ROOT=Path('outputs/ease_paired_advantage_objective_v1')
PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')


class Advantage(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(139,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,1),nn.Tanh())
    def forward(self,x):return self.net(x).squeeze(-1)


def predict(models,mean,scale,x):
    with torch.no_grad():
        xx=torch.tensor(np.clip((np.asarray(x,np.float32)-mean)/scale,-10,10))
        return np.stack([m(xx).numpy() for m in models])


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','paired'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json','fold_audit.json'):shutil.copy2(PRIOR/name,ROOT/name)
    inputs=[DATA/n for n in ('gate_dataset.npz','gate_rows.json','cases.json','verification.json')]
    inputs += [COLLECT/'records.json',PRIOR/'protocol.json',PRIOR/'oof_predictions.npz',PRIOR/'validation_grid.json',PRIOR/'frozen_selection.json',OLD/'records.json',OLD/'summary.json']
    inputs += [p for p in (PRIOR/'models').rglob('*') if p.is_file()]
    save(ROOT/'protocol.json',dict(experiment='E129',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},
        design='Four identical E126 ancestral-family folds, same139 features and fold-local normalization. Same3 seeds126100..126102,1500 full-batch AdamW updates, lr.001 weight_decay.001. No resampling/class weighting, no early stopping or model selection. Existing two-head checkpoints reused and audited.',
        objectives='Two-head:139-64-32-2 tanh-hidden MLP with paired final-success BCE; derive advantage sigmoid(off)-sigmoid(on). Direct:139-64-32-1 tanh-hidden MLP with final tanh, MSE to y_off-y_on in{-1,0,1}. Hidden-layer initialization matched exactly by seed. Output parameterization differs as required for one scalar; no other representation change.',
        primary='Compare out-of-fold prediction metrics and the same prespecified min-advantage0.1/max-spread2 first-clean-phase intervention rule over144 fitting placement/estimator episodes. Report each fold and seed; zero-advantage reference included. No original held-placement benchmark or unseen-body evaluation.',
        secondary='Apply the same preexisting E126 threshold/spread grid to both objectives. Select zero-E120-loss rules by gains, then fewer switches, higher advantage threshold, smaller spread. These are tuning-validation scores, not independent confirmation. No new grid after results.',
        outcomes='Reconstruct complete-task success using verified matched continuations: first accepted clean phase, up-to8-step off pulse, then E120 resumes. Prior-collision starts omitted because task success is irreversibly false. This does not claim a fresh online-policy rollout or post-collision trajectory equivalence. Videos are corresponding executed counterfactual continuations.',
        media='Save all first-rule outcome changes versus E120 or between objectives, selected-rule changes, and E128 diagnostic on/off pairs. Outcome-selected diagnostic videos. Preserve original full trajectories and snapshot provenance.',
        baseline='E120 unchanged. No deployment or baseline replacement based on this fitting-only objective comparison.',grid=read(PRIOR/'protocol.json')['grid']))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print('Prepared E129 fixed objective comparison',flush=True)


def data():
    f=np.load(DATA/'gate_dataset.npz');o=np.load(PRIOR/'oof_predictions.npz')
    return f['x'],f['y'],o['row_fold'],read(DATA/'gate_rows.json'),read(ROOT/'cases.json')


def train():
    torch.set_num_threads(4)
    x,y,folds,meta,tasks=data();target=y[:,1]-y[:,0]
    assert all(not tasks[r['case']]['validation'] for r in meta)
    oof=np.empty((3,len(x)),np.float32);audit=[]
    for fold in range(4):
        train_ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not {family(tasks,meta[i]['case']) for i in train_ids}&{family(tasks,meta[i]['case']) for i in val}
        n=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz');mean,scale=n['mean'],n['scale']
        np.testing.assert_array_equal(n['training_indices'],train_ids);np.testing.assert_array_equal(mean,x[train_ids].mean(0));np.testing.assert_array_equal(scale,np.maximum(x[train_ids].std(0),1e-6))
        directory=ROOT/'models'/f'fold_{fold}';directory.mkdir()
        np.savez_compressed(directory/'normalization.npz',mean=mean,scale=scale,training_indices=train_ids)
        xx=torch.tensor(np.clip((x[train_ids]-mean)/scale,-10,10));yy=torch.tensor(target[train_ids]);models=[];losses=[]
        for seed in SEEDS:
            torch.manual_seed(seed);original=Gate()
            torch.manual_seed(seed);m=Advantage()
            for layer in (0,2):
                torch.testing.assert_close(m.net[layer].weight,original.net[layer].weight,rtol=0,atol=0)
                torch.testing.assert_close(m.net[layer].bias,original.net[layer].bias,rtol=0,atol=0)
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
            for _ in range(1500):
                loss=nn.functional.mse_loss(m(xx),yy);opt.zero_grad();loss.backward();opt.step()
            m.eval();torch.save(m.state_dict(),directory/f'{seed}.pt');models.append(m);losses.append(float(loss.detach()))
        oof[:,val]=predict(models,mean,scale,x[val])
        audit.append(dict(fold=fold,train_rows=len(train_ids),validation_rows=len(val),training_mse=losses,hidden_initialization_exact=True))
        save(directory/'training.json',audit[-1]);print('trained direct fold',fold,flush=True)
    np.savez_compressed(ROOT/'direct_oof.npz',advantages=oof,row_fold=folds)
    save(ROOT/'training.json',audit)
    frozen={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    for name in ('protocol.json','direct_oof.npz','training.json','folds.json'):frozen[name]=fingerprint(ROOT/name)
    save(ROOT/'frozen_training.json',frozen)


def score(advantages,meta,tasks,rule):
    accepted=(advantages.min(0)>rule['min_advantage'])&(np.ptp(advantages,axis=0)<=rule['max_spread'])
    lookup={(r['case'],r['seed'],r['off_phase']):r for r in read(COLLECT/'records.json')}
    index={(r['case'],r['seed'],r['phase']):i for i,r in enumerate(meta)};episodes=[]
    for case,t in enumerate(tasks):
        if t['validation']:continue
        for seed in range(3):
            phase=next((p for p in PHASES if (case,seed,p) in index and accepted[index[case,seed,p]]),None)
            base,result=lookup[case,seed,None],lookup[case,seed,phase]
            episodes.append(dict(case=case,seed=seed,phase=phase,baseline_success=base['success'],success=result['success']))
    return dict(rule=rule,n=len(episodes),baseline_success=sum(r['baseline_success'] for r in episodes),success=sum(r['success'] for r in episodes),gains=sum(r['success'] and not r['baseline_success'] for r in episodes),losses=sum(r['baseline_success'] and not r['success'] for r in episodes),switches=sum(r['phase'] is not None for r in episodes),episodes=episodes)


def select(grid):
    return max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],r['rule']['min_advantage'],-r['rule']['max_spread']))


def metrics(adv,target):
    mean=adv.mean(0);decisive=target!=0
    return dict(rows=len(target),mse=float(np.mean((mean-target)**2)),decisive_mse=float(np.mean((mean[decisive]-target[decisive])**2)),
        neutral_mse=float(np.mean(mean[~decisive]**2)),decisive_sign_accuracy=float(np.mean(np.sign(mean[decisive])==target[decisive])),
        beneficial_reversed=int(np.sum((target>0)&(mean<0))),harmful_reversed=int(np.sum((target<0)&(mean>0))),
        consensus_beneficial=int(np.sum((target>0)&(adv.min(0)>.1))),consensus_harmful=int(np.sum((target<0)&(adv.min(0)>.1))))


def compare():
    x,y,folds,meta,tasks=data();target=y[:,1]-y[:,0]
    p=np.load(PRIOR/'oof_predictions.npz')['probabilities'];two=p[:,:,1]-p[:,:,0];direct=np.load(ROOT/'direct_oof.npz')['advantages']
    grid=read(ROOT/'protocol.json')['grid'];primary_rule=dict(min_advantage=.1,max_spread=2.)
    result={};grids={};primary={};selected={};fold_rows=[];seed_rows=[]
    for name,adv in (('two_head',two),('direct',direct)):
        result[name]=metrics(adv,target);primary[name]=score(adv,meta,tasks,primary_rule);grids[name]=[score(adv,meta,tasks,r) for r in grid];selected[name]=select(grids[name])
        for fold in range(4):
            ids=folds==fold;fold_rows.append(dict(method=name,fold=fold,**metrics(adv[:,ids],target[ids])))
        for i,seed in enumerate(SEEDS):seed_rows.append(dict(method=name,gate_seed=seed,**metrics(adv[i:i+1],target)))
    old=next(r for r in read(PRIOR/'validation_grid.json') if r['rule']==primary_rule)
    assert primary['two_head']==old
    summary=dict(metrics=result,zero_advantage_reference=metrics(np.zeros_like(direct),target),fold_metrics=fold_rows,seed_metrics=seed_rows,
        primary={k:{a:b for a,b in v.items() if a!='episodes'} for k,v in primary.items()},selected={k:{a:b for a,b in v.items() if a!='episodes'} for k,v in selected.items()},held_placements_evaluated=False,new_simulation_rollouts=0)
    save(ROOT/'summary.json',summary);save(ROOT/'primary_outcomes.json',primary);save(ROOT/'validation_grids.json',grids);save(ROOT/'selected_rules.json',selected)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('fold_metrics','seed_metrics')},indent=2),flush=True)


def media():
    x,y,folds,meta,tasks=data();primary=read(ROOT/'primary_outcomes.json');selected=read(ROOT/'selected_rules.json')
    wanted={};reasons={}
    def add(case,seed,phase,reason):
        assert not tasks[case]['validation']
        key=(case,seed,phase);wanted[key]=True;reasons.setdefault(str(key),[]).append(reason)
    maps={name:{(r['case'],r['seed']):r for r in score['episodes']} for name,score in primary.items()}
    for key,a in maps['two_head'].items():
        b=maps['direct'][key]
        if a['success']!=b['success'] or a['success']!=a['baseline_success'] or b['success']!=b['baseline_success']:
            for r in (a,b):add(r['case'],r['seed'],r['phase'],'fixed-rule comparison')
            add(*key,None,'E120 reference')
    for name,s in selected.items():
        for r in s['episodes']:
            if r['success']!=r['baseline_success']:
                add(r['case'],r['seed'],r['phase'],f'{name} selected-rule change');add(r['case'],r['seed'],None,'E120 reference')
    for d in read(Path('outputs/ease_task_value_ranking_diagnosis_v1/diagnostics.json')):
        for phase in (None,d['phase']):add(d['case'],d['seed'],phase,'E128 diagnostic pair')
    records=[];hashes={};pairs=[]
    for case,seed,phase in wanted:
        method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias'
        trial=read(COLLECT/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:
                path=COLLECT/folder/f'{stem}.{ext}';shutil.copy2(path,ROOT/folder/path.name);hashes[str(path)]=fingerprint(path)
        records.append(dict(case=case,method=method,mode='frozen_bias',success=trial['success'],collision=trial['collision']))
        if phase is not None:
            on=read(COLLECT/'branches'/f'{case}_on_{seed}.json');off=read(COLLECT/'branches'/f'{case}_{method}.json');step=off['intervention']
            assert on['signatures'][:step+1]==off['signatures'][:step+1]
            snaps=[np.load(COLLECT/'snapshots'/f'{case}_{m}_{phase}.npz') for m in (f'on_{seed}',method)]
            for key in snaps[0].files:np.testing.assert_array_equal(snaps[0][key],snaps[1][key])
            pairs.append(dict(case=case,seed=seed,phase=phase,exact_prefix_and_snapshot=True))
            for path in [COLLECT/'branches'/f'{case}_{m}.json' for m in (f'on_{seed}',method)]+[COLLECT/'snapshots'/f'{case}_{m}_{phase}.npz' for m in (f'on_{seed}',method)]:hashes[str(path)]=fingerprint(path)
    save(ROOT/'records.json',records);save(ROOT/'media_selection.json',dict(reasons=reasons,paired_checks=pairs));save(ROOT/'media_input_hashes.json',hashes)
    source=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','paired_advantage_objective import ROOT').replace('E128','E129').replace('Task-value ranking diagnosis','Paired-advantage objective comparison').replace('Diagnosis only; no new policy.','Counterfactual continuations selected by fitting-family objective comparison; not fresh policy rollouts.')
    Path(__file__).with_name('paired_advantage_videos.py').write_text(source)
    print('Prepared',len(records),'diagnostic videos',flush=True)


def verify():
    torch.set_num_threads(4);x,y,folds,meta,tasks=data();direct=np.load(ROOT/'direct_oof.npz')['advantages'];two=np.load(PRIOR/'oof_predictions.npz')['probabilities']
    lookup={(r['case'],r['seed'],r['off_phase']):r for r in read(COLLECT/'records.json')}
    for i,r in enumerate(meta):
        assert not tasks[r['case']]['validation'];np.testing.assert_array_equal(y[i],[lookup[r['case'],r['seed'],p]['success'] for p in (None,r['phase'])])
    for fold in range(4):
        ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold);n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz')
        assert not {family(tasks,meta[i]['case']) for i in ids}&{family(tasks,meta[i]['case']) for i in val}
        np.testing.assert_array_equal(n['training_indices'],ids);np.testing.assert_array_equal(n['mean'],x[ids].mean(0));np.testing.assert_array_equal(n['scale'],np.maximum(x[ids].std(0),1e-6))
        models=[];original=[]
        for seed in SEEDS:
            m=Advantage();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));m.eval();models.append(m)
            b=Gate();b.load_state_dict(torch.load(PRIOR/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));b.eval();original.append(b)
        np.testing.assert_array_equal(predict(models,n['mean'],n['scale'],x[val]),direct[:,val]);np.testing.assert_array_equal(probabilities(original,n['mean'],n['scale'],x[val]),two[:,val])
    for name,adv in (('direct',direct),('two_head',two[:,:,1]-two[:,:,0])):
        scores=[score(adv,meta,tasks,r) for r in read(ROOT/'protocol.json')['grid']]
        assert scores==read(ROOT/'validation_grids.json')[name];assert select(scores)==read(ROOT/'selected_rules.json')[name]
    for path,h in read(ROOT/'protocol.json')['inputs'].items():assert fingerprint(path)==h
    for path,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/path)==h
    save(ROOT/'verification.json',dict(labels_verified=1108,held_rows=0,ancestral_splits_disjoint=True,identical_normalization=True,all_oof_predictions_exact=True,all_rule_outcomes_reproduced=True,unchanged_baseline_and_two_head_models=True))
    print('Verified both objectives, labels, folds and complete-outcome reconstructions',flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'compare':compare,'media':media,'verify':verify}[sys.argv[1]]()

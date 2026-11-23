"""E130: richer paired advantage versus two success heads, fitting families only."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .abstaining_task_value_gate import DATA, COLLECT, OLD, SEEDS, PHASES, Gate, read, save, family, probabilities, fingerprint

ROOT=Path('outputs/ease_richer_task_state_v1')
PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')


class RichGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(162,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,x):return self.net(x)


def predict(models,mean,scale,x):
    with torch.no_grad():
        xx=torch.tensor(np.clip((np.asarray(x,np.float32)-mean)/scale,-10,10))
        return np.stack([m(xx).sigmoid().numpy() for m in models])


def extract(meta,tasks):
    import mujoco
    from .arm_pick_place import rotation_error
    from .broad_task_value_gate import features
    cache={};extra=[];provenance={}
    base=np.load(DATA/'gate_dataset.npz')['x']
    for i,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];key=(case,seed)
        assert not tasks[case]['validation']
        if key not in cache:
            path=COLLECT/'states'/f'{case}_on_{seed}_frozen_bias.xml'
            m=mujoco.MjModel.from_xml_path(str(path.resolve()));d=mujoco.MjData(m)
            site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
            joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'gripper_{side}_slide') for side in ('left','right')]
            assert all(j>=0 for j in joints)
            actuators=[int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in joints]
            cache[key]=(m,d,site,joints,actuators);provenance[str(path)]=fingerprint(path)
        m,d,site,joints,actuators=cache[key]
        path=COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz';snap=np.load(path);provenance[str(path)]=fingerprint(path)
        mujoco.mj_setState(m,d,snap['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        n=int(base[i,9:16].sum());position=d.site_xpos[site].copy();rot=d.site_xmat[site].reshape(3,3)
        np.testing.assert_allclose(base[i,16:16+n],d.qpos[:n],rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(base[i,30:33],snap['goal']-position,rtol=1e-5,atol=1e-7)
        # Full base feature equality ensures the reconstruction is at the decision time.
        rebuilt=features(n,snap['gram'],snap['cross'],snap['bias'],d.qpos[:n],position,snap['goal'],phase,bool(base[i,138]),base[i,124:124+n],base[i,131:131+n])
        np.testing.assert_array_equal(rebuilt,base[i])
        dq=np.zeros(7);dx=np.zeros(6)
        if len(snap['history_dq']):dq[:n]=snap['history_dq'][-1]
        if len(snap['history_dx']):dx[:]=snap['history_dx'][-1]
        encoder=np.array([d.qpos[m.jnt_qposadr[j]] for j in joints]);command=d.ctrl[actuators].copy()
        value=np.r_[position,rotation_error(np.eye(3),rot),dq,dx,encoder,command].astype(np.float32)
        assert value.shape==(23,) and np.isfinite(value).all()
        extra.append(value)
    return np.stack(extra),provenance


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','paired'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json','fold_audit.json'):shutil.copy2(PRIOR/name,ROOT/name)
    inputs=[DATA/n for n in ('gate_dataset.npz','gate_rows.json','cases.json','verification.json')]
    inputs += [COLLECT/'records.json',PRIOR/'protocol.json',PRIOR/'oof_predictions.npz',PRIOR/'validation_grid.json',PRIOR/'frozen_selection.json',OLD/'records.json',OLD/'summary.json']
    inputs += [p for p in (PRIOR/'models').rglob('*') if p.is_file()]
    save(ROOT/'protocol.json',dict(experiment='E130',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},
        design='Same E126 two-success-head BCE objective,4 ancestral fitting-family folds,3 seeds126100..126102,1500 full-batch AdamW updates, lr.001 weight_decay.001, no weighting/resampling/model selection. Same64/32 tanh hidden layers and2 output logits. Input grows139 to162.',
        initialization='Copy original139-column first-layer weights, all original biases, hidden and output weights from the same-seed baseline initialization. Added23 input columns start at zero. Thus existing pathways and initial behavior match; only new input weights/capacity are added.',
        features='Append tool position3, tool orientation error3, latest measured joint increment padded7, latest measured Cartesian/rotation increment6, two gripper encoder positions, two preceding gripper actuator commands. Reconstruct from the pre-action E120-on integration/history snapshot. No post-action or future state.',
        observability='Ideal noise-free simulated observations. Tool pose and joint history already available to controller; gripper positions model encoder readings, prior commands are known controller signals. No object pose, contact/grasp/lift/bad flags, object velocity, privileged Jacobian, or physical parameters. Gripper sensing is not a tested hardware sensor pipeline.',
        normalization='Original139 columns retain exact original fold-local normalization; added23 mean/std from that fold training families only, scale floor1e-6; all features clipped to[-10,10].',
        primary='Same predeclared minimum-advantage0.1/max-spread2 first-clean-phase rule on144 fitting-placement/estimator episodes. Secondary unchanged E126 grid and zero-loss selection. No original held-placement or unseen-body evaluation.',
        outcomes='Reconstruct success using existing exact paired continuations: first accepted clean phase gets up-to8 off-pulse steps, then E120 resumes. Post-collision starts omitted because success cannot recover. No fresh online-policy rollout or post-collision trajectory equivalence claim.',
        baseline='E120 unchanged; no promotion on fitting-only results. Same success/failure-video provenance workflow.',grid=read(PRIOR/'protocol.json')['grid']))
    tasks,meta=read(ROOT/'cases.json'),read(DATA/'gate_rows.json')
    new,provenance=extract(meta,tasks)
    base=np.load(DATA/'gate_dataset.npz')
    np.savez_compressed(ROOT/'features.npz',x=np.c_[base['x'],new],y=base['y'])
    save(ROOT/'feature_provenance.json',provenance)
    for name in ('richer_task_state.py','richer_task_state_features.py','build_richer_task_state.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print('Prepared162 inputs from1108 pre-action snapshots',flush=True)


def data():
    f=np.load(ROOT/'features.npz');o=np.load(PRIOR/'oof_predictions.npz')
    return f['x'],f['y'],o['row_fold'],read(DATA/'gate_rows.json'),read(ROOT/'cases.json')


def verify_features():
    x,y,folds,meta,tasks=data();new,provenance=extract(meta,tasks)
    np.testing.assert_array_equal(new,x[:,139:]);np.testing.assert_array_equal(x[:,:139],np.load(DATA/'gate_dataset.npz')['x'])
    assert provenance==read(ROOT/'feature_provenance.json')
    for fold in range(4):
        n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz');old=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz')
        np.testing.assert_array_equal(n['mean'][:139],old['mean']);np.testing.assert_array_equal(n['scale'][:139],old['scale'])


def train():
    torch.set_num_threads(4)
    x,y,folds,meta,tasks=data();target=y
    assert all(not tasks[r['case']]['validation'] for r in meta)
    oof=np.empty((3,len(x),2),np.float32);audit=[]
    for fold in range(4):
        train_ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not {family(tasks,meta[i]['case']) for i in train_ids}&{family(tasks,meta[i]['case']) for i in val}
        n=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz');mean=np.r_[n['mean'],x[train_ids,139:].mean(0)];scale=np.r_[n['scale'],np.maximum(x[train_ids,139:].std(0),1e-6)]
        np.testing.assert_array_equal(n['training_indices'],train_ids);np.testing.assert_array_equal(mean[:139],n['mean']);np.testing.assert_array_equal(scale[:139],n['scale'])
        richerory=ROOT/'models'/f'fold_{fold}';richerory.mkdir()
        np.savez_compressed(richerory/'normalization.npz',mean=mean,scale=scale,training_indices=train_ids)
        xx=torch.tensor(np.clip((x[train_ids]-mean)/scale,-10,10));yy=torch.tensor(target[train_ids]);models=[];losses=[]
        for seed in SEEDS:
            torch.manual_seed(seed);original=Gate()
            torch.manual_seed(seed);m=RichGate()
            with torch.no_grad():
                m.net[0].weight[:,:139].copy_(original.net[0].weight)
                m.net[0].weight[:,139:].zero_()
                m.net[0].bias.copy_(original.net[0].bias)
                for layer in (2,4):m.net[layer].load_state_dict(original.net[layer].state_dict())
            for layer in (2,4):
                torch.testing.assert_close(m.net[layer].weight,original.net[layer].weight,rtol=0,atol=0)
            torch.testing.assert_close(m.net[0].weight[:,:139],original.net[0].weight,rtol=0,atol=0)
            assert torch.count_nonzero(m.net[0].weight[:,139:])==0
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.001)
            for _ in range(1500):
                loss=nn.functional.binary_cross_entropy_with_logits(m(xx),yy);opt.zero_grad();loss.backward();opt.step()
            m.eval();torch.save(m.state_dict(),richerory/f'{seed}.pt');models.append(m);losses.append(float(loss.detach()))
        oof[:,val]=predict(models,mean,scale,x[val])
        audit.append(dict(fold=fold,train_rows=len(train_ids),validation_rows=len(val),training_bce=losses,original_weights_initialization_exact=True,added_weights_initially_zero=True))
        save(richerory/'training.json',audit[-1]);print('trained richer fold',fold,flush=True)
    np.savez_compressed(ROOT/'richer_oof.npz',advantages=oof[:,:,1]-oof[:,:,0],probabilities=oof,row_fold=folds)
    save(ROOT/'training.json',audit)
    frozen={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    for name in ('protocol.json','richer_oof.npz','training.json','folds.json','features.npz','feature_provenance.json'):frozen[name]=fingerprint(ROOT/name)
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
    p=np.load(PRIOR/'oof_predictions.npz')['probabilities'];two=p[:,:,1]-p[:,:,0];richer=np.load(ROOT/'richer_oof.npz')['advantages']
    grid=read(ROOT/'protocol.json')['grid'];primary_rule=dict(min_advantage=.1,max_spread=2.)
    result={};grids={};primary={};selected={};fold_rows=[];seed_rows=[]
    for name,adv in (('two_head',two),('richer',richer)):
        result[name]=metrics(adv,target);primary[name]=score(adv,meta,tasks,primary_rule);grids[name]=[score(adv,meta,tasks,r) for r in grid];selected[name]=select(grids[name])
        for fold in range(4):
            ids=folds==fold;fold_rows.append(dict(method=name,fold=fold,**metrics(adv[:,ids],target[ids])))
        for i,seed in enumerate(SEEDS):seed_rows.append(dict(method=name,gate_seed=seed,**metrics(adv[i:i+1],target)))
    old=next(r for r in read(PRIOR/'validation_grid.json') if r['rule']==primary_rule)
    assert primary['two_head']==old
    summary=dict(metrics=result,zero_advantage_reference=metrics(np.zeros_like(richer),target),fold_metrics=fold_rows,seed_metrics=seed_rows,
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
        b=maps['richer'][key]
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
    source=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','richer_task_state import ROOT').replace('E128','E130').replace('Task-value ranking diagnosis','Richer measured task-state comparison').replace('Diagnosis only; no new policy.','Counterfactual continuations selected by fitting-family feature comparison; not fresh policy rollouts.')
    Path(__file__).with_name('richer_task_state_videos.py').write_text(source)
    print('Prepared',len(records),'diagnostic videos',flush=True)


def verify():
    torch.set_num_threads(4);x,y,folds,meta,tasks=data();richer=np.load(ROOT/'richer_oof.npz')['advantages'];two=np.load(PRIOR/'oof_predictions.npz')['probabilities']
    lookup={(r['case'],r['seed'],r['off_phase']):r for r in read(COLLECT/'records.json')}
    for i,r in enumerate(meta):
        assert not tasks[r['case']]['validation'];np.testing.assert_array_equal(y[i],[lookup[r['case'],r['seed'],p]['success'] for p in (None,r['phase'])])
    for fold in range(4):
        ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold);n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz')
        assert not {family(tasks,meta[i]['case']) for i in ids}&{family(tasks,meta[i]['case']) for i in val}
        np.testing.assert_array_equal(n['training_indices'],ids);np.testing.assert_array_equal(n['mean'][139:],x[ids,139:].mean(0));np.testing.assert_array_equal(n['scale'][139:],np.maximum(x[ids,139:].std(0),1e-6))
        models=[];original=[]
        for seed in SEEDS:
            m=RichGate();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));m.eval();models.append(m)
            b=Gate();b.load_state_dict(torch.load(PRIOR/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));b.eval();original.append(b)
        rp=predict(models,n['mean'],n['scale'],x[val]);np.testing.assert_array_equal(rp,np.load(ROOT/'richer_oof.npz')['probabilities'][:,val]);np.testing.assert_array_equal(rp[:,:,1]-rp[:,:,0],richer[:,val]);np.testing.assert_array_equal(probabilities(original,n['mean'][:139],n['scale'][:139],x[val,:139]),two[:,val])
    for name,adv in (('richer',richer),('two_head',two[:,:,1]-two[:,:,0])):
        scores=[score(adv,meta,tasks,r) for r in read(ROOT/'protocol.json')['grid']]
        assert scores==read(ROOT/'validation_grids.json')[name];assert select(scores)==read(ROOT/'selected_rules.json')[name]
    for path,h in read(ROOT/'protocol.json')['inputs'].items():assert fingerprint(path)==h
    for path,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/path)==h
    verify_features()
    save(ROOT/'verification.json',dict(measured_features_reconstructed=True,labels_verified=1108,held_rows=0,ancestral_splits_disjoint=True,identical_normalization=True,all_oof_predictions_exact=True,all_rule_outcomes_reproduced=True,unchanged_baseline_and_two_head_models=True))
    print('Verified both objectives, labels, folds and complete-outcome reconstructions',flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'compare':compare,'media':media,'verify':verify}[sys.argv[1]]()

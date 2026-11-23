"""E131: privileged paired advantage versus two success heads, fitting families only."""
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .abstaining_task_value_gate import DATA, COLLECT, OLD, SEEDS, PHASES, Gate, read, save, family, probabilities, fingerprint

ROOT=Path('outputs/ease_privileged_task_state_v1')
PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')
MEASURED=Path('outputs/ease_richer_task_state_v1')


class PrivilegedGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(182,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,x):return self.net(x)


def predict(models,mean,scale,x):
    with torch.no_grad():
        xx=torch.tensor(np.clip((np.asarray(x,np.float32)-mean)/scale,-10,10))
        return np.stack([m(xx).sigmoid().numpy() for m in models])


def extract(meta,tasks):
    import mujoco
    cache={};extra=[];provenance={}
    base=np.load(MEASURED/'features.npz')['x']
    for i,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];key=(case,seed)
        assert not tasks[case]['validation']
        if key not in cache:
            xml=COLLECT/'states'/f'{case}_on_{seed}_frozen_bias.xml';m=mujoco.MjModel.from_xml_path(str(xml.resolve()));d=mujoco.MjData(m)
            site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object')
            gids={name:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ('cube','table','gripper_left_pad','gripper_right_pad')}
            assert min([site,obj,*gids.values()])>=0
            branch_path=COLLECT/'branches'/f'{case}_on_{seed}.json';branch=read(branch_path)
            cache[key]=(m,d,site,obj,gids,branch);provenance[str(xml)]=fingerprint(xml);provenance[str(branch_path)]=fingerprint(branch_path)
        m,d,site,obj,gids,branch=cache[key]
        path=COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz';snap=np.load(path);provenance[str(path)]=fingerprint(path)
        mujoco.mj_setState(m,d,snap['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        tool=d.site_xpos[site];position=d.xpos[obj];rotation=d.xmat[obj].reshape(3,3)
        np.testing.assert_allclose(base[i,139:142],tool,atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(base[i,30:33],snap['goal']-tool,atol=1e-7,rtol=1e-5)
        contacts=np.zeros(3)
        for c in d.contact[:d.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            if gids['cube'] not in pair:continue
            for j,name in enumerate(('gripper_left_pad','gripper_right_pad','table')):
                if gids[name] in pair:contacts[j]=1
        entry=branch['entries'][phase]
        assert entry['signature']==r['signature']
        # Cumulative grasp flag is captured before the candidate action, not from final outcomes.
        grasped=float(entry['metadata']['grasped']);current_bilateral=float(contacts[0] and contacts[1])
        target=np.r_[tasks[case]['target'],.215]
        value=np.r_[position-tool,rotation.ravel(),target-position,contacts,grasped,current_bilateral].astype(np.float32)
        assert value.shape==(20,) and np.isfinite(value).all()
        extra.append(value)
    return np.stack(extra),provenance


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','paired'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json','fold_audit.json'):shutil.copy2(PRIOR/name,ROOT/name)
    inputs=[DATA/n for n in ('gate_dataset.npz','gate_rows.json','cases.json','verification.json')]
    inputs += [COLLECT/'records.json',PRIOR/'protocol.json',PRIOR/'oof_predictions.npz',PRIOR/'validation_grid.json',PRIOR/'frozen_selection.json',OLD/'records.json',OLD/'summary.json']
    inputs += [MEASURED/n for n in ('features.npz','richer_oof.npz','primary_outcomes.json','validation_grids.json','verification.json')]
    inputs += [p for base in (PRIOR,MEASURED) for p in (base/'models').rglob('*') if p.is_file()]
    save(ROOT/'protocol.json',dict(experiment='E131',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},
        label='PRIVILEGED SIMULATOR-STATE INFORMATION UPPER BOUND. Not a measured-sensor or deployable policy result; extra information does not guarantee improved performance.',
        design='Same two-head BCE,4 ancestral-family fitting folds,3 seeds126100..126102,1500 full-batch AdamW steps, lr.001 weight_decay.001, widths64/32. Input expands measured162 to182. No reweighting/resampling/early stopping or held-driven selection.',
        initialization='Same-seed original139-column weights, biases, hidden and2-output weights copied exactly; all43 extra measured/privileged columns start at zero, matching E130 initial function. Input capacity grows with features.',
        features='Measured162 unchanged. Append20 privileged inputs: object-to-tool position3, world object rotation matrix9, destination-minus-object position3 (task landing height0.215m), instantaneous object contacts with left pad/right pad/table3, pre-action cumulative grasp flag1, instantaneous bilateral finger contact1.',
        timing='All added values use E120-on pre-action integration/history snapshot and same-time branch metadata. Grasp flag is from entry metadata, never final trial success/grasp/lift. No future state, final label, collision/bad flag, physical parameters, or true Jacobian in features.',
        normalization='Measured162 normalization retained exactly per fold from E130. Added20 mean/std fitted only on training families, scale floor1e-6; clipping[-10,10].',
        primary='Same min-advantage0.1/max-spread2 first-clean-phase intervention rule,144 fitting-placement/estimator episodes. Compare original139, measured162 and privileged182. Secondary same E126 zero-loss grid, tuning-validation only.',
        outcomes='Reconstruct success from existing exact paired full-task continuations: one up-to8-step correction-off pulse then E120. Post-collision starts omitted because success irreversibly false. No fresh online-policy rollout or post-collision trajectory equivalence claim.',
        baseline='E120 unchanged. No original held-placement benchmark, unseen bodies, sensing deployment, or promotion from this comparison.',grid=read(PRIOR/'protocol.json')['grid']))
    tasks,meta=read(ROOT/'cases.json'),read(DATA/'gate_rows.json');new,provenance=extract(meta,tasks)
    base=np.load(MEASURED/'features.npz');np.savez_compressed(ROOT/'features.npz',x=np.c_[base['x'],new],y=base['y'])
    save(ROOT/'feature_provenance.json',provenance)
    for name in ('privileged_task_state.py','privileged_task_state_features.py','build_privileged_task_state.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print('Prepared182 inputs:162 measured plus20 PRIVILEGED',flush=True)


def data():
    f=np.load(ROOT/'features.npz');o=np.load(PRIOR/'oof_predictions.npz')
    return f['x'],f['y'],o['row_fold'],read(DATA/'gate_rows.json'),read(ROOT/'cases.json')


def verify_features():
    x,y,folds,meta,tasks=data();new,provenance=extract(meta,tasks)
    np.testing.assert_array_equal(new,x[:,162:]);np.testing.assert_array_equal(x[:,:162],np.load(MEASURED/'features.npz')['x'])
    assert provenance==read(ROOT/'feature_provenance.json')
    for fold in range(4):
        n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz');old=np.load(MEASURED/'models'/f'fold_{fold}'/'normalization.npz')
        np.testing.assert_array_equal(n['mean'][:162],old['mean']);np.testing.assert_array_equal(n['scale'][:162],old['scale'])


def train():
    torch.set_num_threads(4)
    x,y,folds,meta,tasks=data();target=y
    assert all(not tasks[r['case']]['validation'] for r in meta)
    oof=np.empty((3,len(x),2),np.float32);audit=[]
    for fold in range(4):
        train_ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold)
        assert not {family(tasks,meta[i]['case']) for i in train_ids}&{family(tasks,meta[i]['case']) for i in val}
        n=np.load(MEASURED/'models'/f'fold_{fold}'/'normalization.npz');mean=np.r_[n['mean'],x[train_ids,162:].mean(0)];scale=np.r_[n['scale'],np.maximum(x[train_ids,162:].std(0),1e-6)]
        np.testing.assert_array_equal(n['training_indices'],train_ids);np.testing.assert_array_equal(mean[:162],n['mean']);np.testing.assert_array_equal(scale[:162],n['scale'])
        directory=ROOT/'models'/f'fold_{fold}';directory.mkdir()
        np.savez_compressed(directory/'normalization.npz',mean=mean,scale=scale,training_indices=train_ids)
        xx=torch.tensor(np.clip((x[train_ids]-mean)/scale,-10,10));yy=torch.tensor(target[train_ids]);models=[];losses=[]
        for seed in SEEDS:
            torch.manual_seed(seed);original=Gate()
            torch.manual_seed(seed);m=PrivilegedGate()
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
            m.eval();torch.save(m.state_dict(),directory/f'{seed}.pt');models.append(m);losses.append(float(loss.detach()))
        oof[:,val]=predict(models,mean,scale,x[val])
        audit.append(dict(fold=fold,train_rows=len(train_ids),validation_rows=len(val),training_bce=losses,original_weights_initialization_exact=True,added_weights_initially_zero=True))
        save(directory/'training.json',audit[-1]);print('trained privileged fold',fold,flush=True)
    np.savez_compressed(ROOT/'privileged_oof.npz',advantages=oof[:,:,1]-oof[:,:,0],probabilities=oof,row_fold=folds)
    save(ROOT/'training.json',audit)
    frozen={str(p.relative_to(ROOT)):fingerprint(p) for p in (ROOT/'models').rglob('*') if p.is_file()}
    for name in ('protocol.json','privileged_oof.npz','training.json','folds.json','features.npz','feature_provenance.json'):frozen[name]=fingerprint(ROOT/name)
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
    p=np.load(PRIOR/'oof_predictions.npz')['probabilities'];two=p[:,:,1]-p[:,:,0];privileged=np.load(ROOT/'privileged_oof.npz')['advantages']
    grid=read(ROOT/'protocol.json')['grid'];primary_rule=dict(min_advantage=.1,max_spread=2.)
    result={};grids={};primary={};selected={};fold_rows=[];seed_rows=[]
    for name,adv in (('two_head',two),('measured',np.load(MEASURED/'richer_oof.npz')['advantages']),('privileged',privileged)):
        result[name]=metrics(adv,target);primary[name]=score(adv,meta,tasks,primary_rule);grids[name]=[score(adv,meta,tasks,r) for r in grid];selected[name]=select(grids[name])
        for fold in range(4):
            ids=folds==fold;fold_rows.append(dict(method=name,fold=fold,**metrics(adv[:,ids],target[ids])))
        for i,seed in enumerate(SEEDS):seed_rows.append(dict(method=name,gate_seed=seed,**metrics(adv[i:i+1],target)))
    old=next(r for r in read(PRIOR/'validation_grid.json') if r['rule']==primary_rule)
    assert primary['two_head']==old
    assert primary['measured']==read(MEASURED/'primary_outcomes.json')['richer']
    assert grids['measured']==read(MEASURED/'validation_grids.json')['richer']
    summary=dict(metrics=result,zero_advantage_reference=metrics(np.zeros_like(privileged),target),fold_metrics=fold_rows,seed_metrics=seed_rows,
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
        b=maps['privileged'][key]
        if a['success']!=b['success'] or a['success']!=a['baseline_success'] or b['success']!=b['baseline_success']:
            for r in (a,b):add(r['case'],r['seed'],r['phase'],'fixed-rule comparison')
            add(*key,None,'E120 reference')
    for key,a in maps['measured'].items():
        b=maps['privileged'][key]
        if a['success']!=b['success']:
            for r in (a,b):add(r['case'],r['seed'],r['phase'],'measured versus privileged')
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
    source=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','privileged_task_state import ROOT').replace('E128','E131').replace('Task-value ranking diagnosis','PRIVILEGED simulator task-state comparison').replace('Diagnosis only; no new policy.','Counterfactual continuations selected by fitting-family feature comparison; not fresh policy rollouts.')
    source=source.replace('Measured-history control | No physics selector or geometry filter','PRIVILEGED simulator-state study | Counterfactual replay')
    Path(__file__).with_name('privileged_task_state_videos.py').write_text(source)
    print('Prepared',len(records),'diagnostic videos',flush=True)


def verify():
    torch.set_num_threads(4);x,y,folds,meta,tasks=data();privileged=np.load(ROOT/'privileged_oof.npz')['advantages'];two=np.load(PRIOR/'oof_predictions.npz')['probabilities']
    lookup={(r['case'],r['seed'],r['off_phase']):r for r in read(COLLECT/'records.json')}
    for i,r in enumerate(meta):
        assert not tasks[r['case']]['validation'];np.testing.assert_array_equal(y[i],[lookup[r['case'],r['seed'],p]['success'] for p in (None,r['phase'])])
    for fold in range(4):
        ids,val=np.flatnonzero(folds!=fold),np.flatnonzero(folds==fold);n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz')
        assert not {family(tasks,meta[i]['case']) for i in ids}&{family(tasks,meta[i]['case']) for i in val}
        np.testing.assert_array_equal(n['training_indices'],ids);np.testing.assert_array_equal(n['mean'][162:],x[ids,162:].mean(0));np.testing.assert_array_equal(n['scale'][162:],np.maximum(x[ids,162:].std(0),1e-6))
        models=[];original=[]
        for seed in SEEDS:
            m=PrivilegedGate();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));m.eval();models.append(m)
            b=Gate();b.load_state_dict(torch.load(PRIOR/'models'/f'fold_{fold}'/f'{seed}.pt',weights_only=True));b.eval();original.append(b)
        rp=predict(models,n['mean'],n['scale'],x[val]);np.testing.assert_array_equal(rp,np.load(ROOT/'privileged_oof.npz')['probabilities'][:,val]);np.testing.assert_array_equal(rp[:,:,1]-rp[:,:,0],privileged[:,val]);np.testing.assert_array_equal(probabilities(original,n['mean'][:139],n['scale'][:139],x[val,:139]),two[:,val])
    for name,adv in (('privileged',privileged),('two_head',two[:,:,1]-two[:,:,0])):
        scores=[score(adv,meta,tasks,r) for r in read(ROOT/'protocol.json')['grid']]
        assert scores==read(ROOT/'validation_grids.json')[name];assert select(scores)==read(ROOT/'selected_rules.json')[name]
    for path,h in read(ROOT/'protocol.json')['inputs'].items():assert fingerprint(path)==h
    for path,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/path)==h
    verify_features()
    save(ROOT/'verification.json',dict(privileged_features_reconstructed=True,measured_features_unchanged=True,labels_verified=1108,held_rows=0,ancestral_splits_disjoint=True,identical_normalization=True,all_oof_predictions_exact=True,all_rule_outcomes_reproduced=True,unchanged_baseline_and_two_head_models=True))
    print('Verified both objectives, labels, folds and complete-outcome reconstructions',flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'compare':compare,'media':media,'verify':verify}[sys.argv[1]]()

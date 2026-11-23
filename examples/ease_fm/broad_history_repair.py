"""E118: broader training-only histories with phase and success retention."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from . import training_estimator_core as core
from .training_estimator_repair import setup as original_setup,predict,batch,analytical_targets,action_tensor
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint
ROOT=Path('outputs/ease_broad_history_v1');OLD=Path('outputs/ease_estimator_repair_v1')

def setup():
    tasks,parents,hashes=original_setup();models={'analytical':None}
    for seed in range(3):
        p=OLD/f'seed_{seed}/policy.pt';m=MotionTransformer().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));models[f'reference_{seed}']=m.eval();hashes[str(p)]=fingerprint(p)
    if (ROOT/'cases.json').exists():tasks=json.loads((ROOT/'cases.json').read_text())
    core.ROOT=ROOT
    return tasks,parents,models,hashes

def prepare():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','source'):(ROOT/d).mkdir()
    tasks,parents,models,hashes=setup()
    for i,t in enumerate(tasks):t.update(validation=i%3==2,group='original_held' if i%3==2 else 'original_train',source_case=i)
    expanded=[]
    for i,t in enumerate(tasks):
        new=json.loads(json.dumps(t));offset=np.array([.012 if i%2==0 else -.012,.008 if i%2==0 else -.008])
        new['xy']=(np.array(t['xy'])+offset).tolist();new['target']=(np.array(t['target'])+offset).tolist();new['group']='fresh_held' if t['validation'] else 'broad_train';expanded.append(new)
    tasks+=expanded;assert len(tasks)==24 and all(t['arm']['split']=='train' for t in tasks)
    (ROOT/'cases.json').write_text(json.dumps(tasks,indent=2))
    inputs={str(OLD/n):fingerprint(OLD/n) for n in ('contexts.npz','context_provenance.json','records.json','summary.json')}
    spec=dict(experiment='E118',inputs=inputs,checkpoints=hashes,
        design='Keep all12 original tasks; add one prespecified shifted task per original on the same4 training bodies (x +/-12mm,y +/-8mm with parity). Retain original reset q; no success/IK filtering of shifts. Newcase calibration seeds38000+case differ. Eight shifted fitting placements and four fresh held placements; original four held placements remain excluded. No unseen-body execution or data.',
        collection='Reuse48 original E99/analytical trajectories and36 E117 candidates. Add48 trajectories on12 shifted cases using analytical/E117 three seeds; record all successes/failures and histories. Reuse original E117 context dataset; only clean fitting histories enter loss. Fresh and original held histories never enter optimizer.',
        training='Three fixed2000-update runs from E117 final checkpoints, AdamW1e-5, clip1, batch128:32 original fitting rows,32 shifted descend/lower rows,32 shifted other rows,32 synthetic. Main ridge MSE +0.10 normalized task action loss (descent/lower weight3) +0.02 measured-response loss +0.25 matrix preservation on protected rows. Successful E117 reference task: preserve its same-history action/matrix. On original fitting cases where E117 failed but E99 succeeded: preserve E99. Otherwise analytical targets. No action objective during explicit probe. Same-history targets, no privileged inference or selector. No held-based checkpoint selection.',
        evaluation='72 candidate full episodes on all24 cases. Compare original successes against E99 and E117 union, shifted against E117. Require overall/reference improvement, zero lost reference successes, no joint-count/seed decline, no collision increase; original and fresh held results reported separately. No unseen testing in this experiment, even if gate passes.',
        videos='First success/failure per reference and candidate seed; paired first gain/loss per seed and first fresh-held gain/loss. Keep source models, predictions, full states and all outcomes.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));rows=[]
    for r in json.loads((OLD/'records.json').read_text()):
        oldmethod=r['method'];method=oldmethod.replace('learned_','parent_').replace('candidate_','reference_');r['method']=method
        a=f"{r['case']}_{oldmethod}_frozen_bias";b=f"{r['case']}_{method}_frozen_bias"
        full=json.loads((OLD/'trials'/f'{a}.json').read_text());full['method']=method;(ROOT/'trials'/f'{b}.json').write_text(json.dumps(full))
        for ext in ('xml','npz'):shutil.copy2(OLD/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    samples=[];provenance=[]
    for case in range(12,24):
        for method,model in models.items():
            start=len(samples);r=core.evaluate(case,tasks[case],method,model,'frozen_bias',save_states=True,contexts=samples)
            for x in samples[start:]:x['validation']=tasks[case]['validation']
            provenance.append(dict(start=start,end_exclusive=len(samples),case=case,driver=method,body=r['body'],success=r['success'],validation=tasks[case]['validation']))
            (ROOT/'trials'/f'{case}_{method}_frozen_bias.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('broad data case',case,flush=True)
    f=np.load(OLD/'contexts.npz');arrays={k:np.concatenate([f[k],np.stack([r[k] for r in samples])]) for k in f.files}
    assert all(not tasks[int(c)]['validation'] for c in arrays['case'][~arrays['validation']]);assert set(arrays['case'][~arrays['validation']]).isdisjoint(set(arrays['case'][arrays['validation']]))
    np.savez_compressed(ROOT/'contexts.npz',**arrays);(ROOT/'new_context_provenance.json').write_text(json.dumps(provenance,indent=2))
    (ROOT/'data_audit.json').write_text(json.dumps(dict(original_rows=len(f['g']),new_rows=len(samples),rows=len(arrays['g']),clean_fitting=int(np.sum(arrays['clean']&~arrays['validation'])),clean_held=int(np.sum(arrays['clean']&arrays['validation'])),cases_sha256=fingerprint(ROOT/'cases.json'),contexts_sha256=fingerprint(ROOT/'contexts.npz'),held_cases=sorted(set(int(c) for c in arrays['case'][arrays['validation']])),fitting_cases=sorted(set(int(c) for c in arrays['case'][~arrays['validation']])),body_ids=sorted(set(t['arm']['seed'] for t in tasks))),indent=2))
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    for n in ('broad_history_repair.py','training_estimator_core.py','local_history.py','pick_place_control.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)

def train():
    torch.set_num_threads(4);tasks,parents,models,hashes=setup()
    with np.load(ROOT/'contexts.npz') as archive:f={k:archive[k] for k in archive.files}
    v={k:torch.tensor(f[k],device='cuda') for k in f if k!='phase'}
    v['target']=analytical_targets(v['g'],v['c'],v['mask']);valid=v['clean']&~v['validation'];special=torch.tensor(np.isin(f['phase'],['descend','lower']),device='cuda');action_eligible=torch.tensor(f['phase']!='probe',device='cuda')
    pools=[torch.where(valid&(v['case']<12))[0],torch.where(valid&(v['case']>=12)&special)[0],torch.where(valid&(v['case']>=12)&~special)[0]];assert all(len(p)>0 for p in pools)
    rows=json.loads((ROOT/'records.json').read_text());lookup={(r['case'],r['method']):r for r in rows};held=torch.where(v['clean']&v['validation'])[0]
    def metrics(model):
        model.eval();stats=[]
        with torch.no_grad():
            for ids in held.split(256):
                p=predict(model,v['g'][ids],v['c'][ids],v['mask'][ids]);a=action_tensor(p,v['error'][ids],v['bias'][ids],v['mask'][ids]);at=action_tensor(v['target'][ids],v['error'][ids],v['bias'][ids],v['mask'][ids]);dx=torch.einsum('bnd,bn->bd',p,v['dq'][ids]);ar=torch.sqrt(((a-at).square()*v['mask'][ids]).sum(-1)/v['mask'][ids].sum(-1));er=torch.linalg.vector_norm(dx[:,:3]-v['dx'][ids,:3],dim=-1)
                for j,i in enumerate(ids.tolist()):stats.append(dict(case=int(f['case'][i]),dof=int(f['mask'][i].sum()),phase=str(f['phase'][i]),action_rmse=float(ar[j]) if f['phase'][i]!='probe' else None,xyz_error=float(er[j])))
        return {group:{k:float(np.mean([r[k] for r in stats if (group=='all' or (group=='fresh')==(r['case']>=12)) and r[k] is not None])) for k in ('action_rmse','xyz_error')} for group in ('all','original','fresh')}
    for seed in range(3):
        if (ROOT/f'seed_{seed}/training.json').exists():
            assert fingerprint(ROOT/f'seed_{seed}/policy.pt')==json.loads((ROOT/f'seed_{seed}/training.json').read_text())['policy_sha256']
            print('preserving completed seed',seed,flush=True);continue
        teacher=models[f'reference_{seed}'];parent=parents[f'learned_{seed}'];targets=v['target'].clone();protected=torch.zeros(len(f['g']),device='cuda',dtype=torch.bool)
        with torch.no_grad():
            for case in range(24):
                if tasks[case]['validation']:continue
                ref=teacher if lookup[case,f'reference_{seed}']['success'] else parent if case<12 and lookup[case,f'parent_{seed}']['success'] else None
                if ref is None:continue
                ids=torch.where(valid&(v['case']==case))[0];protected[ids]=True
                for part in ids.split(256):targets[part]=predict(ref,v['g'][part],v['c'][part],v['mask'][part])
        desired=action_tensor(targets,v['error'],v['bias'],v['mask']);torch.manual_seed(118000+seed);model=MotionTransformer().cuda();model.load_state_dict(teacher.state_dict());initial=metrics(model);model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-5);gen=torch.Generator(device='cuda').manual_seed(118100+seed);losses=[];digest=hashlib.sha256()
        for step in range(2000):
            ids=torch.cat([pool[torch.randint(len(pool),(32,),device='cuda',generator=gen)] for pool in pools]);sg,sc,sm,st,_=batch(32,'cuda',gen)
            g=torch.cat([v['g'][ids],sg]);c=torch.cat([v['c'][ids],sc]);m=torch.cat([v['mask'][ids],sm]);p=predict(model,g,c,m);fit=((p-torch.cat([v['target'][ids],st])).square()*m[:,:,None]).sum()/(6*m.sum())
            a=action_tensor(p[:96],v['error'][ids],v['bias'][ids],m[:96]);w=m[:96]*action_eligible[ids,None]*torch.where(special[ids,None],3.,1.);action_loss=((a-desired[ids]).square()*w).sum()/(w.sum()*.04**2)
            response=(torch.einsum('bnd,bn->bd',p[:96],v['dq'][ids])-v['dx'][ids]).square().mean()/.003**2
            pm=m[:96]*protected[ids,None];retain=((p[:96]-targets[ids]).square()*pm[:,:,None]).sum()/(6*pm.sum().clamp_min(1));loss=fit+.10*action_loss+.02*response+.25*retain
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();digest.update(ids.cpu().numpy().tobytes());losses.append([float(x.detach()) for x in (fit,action_loss,response,retain)])
            if (step+1)%500==0:print('train',seed,step+1,losses[-1],flush=True)
        folder=ROOT/f'seed_{seed}';folder.mkdir();model.eval();torch.save(model.state_dict(),folder/'policy.pt');(folder/'training.json').write_text(json.dumps(dict(seed=seed,steps=2000,initial=initial,final=metrics(model),losses=losses,protected_contexts=int(protected.sum()),batch_sha256=digest.hexdigest(),policy_sha256=fingerprint(folder/'policy.pt')),indent=2));print('trained',seed,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

def evaluation():
    torch.set_num_threads(4);tasks,parents,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==132
    candidates={}
    for seed in range(3):
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(ROOT/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));candidates[f'candidate_{seed}']=m.eval()
    for case,t in enumerate(tasks):
        for method,model in candidates.items():
            r=core.evaluate(case,t,method,model,'frozen_bias',save_states=True);old=json.loads((ROOT/'trials'/f'{case}_reference_{method[-1]}_frozen_bias.json').read_text());assert r['calibration']==old['calibration'] and r['initial_estimator']==old['initial_estimator']
            (ROOT/'trials'/f'{case}_{method}_frozen_bias.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('eval case',case,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'evaluate':evaluation}[sys.argv[1]]()

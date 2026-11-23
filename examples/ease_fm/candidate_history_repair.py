"""E119: fit corrective estimates on the failed candidates' own histories."""
import json,shutil,hashlib
from pathlib import Path
import numpy as np
import torch
from . import broad_history_repair as prior
from .broad_history_repair import core,predict,batch,analytical_targets,action_tensor,MotionTransformer,fingerprint
ROOT=Path('outputs/ease_candidate_history_v1');OLD=prior.ROOT

def setup():
    tasks,parents,older,hashes=prior.setup();models={'analytical':None}
    for seed in range(3):
        p=OLD/f'seed_{seed}/policy.pt';m=MotionTransformer().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));models[f'reference_{seed}']=m.eval();hashes[str(p)]=fingerprint(p)
    core.ROOT=ROOT
    return tasks,parents,models,hashes

def prepare():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','source'):(ROOT/d).mkdir()
    tasks,parents,models,hashes=setup();shutil.copy2(OLD/'cases.json',ROOT/'cases.json');inputs={str(OLD/n):fingerprint(OLD/n) for n in ('contexts.npz','records.json','summary.json','cases.json')}
    spec=dict(experiment='E119',inputs=inputs,checkpoints=hashes,
        collection='Exactly replay all48 fitting-case E118 candidate trajectories (16 fitting placements x3 checkpoints), including successes and failures, collecting every pre-update context. Do not collect held candidate histories for fitting. Check every baseline trace and state against E118. Append to E118 contexts; inherit held histories for fixed offline diagnostics only.',
        training='Three fixed1500-update runs from E118 final checkpoints; AdamW1e-5, gradient clip1,128 rows:32 own failed-task clean contexts,32 own successful-task clean contexts,32 earlier fitting contexts,32 synthetic. On cases already successful for this seed, preserve current E118 same-history response/action. Failed cases: use successful E117 if available, then successful E99 on original cases, otherwise analytical same-history ridge. Matrix/action objectives use these coherent targets; measured-response loss only on unprotected failed cases. No conflicting analytical-matrix objective on protected successful cases. Loss matrix MSE +0.1 normalized action MSE (descent/lower3x; probe excluded)+0.02 measured-response loss. No held or unseen optimization, no checkpoint selection.',
        evaluation='Reuse204 earlier results;72 new candidate full episodes. Require net improvement, no lost E118 success, no additional collisions, and retention of older successful baselines. Report targeted fitting failures and both held groups separately. No physics selector, geometry filter or privileged Jacobian; same frozen calibrated bias and response updates. No unseen evaluation.',
        videos='First success/failure per reference/candidate seed; paired first gain/loss overall and fresh-held per seed. All trials, contexts, state sequences, checkpoints and failures retained.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));rows=[]
    for r in json.loads((OLD/'records.json').read_text()):
        original=r['method'];method=original.replace('reference_','older_').replace('candidate_','reference_');r['method']=method
        a=f"{r['case']}_{original}_frozen_bias";b=f"{r['case']}_{method}_frozen_bias";full=json.loads((OLD/'trials'/f'{a}.json').read_text());full['method']=method;(ROOT/'trials'/f'{b}.json').write_text(json.dumps(full))
        for ext in ('npz','xml'):shutil.copy2(OLD/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));samples=[];drivers=[];provenance=[];exact=[]
    for case,t in enumerate(tasks):
        if t['validation']:continue
        for seed in range(3):
            method=f'reference_{seed}';states=[];start=len(samples)
            def capture(env):states.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
            r=core.evaluate(case,t,method,models[method],'frozen_bias',capture=capture,contexts=samples)
            old=json.loads((ROOT/'trials'/f'{case}_{method}_frozen_bias.json').read_text());assert r==old
            np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{case}_{method}_frozen_bias.npz')['states'])
            for x in samples[start:]:x['validation']=False
            drivers.extend([seed]*(len(samples)-start));provenance.append(dict(case=case,seed=seed,start=start,end_exclusive=len(samples),success=r['success']));exact.append(f'{case}_{method}')
        print('candidate histories case',case,flush=True)
    with np.load(OLD/'contexts.npz') as f:oldarrays={k:f[k] for k in f.files}
    arrays={k:np.concatenate([oldarrays[k],np.stack([r[k] for r in samples])]) for k in oldarrays};arrays['driver']=np.r_[np.full(len(oldarrays['g']),-1),drivers]
    assert not arrays['validation'][arrays['driver']>=0].any();np.savez_compressed(ROOT/'contexts.npz',**arrays)
    (ROOT/'context_provenance.json').write_text(json.dumps(provenance,indent=2));(ROOT/'data_audit.json').write_text(json.dumps(dict(original_rows=len(oldarrays['g']),candidate_rows=len(samples),total_rows=len(arrays['g']),exact_candidate_replays=exact,held_candidate_histories_excluded=True,contexts_sha256=fingerprint(ROOT/'contexts.npz')),indent=2))
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    for n in ('candidate_history_repair.py','training_estimator_core.py','local_history.py','pick_place_control.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)

def train():
    torch.set_num_threads(4);tasks,parents,models,hashes=setup();_,_,older,_=prior.setup();core.ROOT=ROOT
    with np.load(ROOT/'contexts.npz') as archive:f={k:archive[k] for k in archive.files}
    v={k:torch.tensor(f[k],device='cuda') for k in f if k!='phase'};ridge=analytical_targets(v['g'],v['c'],v['mask']);valid=v['clean']&~v['validation'];special=torch.tensor(np.isin(f['phase'],['descend','lower']),device='cuda');eligible=torch.tensor(f['phase']!='probe',device='cuda')
    rows=json.loads((ROOT/'records.json').read_text());lookup={(r['case'],r['method']):r for r in rows};held=torch.where(v['clean']&v['validation'])[0]
    def metrics(model):
        model.eval();stats=[]
        with torch.no_grad():
            for ids in held.split(256):
                m=v['mask'][ids];p=predict(model,v['g'][ids],v['c'][ids],m);a=action_tensor(p,v['error'][ids],v['bias'][ids],m);target=action_tensor(ridge[ids],v['error'][ids],v['bias'][ids],m);dx=torch.einsum('bnd,bn->bd',p,v['dq'][ids]);ar=torch.sqrt(((a-target).square()*m).sum(-1)/m.sum(-1)).cpu().numpy();pe=torch.linalg.vector_norm(dx[:,:3]-v['dx'][ids,:3],dim=-1).cpu().numpy()
                for j,i in enumerate(ids.tolist()):stats.append(dict(case=int(f['case'][i]),action_rmse=float(ar[j]) if f['phase'][i]!='probe' else None,xyz_error=float(pe[j])))
        return {group:{k:float(np.mean([r[k] for r in stats if (group=='all' or (group=='fresh')==(r['case']>=12)) and r[k] is not None])) for k in ('action_rmse','xyz_error')} for group in ('all','original','fresh')}
    for seed in range(3):
        teacher=models[f'reference_{seed}'];targets=ridge.clone();protected=torch.zeros(len(f['g']),dtype=torch.bool,device='cuda');case_success=torch.tensor([lookup[int(c),f'reference_{seed}']['success'] for c in f['case']],device='cuda')
        with torch.no_grad():
            for case,t in enumerate(tasks):
                if t['validation']:continue
                ids=torch.where(valid&(v['case']==case))[0]
                if lookup[case,f'reference_{seed}']['success']:ref=teacher;protected[ids]=True
                elif lookup[case,f'older_{seed}']['success']:ref=older[f'reference_{seed}']
                elif case<12 and lookup[case,f'parent_{seed}']['success']:ref=parents[f'learned_{seed}']
                else:ref=None
                if ref is not None:
                    for part in ids.split(256):targets[part]=predict(ref,v['g'][part],v['c'][part],v['mask'][part])
        pools=[torch.where(valid&(v['driver']==seed)&~case_success)[0],torch.where(valid&(v['driver']==seed)&case_success)[0],torch.where(valid&(v['driver']==-1))[0]];assert all(len(p)>0 for p in pools)
        desired=action_tensor(targets,v['error'],v['bias'],v['mask']);target_ids=torch.where(valid)[0]
        np.savez_compressed(ROOT/f'targets_seed_{seed}.npz',row_ids=target_ids.cpu().numpy(),matrix=targets[target_ids].cpu().numpy(),action=desired[target_ids].cpu().numpy(),protected=protected[target_ids].cpu().numpy())
        torch.manual_seed(119000+seed);model=MotionTransformer().cuda();model.load_state_dict(teacher.state_dict());initial=metrics(model);model.train();opt=torch.optim.AdamW(model.parameters(),lr=1e-5);gen=torch.Generator(device='cuda').manual_seed(119100+seed);history=[];digest=hashlib.sha256()
        for step in range(1500):
            ids=torch.cat([p[torch.randint(len(p),(32,),device='cuda',generator=gen)] for p in pools]);sg,sc,sm,st,_=batch(32,'cuda',gen);m=torch.cat([v['mask'][ids],sm]);pred=predict(model,torch.cat([v['g'][ids],sg]),torch.cat([v['c'][ids],sc]),m)
            fit=((pred-torch.cat([targets[ids],st])).square()*m[:,:,None]).sum()/(6*m.sum());a=action_tensor(pred[:96],v['error'][ids],v['bias'][ids],m[:96]);w=m[:96]*eligible[ids,None]*torch.where(special[ids,None],3.,1.);al=((a-desired[ids]).square()*w).sum()/(w.sum()*.04**2)
            residual=(torch.einsum('bnd,bn->bd',pred[:96],v['dq'][ids])-v['dx'][ids]).square();rw=(~protected[ids]).float();rl=(residual*rw[:,None]).sum()/(6*rw.sum().clamp_min(1)*.003**2);loss=fit+.1*al+.02*rl
            assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();digest.update(ids.cpu().numpy().tobytes());history.append([float(x.detach()) for x in (fit,al,rl)])
            if (step+1)%500==0:print('train',seed,step+1,history[-1],flush=True)
        folder=ROOT/f'seed_{seed}';folder.mkdir();model.eval();torch.save(model.state_dict(),folder/'policy.pt');(folder/'training.json').write_text(json.dumps(dict(seed=seed,steps=1500,initial=initial,final=metrics(model),losses=history,pool_sizes=[len(p) for p in pools],batch_sha256=digest.hexdigest(),targets_sha256=fingerprint(ROOT/f'targets_seed_{seed}.npz'),policy_sha256=fingerprint(folder/'policy.pt')),indent=2));print('trained',seed,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

def evaluation():
    torch.set_num_threads(4);tasks,_,_,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==204;models={}
    for seed in range(3):
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(ROOT/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));models[f'candidate_{seed}']=m.eval()
    for case,t in enumerate(tasks):
        for method,model in models.items():
            r=core.evaluate(case,t,method,model,'frozen_bias',save_states=True);old=json.loads((ROOT/'trials'/f'{case}_reference_{method[-1]}_frozen_bias.json').read_text());assert r['calibration']==old['calibration'] and r['initial_estimator']==old['initial_estimator'];(ROOT/'trials'/f'{case}_{method}_frozen_bias.json').write_text(json.dumps(r));rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('eval case',case,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train,'evaluate':evaluation}[sys.argv[1]]()

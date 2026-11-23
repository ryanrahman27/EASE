"""One-time E159 fresh placement evaluation of frozen estimators and matched controls."""
from unittest.mock import patch
import math
import numpy as np
import torch
from .fresh_fixed_weights import ROOT,SEEDS
from .anchored_history_weights import AnchoredWeights
from .learned_history_weights import token_features
from .history_estimator_control import history
from . import broad_task_value_gate as broad
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .arm_pick_place import ROT_SCALE
METHODS=('uniform32','analytical','e120_ensemble','learned_final')

def verify_frozen():
    for p,h in {**read(ROOT/'model_freeze.json'),**read(ROOT/'evaluation_freeze.json'),**read(ROOT/'reference_checkpoints.json')}.items():assert fingerprint(p)==h

def evaluate(index,t,method,models,learned):
    case=10000+index;previous=broad.previous;core=previous.core;original_predict=core.LocalPose.predict;original_rollout=core.probe_rollout;cache={};logs=[];contexts=[];correction_ids={id(models[f'correction_{s}']) for s in range(3)}
    def prediction(c):
        if method=='e120_ensemble':
            group='correction' if id(c.model) in correction_ids else 'reference';key=(group,c.n,c.gram.tobytes(),c.cross.tobytes())
            if cache.get(group+'_key')!=key:
                old=c.model;js=[]
                try:
                    for s in range(3):c.model=models[f'{group}_{s}'];js.append(original_predict(c)[0])
                finally:c.model=old
                cache[group+'_key']=key;cache[group+'_j']=np.mean(js,axis=0)
            return cache[group+'_j'].copy(),c.bias
        if method=='analytical':return original_predict(c)
        hq,hx=history(c);key=(hq.tobytes(),hx.tobytes(),c.n)
        if cache.get('key')!=key:
            q=torch.tensor(hq[None],device='cuda');x=torch.tensor(hx[None],device='cuda');mask=torch.arange(7,device='cuda')[None]<c.n
            with torch.no_grad():
                if method=='learned_final':
                    feature=token_features(q,x,mask);j=torch.stack([m(q,x,mask,feature)[0] for m in learned]).mean(0)
                else:
                    w=.98**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);qq,xx=q.double(),x.double();g=qq.transpose(1,2)@(qq*w[None,:,None])+1e-6*torch.eye(7,device='cuda',dtype=torch.float64)[None];j=torch.linalg.solve(g,qq.transpose(1,2)@(xx*w[None,:,None]))
            cache['key']=key;cache['j']=j[0,:c.n].cpu().numpy().T.copy()
        return cache['j'].copy(),c.bias
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            before=env.observe();phase=env.phase;clean=not env.bad;hq,hx=history(c);after,command,info=original_step(action,aperture);dq=np.zeros(7,np.float32);dq[:c.n]=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])].astype(np.float32);contexts.append(dict(dq_history=hq,dx_history=hx,query=dq,target=dx,clean=clean,phase=(*PHASES,'probe').index(phase)));logs.append(dict(phase=phase,q_before=before['q'].tolist(),position_before=before['position'].tolist(),nominal_action=np.asarray(action).tolist(),executed_command=command.tolist(),aperture=float(aperture),bad=info['bad']));return after,command,info
        env.step=step;return original_rollout(env,home,c,**kwargs)
    with patch.object(core.LocalPose,'predict',prediction),patch.object(core,'probe_rollout',observer):
        if method=='e120_ensemble':r,_=previous.evaluate(case,t,method,models['reference_0'],models['correction_0'],'bounded',save=True)
        else:r=core.evaluate(case,t,method,None,'frozen_bias',save_states=True)
    r.update(fresh_index=index,ancestral_source=t['source_case']);stem=f'{case}_{method}_frozen_bias';save(ROOT/'trials'/f'{stem}.json',r);save(ROOT/'actions'/f'{stem}.json',logs);np.savez_compressed(ROOT/'histories'/f'{stem}.npz',**{k:np.stack([c[k] for c in contexts]) for k in contexts[0]},mask=np.arange(7)<len(t['home']));assert r['initial_estimator']['bias']==r['final_estimator']['bias'];return {k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}

def run(worker):
    verify_frozen();_,models,_=broad.setup();broad.previous.ROOT=ROOT;broad.previous.core.ROOT=ROOT;learned=[]
    if worker=='learned':
        for seed in SEEDS:
            m=AnchoredWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'{seed}.pt',weights_only=True));learned.append(m)
    methods=('uniform32','analytical') if worker=='references' else ('learned_final',) if worker=='learned' else ('e120_ensemble',);assert worker in ('references','learned','e120_ensemble');path=ROOT/f'records_{worker}.json';rows=read(path) if path.exists() else [];done={(r['fresh_index'],r['method']) for r in rows}
    for index,t in enumerate(read(ROOT/'cases.json')):
        for method in methods:
            if (index,method) in done:continue
            r=evaluate(index,t,method,models,learned);rows.append(r);temporary=path.with_suffix('.tmp');save(temporary,rows);temporary.replace(path);print('E159 fresh',index,method,r['success'],flush=True)
    assert len(rows)==48*len(methods);verify_frozen()

def summarize():
    verify_frozen();rows=sum([read(ROOT/f'records_{w}.json') for w in ('references','learned','e120_ensemble')],[]);assert len(rows)==192;stats={};paired={};candidate=[r for r in rows if r['method']=='learned_final'];lookup={(r['case'],r['method']):r for r in rows}
    for method in METHODS:
        rr=[r for r in rows if r['method']==method];stats[method]={str(n):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for n in ('all',6,7) if (ss:=[r for r in rr if n=='all' or r['dof']==n])};stats[method]['bodies']={str(body):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for body in sorted(set(r['body'] for r in rr)) if (ss:=[r for r in rr if r['body']==body])}
        if method!='learned_final':
            pairs=[dict(case=r['case'],fresh_index=r['fresh_index'],body=r['body'],dof=r['dof'],before=lookup[r['case'],method]['success'],after=r['success']) for r in candidate];g=sum(r['after'] and not r['before'] for r in pairs);l=sum(r['before'] and not r['after'] for r in pairs);body_diffs=np.array([np.mean([int(r['after'])-int(r['before']) for r in pairs if r['body']==b]) for b in sorted(set(r['body'] for r in pairs))]);rng=np.random.default_rng(159000);boot=rng.choice(body_diffs,(10000,len(body_diffs)),replace=True).mean(1);paired[method]=dict(gains=g,losses=l,net=g-l,no_regressions=l==0,changes=[r for r in pairs if r['before']!=r['after']],body_success_rate_differences=body_diffs.tolist(),body_cluster_bootstrap_95_percentile=np.quantile(boot,[.025,.975]).tolist(),uncertainty_caution='Only four known body clusters; exploratory interval, not evidence of new-morphology generalization.')
    a=stats['learned_final'];positive=all(a['all']['success']>stats[m]['all']['success'] and a['all']['collisions']<=stats[m]['all']['collisions'] and all(a[str(n)]['success']>=stats[m][str(n)]['success'] for n in (6,7)) for m in ('uniform32','e120_ensemble'));save(ROOT/'records.json',rows);save(ROOT/'summary.json',dict(methods=stats,paired=paired,positive_net_pattern=positive,prior_retention_gate_passed=False,amended_prospective_question=True,known_bodies_new_placements=True,unseen_embodiment_test=False,E120_unchanged=True));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:summarize()

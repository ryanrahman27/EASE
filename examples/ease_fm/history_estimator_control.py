"""E144: complete fitting episodes for direct history estimation and matched controls."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .anchored_history_weights import ROOT as LEARNED,AnchoredWeights
from .history_weight_ensemble import ROOT as ENSEMBLE
from .learned_history_weights import token_features,SEEDS
from .expanded_structured_pairs import ROOT as DATA_ROOT,COLLECT
from .abstaining_task_value_gate import read,save,fingerprint,DATA,PHASES
from . import broad_task_value_gate as broad
from .arm_pick_place import ROT_SCALE

ROOT=Path('outputs/ease_history_estimator_control_v1')
METHODS=('analytical','uniform32','constant','e120_ensemble','learned_ensemble')

def prepare():
    assert read(ENSEMBLE/'summary.json')['qualifies_for_control'];assert read(ENSEMBLE/'verification.json')['all_seed_average_reproduced'];assert read(LEARNED/'verification.json')['neural_predictions_reproduced']
    ROOT.mkdir(exist_ok=False)
    for folder in ('trials','states','actions','histories','source','models'):(ROOT/folder).mkdir()
    shutil.copy2(DATA_ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(LEARNED/'reference_choices.json',ROOT/'reference_choices.json');shutil.copy2('outputs/ease_structured_response_v1/folds.json',ROOT/'original_folds.json')
    from .abstaining_task_value_gate import family
    tasks=read(ROOT/'cases.json');oldfolds=read(ROOT/'original_folds.json');save(ROOT/'folds.json',{i:oldfolds[str(family(tasks,i))] for i,t in enumerate(tasks) if not t['validation']})
    shutil.copytree(LEARNED/'models',ROOT/'models',dirs_exist_ok=True);_,_,hashes=broad.setup()
    inputs=[LEARNED/'summary.json',LEARNED/'verification.json',ENSEMBLE/'summary.json',ENSEMBLE/'verification.json',DATA_ROOT/'cases.json',DATA/'collection/records.json',COLLECT/'records.json']
    save(ROOT/'protocol.json',dict(experiment='E144',inputs={str(p):fingerprint(p) for p in inputs},checkpoints=hashes,
        methods='Five fixed controllers on all96 fitting placements: original analytical history (full statistics until close, then local32); uniform32 .98/ridge1e-6; fold-fitted constant kernel; E120 ensemble; learned three-model history-weight ensemble. One actual episode per method/placement,480 new episodes. Do not count deterministic references as three independent seeds. Existing E120 individual-seed results195/288 remain separately reported.',
        ensemble='Learned uses mean of all three E142 Jacobians from the same history; no best-seed selection. E120 ensemble averages its three reference Jacobians and separately its three correction Jacobians, then uses unchanged DLS and E120 bounded residual. This controls for ensembling; it is a new reference, not the old individual-seed E120 score.',
        controller='Same32-step initial calibration and close-phase probe, goals, DLS damping.015 and gain.35, +/- .04 action bounds, frozen measured bias, task success and collision definitions. Hybrid methods replace only the local Jacobian estimator; no correction-off gate or physics selector. All methods consume measured history only. Constant/learned float64 solve uses histories cast to float32 then float64, matching training. Local networks receive no phase, body ID, object/contact state or future data.',
        fitting='Use four family-excluded kernel/neural models according to the original ancestry folds. No per-task fitting, online gradient update or controller hyperparameter search. Save all outcomes, complete states/actions and ordered pre-action histories, including failures. Prediction cache uses exact history/statistic bytes and returns live bias.',
        next_test='Assess fixed learned ensemble overall, both joint counts, collisions, paired rescues/losses versus E120 ensemble and constant, and retention of E120 fitting gain placements1,13,18. A fresh placement test is justified only if overall success exceeds E120 ensemble, neither joint count declines, collisions do not rise, and performance is at least the constant-kernel reference. Report all lost successes; E120 is never overwritten or automatically promoted. If this screen fails, diagnose on fitting data and keep the reserved fresh set closed.',
        limitation='Full estimator replacement can trade successes differently from an isolated gate. Zero-loss improvement remains preferred; any research gain with regressions must be labeled and does not replace E120. Known training bodies, not unseen morphology.'))
    for name in ('history_estimator_control.py','anchored_history_weights.py','learned_history_weights.py','training_estimator_core.py','bounded_history_correction.py','local_history.py','pick_place_control.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',ROOT/'folds.json',ROOT/'reference_choices.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*.py')] if p.is_file()});print('Prepared480 fixed fitting episodes, five controllers',flush=True)

def load_neural():
    models={}
    for f in range(4):
        models[f]=[]
        for seed in SEEDS:
            m=AnchoredWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{f}'/f'{seed}.pt',weights_only=True));models[f].append(m)
    return models

def history(c):
    h=list(c.history);assert len(h)==32;q=np.zeros((32,7),np.float32);x=np.asarray([r[1] for r in h],np.float32);q[:,:c.n]=np.asarray([r[0] for r in h],np.float32);return q,x

def evaluate(case,t,method,models,learned,choice):
    previous=broad.previous;core=previous.core;original_predict=core.LocalPose.predict;original_rollout=core.probe_rollout;cache={};logs=[];contexts=[];correction_ids={id(models[f'correction_{s}']) for s in range(3)}
    def prediction(c):
        if method=='e120_ensemble':
            group='correction' if id(c.model) in correction_ids else 'reference';key=(group,c.n,c.gram.tobytes(),c.cross.tobytes())
            if cache.get(group+'_key')!=key:
                old=c.model;js=[]
                try:
                    for s in range(3):c.model=models[f'{group}_{s}'];js.append(original_predict(c)[0])
                finally:c.model=old
                # Keep both same-history roles cached through controller.update.
                if cache.get(group+'_key')!=key:cache[group+'_key']=key;cache[group+'_j']=np.mean(js,axis=0)
            return cache[group+'_j'].copy(),c.bias
        if method=='analytical':return original_predict(c)
        hq,hx=history(c);key=(hq.tobytes(),hx.tobytes(),c.n)
        if cache.get('key')!=key:
            q=torch.tensor(hq[None],device='cuda');x=torch.tensor(hx[None],device='cuda');mask=torch.arange(7,device='cuda')[None]<c.n
            with torch.no_grad():
                if method=='learned_ensemble':
                    features=token_features(q,x,mask);j=torch.stack([m(q,x,mask,features)[0] for m in learned]).mean(0)
                else:
                    forgetting,ridge=(.98,1e-6) if method=='uniform32' else (choice['forgetting'],choice['ridge']);w=forgetting**torch.arange(31,-1,-1,device='cuda',dtype=torch.float64);qq,xx=q.double(),x.double();g=qq.transpose(1,2)@(qq*w[None,:,None])+ridge*torch.eye(7,device='cuda',dtype=torch.float64)[None];cross=qq.transpose(1,2)@(xx*w[None,:,None]);j=torch.linalg.solve(g,cross)
            cache['key']=key;cache['j']=j[0,:c.n].cpu().numpy().T.copy()
        return cache['j'].copy(),c.bias
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            before=env.observe();phase=env.phase;clean=not env.bad;hq,hx=history(c);g=np.zeros((7,7));g[:c.n,:c.n]=c.gram;cross=np.zeros((7,6));cross[:c.n]=c.cross
            after,command,info=original_step(action,aperture);dq=np.zeros(7);dq[:c.n]=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
            contexts.append(dict(dq_history=hq,dx_history=hx,query=dq.astype(np.float32),target=dx.astype(np.float32),gram=g.astype(np.float32),cross=cross.astype(np.float32),clean=clean,phase=(*PHASES,'probe').index(phase)))
            logs.append(dict(phase=phase,q_before=before['q'].tolist(),position_before=before['position'].tolist(),nominal_action=np.asarray(action).tolist(),executed_command=command.tolist(),aperture=float(aperture),bad=info['bad']));return after,command,info
        env.step=step;return original_rollout(env,home,c,**kwargs)
    with patch.object(core.LocalPose,'predict',prediction),patch.object(core,'probe_rollout',observer):
        if method=='e120_ensemble':r,_=previous.evaluate(case,t,method,models['reference_0'],models['correction_0'],'bounded',save=True)
        else:r=core.evaluate(case,t,method,None,'frozen_bias',save_states=True)
    stem=f'{case}_{method}_frozen_bias';save(ROOT/'trials'/f'{stem}.json',r);save(ROOT/'actions'/f'{stem}.json',logs);arrays={k:np.stack([c[k] for c in contexts]) for k in contexts[0]};np.savez_compressed(ROOT/'histories'/f'{stem}.npz',**arrays,mask=np.arange(7)<len(t['home']))
    source=DATA/'collection' if case<56 else COLLECT;baseline=read(source/'trials'/f'{case}_on_0_frozen_bias.json');assert r['calibration']==baseline['calibration'] and r['initial_estimator']==baseline['initial_estimator']
    return {k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}

def run(worker):
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    _,models,hashes=broad.setup();broad.previous.ROOT=ROOT;broad.previous.core.ROOT=ROOT;tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');choices=read(ROOT/'reference_choices.json');neural=load_neural() if worker=='learned' else {};methods=('analytical','uniform32') if worker=='references' else ('learned_ensemble',) if worker=='learned' else (worker,);assert all(m in METHODS for m in methods);path=ROOT/f'records_{worker}.json';rows=read(path) if path.exists() else [];done={(r['case'],r['method']) for r in rows}
    for case,t in enumerate(tasks):
        if t['validation']:continue
        f=folds[str(case)]
        for method in methods:
            if (case,method) in done:continue
            r=evaluate(case,t,method,models,neural.get(f),choices[f]);rows.append(r);temp=path.with_suffix('.tmp');save(temp,rows);temp.replace(path);print('E144',worker,case,method,r['success'],flush=True)
    assert len(rows)==96*len(methods)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h

def summarize():
    rows=sum([read(ROOT/f'records_{w}.json') for w in ('references','constant','e120_ensemble','learned')],[]);assert len(rows)==480;lookup={(r['case'],r['method']):r for r in rows};summary={}
    for method in METHODS:
        summary[method]={}
        for dof in ('all',6,7):
            rr=[r for r in rows if r['method']==method and (dof=='all' or r['dof']==dof)];summary[method][str(dof)]=dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr))
    changes={}
    for base in ('e120_ensemble','constant','analytical','uniform32'):
        pairs=[dict(case=r['case'],dof=r['dof'],before=lookup[r['case'],base]['success'],after=r['success']) for r in rows if r['method']=='learned_ensemble'];changes[base]=dict(gains=sum(r['after'] and not r['before'] for r in pairs),losses=sum(r['before'] and not r['after'] for r in pairs),changes=[r for r in pairs if r['before']!=r['after']])
    l,b,c=summary['learned_ensemble'],summary['e120_ensemble'],summary['constant'];retained={str(case):lookup[case,'learned_ensemble']['success'] for case in (1,13,18)};qualified=l['all']['success']>b['all']['success'] and l['all']['success']>=c['all']['success'] and all(l[str(n)]['success']>=b[str(n)]['success'] for n in (6,7)) and l['all']['collisions']<=b['all']['collisions'] and all(retained.values())
    save(ROOT/'records.json',rows);save(ROOT/'summary.json',dict(methods=summary,paired=changes,E120_fitting_gain_placements=retained,qualifies_for_fresh_test=bool(qualified),fresh_evaluated=False,E120_unchanged=True));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:{'prepare':prepare,'summarize':summarize}[sys.argv[1]]()

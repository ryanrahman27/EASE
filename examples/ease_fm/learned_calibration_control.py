"""E147: fixed learned calibration-prior estimator on full fitting episodes."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import history_estimator_control as old
from . import calibration_prior_learning as learning
from .calibration_prior_control import ROOT as ANALYTICAL
from .abstaining_task_value_gate import read,save,fingerprint,PHASES
from .arm_pick_place import ROT_SCALE
ROOT=Path('outputs/ease_learned_calibration_control_v1')

def prepare():
    assert read(learning.ROOT/'summary.json')['qualifies_for_control'];assert read(learning.ROOT/'verification.json')['all_predictions_exact'];ROOT.mkdir(exist_ok=False)
    for folder in ('source','trials','states','actions','histories'):(ROOT/folder).mkdir()
    shutil.copy2(old.ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(old.ROOT/'folds.json',ROOT/'folds.json');shutil.copytree(learning.ROOT/'models',ROOT/'models');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);shutil.copy2(Path(learning.__file__),ROOT/'source'/Path(learning.__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E147',inputs={str(learning.ROOT/p):fingerprint(learning.ROOT/p) for p in ('summary.json','verification.json','frozen.json')},controller='Fixed mean of all three E146 Jacobians per excluded-family fold. Measured initial calibration prior, relative learned history weights, .98 recency,1e-6 ridge. Same task, probe, DLS, action bounds and frozen bias as E145. No checkpoint/seed selection or online gradient update. No physics selector or privileged runtime inputs.',evaluation='All96 fitting placements, known bodies. Compare E145 analytical prior and E144 constant/analytical/uniform/E120 ensemble; also retain original E120195/288 individual-seed rate separately. Save all outcomes and ordered histories. No new held or fresh placements.',advancement='Require more successes than E120 ensemble, at least as many as analytical prior and constant, no lower success in either joint count than E120 ensemble, no more collision episodes, and retain fitting gain placements1,13,18. Report every lost success; zero-loss improvement preferred and no automatic E120 replacement. Only a passing fixed controller warrants reserved fresh evaluation.'))
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',ROOT/'folds.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*')] if p.is_file()})

def run():
    _,_,hashes=old.broad.setup();core=old.broad.previous.core;core.ROOT=ROOT;original_rollout=core.probe_rollout;folds=read(ROOT/'folds.json');models={}
    for f in range(4):
        models[f]=[]
        for seed in learning.SEEDS:
            m=learning.PriorWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{f}'/f'{seed}.pt',weights_only=True));models[f].append(m)
    path=ROOT/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows}
    for case,t in enumerate(read(ROOT/'cases.json')):
        if t['validation'] or case in done:continue
        cache={};logs=[];contexts=[]
        def predict(c):
            if 'prior' not in cache:
                prior=np.zeros((7,6));prior[:c.n]=np.linalg.solve(c.gram,c.cross);cache['prior']=prior;cache['prior_tensor']=torch.tensor(prior[None],device='cuda')
            q,x=old.history(c);key=(q.tobytes(),x.tobytes())
            if cache.get('key')!=key:
                qt=torch.tensor(q[None],device='cuda');xt=torch.tensor(x[None],device='cuda');mask=torch.arange(7,device='cuda')[None]<c.n
                with torch.no_grad():
                    feature=learning.features(qt,xt,mask,cache['prior_tensor']);j=torch.stack([m(qt,xt,mask,cache['prior_tensor'],feature)[0] for m in models[folds[str(case)]]]).mean(0)
                cache['j']=j[0,:c.n].cpu().numpy().T.copy();cache['key']=key
            return cache['j'].copy(),c.bias
        def observer(env,home,c,**kwargs):
            original_step=env.step
            def step(action,aperture):
                before=env.observe();q,x=old.history(c);clean=not env.bad;phase=env.phase;after,command,info=original_step(action,aperture);dq=np.zeros(7,np.float32);dq[:c.n]=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])].astype(np.float32);contexts.append(dict(dq_history=q,dx_history=x,query=dq,target=dx,clean=clean,phase=(*PHASES,'probe').index(phase)));logs.append(dict(phase=phase,q_before=before['q'].tolist(),nominal_action=np.asarray(action).tolist(),executed_command=command.tolist(),bad=info['bad']));return after,command,info
            env.step=step;return original_rollout(env,home,c,**kwargs)
        with patch.object(core.LocalPose,'predict',predict),patch.object(core,'probe_rollout',observer):r=core.evaluate(case,t,'learned_prior',None,'frozen_bias',save_states=True)
        reference=read(ANALYTICAL/'trials'/f'{case}_calibration_prior_frozen_bias.json');assert reference['calibration']==r['calibration'] and reference['initial_estimator']==r['initial_estimator'];assert r['initial_estimator']['bias']==r['final_estimator']['bias'];np.testing.assert_array_equal(cache['prior'][:len(t['home'])],np.linalg.solve(r['initial_estimator']['gram'],r['initial_estimator']['cross']))
        stem=f'{case}_learned_prior_frozen_bias';save(ROOT/'trials'/f'{stem}.json',r);save(ROOT/'actions'/f'{stem}.json',logs);np.savez_compressed(ROOT/'histories'/f'{stem}.npz',**{k:np.stack([c[k] for c in contexts]) for k in contexts[0]});rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')});save(path,rows);print('E147',case,r['success'],flush=True)
    assert len(rows)==96
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(calibration_and_initial_statistics_identical=True,prior_is_actual_initial_measured_solve=True,frozen_bias=True,source_and_E120_hashes_unchanged=True))

def summarize():
    rows=read(ROOT/'records.json');assert len(rows)==96;refs=read(old.ROOT/'records.json')+read(ANALYTICAL/'records.json');summary={};paired={}
    for method in (*old.METHODS,'calibration_prior','learned_prior'):
        rr=[r for r in refs+rows if r['method']==method];summary[method]={str(n):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for n in ('all',6,7) if (ss:=[r for r in rr if n=='all' or r['dof']==n])}
        if method=='learned_prior':continue
        pairs=[dict(case=r['case'],dof=r['dof'],before=next(s['success'] for s in rr if s['case']==r['case']),after=r['success']) for r in rows];paired[method]=dict(gains=sum(r['after'] and not r['before'] for r in pairs),losses=sum(r['before'] and not r['after'] for r in pairs),changes=[r for r in pairs if r['before']!=r['after']])
    l,b=summary['learned_prior'],summary['e120_ensemble'];retained={str(c):next(r['success'] for r in rows if r['case']==c) for c in (1,13,18)};qualified=l['all']['success']>b['all']['success'] and all(l['all']['success']>=summary[m]['all']['success'] for m in ('constant','calibration_prior')) and all(l[str(n)]['success']>=b[str(n)]['success'] for n in (6,7)) and l['all']['collisions']<=b['all']['collisions'] and all(retained.values());save(ROOT/'summary.json',dict(methods=summary,paired=paired,gain_retention=retained,qualifies_for_fresh_test=qualified,E120_unchanged=True,fresh_evaluated=False));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run,'summarize':summarize}[sys.argv[1]]()

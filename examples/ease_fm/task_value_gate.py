"""E124: fitting-only expanded paired outcomes and one-pulse learned task value."""
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from . import matched_correction as paired
from .matched_correction import PHASES,fingerprint
from . import bounded_history_correction as previous
from .arm_pick_place import ROT_SCALE,rotation_error
ROOT=Path('outputs/ease_task_value_gate_v1')
COLLECT=ROOT/'collection'
OLD=Path('outputs/ease_bounded_history_v1')
REUSE=Path('outputs/ease_matched_correction_v1')

def setup(collection=False):
    paired.ROOT=COLLECT
    tasks,models,hashes=paired.setup()
    previous.ROOT=COLLECT if collection else ROOT
    previous.core.ROOT=previous.ROOT
    return tasks,models,hashes

def features(n,g,c,bias,q,position,goal,phase,local,a,b):
    pad=lambda x:np.pad(np.asarray(x),(0,7-n))
    gg=np.zeros((7,7));gg[:n,:n]=g;cc=np.zeros((7,6));cc[:n]=c
    x=np.r_[np.eye(len(PHASES))[PHASES.index(phase)],np.arange(7)<n,pad(q),pad(bias),np.asarray(goal)-position,gg.ravel(),cc.ravel(),pad(a),pad(b),float(local)].astype(np.float32)
    assert x.shape==(139,) and np.isfinite(x).all()
    return x

class Gate(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(139,64),nn.Tanh(),nn.Linear(64,32),nn.Tanh(),nn.Linear(32,2))
    def forward(self,x):return self.net(x)

def prepare():
    tasks,models,hashes=setup();ROOT.mkdir(exist_ok=False);COLLECT.mkdir()
    for root in (ROOT,COLLECT):
        for name in ('trials','states','actions','source','branches','snapshots'):(root/name).mkdir()
    shutil.copy2(OLD/'cases.json',ROOT/'cases.json');pairs=[];reuse_rows=[]
    oldrows=json.loads((OLD/'records.json').read_text());lookup={(r['case'],r['method']):r for r in oldrows}
    for case,t in enumerate(tasks):
        if t['validation']:continue
        for seed in range(3):pairs.append(dict(case=case,seed=seed,e120_gain=lookup[case,f'candidate_{seed}']['success'] and not lookup[case,f'reference_{seed}']['success'],group=t['group']))
    assert len(pairs)==48
    for r in json.loads((REUSE/'records.json').read_text()):
        if tasks[r['case']]['validation']:continue
        stem=f"{r['case']}_{r['method']}_frozen_bias"
        for folder,ext in (('trials','json'),('states','npz'),('states','xml'),('actions','json')):shutil.copy2(REUSE/folder/f'{stem}.{ext}',COLLECT/folder/f'{stem}.{ext}')
        shutil.copy2(REUSE/'branches'/f"{r['case']}_{r['method']}.json",COLLECT/'branches'/f"{r['case']}_{r['method']}.json")
        for p in (REUSE/'snapshots').glob(f"{r['case']}_{r['method']}_*.npz"):shutil.copy2(p,COLLECT/'snapshots'/p.name)
        reuse_rows.append(r)
    assert len(reuse_rows)==70
    (COLLECT/'reused_records.json').write_text(json.dumps(reuse_rows,indent=2));(COLLECT/'pairs.json').write_text(json.dumps(pairs,indent=2))
    rows=[]
    for r in oldrows:
        if r['method'].startswith('unbounded_'):continue
        source=r['method'];method=source.replace('candidate_','unbounded_');r['method']=method;a=f"{r['case']}_{source}_frozen_bias";b=f"{r['case']}_{method}_frozen_bias"
        trial=json.loads((OLD/'trials'/f'{a}.json').read_text());trial['method']=method;(ROOT/'trials'/f'{b}.json').write_text(json.dumps(trial))
        for ext in ('npz','xml'):shutil.copy2(OLD/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    assert len(rows)==168;(ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    spec=dict(experiment='E124',checkpoints=hashes,inputs={str(OLD/n):fingerprint(OLD/n) for n in ('records.json','summary.json','cases.json')},reused_inputs={str(REUSE/'records.json'):fingerprint(REUSE/'records.json')},
        collection='All16 fitting placements x3 frozen estimator seeds =48 pairs. Nine independent phase-entry off-pulses plus on =480 trajectories. Reuse70 E123 fitting trajectories only, generate410. Assert exact E120 on traces/states and all432 intervention prefixes. Preserve contaminated branches but exclude any with prior forbidden contact from fitting. No held trajectory or outcome in fitting.',
        features='139 observable dimensions: phase one-hot, joint mask, measured padded q/bias, current position-goal error, Gram/cross history statistics, both proposed clipped DLS actions, local-history flag. No case/body/seed ID, XML, integration state, collision flags, outcome, future rollout, object oracle, or held-derived normalization.',
        model='One shared139-64-32-2 tanh MLP predicts success of correction-on and one8-step off-pulse followed by E120. Standardization on fitting rows only. Fixed seed124000, AdamW learning rate0.001 weight decay0.001, full-batch BCE,1500 updates. No held-based checkpoint/threshold selection. Keep E118/E119 estimators frozen.',
        inference='At each phase entry before any prior intervention, choose one off-pulse only if predicted P(success|off)-P(success|on)>0.10. Up to8 steps within that phase, then E120 permanently resumes. At most one pulse per episode, matching label continuation. No physics selector or inference simulator branching.',
        evaluation='All24 placements x3 estimator seeds =72 episodes. Report E120 gains/failures, E118 retention and held placements separately. Do not promote if earlier successes are lost. No unseen bodies.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for n in ('task_value_gate.py','matched_correction.py','bounded_history_correction.py','training_estimator_core.py','local_probe.py','local_history.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)
    print('Prepared48 fitting pairs; reused70 episodes',flush=True)

def collect(seed):
    tasks,models,hashes=setup(True);pairs=[p for p in json.loads((COLLECT/'pairs.json').read_text()) if p['seed']==seed]
    progress=COLLECT/f'records_seed_{seed}.json'
    rows=json.loads(progress.read_text()) if progress.exists() else [r for r in json.loads((COLLECT/'reused_records.json').read_text()) if r['seed']==seed]
    done={(r['case'],r['off_phase']) for r in rows}
    # Prefix replay repeats exactly the same frozen neural estimates. Cache only
    # by model identity and exact statistic bytes; bias is returned live.
    original_predict=previous.core.LocalPose.predict;cache={}
    def cached_predict(c):
        if c.model is None:return original_predict(c)
        key=(id(c.model),c.n,c.gram.tobytes(),c.cross.tobytes())
        if key not in cache:cache[key]=original_predict(c)[0].copy()
        return cache[key].copy(),c.bias
    predictor_patch=patch.object(previous.core.LocalPose,'predict',cached_predict);predictor_patch.start()
    for p in pairs:
        cache.clear()
        case=p['case']
        for phase in (None,*PHASES):
            if (case,phase) in done:continue
            r,b=paired.evaluate(case,tasks[case],seed,models[f'reference_{seed}'],models[f'correction_{seed}'],phase)
            if phase is None:
                old=json.loads((OLD/'trials'/f'{case}_candidate_{seed}_frozen_bias.json').read_text());old['method']=r['method'];assert r==old
                np.testing.assert_array_equal(np.load(COLLECT/'states'/f"{case}_{r['method']}_frozen_bias.npz")['states'],np.load(OLD/'states'/f'{case}_candidate_{seed}_frozen_bias.npz')['states'])
            rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(seed=seed,off_phase=phase,e120_gain=p['e120_gain']))
            temp=progress.with_suffix('.tmp');temp.write_text(json.dumps(rows,indent=2));temp.replace(progress)
        print('collected',seed,case,'episodes',len(rows),flush=True)
    assert len(rows)==160
    predictor_patch.stop();progress.write_text(json.dumps(rows,indent=2))
    for p,h in hashes.items():assert fingerprint(p)==h

def train():
    torch.set_num_threads(4);tasks,models,hashes=setup();rows=sum([json.loads((COLLECT/f'records_seed_{s}.json').read_text()) for s in range(3)],[]);assert len(rows)==480
    (COLLECT/'records.json').write_text(json.dumps(rows,indent=2));lookup={(r['case'],r['seed'],r['off_phase']):r for r in rows};xs=[];ys=[];meta=[];checks=0;excluded=[]
    for p in json.loads((COLLECT/'pairs.json').read_text()):
        case,seed=p['case'],p['seed'];assert not tasks[case]['validation'];on=json.loads((COLLECT/'branches'/f'{case}_on_{seed}.json').read_text());actions=json.loads((COLLECT/'actions'/f'{case}_on_{seed}_frozen_bias.json').read_text())
        for phase in PHASES:
            off=json.loads((COLLECT/'branches'/f'{case}_off_{phase}_{seed}.json').read_text());i=off['intervention'];assert on['signatures'][:i+1]==off['signatures'][:i+1];a=np.load(COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz');b=np.load(COLLECT/'snapshots'/f'{case}_off_{phase}_{seed}_{phase}.npz')
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
            checks+=1
            if not on['metrics'][i]['clean_before']:excluded.append(dict(case=case,seed=seed,phase=phase));continue
            action=actions[i];n=len(action['q_before']);x=features(n,a['gram'],a['cross'],a['bias'],action['q_before'],action['position_before'],a['goal'],phase,on['entries'][phase]['metadata']['local'],action['base_action'],action['correction_action'])
            y=[float(lookup[case,seed,None]['success']),float(lookup[case,seed,phase]['success'])];xs.append(x);ys.append(y);meta.append(dict(case=case,seed=seed,phase=phase,signature=on['signatures'][i],labels=y))
    x=np.stack(xs);y=np.array(ys,np.float32);mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);np.savez_compressed(ROOT/'gate_dataset.npz',x=x,y=y,mean=mean,scale=scale);(ROOT/'gate_rows.json').write_text(json.dumps(meta,indent=2))
    torch.manual_seed(124000);model=Gate();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);xx=torch.tensor(np.clip((x-mean)/scale,-10,10));yy=torch.tensor(y);losses=[]
    for step in range(1500):
        logits=model(xx);loss=nn.functional.binary_cross_entropy_with_logits(logits,yy);opt.zero_grad();loss.backward();opt.step();losses.append(float(loss.detach()))
    model.eval();torch.save(model.state_dict(),ROOT/'gate.pt')
    with torch.no_grad():prob=model(xx).sigmoid().numpy()
    off=prob[:,1]-prob[:,0]>.1
    (ROOT/'training.json').write_text(json.dumps(dict(rows=len(x),features=139,independent_pairs=48,prefix_snapshot_checks=checks,excluded_contaminated=excluded,steps=1500,seed=124000,losses=losses,training_head_accuracy=float(np.mean((prob>.5)==y)),training_switches=int(off.sum()),training_switch_positive_advantage=int(np.sum(off&(y[:,1]>y[:,0]))),training_switch_negative_advantage=int(np.sum(off&(y[:,1]<y[:,0]))),policy_sha256=fingerprint(ROOT/'gate.pt'),dataset_sha256=fingerprint(ROOT/'gate_dataset.npz'),held_rows=0),indent=2))
    print('trained',len(x),'clean rows; switches',int(off.sum()),flush=True)

def load_gate():
    m=Gate();m.load_state_dict(torch.load(ROOT/'gate.pt',weights_only=True));m.eval();f=np.load(ROOT/'gate_dataset.npz');return m,f['mean'],f['scale']

def evaluate(case,t,seed,models,gate,mean,scale,save=True):
    method=f'candidate_{seed}';original_rollout=previous.core.probe_rollout;original_action=previous.bounded_action;counts={};chosen=None;off=False;decisions=[];step_index=0
    def bounded(a,b):return a.copy() if off else original_action(a,b)
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            nonlocal chosen,off,step_index
            phase=env.phase;k=counts.get(phase,0);counts[phase]=k+1
            if phase in PHASES and k==0 and chosen is None:
                obs=env.observe();old=c.model
                try:c.model=models[f'correction_{seed}'];j,bias=c.predict()
                finally:c.model=old
                error=np.r_[env.goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])];proposal=np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
                x=features(c.n,c.gram,c.cross,c.bias,obs['q'],obs['position'],env.goal,phase,c.local,action,proposal)
                with torch.no_grad():prob=gate(torch.tensor(np.clip((x-mean)/scale,-10,10))).sigmoid().numpy()
                selected=bool(prob[1]-prob[0]>.1)
                decisions.append(dict(step=step_index,phase=phase,features=x.tolist(),on_value=float(prob[0]),off_value=float(prob[1]),selected=selected))
                if selected:chosen=phase
            off=phase==chosen and k<8;step_index+=1
            return original_step(action,aperture)
        env.step=step
        return original_rollout(env,home,c,**kwargs)
    with patch.object(previous.core,'probe_rollout',observer),patch.object(previous,'bounded_action',bounded):r,logs=previous.evaluate(case,t,method,models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded',save=save)
    if save:(ROOT/'actions'/f'{case}_{method}_gate.json').write_text(json.dumps(dict(chosen=chosen,decisions=decisions)))
    return r,dict(chosen=chosen,decisions=decisions)

def run():
    tasks,models,hashes=setup();gate,mean,scale=load_gate();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==168
    for case,t in enumerate(tasks):
        for seed in range(3):
            r,d=evaluate(case,t,seed,models,gate,mean,scale);old=json.loads((ROOT/'trials'/f'{case}_unbounded_{seed}_frozen_bias.json').read_text());assert r['calibration']==old['calibration'] and r['initial_estimator']==old['initial_estimator']
            if d['chosen'] is None:
                old['method']=r['method'];assert old==r
            rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(gate_phase=d['chosen']))
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('evaluated',case,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    cmd=sys.argv[1]
    if cmd=='collect':collect(int(sys.argv[2]))
    else:{'prepare':prepare,'train':train,'evaluate':run}[cmd]()

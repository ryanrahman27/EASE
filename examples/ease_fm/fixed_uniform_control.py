"""E154: learned fixed-uniform weights, with and without fixed posture objective."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .fixed_uniform_learning import ROOT as MODEL
from . import history_estimator_control as old
from . import home_posture_control as posture
from .arm_pick_place import PickPlace
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_fixed_uniform_control_v1')

def prepare():
    assert read(MODEL/'summary.json')['qualifies_for_control'];assert read(MODEL/'verification.json')['all_predictions_exact'];ROOT.mkdir(exist_ok=False);(ROOT/'source').mkdir();shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    save(ROOT/'protocol.json',dict(experiment='E154',design='Fixed E153 three-model average, four excluded-family models, all96 fitting placements. Compare learned relative weights to identical uniform32 kernel and E120 ensemble under both original controller and fixed E151 home-posture objective. The posture condition is primary; standard control is a disclosed factorial comparison. No training or coefficient sweep during rollout.',isolation='Same.98recency, zero-centered1e-6ridge,32measured transitions, mass normalization and DLS/task/probe/frozen-bias settings. Only relative history weighting is learned. No calibration-prior solve term, prior features, simulator Jacobian, contact input or physics selector. Explicit weighted analytical solve remains part of architecture.',screen='For either fixed controller context: learned success must exceed both matched uniform32 and E120 ensemble, neither joint-count success may decline against either reference, collisions may not exceed either reference, and fitting gain placements1,13,18 must succeed. Report all paired rescues/losses; zero-loss improvement preferred and no automatic baseline replacement. If both pass, choose primary posture; otherwise choose the sole passing context. If neither passes, fresh stays closed. Selection is declared before these outcomes.',inputs={str(MODEL/p):fingerprint(MODEL/p) for p in ('summary.json','verification.json','frozen.json')}))
    for context in ('standard','posture'):
        root=ROOT/context;root.mkdir()
        for folder in ('trials','states','actions','histories','source','posture'):(root/folder).mkdir()
        shutil.copy2(MODEL/'cases.json',root/'cases.json');shutil.copy2(old.ROOT/'folds.json',root/'folds.json');shutil.copytree(MODEL/'models',root/'models');save(root/'reference_choices.json',[dict(forgetting=.98,ridge=1e-6) for _ in range(4)]);shutil.copy2(ROOT/'protocol.json',root/'protocol.json');shutil.copy2(Path(old.__file__),root/'source'/Path(old.__file__).name)
        save(root/'frozen.json',{str(p):fingerprint(p) for p in [root/'protocol.json',root/'cases.json',root/'folds.json',root/'reference_choices.json',*(root/'models').rglob('*'),*(root/'source').glob('*')] if p.is_file()})
    save(ROOT/'frozen.json',{str(ROOT/p):fingerprint(ROOT/p) for p in ('protocol.json','source/fixed_uniform_control.py')})

def run(context):
    assert context in ('standard','posture');root=ROOT/context;episodes=[];core=old.broad.previous.core;original_rollout=core.probe_rollout
    class PostureArm(PickPlace):
        def step(self,action,aperture):
            c=getattr(self,'posture_controller',None)
            if c is not None:
                original=np.asarray(action).copy();delta=np.zeros(self.n);j=None
                if self.phase!='probe':
                    j,_=c.predict();q=self.observe()['q'];projector=np.eye(self.n)-j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),j);delta=projector@(.05*(self.posture_home-q));delta*=min(1.,.01/max(float(np.max(np.abs(delta))),1e-12));action=np.clip(original+delta,-.04,.04);assert np.max(np.abs(action-original))<=.010000000001
                self.posture_logs.append(dict(phase=self.phase,base_action=original.tolist(),posture_delta=delta.tolist(),final_action=np.asarray(action).tolist(),task_space_delta=None if j is None else (j@delta).tolist()))
            return super().step(action,aperture)
    def rollout(env,home,c,**kwargs):
        env.posture_controller=c;env.posture_home=np.asarray(home);env.posture_logs=[];r=original_rollout(env,home,c,**kwargs);episodes.append(env.posture_logs);return r
    a=Path(posture.__file__).read_text();b=Path(__file__).read_text();start="            c=getattr(self,'posture_controller',None)";end='            return super().step(action,aperture)';assert a[a.index(start):a.index(end)]==b[b.index(start):b.index(end)]
    with patch.object(old,'ROOT',root):
        if context=='posture':
            with patch.object(core,'PickPlace',PostureArm),patch.object(core,'probe_rollout',rollout):old.run('learned')
        else:old.run('learned')
    rows=read(root/'records_learned.json');assert len(rows)==96;save(root/'records.json',rows)
    if context=='posture':
        assert len(episodes)==96
        for r,log in zip(rows,episodes):save(root/'posture'/f"case_{r['case']}.json",log);assert all(np.array_equal(a['base_action'],a['final_action']) for a in log if a['phase']=='probe')
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(root/'verification.json',dict(episodes=96,unchanged_models_and_sources=True,posture_rule_identical_to_E151=context=='posture',calibration_unchanged=True));print('E154',context,sum(r['success'] for r in rows),'/96',flush=True)

def summarize():
    summary={};eligible=[]
    for context in ('standard','posture'):
        rows=read(ROOT/context/'records.json');refs=read(old.ROOT/'records.json') if context=='standard' else read(posture.ROOT/'records.json');methods={};paired={}
        for method in ('uniform32','e120_ensemble','learned_fixed_uniform'):
            rr=rows if method=='learned_fixed_uniform' else [r for r in refs if r['method']==method];methods[method]={str(n):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for n in ('all',6,7) if (ss:=[r for r in rr if n=='all' or r['dof']==n])}
            if method!='learned_fixed_uniform':
                pairs=[dict(case=r['case'],dof=r['dof'],before=next(v['success'] for v in rr if v['case']==r['case']),after=r['success']) for r in rows];paired[method]=dict(gains=sum(r['after'] and not r['before'] for r in pairs),losses=sum(r['before'] and not r['after'] for r in pairs),changes=[r for r in pairs if r['before']!=r['after']])
        a=methods['learned_fixed_uniform'];retained={str(c):next(r['success'] for r in rows if r['case']==c) for c in (1,13,18)};qualified=all(a['all']['success']>methods[m]['all']['success'] and a['all']['collisions']<=methods[m]['all']['collisions'] and all(a[str(n)]['success']>=methods[m][str(n)]['success'] for n in (6,7)) for m in ('uniform32','e120_ensemble')) and all(retained.values());summary[context]=dict(methods=methods,paired=paired,gain_retention=retained,qualifies=qualified)
        if qualified:eligible.append(context)
    selected='posture' if 'posture' in eligible else eligible[0] if eligible else None;save(ROOT/'summary.json',dict(contexts=summary,selected=selected,qualifies_for_fresh_test=bool(eligible),E120_unchanged=True,fresh_evaluated=False));print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':run(sys.argv[2])
    else:{'prepare':prepare,'summarize':summarize}[sys.argv[1]]()

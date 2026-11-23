"""E158: isolate posture-map quality from the primary task-motion estimator."""
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .conservative_history_weights import ROOT as CONSERVATIVE,ConservativeWeights
from .learned_history_weights import SEEDS
from . import history_estimator_control as old
from .fixed_uniform_learning import ROOT as MODEL
from .arm_pick_place import PickPlace
from .abstaining_task_value_gate import read,save,fingerprint
ROOT=Path('outputs/ease_conservative_lift_control_v1')
METHODS=('uniform32','learned_ensemble','e120_ensemble')
DIAGNOSTIC_CASES=(13,1,6,10)

def prepare():
    assert read(CONSERVATIVE/'summary.json')['qualifies_for_diagnostic']
    ROOT.mkdir(exist_ok=False)
    for folder in ('source','trials','states','actions','histories','posture'):(ROOT/folder).mkdir()
    shutil.copy2(old.ROOT/'cases.json',ROOT/'cases.json');shutil.copy2(old.ROOT/'folds.json',ROOT/'folds.json');shutil.copytree(MODEL/'models',ROOT/'models');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);shutil.copy2(Path(__file__).with_name('conservative_history_weights.py'),ROOT/'source/conservative_history_weights.py')
    save(ROOT/'protocol.json',dict(experiment='E158',hypothesis='E154 standard learned weights improve77/96 vs uniform74/E12072 but regress cases. E156 lift-only projection rescues case13 with uniform, not full learned weights. E157 attenuates learned log weights by fixed.25 and renormalizes mass, retaining lower prediction error. Test this one conservative model under the unchanged lift-only rule; no temperature sweep or new weight training. Compare every pre-lift state with an actual matched conservative standard rollout.',rule='Only during lift, use the fixed E151 home objective.05(home-q),.015damping,maxabs.01rad and final+/-.04 clamp. Projector map is current32 observed transitions,.98recency,1e-6ridge toward initial measured calibration. All other phases execute the unmodified primary task controller. Primary maps are uniform32, E157 conservative learned fixed-uniform ensemble, E120 ensemble. No privileged geometry, simulator Jacobian or physics selector. No coefficient sweep.',diagnostic='Outcome-informed fitting cases13(persistent six-joint loss),1(previous posture regression),6(seven-joint success),10(seven-joint failure), all three methods. These12 episodes are diagnostic, not representative rates. Continue to all96 fitting placements only if learned succeeds on13,1,6; retain case10 regardless. No coefficient or checkpoint tuning. This diagnostic gate is declared before new-projector outcomes.',full_gate='If full fitting proceeds, learned must exceed both matched uniform and E120 ensembles overall, neither joint count decline, no higher collision count, and retain1,13,18. Report every regression. Otherwise no fresh test. E120 immutable.',inputs={str(CONSERVATIVE/p):fingerprint(CONSERVATIVE/p) for p in ('summary.json','verification.json','frozen.json')}))
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',ROOT/'folds.json',*(ROOT/'models').rglob('*'),*(ROOT/'source').glob('*')] if p.is_file()})

def run(stage):
    assert stage in ('diagnostic','full');path=ROOT/'records.json';rows=read(path) if path.exists() else [];done={(r['case'],r['method']) for r in rows}
    if stage=='full':assert read(ROOT/'diagnostic_summary.json')['ready_for_full']
    tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');_,models,hashes=old.broad.setup();core=old.broad.previous.core;old.broad.previous.ROOT=ROOT;core.ROOT=ROOT;original_rollout=core.probe_rollout
    neural=load_neural()
    class PostureArm(PickPlace):
        def step(self,action,aperture):
            c=getattr(self,'posture_controller',None)
            if c is not None:
                original=np.asarray(action).copy();delta=np.zeros(self.n)
                if self.phase=='lift':
                    q,x=old.history(c);q=q[:,:self.n].astype(float);x=x.astype(float);w=.98**np.arange(31,-1,-1);g=q.T@(q*w[:,None])+1e-6*np.eye(self.n);cross=q.T@(x*w[:,None])+1e-6*self.calibration_prior;j=np.linalg.solve(g,cross).T;projector=np.eye(self.n)-j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),j);delta=projector@(.05*(self.posture_home-self.observe()['q']));delta*=min(1.,.01/max(float(np.max(np.abs(delta))),1e-12));action=np.clip(original+delta,-.04,.04)
                self.posture_logs.append(dict(phase=self.phase,base_action=original.tolist(),posture_delta=delta.tolist(),final_action=np.asarray(action).tolist()))
            return super().step(action,aperture)
    selected=DIAGNOSTIC_CASES if stage=='diagnostic' else [i for i,t in enumerate(tasks) if not t['validation']]
    for case in selected:
        assert not tasks[case]['validation']
        for method in METHODS:
            if (case,method) in done:continue
            saved=[]
            def rollout(env,home,c,**kwargs):
                env.posture_controller=c;env.posture_home=np.asarray(home);env.calibration_prior=np.linalg.solve(c.gram,c.cross);env.posture_logs=[];r=original_rollout(env,home,c,**kwargs);saved.extend(env.posture_logs);return r
            with patch.object(old,'ROOT',ROOT),patch.object(core,'PickPlace',PostureArm),patch.object(core,'probe_rollout',rollout):r=old.evaluate(case,tasks[case],method,models,neural[folds[str(case)]],dict(forgetting=.98,ridge=1e-6))
            trial=read(ROOT/'trials'/f'{case}_{method}_frozen_bias.json');base_root=ROOT/'standard' if method=='learned_ensemble' else Path('outputs/ease_history_estimator_control_v1');baseline=read(base_root/'trials'/f'{case}_{method}_frozen_bias.json');prefix=lambda trace:trace[:next(i for i,x in enumerate(trace) if x['phase']=='lift')];assert prefix(trial['trace'])==prefix(baseline['trace']);count=len(prefix(trial['trace']));np.testing.assert_array_equal(np.load(ROOT/'states'/f'{case}_{method}_frozen_bias.npz')['states'][:count],np.load(base_root/'states'/f'{case}_{method}_frozen_bias.npz')['states'][:count]);assert all(np.array_equal(a['base_action'],a['final_action']) for a in saved if a['phase']!='lift')
            rows.append(r);save(path,rows);save(ROOT/'posture'/f'{case}_{method}.json',saved);print('E158',stage,case,method,r['success'],'exact pre-lift prefix',flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h
    if stage=='diagnostic':
        ready=all(next(r['success'] for r in rows if r['case']==c and r['method']=='learned_ensemble') for c in (13,1,6));save(ROOT/'diagnostic_summary.json',dict(ready_for_full=ready,outcome_selected_fitting_cases=list(DIAGNOSTIC_CASES),rows=rows,fresh_evaluated=False));print('ready_for_full',ready,flush=True)
    else:
        assert len(rows)==288;stats={}
        for method in METHODS:
            rr=[r for r in rows if r['method']==method];stats[method]={str(n):dict(n=len(ss),success=sum(r['success'] for r in ss),collisions=sum(r['collision'] for r in ss)) for n in ('all',6,7) if (ss:=[r for r in rr if n=='all' or r['dof']==n])}
        a=stats['learned_ensemble'];retained=all(next(r['success'] for r in rows if r['case']==c and r['method']=='learned_ensemble') for c in (1,13,18));qualified=retained and all(a['all']['success']>stats[m]['all']['success'] and a['all']['collisions']<=stats[m]['all']['collisions'] and all(a[str(n)]['success']>=stats[m][str(n)]['success'] for n in (6,7)) for m in ('uniform32','e120_ensemble'));save(ROOT/'summary.json',dict(methods=stats,qualifies_for_fresh_test=qualified,fresh_evaluated=False,E120_unchanged=True));print(read(ROOT/'summary.json'),flush=True)


def load_neural():
    neural={}
    for f in range(4):
        neural[f]=[]
        for seed in SEEDS:
            m=ConservativeWeights().cuda().eval();m.load_state_dict(torch.load(ROOT/'models'/f'fold_{f}'/f'{seed}.pt',weights_only=True));neural[f].append(m)
    return neural

def baseline(scope):
    assert scope in ('diagnostic','full')
    if scope=='full':assert read(ROOT/'diagnostic_summary.json')['ready_for_full']
    root=ROOT/'standard';root.mkdir(exist_ok=True)
    for folder in ('trials','states','actions','histories'):(root/folder).mkdir(exist_ok=True)
    path=root/'records.json';rows=read(path) if path.exists() else [];done={r['case'] for r in rows};tasks=read(ROOT/'cases.json');folds=read(ROOT/'folds.json');_,models,hashes=old.broad.setup();old.broad.previous.ROOT=root;old.broad.previous.core.ROOT=root;neural=load_neural();selected=DIAGNOSTIC_CASES if scope=='diagnostic' else [i for i,t in enumerate(tasks) if not t['validation']]
    for case in selected:
        if case in done:continue
        with patch.object(old,'ROOT',root):r=old.evaluate(case,tasks[case],'learned_ensemble',models,neural[folds[str(case)]],dict(forgetting=.98,ridge=1e-6))
        rows.append(r);save(path,rows);print('E158 standard',case,r['success'],flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json')}.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='baseline':baseline(sys.argv[2])
    else:run(sys.argv[1])

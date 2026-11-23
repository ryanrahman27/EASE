"""E137: family-deletion uncertainty for the structured conjunction."""
import shutil
from pathlib import Path
import numpy as np
from .structured_response import fit,ROOT as MOTION
from .compact_response_value import ROOT as VALUE
from .short_horizon_paired import data,score
from .abstaining_task_value_gate import read,save,fingerprint,family

ROOT=Path('outputs/ease_family_consensus_gate_v1')
MARGINS=(0.,.025,.05,.1,1e9)

def ridge(x,y):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-6);z=np.c_[np.ones(len(x)),np.clip((x-mean)/scale,-10,10)];reg=np.eye(z.shape[1]);reg[0,0]=0
    return dict(mean=mean,scale=scale,weights=np.linalg.solve(z.T@z/len(x)+reg,z.T@y/len(x)))

def predict(model,scalar,x,effect,common,error):
    adv=np.c_[np.ones(len(x)),np.clip((x-model['mean'])/model['scale'],-10,10)]@model['weights'];direction=error-scalar[0]*common;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
    return adv,np.sum(direction*(scalar[0]*effect),axis=1)

def run():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','actions'):(ROOT/folder).mkdir()
    inputs=[MOTION/'dataset.npz',VALUE/'dataset.npz',Path('outputs/ease_combined_response_value_v1/rollout_summary.json')]
    save(ROOT/'protocol.json',dict(experiment='E137',inputs={str(p):fingerprint(p) for p in inputs},motivation='E136 held15/24 equals baseline with1 gain/1 loss, so not promoted. Do not tune on those examples. Quantify training-data uncertainty by family deletion, not neural initialization.',
        design='For each of four outer fitting-family folds, fit six scalar-response/ridge-value pairs, each excluding one additional ancestral training family. Require every inner model to predict positive local progress above10um and task advantage above margin. Same features, ridge1, target and8-step pulse as E136. Primary value margin.05; fixed grid0,.025,.05,.1 plus abstain. Select zero outer-fitting losses, most gains, fewer switches, higher margin. Final ensemble comprises eight leave-one-family-out models fitted on the original fitting pool only.',
        fresh_test='Predeclare48 new randomized placements, six per ancestral fitting source, same four known bodies/homes. RNG137000; XY uniform[.275,.425]x[-.10,.10], destination source displacement plus independent20mm jitter. No reachability/success filtering. All3 E120 estimator seeds, paired baseline/candidate=288 fresh episodes. Geometry saved before model outcomes. Only execute a qualifying nontrivial no-loss fitting candidate. Do not reuse original held results for model selection; fresh outcomes will not tune this candidate.',
        scope='Development-family validation remains adaptive. Fresh placements are a new placement test on known bodies, not unseen morphology. E120 immutable.'))
    _,_,_,folds,meta,tasks=data();save(ROOT/'cases.json',tasks)
    families=np.array([family(tasks,r['case']) for r in meta]);roots=sorted(set(families.tolist()));rng=np.random.default_rng(137000);fresh=[]
    for source in roots:
        for variant in range(6):
            t=read_copy(tasks[source]);xy=rng.uniform([.275,-.1],[.425,.1]);delta=xy-np.asarray(t['xy']);t['xy']=xy.tolist();t['target']=(np.asarray(t['target'])+delta+rng.uniform(-.02,.02,2)).tolist();t.update(validation=True,group='fresh_e137',source_case=source,variant=variant);fresh.append(t)
    save(ROOT/'fresh_cases.json',fresh)
    d=np.load(MOTION/'dataset.npz');v=np.load(VALUE/'dataset.npz');x,y=v['x'],v['y'];assert len(fresh)==48
    adv=np.zeros((6,len(x)));progress=np.zeros_like(adv)
    for f in range(4):
        va=folds==f;inner=sorted(set(families[~va]));assert len(inner)==6
        for k,excluded in enumerate(inner):
            tr=(~va)&(families!=excluded);assert not set(families[tr])&set(families[va]);m=ridge(x[tr],y[tr]);s=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);np.savez(ROOT/'models'/f'fold_{f}_exclude_{excluded}.npz',**m,scalar=s,training=np.flatnonzero(tr),excluded=excluded)
            adv[k,va],progress[k,va]=predict(m,s,x[va],d['effect'][va],d['common'][va],d['error'][va])
    grid=[score((adv.min(0)>margin)&(progress.min(0)>1e-5),dict(value_margin=margin,motion_margin=1e-5),meta,tasks) for margin in MARGINS];selected=max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],r['rule']['value_margin']))
    primary=next(r for r in grid if r['rule']['value_margin']==.05)
    for excluded in roots:
        tr=families!=excluded;m=ridge(x[tr],y[tr]);s=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);np.savez(ROOT/'models'/f'final_exclude_{excluded}.npz',**m,scalar=s,training=np.flatnonzero(tr),excluded=excluded)
    np.savez_compressed(ROOT/'oof.npz',advantage=adv,progress=progress,fold=folds,family=families);save(ROOT/'grid.json',grid);save(ROOT/'selected.json',selected);save(ROOT/'primary.json',primary)
    summary=dict(primary={k:v for k,v in primary.items() if k!='episodes'},selected={k:v for k,v in selected.items() if k!='episodes'},fresh_evaluated=False,E120_unchanged=True);save(ROOT/'summary.json',summary)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'fresh_cases.json',ROOT/'selected.json',*(ROOT/'models').glob('*.npz'),*(ROOT/'source').glob('*.py')]});print(summary,flush=True)

def read_copy(x):
    import json
    return json.loads(json.dumps(x))

if __name__=='__main__':run()

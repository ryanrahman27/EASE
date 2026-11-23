"""E133: measured-history structured one-step paired response, grouped OOF."""
import json
import shutil
from pathlib import Path
import numpy as np
from .short_horizon_paired import data, score, ROOT as SHORT
from .abstaining_task_value_gate import COLLECT, DATA, OLD, read, save, fingerprint, family, PHASES
from .bounded_history_correction import bounded_action

ROOT=Path('outputs/ease_structured_response_v1')
KINDS=('analytical','scalar','phase_scalar','linear')
MARGINS=(0.,5e-6,1e-5,2.5e-5,5e-5,1e-4,1e9)


def preaction(x):
    n=int(x[9:16].sum());q=x[16:16+n];bias=x[23:23+n]
    g=x[33:82].reshape(7,7)[:n,:n];c=x[82:124].reshape(7,6)[:n]
    j=np.linalg.solve(g,c).T[:3];a=x[124:124+n];b=x[131:131+n]
    on=np.clip(q+bounded_action(a,b),-2.5,2.5)-q
    off=np.clip(q+a,-2.5,2.5)-q
    return j@(off-on),j@(on-bias),x[30:33]


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json'):shutil.copy2(SHORT/name,ROOT/name)
    inputs={str(p):fingerprint(p) for p in [SHORT/'targets.npz',SHORT/'target_rows.json',DATA/'gate_dataset.npz',DATA/'gate_rows.json',OLD/'summary.json',OLD/'records.json']}
    save(ROOT/'protocol.json',dict(experiment='E133',inputs=inputs,
        design='Same1108 clean matched starts and four ancestral fitting-family folds. Predict off-minus-on one-step Cartesian displacement from estimated history Jacobian times known clipped command difference. No object/contact state, simulator Jacobian, future inputs, or final success training labels. Uses original139 measured input columns only.',
        models='Analytical raw response; learned one global scalar; nine phase-specific scalars; global3x3 linear response. Zero intercept ensures zero prediction for identical actions. Ridge1e-6 times mean squared design magnitude, training folds only. No neural seed selection. Projection onto goal direction gives first-order progress benefit. Baseline common movement estimated from J*(on-bias), scaled by learned response where applicable. No collision-risk or physics selector.',
        selection='Primary scalar >10 micrometers at first clean phase, up to8-step off pulse then E120. Secondary fixed model/margin grid; choose zero E120 task losses, maximum gains, fewest switches, simplest model then highest margin. This is adaptive development after E132, not an independent confirmatory test.',
        margins=list(MARGINS),kinds=list(KINDS),held_evaluated=False,E120_unchanged=True))
    x,y,c,folds,meta,tasks=data();features=[];target=[];hashes={};cache={}
    for i,r in enumerate(meta):
        key=r['case'],r['seed'];phase=r['phase']
        if key not in cache:
            p=COLLECT/'trials'/f'{key[0]}_on_{key[1]}_frozen_bias.json';cache[key]=read(p);hashes[str(p)]=fingerprint(p)
        p=COLLECT/'trials'/f'{key[0]}_off_{phase}_{key[1]}_frozen_bias.json';off=read(p);hashes[str(p)]=fingerprint(p)
        entry=read(SHORT/'target_rows.json')[i]['entry_step']
        target.append(np.asarray(off['trace'][entry]['position'])-cache[key]['trace'][entry]['position'])
        features.append(preaction(x[i].astype(np.float64)))
    a=np.asarray(features);np.savez_compressed(ROOT/'dataset.npz',effect=a[:,0],common=a[:,1],error=a[:,2],target=target,phase=x[:,:9].argmax(1),fold=folds)
    save(ROOT/'target_input_hashes.json',hashes);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


def fit(kind,z,t,phase):
    if kind=='analytical':return np.ones(1)
    if kind=='scalar':return np.array([(z*t).sum()/(np.square(z).sum()+1e-18)])
    if kind=='phase_scalar':return np.array([(z[phase==p]*t[phase==p]).sum()/(np.square(z[phase==p]).sum()+1e-18) for p in range(9)])
    gram=z.T@z;ridge=max(float(np.trace(gram)/3)*1e-6,1e-18)
    return np.linalg.solve(gram+ridge*np.eye(3),z.T@t)


def apply(kind,w,z,phase):
    if kind=='linear':return z@w
    return z*(w[phase,None] if kind=='phase_scalar' else w[0])


def train():
    d=np.load(ROOT/'dataset.npz');x,y,c,folds,meta,tasks=data();z,t,phase=d['effect'],d['target'],d['phase'];pred={};progress={};metrics={}
    for kind in KINDS:
        out=np.zeros_like(t);common=np.zeros_like(t)
        for f in range(4):
            tr,va=folds!=f,folds==f
            assert not {family(tasks,meta[i]['case']) for i in np.flatnonzero(tr)}&{family(tasks,meta[i]['case']) for i in np.flatnonzero(va)}
            w=fit(kind,z[tr],t[tr],phase[tr]);np.savez(ROOT/'models'/f'{kind}_{f}.npz',weights=w,training=np.flatnonzero(tr))
            out[va]=apply(kind,w,z[va],phase[va]);common[va]=apply(kind,w,d['common'][va],phase[va])
        direction=d['error']-common;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
        p=np.sum(direction*out,axis=1);nz=abs(y[:,0])>1e-6
        metrics[kind]=dict(displacement_rmse_um=float(np.sqrt(np.mean((out-t)**2))*1e6),progress_rmse_um=float(np.sqrt(np.mean((p-y[:,0])**2))*1e6),sign_accuracy=float(np.mean(np.sign(p[nz])==np.sign(y[nz,0]))))
        pred[kind]=out;progress[kind]=p
    metrics['zero']=dict(displacement_rmse_um=float(np.sqrt(np.mean(t*t))*1e6),progress_rmse_um=float(np.sqrt(np.mean(y[:,0]**2))*1e6))
    grid=[score(progress[k]>m,dict(kind=k,margin=m),meta,tasks) for k in KINDS for m in MARGINS]
    best=max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],-KINDS.index(r['rule']['kind']),r['rule']['margin']))
    primary=next(r for r in grid if r['rule']==dict(kind='scalar',margin=1e-5))
    np.savez_compressed(ROOT/'oof.npz',**progress,**{k+'_displacement':v for k,v in pred.items()},fold=folds)
    save(ROOT/'metrics.json',metrics);save(ROOT/'grid.json',grid);save(ROOT/'selected.json',best);save(ROOT/'primary.json',primary)
    save(ROOT/'summary.json',dict(metrics=metrics,primary={k:v for k,v in primary.items() if k!='episodes'},selected={k:v for k,v in best.items() if k!='episodes'},held_evaluated=False,E120_unchanged=True))
    print(json.dumps(read(ROOT/'summary.json'),indent=2),flush=True)


if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train}[sys.argv[1]]()

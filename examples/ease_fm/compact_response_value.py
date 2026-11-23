"""E135: regularized paired task advantage on structured measured features."""
import shutil
from pathlib import Path
import numpy as np
from .structured_response import preaction
from .short_horizon_paired import data, score
from .abstaining_task_value_gate import read,save,fingerprint,DATA,OLD,family

ROOT=Path('outputs/ease_compact_response_value_v1')
PENALTIES=(.01,.1,1.,10.)
MARGINS=(.01,.025,.05,.1,.2,1e9)

def features(x):
    effect,common,error=preaction(x.astype(np.float64));n=int(x[9:16].sum());g=x[33:82].reshape(7,7)[:n,:n]
    norm=np.linalg.norm(error);direction=error/max(norm,1e-9);progress=float(direction@effect)
    # No case/body/seed IDs or outcomes. Position/orientation are pre-action observations.
    physical=np.r_[error,norm,x[141],np.linalg.norm(x[142:145]),effect,progress,np.linalg.norm(effect),np.linalg.norm(common),np.log10(max(np.linalg.cond(g),1)),n-6]
    return np.r_[x[:9],physical,x[:9]*progress]

def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states'):(ROOT/folder).mkdir()
    x,y,c,folds,meta,tasks=data();inputs=[DATA/'gate_rows.json',DATA/'gate_dataset.npz',OLD/'records.json',Path('outputs/ease_privileged_task_state_v1/features.npz')]
    save(ROOT/'protocol.json',dict(experiment='E135',inputs={str(p):fingerprint(p) for p in inputs},design='Same1108 clean starts/four ancestral fitting-family splits. Linear ridge signed final-success advantage regression on compact measured pre-action features, with intercept. No privileged object/contact signals: reads only original139 and measured tool z/orientation at141:145. No case/body/seed IDs. No physics selector.',
        features='Nine phase indicators; goal error3/norm, tool height, orientation-error norm, analytical J times paired command difference3, projected progress, effect norm, estimated common response norm, log10 history-Gram condition number, joint-count minus6; nine phase/progress interactions.32 features total. No learned E133 coefficients, preventing cross-fold feature leakage.',
        training='Closed-form ridge, mean squared final paired advantage plus lambda times squared standardized weights, intercept unpenalized. Feature means/scales fitting-fold only. Four fixed penalties. No class weights or resampling. Primary lambda1 margin.05. Secondary same grid chooses zero E120 losses, most gains, fewer switches, stronger regularization, higher margin. Adaptive fitting-validation search, not independent testing.',
        penalties=list(PENALTIES),margins=list(MARGINS),intervention='First accepted clean phase: up to8 off actions, then E120 resumes. Reconstruct from saved exact paired continuations.',held_evaluated=False))
    xx=np.stack([features(r) for r in x]);assert xx.shape==(1108,32);yy=np.asarray([r['labels'][1]-r['labels'][0] for r in meta]);np.savez_compressed(ROOT/'dataset.npz',x=xx,y=yy,fold=folds)
    save(ROOT/'cases.json',tasks);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def train():
    d=np.load(ROOT/'dataset.npz');x,y,folds=d['x'],d['y'],d['fold'];_,_,_,_,meta,tasks=data();pred={};metrics={};grid=[]
    for penalty in PENALTIES:
        p=np.zeros(len(x))
        for f in range(4):
            tr,va=folds!=f,folds==f
            assert not {family(tasks,meta[i]['case']) for i in np.flatnonzero(tr)}&{family(tasks,meta[i]['case']) for i in np.flatnonzero(va)}
            mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),1e-6);z=np.clip((x[tr]-mean)/scale,-10,10);z=np.c_[np.ones(sum(tr)),z];reg=np.eye(z.shape[1])*penalty;reg[0,0]=0
            w=np.linalg.solve(z.T@z/sum(tr)+reg,z.T@y[tr]/sum(tr));p[va]=np.c_[np.ones(sum(va)),np.clip((x[va]-mean)/scale,-10,10)]@w
            np.savez(ROOT/'models'/f'lambda_{penalty}_fold_{f}.npz',weights=w,mean=mean,scale=scale,training=np.flatnonzero(tr))
        pred[str(penalty)]=p;metrics[str(penalty)]=dict(mse=float(np.mean((p-y)**2)),nonzero_sign_accuracy=float(np.mean(np.sign(p[y!=0])==y[y!=0])))
        grid += [score(p>margin,dict(penalty=penalty,margin=margin),meta,tasks) for margin in MARGINS]
    best=max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],r['rule']['penalty'],r['rule']['margin']))
    primary=next(r for r in grid if r['rule']==dict(penalty=1.,margin=.05));np.savez_compressed(ROOT/'oof.npz',**pred)
    save(ROOT/'grid.json',grid);save(ROOT/'primary.json',primary);save(ROOT/'selected.json',best);summary=dict(metrics=metrics,primary={k:v for k,v in primary.items() if k!='episodes'},selected={k:v for k,v in best.items() if k!='episodes'},held_evaluated=False,E120_unchanged=True);save(ROOT/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'train':train}[sys.argv[1]]()

"""E140: separate helpful/harmful paired effects from neutral-majority regression."""
import shutil
from pathlib import Path
import numpy as np
from .expanded_structured_pairs import ROOT as DATA_ROOT
from .expanded_structured_training import dependencies,score
from .rich_structured_value import ROOT as RICH
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_paired_effect_classifier_v1')
CONFIDENCE=(.65,.75,.85,.95)

def run():
    sklearn,joblib,_=dependencies()
    from sklearn.ensemble import ExtraTreesClassifier
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states'):(ROOT/folder).mkdir()
    inputs=[DATA_ROOT/'dataset.npz',DATA_ROOT/'rows.json',DATA_ROOT/'cases.json',RICH/'features.npz',RICH/'summary.json']
    save(ROOT/'protocol.json',dict(experiment='E140',inputs={str(p):fingerprint(p) for p in inputs},motivation='1994/2208 paired task effects are neutral; only121 beneficial and93 harmful. Regression may shrink toward zero while failing to rank changed outcomes. Test classification as a distinct objective, not more held-out tuning.',
        design='Same expanded family folds and fixed local response predictions. Compare compact32 and richer115 measured features. Three-class balanced ExtraTrees128/depth5/min_leaf12/seed138000 learns -1,0,+1 on all fitting rows. Conditional binary balanced ExtraTrees128/depth5/min_leaf6/seed138000 learns benefit versus harm on nonzero fitting effects only. Also a simple phase-by-joint-count conditional frequency reference with Beta(1,1) smoothing. No privileged inputs or IDs.',
        rule='Always require E138 OOF scalar local progress>10um. Conditional classifiers: benefit vote above .65,.75,.85,.95; primary rich conditional .75. Three-class classifier: benefit-minus-harm vote>.1 and harm vote below .1 or .2. Votes from balanced models are NOT calibrated natural-population probabilities. First clean selected phase receives same8-step off pulse.',
        selection='Fixed development grid; require zero E120 losses, most gains, fewer switches, higher confidence, lexical rule tie break. Neutral-state generalization is explicitly evaluated in complete-task reconstruction; conditional models have no neutral training examples. Fresh and original-held tests remain untouched. No success-based episode removal.'))
    d=np.load(DATA_ROOT/'dataset.npz');features={'compact':d['x'],'rich':np.load(RICH/'features.npz')['x']};y,fold=d['y'].astype(int),d['fold'];meta=read(DATA_ROOT/'rows.json');tasks=read(DATA_ROOT/'cases.json');progress=np.load(DATA_ROOT/'oof.npz')['ridge'][1];pred={};metrics={};allmodels=[]
    for name,x in features.items():
        conditional=np.zeros(len(x));three=np.zeros((len(x),3))
        for f in range(4):
            tr,va=fold!=f,fold==f;mean=x[tr].mean(0);scale=np.maximum(x[tr].std(0),1e-6);z=np.clip((x-mean)/scale,-10,10);nz=tr&(y!=0)
            for mode,ids,labels,leaf in (('conditional',nz,y>0,6),('three',tr,y,12)):
                m=ExtraTreesClassifier(n_estimators=128,max_depth=5,min_samples_leaf=leaf,random_state=138000,class_weight='balanced',n_jobs=4);m.fit(z[ids],labels[ids]);stem=f'{name}_{mode}_fold_{f}';joblib.dump(m,ROOT/'models'/f'{stem}.joblib');np.savez(ROOT/'models'/f'{stem}.npz',mean=mean,scale=scale,training=np.flatnonzero(ids));prob=m.predict_proba(z[va])
                if mode=='conditional':conditional[va]=prob[:,list(m.classes_).index(True)]
                else:
                    assert list(m.classes_)==[-1,0,1];three[va]=prob
        pred[name+'_conditional']=conditional;pred[name+'_three']=three;changed=y!=0;metrics[name]=dict(conditional_sign_accuracy=float(np.mean((conditional[changed]>.5)==(y[changed]>0))),three_changed_sign_accuracy=float(np.mean((three[changed,2]>three[changed,0])==(y[changed]>0))))
    # A task-phase-only conditional reference tests whether rich state adds value.
    phase=d['phase'];dof=d['raw'][:,9:16].sum(1).astype(int);reference=np.zeros(len(y))
    for f in range(4):
        for p in range(9):
            for n in (6,7):
                tr=(fold!=f)&(phase==p)&(dof==n)&(y!=0);va=(fold==f)&(phase==p)&(dof==n);reference[va]=(np.sum(y[tr]>0)+1)/(sum(tr)+2)
    pred['phase_conditional']=reference;grid=[]
    for name in ('compact','rich','phase'):
        for confidence in CONFIDENCE:
            r=score((pred[name+'_conditional']>confidence)&(progress>1e-5),f'{name}_conditional_{confidence}',meta,tasks);r['rule']=dict(features=name,objective='conditional',confidence=confidence);grid.append(r)
    for name in ('compact','rich'):
        for cap in (.1,.2):
            pp=pred[name+'_three'];r=score((pp[:,2]-pp[:,0]>.1)&(pp[:,0]<cap)&(progress>1e-5),f'{name}_three_cap_{cap}',meta,tasks);r['rule']=dict(features=name,objective='three',harm_vote_cap=cap);grid.append(r)
    abstain=score(np.zeros(len(y),bool),'abstain',meta,tasks);abstain['rule']=dict(confidence=1e9);grid.append(abstain);chosen=max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],r['rule'].get('confidence',0),r['kind']));primary=next(r for r in grid if r['kind']=='rich_conditional_0.75')
    np.savez_compressed(ROOT/'oof.npz',**pred,progress=progress,fold=fold);save(ROOT/'cases.json',tasks);save(ROOT/'grid.json',grid);save(ROOT/'selected.json',chosen);save(ROOT/'primary.json',primary);save(ROOT/'summary.json',dict(metrics=metrics,primary={k:v for k,v in primary.items() if k!='episodes'},selected={k:v for k,v in chosen.items() if k!='episodes'},fresh_evaluated=False,E120_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'oof.npz',ROOT/'selected.json',*(ROOT/'models').iterdir(),*(ROOT/'source').glob('*.py')]});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':run()

"""E139: richer measured kinematic state on the fixed expanded dataset."""
import shutil
from pathlib import Path
import numpy as np
from .expanded_structured_pairs import ROOT as PRIOR
from .expanded_structured_training import dependencies,score
from .family_consensus_gate import ridge,predict
from .structured_response import preaction,fit
from .compact_response_value import features as compact
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_rich_structured_value_v1')
KINDS=('ridge','consensus','trees','tree_consensus')

def features(raw):
    n=int(raw[9:16].sum());g=raw[33:82].reshape(7,7)[:n,:n];c=raw[82:124].reshape(7,6)[:n];j=np.zeros((7,6));j[:n]=np.linalg.solve(g.astype(float),c.astype(float));a=raw[124:131];b=raw[131:138];delta=-np.clip(.25*(b-a),-.001,.001)*raw[9:16]
    x=np.r_[compact(raw),raw[16:23],raw[9:16],raw[139:145],j.ravel(),a,delta,raw[23:30]];assert x.shape==(115,) and np.isfinite(x).all();return x

def run():
    _,joblib,ExtraTreesRegressor=dependencies();ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states'):(ROOT/folder).mkdir()
    inputs=[PRIOR/'dataset.npz',PRIOR/'rows.json',PRIOR/'summary.json',PRIOR/'cases.json',PRIOR/'fresh_cases.json',PRIOR/'frozen_training.json']
    save(ROOT/'protocol.json',dict(experiment='E139',inputs={str(p):fingerprint(p) for p in inputs},design='Same2208 rows, outcomes, family folds, local scalar prediction, fixed thresholds and model hyperparameters as E138. Expand task-value features32 to115 with measured joint positions7, mask7, tool position/orientation error6, analytical measured-history Jacobian columns42, current base action7, bounded off-minus-on proposal difference7, frozen measured bias7. No simulator geometry, object/contact truth, body/seed IDs or future inputs.',
        models='Ridge1, family-deletion ridge consensus, and ExtraTrees128/depth5/min_leaf12/random_state138000 exactly as E138. Primary richer trees. Additional predeclared tree-consensus requires all six inner-family-deletion trees to exceed the same task threshold. Response scalar and progress threshold unchanged. Input representation and added tree-consensus are reported separately.',
        selection='Fixed task-value>.05 and progress>10um, first clean phase, same8-step off pulse. Require zero E120 losses on288 fitting episodes; maximize gains, fewer switches; tie favor trees, ridge, consensus, tree_consensus. No threshold sweep. No original-held or reserved fresh outcomes used. Fresh48-placement set remains unexecuted unless a candidate qualifies.',
        limitation='Measured state is ideal simulator observation, not a demonstrated sensor pipeline. New features may improve known-body placement generalization; no unseen-body claim.'))
    d=np.load(PRIOR/'dataset.npz');raw=d['raw'];x=np.stack([features(r) for r in raw]);y,fold,fam=d['y'],d['fold'],d['family'];meta=read(PRIOR/'rows.json');tasks=read(PRIOR/'cases.json');np.savez_compressed(ROOT/'features.npz',x=x);save(ROOT/'cases.json',tasks);shutil.copy2(PRIOR/'fresh_cases.json',ROOT/'fresh_cases.json');out={k:np.zeros((2,len(x))) for k in KINDS}
    def tree(z,yy):
        m=ExtraTreesRegressor(n_estimators=128,max_depth=5,min_samples_leaf=12,random_state=138000,n_jobs=4);m.fit(z,yy);return m
    for f in range(4):
        tr,va=fold!=f,fold==f;m=ridge(x[tr],y[tr]);s=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);np.savez(ROOT/'models'/f'ridge_fold_{f}.npz',**m,scalar=s,training=np.flatnonzero(tr));out['ridge'][:,va]=np.asarray(predict(m,s,x[va],d['effect'][va],d['common'][va],d['error'][va]));z=np.clip((x[tr]-m['mean'])/m['scale'],-10,10);zv=np.clip((x[va]-m['mean'])/m['scale'],-10,10);t=tree(z,y[tr]);joblib.dump(t,ROOT/'models'/f'trees_fold_{f}.joblib');out['trees'][0,va]=t.predict(zv);out['trees'][1,va]=out['ridge'][1,va];aa=[];tt=[]
        for excluded in sorted(set(fam[tr])):
            ids=tr&(fam!=excluded);inner=ridge(x[ids],y[ids]);scalar=fit('scalar',d['effect'][ids],d['target'][ids],d['phase'][ids]);np.savez(ROOT/'models'/f'consensus_fold_{f}_exclude_{excluded}.npz',**inner,scalar=scalar,training=np.flatnonzero(ids));a,p=predict(inner,scalar,x[va],d['effect'][va],d['common'][va],d['error'][va]);aa.append((a,p));t=tree(np.clip((x[ids]-inner['mean'])/inner['scale'],-10,10),y[ids]);joblib.dump(t,ROOT/'models'/f'tree_consensus_fold_{f}_exclude_{excluded}.joblib');tt.append((t.predict(np.clip((x[va]-inner['mean'])/inner['scale'],-10,10)),p))
        out['consensus'][:,va]=np.asarray(aa).min(0);out['tree_consensus'][:,va]=np.asarray(tt).min(0);print('E139 trained fold',f,flush=True)
    results={k:score((v[0]>.05)&(v[1]>1e-5),k,meta,tasks) for k,v in out.items()};results['abstain']=score(np.zeros(len(x),bool),'abstain',meta,tasks);order=('trees','ridge','consensus','tree_consensus','abstain');chosen=max([r for r in results.values() if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],-order.index(r['kind'])))
    np.savez_compressed(ROOT/'oof.npz',**out);save(ROOT/'outcomes.json',results);save(ROOT/'selected.json',chosen);save(ROOT/'summary.json',dict(results={k:{a:b for a,b in r.items() if a!='episodes'} for k,r in results.items()},selected={k:v for k,v in chosen.items() if k!='episodes'},fresh_evaluated=False,E120_unchanged=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'features.npz',ROOT/'oof.npz',ROOT/'selected.json',*(ROOT/'models').iterdir(),*(ROOT/'source').glob('*.py')]});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':run()

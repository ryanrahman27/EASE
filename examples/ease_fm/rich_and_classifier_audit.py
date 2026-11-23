"""Independent audits for E139/E140 before their media are archived."""
import shutil
from pathlib import Path
import numpy as np
from .expanded_structured_training import dependencies,score
from .expanded_structured_pairs import ROOT as DATA_ROOT
from .rich_structured_value import ROOT as RICH,features,KINDS
from .paired_effect_classifier import ROOT as CLASS,CONFIDENCE
from .family_consensus_gate import predict
from .abstaining_task_value_gate import read,save,fingerprint
from .paired_gate_media import finish

def common(root):
    for p,h in {**read(root/'protocol.json')['inputs'],**read(root/'frozen.json')}.items():assert fingerprint(p)==h
    for p in (root/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))

def rich():
    _,joblib,_=dependencies();d=np.load(DATA_ROOT/'dataset.npz');x=np.load(RICH/'features.npz')['x'];np.testing.assert_array_equal(x,np.stack([features(r) for r in d['raw']]));o=np.load(RICH/'oof.npz');fold,fam=d['fold'],d['family'];pred={k:np.zeros_like(o[k]) for k in KINDS}
    for f in range(4):
        tr,va=fold!=f,fold==f;assert not set(fam[tr])&set(fam[va]);m=np.load(RICH/'models'/f'ridge_fold_{f}.npz');np.testing.assert_array_equal(m['training'],np.flatnonzero(tr));np.testing.assert_array_equal(m['mean'],x[tr].mean(0));np.testing.assert_array_equal(m['scale'],np.maximum(x[tr].std(0),1e-6));pred['ridge'][:,va]=np.asarray(predict(m,m['scalar'],x[va],d['effect'][va],d['common'][va],d['error'][va]));t=joblib.load(RICH/'models'/f'trees_fold_{f}.joblib');pred['trees'][0,va]=t.predict(np.clip((x[va]-m['mean'])/m['scale'],-10,10));pred['trees'][1,va]=pred['ridge'][1,va];aa=[];tt=[]
        for excluded in sorted(set(fam[tr])):
            ids=tr&(fam!=excluded);n=np.load(RICH/'models'/f'consensus_fold_{f}_exclude_{excluded}.npz');np.testing.assert_array_equal(n['training'],np.flatnonzero(ids));a,p=predict(n,n['scalar'],x[va],d['effect'][va],d['common'][va],d['error'][va]);aa.append((a,p));t=joblib.load(RICH/'models'/f'tree_consensus_fold_{f}_exclude_{excluded}.joblib');tt.append((t.predict(np.clip((x[va]-n['mean'])/n['scale'],-10,10)),p))
        pred['consensus'][:,va]=np.asarray(aa).min(0);pred['tree_consensus'][:,va]=np.asarray(tt).min(0)
    meta=read(DATA_ROOT/'rows.json');tasks=read(RICH/'cases.json');results=read(RICH/'outcomes.json')
    for k,v in pred.items():np.testing.assert_allclose(v,o[k],rtol=1e-12,atol=1e-12);assert score((v[0]>.05)&(v[1]>1e-5),k,meta,tasks)==results[k]
    common(RICH);save(RICH/'verification.json',dict(features_exact=True,predictions_reproduced=True,family_exclusions_verified=True,outcomes_exact=True,inputs_unchanged=True))
    finish(RICH,'E139','rich_structured_value',results,read(RICH/'selected.json'),'Expanded measured-state features from 32 to 115 on the same 2,208 paired starts and four family folds. Primary richer trees retain the E138 hyperparameters and thresholds. Added tree-family consensus is reported separately. Extra state information did not create a positive-gain zero-loss candidate. No object/contact truth or simulator geometry enters inference.')

def classifier():
    _,joblib,_=dependencies();d=np.load(DATA_ROOT/'dataset.npz');xs={'compact':d['x'],'rich':np.load(RICH/'features.npz')['x']};y,fold,fam=d['y'].astype(int),d['fold'],d['family'];o=np.load(CLASS/'oof.npz');pred={}
    for name,x in xs.items():
        conditional=np.zeros(len(x));three=np.zeros((len(x),3))
        for f in range(4):
            tr,va=fold!=f,fold==f;assert not set(fam[tr])&set(fam[va])
            for mode in ('conditional','three'):
                stem=f'{name}_{mode}_fold_{f}';n=np.load(CLASS/'models'/f'{stem}.npz');ids=tr&(y!=0) if mode=='conditional' else tr;np.testing.assert_array_equal(n['training'],np.flatnonzero(ids));np.testing.assert_array_equal(n['mean'],x[tr].mean(0));np.testing.assert_array_equal(n['scale'],np.maximum(x[tr].std(0),1e-6));m=joblib.load(CLASS/'models'/f'{stem}.joblib');pp=m.predict_proba(np.clip((x[va]-n['mean'])/n['scale'],-10,10))
                if mode=='conditional':conditional[va]=pp[:,list(m.classes_).index(True)]
                else:three[va]=pp
        pred[name+'_conditional']=conditional;pred[name+'_three']=three
    reference=np.zeros(len(y));dof=d['raw'][:,9:16].sum(1).astype(int)
    for f in range(4):
        for p in range(9):
            for n in (6,7):
                tr=(fold!=f)&(d['phase']==p)&(dof==n)&(y!=0);va=(fold==f)&(d['phase']==p)&(dof==n);reference[va]=(np.sum(y[tr]>0)+1)/(sum(tr)+2)
    pred['phase_conditional']=reference
    for k,v in pred.items():np.testing.assert_allclose(v,o[k],rtol=1e-12,atol=1e-12)
    meta=read(DATA_ROOT/'rows.json');tasks=read(CLASS/'cases.json');grid=read(CLASS/'grid.json');progress=o['progress']
    for r in grid:
        rule=r['rule'];name=rule.get('features')
        if r['kind']=='abstain':accept=np.zeros(len(y),bool)
        elif rule['objective']=='conditional':accept=(pred[name+'_conditional']>rule['confidence'])&(progress>1e-5)
        else:pp=pred[name+'_three'];accept=(pp[:,2]-pp[:,0]>.1)&(pp[:,0]<rule['harm_vote_cap'])&(progress>1e-5)
        expected=score(accept,r['kind'],meta,tasks);expected['rule']=rule;assert expected==r
    common(CLASS);save(CLASS/'verification.json',dict(predictions_reproduced=True,conditional_loss_uses_nonzero_only=True,normalization_uses_all_training_rows_only=True,family_exclusions_verified=True,phase_reference_exact=True,outcomes_exact=True,inputs_unchanged=True))
    finish(CLASS,'E140','paired_effect_classifier',{r['kind']:r for r in grid},read(CLASS/'selected.json'),'The paired labels contain 1,994 neutral, 121 beneficial and 93 harmful outcomes. Compare balanced three-class fitting, conditional benefit-versus-harm fitting, and a phase-by-joint-count frequency reference. Conditional classifier losses omit neutral rows; their input scaling uses all training rows, never validation rows. Balanced classifier votes are not calibrated success probabilities. Both feature representations remain near chance on changed-outcome rankings; no positive-gain zero-loss rule qualified.')

if __name__=='__main__':
    import sys
    {'rich':rich,'classifier':classifier}[sys.argv[1]]()

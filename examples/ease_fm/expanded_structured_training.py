"""E138: measured feature extraction and fixed structured model comparison."""
import sys
import shutil
from pathlib import Path
import numpy as np
from .expanded_structured_pairs import ROOT,COLLECT
from .abstaining_task_value_gate import read,save,fingerprint,DATA,family,PHASES
from .structured_response import ROOT as MOTION,preaction,fit
from .compact_response_value import ROOT as VALUE,features
from .family_consensus_gate import ridge,predict
from .broad_task_value_gate import features as original_features
from .arm_pick_place import rotation_error

KINDS=('ridge','consensus','trees')

def extract():
    import mujoco
    records=sum([read(COLLECT/f'records_seed_{s}.json') for s in range(3)],[]);assert len(records)==1440;save(COLLECT/'records.json',records)
    tasks=read(ROOT/'cases.json');oldmeta=read(DATA/'gate_rows.json');oldm=np.load(MOTION/'dataset.npz');oldv=np.load(VALUE/'dataset.npz');oldx=np.load('outputs/ease_privileged_task_state_v1/features.npz')['x'][:,:145]
    xs=[];raw=[];targets=[];effects=[];commons=[];errors=[];ys=[];meta=[];phases=[];excluded=[];checks=0;hashes={};folds=read(MOTION/'folds.json')
    def get(path):hashes[str(path)]=fingerprint(path);return read(path)
    for case in range(56,104):
        for seed in range(3):
            stem=f'{case}_on_{seed}_frozen_bias';on=get(COLLECT/'trials'/f'{stem}.json');onb=get(COLLECT/'branches'/f'{case}_on_{seed}.json');actions=get(COLLECT/'actions'/f'{stem}.json');xml=COLLECT/'states'/f'{stem}.xml';hashes[str(xml)]=fingerprint(xml);m=mujoco.MjModel.from_xml_path(str(xml.resolve()));d=mujoco.MjData(m);site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
            for phase in PHASES:
                offstem=f'{case}_off_{phase}_{seed}_frozen_bias';off=get(COLLECT/'trials'/f'{offstem}.json');branch=get(COLLECT/'branches'/f'{case}_off_{phase}_{seed}.json');i=branch['intervention'];assert i==onb['entries'][phase]['step'];assert onb['signatures'][:i+1]==branch['signatures'][:i+1]
                pa=COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz';pb=COLLECT/'snapshots'/f'{case}_off_{phase}_{seed}_{phase}.npz';a,b=np.load(pa),np.load(pb)
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                hashes[str(pa)]=fingerprint(pa);hashes[str(pb)]=fingerprint(pb);checks+=1
                if not onb['metrics'][i]['clean_before']:excluded.append(dict(case=case,seed=seed,phase=phase));continue
                mujoco.mj_setState(m,d,a['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d);act=actions[i];n=len(act['q_before']);pos=d.site_xpos[site].copy();rot=d.site_xmat[site].reshape(3,3)
                np.testing.assert_allclose(pos,act['position_before'],atol=1e-12,rtol=0);np.testing.assert_array_equal(d.qpos[:n],act['q_before'])
                base=original_features(n,a['gram'],a['cross'],a['bias'],act['q_before'],act['position_before'],a['goal'],phase,onb['entries'][phase]['metadata']['local'],act['base_action'],act['correction_action']);xx=np.r_[base,pos,rotation_error(np.eye(3),rot)].astype(np.float32);effect,common,error=preaction(xx.astype(np.float64))
                raw.append(xx);xs.append(features(xx));effects.append(effect);commons.append(common);errors.append(error);targets.append(np.asarray(off['trace'][i]['position'])-on['trace'][i]['position']);ys.append(float(off['success'])-float(on['success']));phases.append(PHASES.index(phase));meta.append(dict(case=case,seed=seed,phase=phase,signature=onb['signatures'][i],labels=[float(on['success']),float(off['success'])]))
        print('Extracted E138 case',case,'clean starts',len(meta),flush=True)
    assert checks==1296;allmeta=oldmeta+meta;ff=np.asarray([folds[str(family(tasks,r['case']))] for r in allmeta]);families=np.asarray([family(tasks,r['case']) for r in allmeta]);assert not any(tasks[r['case']]['validation'] for r in allmeta)
    arrays=dict(x=np.r_[oldv['x'],np.asarray(xs)],raw=np.r_[oldx,np.asarray(raw)],y=np.r_[oldv['y'],ys],effect=np.r_[oldm['effect'],effects],common=np.r_[oldm['common'],commons],error=np.r_[oldm['error'],errors],target=np.r_[oldm['target'],targets],phase=np.r_[oldm['phase'],phases],fold=ff,family=families)
    np.savez_compressed(ROOT/'dataset.npz',**arrays);save(ROOT/'rows.json',allmeta);save(ROOT/'extraction.json',dict(original_rows=len(oldmeta),new_rows=len(meta),total_rows=len(allmeta),new_pairs_checked=checks,excluded=excluded,held_rows=0));save(ROOT/'extraction_input_hashes.json',hashes)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def score(accept,kind,meta,tasks):
    records=read(DATA/'collection/records.json')+read(COLLECT/'records.json');lookup={(r['case'],r['seed'],r['off_phase']):r for r in records};index={(r['case'],r['seed'],r['phase']):i for i,r in enumerate(meta)};episodes=[]
    for case,t in enumerate(tasks):
        if t['validation']:continue
        for seed in range(3):
            phase=next((p for p in PHASES if (case,seed,p) in index and accept[index[case,seed,p]]),None);on,off=lookup[case,seed,None],lookup[case,seed,phase]
            episodes.append(dict(case=case,seed=seed,phase=phase,baseline_success=on['success'],success=off['success'],baseline_collision=on['collision'],collision=off['collision']))
    return dict(kind=kind,n=len(episodes),baseline_success=sum(r['baseline_success'] for r in episodes),success=sum(r['success'] for r in episodes),gains=sum(r['success'] and not r['baseline_success'] for r in episodes),losses=sum(r['baseline_success'] and not r['success'] for r in episodes),switches=sum(r['phase'] is not None for r in episodes),episodes=episodes)

def dependencies():
    sys.path.insert(0,str((ROOT/'dependencies').resolve()))
    import sklearn,joblib
    from sklearn.ensemble import ExtraTreesRegressor
    return sklearn,joblib,ExtraTreesRegressor

def train():
    sklearn,joblib,ExtraTreesRegressor=dependencies();d=np.load(ROOT/'dataset.npz');x,y,fold,fam=d['x'],d['y'],d['fold'],d['family'];meta=read(ROOT/'rows.json');tasks=read(ROOT/'cases.json');out={k:np.zeros((2,len(x))) for k in KINDS}
    for f in range(4):
        tr,va=fold!=f,fold==f;assert not set(fam[tr])&set(fam[va]);m=ridge(x[tr],y[tr]);s=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);np.savez(ROOT/'models'/f'ridge_fold_{f}.npz',**m,scalar=s,training=np.flatnonzero(tr));out['ridge'][:,va]=np.asarray(predict(m,s,x[va],d['effect'][va],d['common'][va],d['error'][va]))
        model=ExtraTreesRegressor(n_estimators=128,max_depth=5,min_samples_leaf=12,random_state=138000,n_jobs=4);model.fit(np.clip((x[tr]-m['mean'])/m['scale'],-10,10),y[tr]);joblib.dump(model,ROOT/'models'/f'trees_fold_{f}.joblib');out['trees'][0,va]=model.predict(np.clip((x[va]-m['mean'])/m['scale'],-10,10));out['trees'][1,va]=out['ridge'][1,va]
        aa=[]
        for excluded in sorted(set(fam[tr])):
            ids=tr&(fam!=excluded);inner=ridge(x[ids],y[ids]);scalar=fit('scalar',d['effect'][ids],d['target'][ids],d['phase'][ids]);np.savez(ROOT/'models'/f'consensus_fold_{f}_exclude_{excluded}.npz',**inner,scalar=scalar,training=np.flatnonzero(ids));aa.append(predict(inner,scalar,x[va],d['effect'][va],d['common'][va],d['error'][va]))
        out['consensus'][:,va]=np.asarray(aa).min(0);print('Trained expanded fold',f,flush=True)
    results={k:score((v[0]>.05)&(v[1]>1e-5),k,meta,tasks) for k,v in out.items()};results['abstain']=score(np.zeros(len(x),bool),'abstain',meta,tasks)
    candidates=[r for r in results.values() if r['losses']==0];chosen=max(candidates,key=lambda r:(r['gains'],-r['switches'],-((*KINDS,'abstain').index(r['kind']))))
    m=ridge(x,y);s=fit('scalar',d['effect'],d['target'],d['phase']);np.savez(ROOT/'models'/'ridge_final.npz',**m,scalar=s,training=np.arange(len(x)))
    model=ExtraTreesRegressor(n_estimators=128,max_depth=5,min_samples_leaf=12,random_state=138000,n_jobs=4);model.fit(np.clip((x-m['mean'])/m['scale'],-10,10),y);joblib.dump(model,ROOT/'models'/'trees_final.joblib')
    for excluded in sorted(set(fam)):
        tr=fam!=excluded;inner=ridge(x[tr],y[tr]);scalar=fit('scalar',d['effect'][tr],d['target'][tr],d['phase'][tr]);np.savez(ROOT/'models'/f'consensus_final_exclude_{excluded}.npz',**inner,scalar=scalar,training=np.flatnonzero(tr))
    np.savez_compressed(ROOT/'oof.npz',**out);save(ROOT/'outcomes.json',results);save(ROOT/'selected.json',chosen);save(ROOT/'summary.json',dict(results={k:{a:b for a,b in r.items() if a!='episodes'} for k,r in results.items()},selected={k:v for k,v in chosen.items() if k!='episodes'},fresh_evaluated=False,E120_unchanged=True));save(ROOT/'software.json',dict(sklearn=sklearn.__version__,numpy=np.__version__,joblib=joblib.__version__))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'frozen_training.json',{str(p):fingerprint(p) for p in [ROOT/'dataset.npz',ROOT/'rows.json',ROOT/'selected.json',ROOT/'oof.npz',ROOT/'software.json',*(ROOT/'models').iterdir(),*(ROOT/'source').glob('*.py')]});print(read(ROOT/'summary.json'),flush=True)

if __name__=='__main__':
    {'extract':extract,'train':train}[sys.argv[1]]()

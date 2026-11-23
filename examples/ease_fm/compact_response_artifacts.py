"""Audit and archive E135, including representative paired outcome replays."""
import shutil
from pathlib import Path
import numpy as np
from .compact_response_value import ROOT, PENALTIES, MARGINS, features
from .short_horizon_paired import data,score
from .abstaining_task_value_gate import read,save,fingerprint,COLLECT

def run():
    d=np.load(ROOT/'dataset.npz');x,y,fold=d['x'],d['y'],d['fold'];raw,_,_,folds,meta,tasks=data();np.testing.assert_array_equal(x,np.stack([features(r) for r in raw]));np.testing.assert_array_equal(fold,folds);o=np.load(ROOT/'oof.npz');grid=[]
    for penalty in PENALTIES:
        pred=np.zeros(len(x))
        for f in range(4):
            tr,va=fold!=f,fold==f;n=np.load(ROOT/'models'/f'lambda_{penalty}_fold_{f}.npz');np.testing.assert_array_equal(n['training'],np.flatnonzero(tr));np.testing.assert_array_equal(n['mean'],x[tr].mean(0));np.testing.assert_array_equal(n['scale'],np.maximum(x[tr].std(0),1e-6))
            z=np.c_[np.ones(sum(tr)),np.clip((x[tr]-n['mean'])/n['scale'],-10,10)];reg=np.eye(z.shape[1])*penalty;reg[0,0]=0;w=np.linalg.solve(z.T@z/sum(tr)+reg,z.T@y[tr]/sum(tr));np.testing.assert_array_equal(w,n['weights']);pred[va]=np.c_[np.ones(sum(va)),np.clip((x[va]-n['mean'])/n['scale'],-10,10)]@w
        np.testing.assert_array_equal(pred,o[str(penalty)]);grid += [score(pred>m,dict(penalty=penalty,margin=m),meta,tasks) for m in MARGINS]
    assert grid==read(ROOT/'grid.json')
    for p,h in read(ROOT/'protocol.json')['inputs'].items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(features_exact=True,weights_predictions_exact=True,normalization_training_only=True,task_outcomes_exact=True,inputs_unchanged=True,held_rows=0))
    primary=read(ROOT/'primary.json');wanted={}
    for dof in (6,7):
        for gain in (True,False):
            rows=[r for r in primary['episodes'] if r['success']==gain and r['success']!=r['baseline_success'] and len(tasks[r['case']]['home'])==dof]
            for r in rows[:2]:
                for phase in (r['phase'],None):wanted[r['case'],r['seed'],phase]=True
    records=[]
    for case,seed,phase in wanted:
        method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias';r=read(COLLECT/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:shutil.copy2(COLLECT/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        records.append(dict(case=case,method=method,mode='frozen_bias',success=r['success'],collision=r['collision']))
    save(ROOT/'records.json',records);summary=read(ROOT/'summary.json');lines=['# E135: compact regularized task value','','32 structured measured features, same four fitting-family splits. Linear signed paired-advantage regression; fitting-only normalization and fixed regularization grid. No privileged object/contact signals. Training targets are final success differences. This is exploratory development after E133.','',
        '| Ridge penalty | OOF advantage MSE | Nonzero sign accuracy |','|---|---:|---:|']
    for k,m in summary['metrics'].items():lines.append(f"| {k} | {m['mse']:.5f} | {m['nonzero_sign_accuracy']:.1%} |")
    lines += ['',f'Zero-advantage reference MSE: {np.mean(y*y):.5f}. Strong regularization nearly returns that reference; better MSE than older neural heads does not establish useful ranking.','',
        'Primary lambda1, margin0.05:102/144 reconstructed successes vs E12097;9 rescues and4 lost successes,80 switches. Zero-loss grid selection still fully abstains. E120 unchanged; no held-placement or unseen-body test. All numerical predictions/weights/normalization and task outcomes independently reproduced. Representative paired successes/failures preserved.','']
    (ROOT/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E135: compact regularized task value';assert marker not in text
    p.write_text(text+'\n'+marker+'\n\n32 measured structured features, linear ridge paired advantage. Primary102/144 vs E12097 (9 gains/4 losses). Sign52.1%; zero-loss grid abstains. No promotion or held evaluation. Report: `outputs/ease_compact_response_value_v1/report.md`.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));ax.bar(['E120','E135 primary'],[97,102]);ax.set_ylim(0,144);ax.set_ylabel('Reconstructed fitting successes /144');ax.set_title('9 rescues,4 losses: not promoted');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    for name in ('compact_response_value.py','compact_response_artifacts.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','compact_response_value import ROOT').replace('E128','E135').replace('Task-value ranking diagnosis','Compact regularized task value').replace('Diagnosis only; no new policy.','Outcome-selected OOF task-value decisions; no promotion.')
    Path(__file__).with_name('compact_response_videos.py').write_text(s);print('E135 verified; prepared',len(records),'videos',flush=True)

if __name__=='__main__':run()

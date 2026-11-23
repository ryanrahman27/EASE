"""Persist E133 numerical audit and representative executed outcome videos."""
import shutil
from pathlib import Path
import numpy as np
from .structured_response import ROOT, KINDS, MARGINS, fit, apply, preaction
from .short_horizon_paired import data, score
from .abstaining_task_value_gate import COLLECT, read, save, fingerprint

def run():
    d=np.load(ROOT/'dataset.npz');x,y,c,folds,meta,tasks=data();o=np.load(ROOT/'oof.npz');primary=read(ROOT/'primary.json');summary=read(ROOT/'summary.json')
    for i in range(len(x)):
        z,common,error=preaction(x[i].astype(np.float64))
        for a,b in ((z,d['effect'][i]),(common,d['common'][i]),(error,d['error'][i])):np.testing.assert_array_equal(a,b)
    for kind in KINDS:
        out=np.empty_like(d['target']);common=np.empty_like(out)
        for f in range(4):
            tr,va=folds!=f,folds==f;model=np.load(ROOT/'models'/f'{kind}_{f}.npz');w=fit(kind,d['effect'][tr],d['target'][tr],d['phase'][tr]);np.testing.assert_array_equal(model['weights'],w);np.testing.assert_array_equal(model['training'],np.flatnonzero(tr))
            out[va]=apply(kind,w,d['effect'][va],d['phase'][va]);common[va]=apply(kind,w,d['common'][va],d['phase'][va])
        direction=d['error']-common;direction/=np.maximum(np.linalg.norm(direction,axis=1,keepdims=True),1e-12)
        np.testing.assert_array_equal(out,o[kind+'_displacement']);np.testing.assert_array_equal(np.sum(direction*out,axis=1),o[kind])
    grid=[score(o[k]>m,dict(kind=k,margin=m),meta,tasks) for k in KINDS for m in MARGINS];assert grid==read(ROOT/'grid.json')
    for path,h in {**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'target_input_hashes.json')}.items():assert fingerprint(path)==h
    save(ROOT/'verification.json',dict(predictions_reproduced=True,features_preaction=True,fold_models_reproduced=True,task_outcomes_reproduced=True,prior_inputs_unchanged=True,held_rows=0))
    wanted={}
    for dof in (6,7):
        for gain in (True,False):
            rows=[r for r in primary['episodes'] if r['success']==gain and r['success']!=r['baseline_success'] and len(tasks[r['case']]['home'])==dof]
            for r in rows[:2]:
                for phase in (r['phase'],None):wanted[r['case'],r['seed'],phase]=True
    records=[]
    for case,seed,phase in wanted:
        method=f'on_{seed}' if phase is None else f'off_{phase}_{seed}';stem=f'{case}_{method}_frozen_bias';trial=read(COLLECT/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:shutil.copy2(COLLECT/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        records.append(dict(case=case,method=method,mode='frozen_bias',success=trial['success'],collision=trial['collision']))
    save(ROOT/'records.json',records)
    lines=['# E133: structured one-step action response','','Measured-history inputs only; same four fitting-family folds and1108 clean starts. E120 unchanged. No unseen-body or held-placement evaluation.','',
        'Predict paired off-minus-on Cartesian displacement as estimated history Jacobian times command difference, followed by a fitted scalar, phase-specific scalar, or3x3 linear response. Models have no intercept and use no final-success targets. Scalar fits use an additive1e-18 denominator floor; linear fit uses1e-6 times mean Gram diagonal ridge. This clarifies the protocol’s general ridge description.','',
        '| Model | Progress RMSE (micrometers) | Sign accuracy |','|---|---:|---:|']
    for k,m in summary['metrics'].items():lines.append(f"| {k} | {m['progress_rmse_um']:.2f} | {m.get('sign_accuracy',float('nan')):.1%} |")
    lines+=['','Previous E132 MLP one-step RMSE98.61 micrometers, zero reference50.67. Scalar achieves17.69 on the same family splits. Input representation and model class both change, so this is a structured-method comparison rather than an architecture-only ablation. Sign is evaluated on974 rows with absolute truth above1 micrometer.','',
        'Primary scalar gate at10 micrometers:101/144 complete-task successes versus E12097/144,12 rescues and8 losses,131 switches. These outcomes are reconstructed from existing8-step off-pulse continuations. One-step prediction improvements do not establish safe8-step control. All prespecified no-loss grid candidates abstain; E120 remains baseline.','',
        'Next controlled experiment: one-step intervention matching the prediction horizon, keeping scalar/10-micrometer rule fixed. Videos are outcome-selected counterfactual replays, not fresh E133 rollouts. All predictions, pre-action features, fitted weights and task reconstructions reproduced; source inputs unchanged.','']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,4));names=['Zero','E132 MLP','E133 scalar'];vals=[50.67096,98.605,17.69465];ax.bar(names,vals);ax.set_ylabel('OOF one-step progress RMSE (micrometers)');ax.set_title('Same fitting families; measured structured predictor')
    for i,v in enumerate(vals):ax.text(i,v+1,f'{v:.2f}',ha='center')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E133: structured one-step response';assert marker not in text
    p.write_text(text+'\n'+marker+'\n\nMeasured-history scalar response: one-step OOF progress RMSE17.69um vs zero50.67 and E132 MLP98.61;90.86% nonzero sign accuracy. Eight-step pulse reconstruction101/144 vs E12097 (12 gains/8 losses), so not promoted. No-loss grid abstains. Same fitting-family splits, held untouched. Report: `outputs/ease_structured_response_v1/report.md`.\n',encoding='utf-8')
    for name in ('structured_response.py','structured_response_artifacts.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','structured_response import ROOT').replace('E128','E133').replace('Task-value ranking diagnosis','Structured one-step prediction').replace('Diagnosis only; no new policy.','One-step predictor, existing eight-step pulse outcomes; not promoted.')
    Path(__file__).with_name('structured_response_videos.py').write_text(s)
    print('Verified E133; prepared',len(records),'success/failure videos',flush=True)

if __name__=='__main__':run()

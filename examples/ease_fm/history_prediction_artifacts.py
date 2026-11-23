"""E141 prediction-only scientific report and clearly labeled source-trajectory videos."""
import shutil
from pathlib import Path
import numpy as np
from .learned_history_weights import ROOT
from .expanded_structured_pairs import COLLECT
from .abstaining_task_value_gate import read,save,DATA

def run():
    s=read(ROOT/'summary.json');a=read(ROOT/'audit_summary.json');v=read(ROOT/'verification.json');assert v['predictions_reproduced'];d=np.load(ROOT/'dataset.npz');meta=read(ROOT/'rows.json');tasks=read(ROOT/'cases.json');pred=np.load(ROOT/'oof.npz')['prediction'];ref=np.load(ROOT/'references.npz')['uniform'];target=d['target'];benefit=np.mean((ref[:,:,:3]-target[:,:,:3])**2,axis=(1,2))-np.mean((pred[:,:,:,:3]-target[None,:,:,:3])**2,axis=(0,2,3));dof=d['mask'].sum(1);wanted={}
    for n in (6,7):
        ids=np.flatnonzero(dof==n)
        for idx in (ids[np.argmax(benefit[ids])],ids[np.argmin(benefit[ids])]):r=meta[idx];wanted[r['case'],r['seed']]=True
    for n in (6,7):
        for success in (True,False):
            for r in meta:
                if len(tasks[r['case']]['home'])!=n:continue
                source=DATA/'collection' if r['case']<56 else COLLECT;t=read(source/'trials'/f"{r['case']}_on_{r['seed']}_frozen_bias.json")
                if t['success']==success:wanted[r['case'],r['seed']]=True;break
    for folder in ('trials','states'):(ROOT/folder).mkdir()
    rows=[]
    for case,seed in wanted:
        source=DATA/'collection' if case<56 else COLLECT;method=f'on_{seed}';stem=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{stem}.json')
        for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
            for ext in exts:shutil.copy2(source/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        rows.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],collision=t['collision'],role='Recorded E120 source trajectory, not learned-model control'))
    save(ROOT/'records.json',rows)
    metrics={**s['metrics'],**a['extra_reference_metrics']};lines=['# E141: learned history weights for kinematic estimation','','This is a prediction-only study, not a new task-success result. The learned model predicts a local motion map from 32 past measured joint/tool transitions. For supervised evaluation, that map is multiplied by the actual next joint increment to predict tool displacement. The future joint increment is a query used in the loss, not an input to the history-weight network. This tests kinematic estimation, not complete command-to-motion forecasting.','',
        'The Transformer outputs positive transition weights inside an explicit least-squares solve. It is a learned/analytical hybrid, distinct from the older inverse-free MotionTransformer. No object/contact truth, body IDs or privileged Jacobian enters its input. Four fitting-family folds and three fixed seeds; no final task-success labels or checkpoint selection.','',
        '| Estimator | Position RMSE (mm) | Scaled orientation RMSE (mm) |','|---|---:|---:|']
    for name,m in metrics.items():lines.append(f"| {name} | {m['all']['position_rmse_mm']:.4f} | {m['all']['rotation_rmse_mm']:.4f} |")
    lines += ['', 'Orientation error is measured in ROT_SCALE times the tool-up-vector increment, expressed in millimeter-equivalent units; it is not an angle in millimeters. Paired queries share histories and are not independent episodes.','',
        'The learned model reduces position MSE by about62% versus uniform32 and improves both joint counts. However, a stronger training-fold-tuned constant kernel reaches0.874mm, slightly better than learned0.904mm. Axis scaling alone does not explain the improvement. These reference controls were added after the primary result and are labeled diagnostic; their parameters use fitting folds only, never validation targets.', '',
        'All four reference folds selected forgetting1 and ridge1e-7, the low-ridge boundary. This motivates E142: expand that boundary and initialize learned, mass-normalized relative weights at the stronger kernel. Do not credit the constant-kernel gain to attention. No controller, held-placement or fresh-test rollout was performed in E141.', '',
        'Independent verification reproduced all predictions exactly, checked family exclusions and training-only scales, and confirmed zero-initialized weights reproduce uniform32. Source data and E120 are unchanged. Videos show recorded E120 success/failure trajectories supplying the prediction histories; they are NOT demonstrations of learned-estimator control.', '']
    (ROOT/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E141: learned history weights';assert marker not in text;text=text.replace('Revision 126 · 15 November 2026','Revision 129 · 15 November 2026',1)
    p.write_text(text+'\n'+marker+'\n\nDirect measured-motion estimation,32-history Transformer weighting inside explicit ridge solve. Learned OOF position RMSE0.904mm vs uniform1.468 and analytical1.392; both joint counts improve. Stronger fitting-tuned constant kernel0.874mm beats learned, so no attention advantage over that reference. Prediction-only, no task rollout; all references and labels verified. Report: `outputs/ease_learned_history_weights_v1/report.md`.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['uniform32','analytical','axis_scaled_uniform','learned','tuned_constant_kernel'];fig,ax=plt.subplots(figsize=(9,4));vals=[metrics[k]['all']['position_rmse_mm'] for k in names];ax.bar(range(len(names)),vals,color=['#777777','#777777','#777777','#2679b2','#d58432']);ax.set_xticks(range(len(names)),['Uniform32','Analytical','Scaled uniform','Learned weights','Tuned constant']);ax.set_ylabel('Position prediction RMSE (mm)');ax.set_title('Excluded fitting families; no controller rollout')
    for i,val in enumerate(vals):ax.text(i,val+.02,f'{val:.3f}',ha='center')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    for name in ('history_prediction_artifacts.py','history_weights_audit.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    st=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','learned_history_weights import ROOT').replace('E128','E141').replace('Task-value ranking diagnosis','History prediction diagnostics').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Recorded E120 source trajectories for prediction diagnostics. No learned-estimator controller was executed.').replace('Measured-history control | No physics selector or geometry filter','RECORDED E120 | Prediction study only; not new control')
    Path(__file__).with_name('history_prediction_videos.py').write_text(st);print('E141 report saved; prepared',len(rows),'diagnostic success/failure videos',flush=True)

if __name__=='__main__':run()

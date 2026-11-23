"""Archive E142/E143 prediction results without mislabeling source videos as control."""
import shutil
from pathlib import Path
from .abstaining_task_value_gate import read,save,fingerprint
from .anchored_history_weights import ROOT as A
from .history_weight_ensemble import ROOT as B

def run():
    a,b=read(A/'summary.json'),read(B/'summary.json')
    paragraphs={
      'E142': 'Mass-normalized Transformer weights are anchored to a training-fold-selected constant kernel; ridge and forgetting are frozen. Four family splits and all three prespecified seeds are retained. Pooled position RMSE improves from 0.864 to 0.838 mm, but one seed is slightly worse than the reference, so the individual-model robustness screen fails. No task episodes were executed here. The reference grid is finite and includes a boundary solution, not a globally optimal analytical estimator.',
      'E143': 'The fixed mean of all three E142 Jacobians reaches 0.817 mm versus constant-kernel 0.864 mm: 10.5% lower position MSE. Both joint counts improve; each leave-one-model-out diagnostic also beats the constant reference overall. All three models remain in the primary ensemble; no best seed or pair is selected. This passes the prediction screen for a full-controller experiment, not evidence of higher task success.'}
    for experiment,root,metrics in [('E142',A,a['metrics']),('E143',B,{'constant':a['metrics']['constant'],**b['metrics']})]:
        lines=[f'# {experiment}: measured-history motion prediction','',paragraphs[experiment],'','| Estimator | All position RMSE (mm) | Six joints | Seven joints |','|---|---:|---:|---:|']
        for k,m in metrics.items():lines.append(f"| {k} | {m['all']['position_rmse_mm']:.4f} | {m['6']['position_rmse_mm']:.4f} | {m['7']['position_rmse_mm']:.4f} |")
        lines += ['', 'The network sees past measured joint/tool transitions only. Actual next joint motion is the supervised query, not a network input. This is a learned-weighted analytical solve, not a direct flow-matching action policy. The 4,416 paired labels share 2,208 histories and are not independent episodes. Source data, family splits, weight normalization, predictions, and immutable E120 checkpoint hashes were verified.', '', 'Success/failure context videos are the already saved E120 trajectories in [the E141 gallery](../ease_learned_history_weights_v1/videos/index.html). Those clips are source-data examples, not control by this estimator. Full estimator-control videos belong to E144.', '']
        (root/'report.md').write_text('\n'.join(lines))
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,4));names=list(metrics);vals=[metrics[k]['all']['position_rmse_mm'] for k in names];ax.bar(names,vals);ax.set_ylabel('Position RMSE (mm)');ax.set_title(experiment+' | excluded fitting families | prediction only');fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
        save(root/'media_reference.json',{'gallery':'../ease_learned_history_weights_v1/videos/index.html','role':'Recorded E120 successes and failures; no new controller execution in this prediction study'})
        report=Path(__file__).with_name('TECHNICAL_REPORT.md');text=report.read_text(encoding='utf-8');marker=f'## {experiment}: anchored history prediction';assert marker not in text;report.write_text(text+'\n'+marker+'\n\n'+paragraphs[experiment]+f' Report: `{root}/report.md`.\n',encoding='utf-8')
        shutil.copy2(__file__,root/'source'/Path(__file__).name)
        save(root/'artifact_hashes.json',{str(p.relative_to(root)):fingerprint(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print('E142/E143 reports, figures and source-video references archived',flush=True)

if __name__=='__main__':run()

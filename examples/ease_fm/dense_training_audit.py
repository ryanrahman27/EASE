"""Reproduce E150 predictions and save the unsuccessful controlled comparison."""
import shutil
from pathlib import Path
import numpy as np
import torch
from .dense_joint_training import ROOT,DATA,prior,CONDITIONS
from .abstaining_task_value_gate import read,save,fingerprint,family

def run():
    torch.set_num_threads(4);d=dict(np.load(ROOT/'dataset.npz'));source=dict(np.load(DATA/'dataset.npz'));p=dict(np.load(ROOT/'predictions.npz'));tasks=read(ROOT/'cases.json');training=read(ROOT/'training.json');v={k:torch.tensor(d[k],device='cuda') for k in ('dq','dx','mask','prior','query')}
    np.testing.assert_array_equal(d['query'][:,1:],source['query']);np.testing.assert_array_equal(d['target'][:,1:],source['target']);np.testing.assert_array_equal(d['query_mask'][:,1:],source['query_mask'])
    with torch.no_grad():
        f=prior.features(v['dq'],v['dx'],v['mask'],v['prior'])
        for split in range(4):
            ids=np.flatnonzero(d['fold']!=split);va=np.flatnonzero(d['fold']==split);assert not {family(tasks,int(i)) for i in d['case'][ids]}&{family(tasks,int(i)) for i in d['case'][va]}
            for si,seed in enumerate(prior.SEEDS):
                for ci,condition in enumerate(('E146',*CONDITIONS)):
                    path=prior.ROOT/'models'/f'fold_{split}'/f'{seed}.pt' if ci==0 else ROOT/'models'/condition/f'fold_{split}'/f'{seed}.pt';m=prior.PriorWeights().cuda().eval();m.load_state_dict(torch.load(path,weights_only=True));j,_=m(v['dq'][va],v['dx'][va],v['mask'][va],v['prior'][va],f[va]);pred=(v['query'][va].double()@j).cpu().numpy();expected=p['reference'][si,va] if ci==0 else p['trained'][ci-1,si,va];np.testing.assert_array_equal(pred,expected)
                    if ci:
                        log=next(r for r in training if r['condition']==condition and r['fold']==split and r['seed']==seed);np.testing.assert_array_equal(log['train_indices'],ids);np.testing.assert_array_equal(log['validation_indices'],va)
    for path,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['inputs']}.items():assert fingerprint(path)==h
    save(ROOT/'verification.json',dict(all_predictions_exact=True,directional_targets_unchanged=True,family_exclusion=True,same_initial_checkpoints=True,identical_seed_batch_schedules_by_construction=True,source_hashes_unchanged=True))
    s=read(ROOT/'summary.json');lines=['# E150: observed-only versus dense directional fine-tuning','','The same738 histories,96 fitting placements, four ancestry-family folds and three initial E146 checkpoints are used in both conditions. Continued observed-only learning controls for extra updates and context selection; the primary mixed objective adds equal-weight measured signed-joint probes. All final seeds are retained and averaged. No exact-Jacobian training labels.','', '| Condition | Observed-next position RMSE (mm) | Directional position RMSE (mm) | Directional rotation RMSE (mm) |','|---|---:|---:|---:|']
    for k,m in s['metrics'].items():lines.append(f"| {k} | {m['observed']['all']['position_rmse_mm']:.4f} | {m['directional']['all']['position_rmse_mm']:.4f} | {m['directional']['all']['rotation_rmse_mm']:.4f} |")
    lines+=['','The mixed model does not reach the prespecified5% directional position-MSE improvement and slightly worsens seven-joint directional error. Observed-only fine-tuning performs slightly better directionally than mixed. **No complete controller or fresh evaluation is authorized by this result.** All weights, predictions and losses are saved, including the negative result.','', 'These are repeatedly used development folds, not an unbiased final generalization estimate. Collection-policy kernels were cross-fitted once; the entire policy/data-collection pipeline was not nested separately inside each later fold. Network fitting excludes validation-family transition labels, but this does not make the whole research pipeline independent. Reserved fresh placements remain unopened. E120 unchanged.','', 'Source-context success/failure videos: [E144](../ease_history_estimator_control_v1/videos/index.html). The one-step training branches have no full-task success label; before/after states are saved in E148. No new policy-performance video is claimed for this failed prediction-stage candidate.',''];(ROOT/'report.md').write_text('\n'.join(lines));report=Path(__file__).with_name('TECHNICAL_REPORT.md');text=report.read_text(encoding='utf-8');assert '## E150:' not in text;report.write_text(text+'\n## E150: dense directional supervision\n\nMixed directional RMSE0.1762mm versus fixed E1460.1768 and observed-only continuation0.1752; the5% improvement screen fails. No new controller or fresh evaluation. Development folds are adaptive and collection was not nested end-to-end. Report: `outputs/ease_dense_joint_training_v1/report.md`.\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4));names=list(s['metrics']);ax.bar(names,[s['metrics'][k]['directional']['all']['position_rmse_mm'] for k in names]);ax.set_ylabel('Directional position RMSE (mm)');ax.set_title('E150 | same histories and initial checkpoints');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E150 verified and archived',flush=True)

if __name__=='__main__':run()

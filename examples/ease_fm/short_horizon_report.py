"""E132 short-horizon paired-supervision report and artifact finalization."""
import json
import shutil
from pathlib import Path
import numpy as np
from .short_horizon_paired import ROOT,read,save,fingerprint


def report():
    s=read(ROOT/'summary.json');p=read(ROOT/'protocol.json');primary=read(ROOT/'primary_outcomes.json');grid=read(ROOT/'validation_grid.json')
    lines=['# E132: Dense short-horizon paired supervision','',
        '**PRIVILEGED input study.** The learned short-target model does not beat E120. A future-informed short-target diagnostic shows useful headroom but is not a deployable policy, an achieved learned score, or a guarantee of no losses.', '',
        '## Controlled task and supervision','']
    for key in ('inputs_label','target','dense','training','initialization','primary','secondary','evaluation','baseline'):lines += [f"**{key}:** {p[key]}",'']
    lines += ['The original long-target control is E131: same182 privileged features and family splits, with two final-success heads. E132 predicts four local progress differences and eight on/off collision-prefix labels. Thus target horizon, output dimensions and supervised loss change together; no claim is made that one of those components alone causes the result.', '',
        'Each original clean phase-entry pair now supplies12 labels: four continuous effects and eight collision labels. This gives4432 continuous and8864 binary targets, not13296 independent examples. All on/off continuations start from identical integration/history states. No new rollouts or new placements were collected.', '',
        'A positive progress target means the off-pulse ends closer to the phase-entry goal than E120-on. The goal remains fixed for this label even if the actual scripted controller advances to another phase. The window is1/2/4/8 control steps (0.04/0.08/0.16/0.32 seconds); E120 resumes after the off-pulse. All branches had sufficient saved horizon.', '',
        f"Rows crossing a phase boundary within each horizon: {s['phase_transition_rows']}. No rows are dropped based on this future information. Final task success remains the control-evaluation outcome, separate from the local progress target.", '',
        '## Out-of-fold short-horizon prediction','', '| Steps | Nonzero progress pairs (>1 micrometer) | Learned RMSE (mm) | Zero-effect RMSE (mm) | Train-mean RMSE (mm) | Correct nonzero sign |', '|---:|---:|---:|---:|---:|---:|']
    for r in s['horizon_metrics']:lines.append(f"| {r['horizon']} | {r['progress_nonzero_above_1um']} | {r['progress_rmse_mm']:.4f} | {r['zero_rmse_mm']:.4f} | {r['training_mean_rmse_mm']:.4f} | {r['sign_accuracy_nonzero']:.1%} |")
    lines += ['', 'The zero-effect reference predicts identical on/off progress. The train-mean reference is estimated separately inside each training fold. The learned model is worse than both at all four horizons, despite denser continuous supervision.', '',
        '| Steps | On/off collision-positive rows | Learned on/off Brier | Train-frequency on/off Brier |','|---:|---|---|---|']
    for r in s['horizon_metrics']:lines.append(f"| {r['horizon']} | {r['on_collision_count']}/{r['off_collision_count']} | {r['on_brier']:.4f}/{r['off_brier']:.4f} | {r['on_frequency_brier']:.4f}/{r['off_frequency_brier']:.4f} |")
    lines += ['', 'Immediate collision labels remain sparse; denser timestamps do not make collision events common. Learned collision probabilities also underperform training-frequency references and should not be interpreted as calibrated safety guarantees.', '',
        '## Complete-task choices on144 fitting validation episodes','',
        'Primary learned rule requires every model\'s predicted8-step improvement >0.25mm and all predicted off-collision probabilities across horizons/models <0.1. Only the first accepted clean phase receives a pulse. Complete-task successes are reconstructed from previously executed paired continuations. Post-collision entries are omitted because task success is already irreversibly false.', '',
        '**TRUE-SHORT DIAGNOSTIC:** applies that rule to actual future progress/collision labels. It tests the usefulness of this proxy with perfect short-target knowledge. It is not an inference-time physics selector or a newly trained policy, and its106/144 score must not be presented as learned performance.', '',
        '| Condition | Success /144 | Rescues vs E120 | Lost successes | Switches |','|---|---:|---:|---:|---:|','| E120 | 97 | 0 | 0 | 0 |']
    for name,r in s['primary'].items():lines.append(f"| {name} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', 'The true-short diagnostic improves aggregate task success but still loses four E120 successes. Thus there are two separate limitations: the learned short model predicts poorly, and the short proxy itself is not sufficient for the no-loss objective. Neither the learned96/144 nor the diagnostic106/144 justifies replacing E120.', '',
        '| Learned-short case/seed | Selected phase | E120 success | Learned-short success |','|---|---|---|---|']
    for r in primary['learned_short']['episodes']:
        if r['success']!=r['baseline_success']:lines.append(f"| {r['case']}/{r['seed']} | {r['phase']} | {r['baseline_success']} | {r['success']} |")
    lines += ['', '## Predeclared secondary margin/risk grid','', '| Progress margin (mm) | Risk cap | Success /144 | Gains | Losses | Switches |','|---:|---:|---:|---:|---:|---:|']
    for r in grid:
        margin='abstain' if r['rule']['progress_margin']>1 else f"{1000*r['rule']['progress_margin']:.2f}"
        lines.append(f"| {margin} | {r['rule']['risk_cap']} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    chosen=s['selected']
    lines += ['',f"Selected zero-loss rule: `{chosen['rule']}`; {chosen['success']}/144, {chosen['gains']} gains/{chosen['losses']} losses. It abstains completely. These are fitting-validation selection scores, not independent final-test results.",'',
        '## Interpretation and next controlled comparison','',
        'Shortening the horizon supplies more nonzero paired labels but does not solve out-of-family prediction with this MLP. At the same time, the future-informed diagnostic shows that accurate local predictions could guide some useful choices. This separates a concrete prediction-generalization failure from the additional proxy-alignment problem.', '',
        'A next controlled comparison should benchmark a structured action-response predictor against this generic MLP at one step before extending the horizon. Keep the same inputs, families and dense labels, report zero-effect and fitting-only references, and continue scoring final-task losses. No such alternative model was trained here.', '',
        'All inputs remain explicitly privileged simulator data. E120, original held-placement results, and prior checkpoints stay unchanged. No new held-placement, unseen-body, hardware, or sensor-deployment claim is made. Videos are outcome-selected counterfactual continuations: all learned/long-target outcome changes, representative true-short diagnostic changes, and earlier diagnosis pairs.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    steps=[r['horizon'] for r in s['horizon_metrics']]
    axes[0].plot(steps,[r['progress_rmse_mm'] for r in s['horizon_metrics']],'-o',label='Learned short model')
    axes[0].plot(steps,[r['zero_rmse_mm'] for r in s['horizon_metrics']],'-o',label='Zero-effect reference')
    axes[0].set_xticks(steps);axes[0].set_xlabel('Control steps');axes[0].set_ylabel('Progress-difference RMSE (mm)');axes[0].legend();axes[0].set_title('Excluded fitting families')
    vals=[97,s['primary']['long_target']['success'],s['primary']['learned_short']['success'],s['primary']['true_short_diagnostic']['success']]
    labels=['E120','Long-target\nlearned','Short-target\nlearned','TRUE short\ndiagnostic']
    axes[1].bar(labels,vals,color=['#777777','#2679b2','#d58432','#9255a0'])
    for i,v in enumerate(vals):axes[1].text(i,v+2,f'{v}/144',ha='center')
    axes[1].set_ylim(0,144);axes[1].set_ylabel('Reconstructed task successes');axes[1].set_title('Last column uses future information')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');text=path.read_text(encoding='utf-8');marker='## E132: dense short-horizon paired targets';assert marker not in text
    text=text.replace('Revision 119 · 15 November 2026','Revision 120 · 15 November 2026',1)
    text+='\n'+marker+'\n\nSame privileged182 inputs/family folds;4432 progress and8864 collision-prefix labels at1/2/4/8 steps from1108 matched starts. Learned progress RMSE exceeds zero-effect reference at every horizon (8-step7.28mm vs3.36mm). Reconstructed task success: E12097/144, long-target96/144, learned-short96/144 (2 gains/3 losses), TRUE-short future-informed diagnostic106/144 (13/4). Diagnostic is not learned performance and still loses successes. Selected zero-loss learned rule abstains. No new rollouts, held-placement test or E120 changes. Full report: `outputs/ease_short_horizon_paired_v1/report.md`.\n'
    path.write_text(text,encoding='utf-8')
    for name in ('short_horizon_paired.py','short_horizon_report.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)


def finish():
    v=read(ROOT/'verification.json');assert v['all_dense_targets_reconstructed'] and v['oof_predictions_exact']
    for path,h in {**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'target_input_hashes.json'),**read(ROOT/'media_input_hashes.json')}.items():assert fingerprint(path)==h
    for path,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/path)==h
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    manifest=read(ROOT/'videos'/'manifest.json')
    for r in manifest['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    save(ROOT/'status.json',dict(status='complete',privileged_inputs=True,new_fold_models=12,new_rollouts=0,matched_starts=1108,progress_labels=4432,collision_labels=8864,held_evaluated=False,E120_unchanged=True,zero_loss_rule_abstains=True,videos=len(manifest['videos'])))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nVerification: all dense labels independently regenerated; all1108 paired prefixes/snapshots exact; fold/target normalizations training-only; all OOF predictions and outcome reconstructions reproduced; E120/prior models unchanged. Saved{len(manifest['videos'])} trace-verified success/failure videos.\n")
    path=ROOT/'videos'/'index.html';path.write_text(path.read_text().replace('</h1>','</h1><p>PRIVILEGED input study. The TRUE-short diagnostic uses future outcomes and is not learned performance. Videos replay saved on/off continuations; see the report for selection provenance.</p>',1))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(read(ROOT/'status.json'),indent=2))


if __name__=='__main__':
    import sys
    {'report':report,'finish':finish}[sys.argv[1]]()

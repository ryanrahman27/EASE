"""E129 scientific report and final artifact checks."""
import json
import shutil
from pathlib import Path
import numpy as np
from .paired_advantage_objective import ROOT, read, save, fingerprint


def report():
    s=read(ROOT/'summary.json');training=read(ROOT/'training.json');primary=s['primary'];grids=read(ROOT/'validation_grids.json')
    lines=['# E129: Direct paired advantage versus two success heads','',
        'The tested direct signed-advantage objective generalizes worse than the two-head baseline on the same fitting-family folds. E120 remains unchanged. No original held-placement benchmark or unseen bodies were evaluated.', '',
        '## Controlled comparison','',
        'Four identical ancestral-placement folds,1108 clean paired phase entries,48 fitting placements and three frozen motion-estimator seeds. Each complete source family (original, shifted, and descendants) stays in one fold. Reuse the E126 two-head models; train12 direct models (three gate seeds per fold). All same139 inputs, hidden widths64/32 with tanh, identical hidden initialization by seed, fold-local normalization/clipping,1500 full-batch AdamW updates, lr0.001, weight decay0.001. No class weighting, resampling, additional data, early stopping, or held-driven selection.', '',
        'Two-head objective: paired binary cross-entropy for final on/off success, with advantage computed as sigmoid(off)-sigmoid(on). Direct objective: one tanh-bounded scalar trained by MSE to observed off-minus-on success (-1,0,+1). The output layer and objective necessarily differ; this comparison does not cover every possible direct-ranking objective or output parameterization.', '',
        'The primary rule was fixed before training: every gate model predicts advantage >0.1, with inter-model spread <=2.0. First accepted clean phase gets up to8 correction-off steps, then E120 resumes. Both objectives are scored with the same rule. Secondary analysis uses the unchanged E126 threshold/spread grid and zero-E120-loss selection criterion; those selected scores use tuning-validation data.', '',
        'Complete-task outcomes are reconstructed from previously executed, exact-state paired continuations. Starts already contaminated by collision are omitted because success is irreversibly false. This is not a newly executed online-policy benchmark, and does not assert trajectory equivalence for post-collision decisions. All selected video branches were actually simulated previously. No physics selector is introduced.', '',
        'The144 validation episodes use the expanded fitting-placement set. Their97/144 E120 baseline must not be compared directly with52/72 on the different original evaluation set. Rows/phases share trajectories and are statistically dependent.', '',
        '## Out-of-fold prediction quality','',
        '| Objective | Advantage MSE | Decisive-pair MSE | Neutral-pair MSE | Correct decisive sign | Beneficial reversed /65 | Harmful reversed /54 |', '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in s['metrics'].items():lines.append(f"| {name} | {r['mse']:.4f} | {r['decisive_mse']:.4f} | {r['neutral_mse']:.4f} | {r['decisive_sign_accuracy']:.1%} | {r['beneficial_reversed']} | {r['harmful_reversed']} |")
    lines += ['',f"A constant zero-advantage reference has MSE{s['zero_advantage_reference']['mse']:.4f}, decisive-pair MSE1.0, and neutral-pair MSE0. It abstains rather than ranking choices. There are989 neutral and119 decisive pairs. Report decisive metrics separately to avoid rewarding only neutral predictions.", '',
        '## Fixed-rule complete-task validation','', '| Method | Success /144 | Gains vs E120 | Losses vs E120 | Interventions |', '|---|---:|---:|---:|---:|', '| E120 | 97 | 0 | 0 | 0 |']
    for name,r in primary.items():lines.append(f"| {name} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', 'The direct model changes more decisions:107 interventions versus23. It rescues seven E120 failures but breaks ten successes. The two-head model rescues one and breaks two. More rescues alone are not a better policy.', '',
        '## Every fold and gate seed', '', '| Method | Fold | MSE | Correct decisive sign |', '|---|---:|---:|---:|']
    for r in s['fold_metrics']:lines.append(f"| {r['method']} | {r['fold']} | {r['mse']:.4f} | {r['decisive_sign_accuracy']:.1%} |")
    lines += ['', 'Direct regression has higher OOF advantage MSE in all four folds.', '', '| Method | Gate seed | MSE | Correct decisive sign |', '|---|---:|---:|---:|']
    for r in s['seed_metrics']:lines.append(f"| {r['method']} | {r['gate_seed']} | {r['mse']:.4f} | {r['decisive_sign_accuracy']:.1%} |")
    lines += ['', '| Direct training fold | Last optimization MSE by seed |', '|---:|---|']
    for r in training:lines.append(f"| {r['fold']} | {', '.join(f'{v:.5f}' for v in r['training_mse'])} |")
    lines += ['', 'The direct model fits its training targets closely (last-step MSE about0.004–0.011), while excluded-family MSE is0.535. Switching the objective did not remove the training-to-family generalization gap.', '',
        '## Secondary identical-grid comparison','', '| Minimum advantage | Maximum spread | Two-head gains/losses/switches | Direct gains/losses/switches |', '|---:|---:|---|---|']
    for a,b in zip(grids['two_head'],grids['direct']):
        assert a['rule']==b['rule'];r=a['rule']
        lines.append(f"| {r['min_advantage']} | {r['max_spread']} | {a['gains']}/{a['losses']}/{a['switches']} | {b['gains']}/{b['losses']}/{b['switches']} |")
    lines += ['', 'Both objectives select full abstention under the zero-loss criterion, giving97/144 and zero rescues. This is not an improvement over E120. No full-fitting final model was promoted or tested on original held placements.', '',
        '## Interpretation and next controlled test','',
        'This direct tanh-MSE formulation makes stronger, less reliable intervention predictions outside its training families. The objective change alone did not fix task-value generalization. The result does not rule out alternative ranking losses, but provides no reason to replace E120 or promote this direct model.', '',
        'The next separate ablation should test richer measured task-state information under the same family splits while retaining the two-head objective as the comparison. Object-relative pose, grasp/contact state where observable, and orientation error are candidate inputs; simulator-only quantities must be labeled privileged. Do not combine representation changes with a new loss and then attribute the result to one factor.', '',
        'All outcomes, fold/seed metrics, selected rules, checkpoints, normalization, input hashes and diagnostic videos are saved. Media include every fixed-rule outcome change and the earlier E128 diagnostic pairs. Videos are outcome-selected counterfactual continuations, not an unbiased video sample.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=['E120','Two-head','Direct advantage'];values=[97,primary['two_head']['success'],primary['direct']['success']]
    axes[0].bar(labels,values,color=['#777777','#2779b2','#d37c2f'])
    for i,v in enumerate(values):axes[0].text(i,v+2,f'{v}/144',ha='center')
    axes[0].set_ylim(0,144);axes[0].set_ylabel('Reconstructed task successes');axes[0].set_title('Fixed rule / fitting-family validation')
    xx=np.arange(2)
    for j,(key,label) in enumerate((('gains','Rescued failures'),('losses','Lost successes'))):
        vals=[primary[n][key] for n in ('two_head','direct')];axes[1].bar(xx+(j-.5)*.32,vals,.3,label=label)
        for x,v in zip(xx+(j-.5)*.32,vals):axes[1].text(x,v+.2,str(v),ha='center')
    axes[1].set_xticks(xx,['Two-head','Direct advantage']);axes[1].set_ylim(0,13);axes[1].legend();axes[1].set_title('Paired changes versus E120')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');text=path.read_text(encoding='utf-8');marker='## E129: direct paired-advantage objective';assert marker not in text
    text=text.replace('Revision 116 · 15 November 2026','Revision 117 · 15 November 2026',1)
    text+=f'\n{marker}\n\nSame four fitting-family folds, inputs, seeds, hidden initialization and training budget. Direct tanh-MSE advantage regressor vs two-head BCE: OOF advantage MSE0.535 vs0.200; decisive sign47.9% vs50.4%. Fixed-rule reconstructed task success94/144 vs96/144, E12097/144. Direct7 gains/10 losses; two-head1/2. Both zero-loss selections abstain. No original held-placement retest, unseen bodies, or baseline replacement. Full report: `outputs/ease_paired_advantage_objective_v1/report.md`.\n'
    path.write_text(text,encoding='utf-8')
    for name in ('paired_advantage_objective.py','paired_advantage_report.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)


def finish():
    verification=read(ROOT/'verification.json');assert verification['all_oof_predictions_exact'] and verification['all_rule_outcomes_reproduced']
    for name,h in {**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'media_input_hashes.json')}.items():assert fingerprint(name)==h
    for name,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/name)==h
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    manifest=read(ROOT/'videos'/'manifest.json')
    for r in manifest['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    save(ROOT/'status.json',dict(status='complete',objective_comparison=True,training_folds=4,gate_seeds=3,new_fold_models=12,new_simulation_rollouts=0,held_placement_evaluation=False,E120_unchanged=True,videos=len(manifest['videos']),both_zero_loss_rules_abstain=True))
    path=ROOT/'report.md';path.write_text(path.read_text()+f"\nVerification: all1108 labels and both objectives' OOF predictions reproduced; ancestral splits and normalization checked; all rule outcomes reconstructed identically; saved video-pair snapshots/prefixes exact. E120 data and prior two-head checkpoints unchanged. Saved{len(manifest['videos'])} trace-verified success/failure videos.\n")
    hashes={str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
    save(ROOT/'artifact_hashes.json',hashes);print(json.dumps(read(ROOT/'status.json'),indent=2))


if __name__=='__main__':
    import sys
    {'report':report,'finish':finish}[sys.argv[1]]()

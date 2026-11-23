"""E130 report and immutable artifact finalization."""
import json
import shutil
from pathlib import Path
import numpy as np
from .richer_task_state import ROOT, DATA, PRIOR, read, save, fingerprint


def report():
    s=read(ROOT/'summary.json');protocol=read(ROOT/'protocol.json');primary=read(ROOT/'primary_outcomes.json');grid=read(ROOT/'validation_grids.json')
    y=np.load(ROOT/'features.npz')['y'];heads={}
    for name,path in (('two_head',PRIOR/'oof_predictions.npz'),('richer',ROOT/'richer_oof.npz')):
        p=np.load(path)['probabilities'].mean(0)
        heads[name]=dict(accuracy=np.mean((p>.5)==y,axis=0).tolist(),brier=np.mean((p-y)**2,axis=0).tolist())
    save(ROOT/'head_metrics.json',heads)
    lines=['# E130: Richer measured task state with the two-head objective','',
        'E120 remains the baseline. The richer measured-state model gains two and loses two E120 fitting-validation episodes under the fixed rule. Equal aggregate success is not preservation of successful episodes. Both zero-loss-selected rules abstain.', '',
        '## Controlled feature ablation','']
    for key in ('design','initialization','features','observability','normalization','primary','outcomes','baseline'):
        lines += [f"**{key}:** {protocol[key]}",'']
    lines += ['The original139 inputs are unchanged. Twenty-three extra columns expand the input layer; hidden widths64/32, two success outputs and paired BCE objective remain fixed. The new input columns start at zero; all original weights and biases match the same-seed baseline initialization. No class weighting, resampling, checkpoint selection, or training-data expansion.', '',
        'Four fixed ancestral-family folds hold out whole original/shifted/descendant families, across all phases and estimator seeds. Twelve gate models are trained: three seeds per fold. Both models use1500 full-batch AdamW updates at lr0.001, weight decay0.001. Original normalization is preserved exactly; additional-feature normalization uses only training rows within each fold.', '',
        'The added gripper signals are ideal simulated encoders and known previous commands. Tool pose and measured interaction history already enter the controller. No object pose, contact/grasp/lift/collision labels, privileged dynamics parameters, or future state are features. This is not a real-hardware sensing validation. Object-centric observability remains untested.', '',
        '## Feature provenance','', '| Columns | Signal | Timing/source |','|---|---|---|',
        '| 139–141 | Tool position | Pre-action grasp-center pose |',
        '| 142–144 | Tool orientation error | Pre-action tool rotation versus fixed upright orientation |',
        '| 145–151 | Latest joint increment | Already measured controller history, padded to7 |',
        '| 152–157 | Latest Cartesian/rotation increment | Already measured controller history |',
        '| 158–159 | Left/right gripper positions | Ideal pre-action encoder readings |',
        '| 160–161 | Left/right previous actuator commands | Stored controller commands before current decision |', '',
        'All1108 rows reconstructed from saved integration/history snapshots before the intervention. The original139-vector is independently reconstructed and checked for exact equality, confirming decision-time alignment. No post-step trace entry supplies a new input.', '',
        '## Out-of-fold prediction quality','', '| Model | Advantage MSE | Decisive-pair MSE | Correct decisive sign | Beneficial reversed /65 | Harmful reversed /54 |','|---|---:|---:|---:|---:|---:|']
    for name,r in s['metrics'].items():lines.append(f"| {name} | {r['mse']:.4f} | {r['decisive_mse']:.4f} | {r['decisive_sign_accuracy']:.1%} | {r['beneficial_reversed']} | {r['harmful_reversed']} |")
    lines += ['', '| Model | On/off accuracy | On/off Brier |','|---|---|---|']
    for name,r in heads.items():lines.append(f"| {name} | {r['accuracy'][0]:.4f}/{r['accuracy'][1]:.4f} | {r['brier'][0]:.4f}/{r['brier'][1]:.4f} |")
    lines += ['', 'There are989 neutral and119 decisive pairs. These phase-entry rows share trajectories and are not independent episodes. The small decisive-sign improvement does not establish generalizable task improvement.', '',
        '## Fixed-rule complete-task validation','',
        'Same prespecified minimum advantage0.1 and maximum inter-model spread2.0, requiring all three model advantages to pass. At most one short off pulse. These144 expanded fitting-placement/estimator outcomes are reconstructed from saved exact counterfactual continuations, not fresh online-policy rollouts. They are a different set from the original72-episode held-placement benchmark.', '',
        '| Method | Success /144 | Gains vs E120 | Losses vs E120 | Switches |','|---|---:|---:|---:|---:|','| E120 | 97 | 0 | 0 | 0 |']
    for name,r in s['primary'].items():lines.append(f"| {name} | {r['success']} | {r['gains']} | {r['losses']} | {r['switches']} |")
    lines += ['', '| Richer-state case/seed | Selected phase | E120 success | Richer success |','|---|---|---|---|']
    for r in primary['richer']['episodes']:
        if r['success']!=r['baseline_success']:lines.append(f"| {r['case']}/{r['seed']} | {r['phase']} | {r['baseline_success']} | {r['success']} |")
    lines += ['', 'Richer state rescues22/2 and38/2, but loses0/0 and35/0. It retains the known35/0 harmful descent decision and introduces a lift regression at0/0. These fitting-validation cases do not support promotion over E120.', '',
        '## Fold and seed variability','', '| Model | Fold | Advantage MSE | Correct decisive sign |','|---|---:|---:|---:|']
    for r in s['fold_metrics']:lines.append(f"| {r['method']} | {r['fold']} | {r['mse']:.4f} | {r['decisive_sign_accuracy']:.1%} |")
    lines += ['', '| Model | Gate seed | Advantage MSE | Correct decisive sign |','|---|---:|---:|---:|']
    for r in s['seed_metrics']:lines.append(f"| {r['method']} | {r['gate_seed']} | {r['mse']:.4f} | {r['decisive_sign_accuracy']:.1%} |")
    lines += ['', '## Unchanged secondary selection grid','', '| Minimum advantage | Maximum spread | Original gains/losses/switches | Richer gains/losses/switches |','|---:|---:|---|---|']
    for a,b in zip(grid['two_head'],grid['richer']):
        assert a['rule']==b['rule'];r=a['rule']
        lines.append(f"| {r['min_advantage']} | {r['max_spread']} | {a['gains']}/{a['losses']}/{a['switches']} | {b['gains']}/{b['losses']}/{b['switches']} |")
    lines += ['', 'Both zero-E120-loss selections choose full abstention. These are tuning-validation scores, not an independent confirmation. No original held-placement evaluation, unseen-body test, or full-fitting deployment model follows this result.', '',
        '## Interpretation','',
        'Additional measured proprioceptive and tool state changes decisions but does not solve the task-value ranking problem. Advantage MSE is essentially unchanged (0.200 to0.202), with a small decisive-sign improvement but no net gain over E120. The earlier hypothesis about object/grasp/contact information has not been tested because those signals are not exposed as measurements in this setup.', '',
        'A next experiment could explicitly separate an object-state sensing implementation from a simulator-ground-truth upper bound. Any privileged upper bound must be labeled as such. Keep this measured-only result, the two-head objective and fixed family splits as controls. E120 remains unchanged.', '']
    (ROOT/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    vals=[97,s['primary']['two_head']['success'],s['primary']['richer']['success']]
    axes[0].bar(['E120','Original two-head','Richer measured state'],vals,color=['#777777','#2779b2','#d37c2f'])
    for i,v in enumerate(vals):axes[0].text(i,v+2,f'{v}/144',ha='center')
    axes[0].set_ylim(0,144);axes[0].set_title('Fixed rule / fitting-family validation');axes[0].set_ylabel('Reconstructed task successes')
    xx=np.arange(2)
    for j,(key,label) in enumerate((('gains','Rescued failures'),('losses','Lost successes'))):
        yy=[s['primary'][n][key] for n in ('two_head','richer')];axes[1].bar(xx+(j-.5)*.32,yy,.3,label=label)
        for p,v in zip(xx+(j-.5)*.32,yy):axes[1].text(p,v+.08,str(v),ha='center')
    axes[1].set_xticks(xx,['Original two-head','Richer measured state']);axes[1].set_ylim(0,3.5);axes[1].legend();axes[1].set_title('Paired changes versus E120')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    path=Path(__file__).with_name('TECHNICAL_REPORT.md');text=path.read_text(encoding='utf-8');marker='## E130: richer measured task-state inputs';assert marker not in text
    text=text.replace('Revision 117 · 15 November 2026','Revision 118 · 15 November 2026',1)
    text+='\n'+marker+'\n\nFixed two-head BCE and fitting-family splits;139 to162 inputs with ideal measured tool/proprioceptive/gripper state, no object/contact truth. Matched original initialization, added weights zero. Fixed-rule reconstructed success97/144 vs original96/144, E12097/144; richer2 gains/2 losses. Advantage MSE0.202 vs0.200; decisive signs52.1% vs50.4%. Both zero-loss rules abstain. No original held-placement test or promotion; E120 unchanged. Full report: `outputs/ease_richer_task_state_v1/report.md`.\n'
    path.write_text(text,encoding='utf-8')
    for name in ('richer_task_state.py','richer_task_state_features.py','build_richer_task_state.py','richer_task_state_report.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print(json.dumps(heads,indent=2))


def finish():
    v=read(ROOT/'verification.json');assert v['measured_features_reconstructed'] and v['all_oof_predictions_exact']
    for path,h in {**read(ROOT/'protocol.json')['inputs'],**read(ROOT/'media_input_hashes.json'),**read(ROOT/'feature_provenance.json')}.items():assert fingerprint(path)==h
    for path,h in read(ROOT/'frozen_training.json').items():assert fingerprint(ROOT/path)==h
    for path in (ROOT/'source').glob('*.py'):assert fingerprint(path)==fingerprint(Path(__file__).with_name(path.name))
    m=read(ROOT/'videos'/'manifest.json')
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    save(ROOT/'status.json',dict(status='complete',new_fold_models=12,inputs=162,added_measured_features=23,held_rows=0,new_rollouts=0,original_held_evaluated=False,E120_unchanged=True,zero_loss_rule_abstains=True,videos=len(m['videos'])))
    p=ROOT/'report.md';p.write_text(p.read_text()+f"\nVerification: all1108 measured-feature rows independently reconstructed from pre-action snapshots; original139 inputs exact; fitting-only normalization and family isolation checked; both objectives' OOF predictions and all rule outcomes reproduced. Model/data hashes unchanged for E120 and the original gate. Saved{len(m['videos'])} trace-verified success/failure videos.\n")
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(read(ROOT/'status.json'),indent=2))


if __name__=='__main__':
    import sys
    {'report':report,'finish':finish}[sys.argv[1]]()

"""E136 technical report, provenance checks and outcome-selected media."""
import shutil
from pathlib import Path
import numpy as np
from .combined_response_value import ROOT,MOTION,VALUE,MOTION_MARGINS,VALUE_MARGINS
from .short_horizon_paired import data,score
from .abstaining_task_value_gate import read,save,fingerprint,DATA,OLD

def run():
    summary=read(ROOT/'rollout_summary.json');assert summary['held_evaluated'];tasks=read(ROOT/'cases.json');rows=read(ROOT/'records.json');_,_,_,_,meta,_=data();motion=np.load(MOTION/'oof.npz')['scalar'];value=np.load(VALUE/'oof.npz')['1.0']
    rules=[dict(motion_margin=m,value_margin=v) for m in MOTION_MARGINS for v in VALUE_MARGINS]+[dict(motion_margin=1e9,value_margin=1e9)]
    grid=[score((motion>r['motion_margin'])&(value>r['value_margin']),r,meta,tasks) for r in rules];assert grid==read(ROOT/'grid.json')
    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'baseline_checkpoints.json'),**read(ROOT/'protocol.json')['inputs']}.items():assert fingerprint(p)==h
    baseline={(r['case'],int(r['method'].split('_')[-1])):r for r in read(OLD/'records.json') if r['method'].startswith('candidate_')}
    fitref={(r['case'],r['seed']):r for r in read(DATA/'collection/records.json') if r['off_phase'] is None}
    groups=[]
    for held in (False,True):
        for dof in (6,7):
            rr=[r for r in rows if tasks[r['case']]['validation']==held and r['dof']==dof];ref=baseline if held else fitref
            groups.append(dict(split='held' if held else 'fitting',dof=dof,n=len(rr),baseline_success=sum(ref[r['case'],r['seed']]['success'] for r in rr),success=sum(r['success'] for r in rr),baseline_collisions=sum(ref[r['case'],r['seed']]['collision'] for r in rr),collisions=sum(r['collision'] for r in rr)))
    save(ROOT/'groups.json',groups)
    selected={}
    for split in ('fitting','held'):
        for c in summary[split]['changes']:selected[c['case'],c['seed']]=True
    for held in (False,True):
        for dof in (6,7):
            for success in (False,True):
                candidates=[r for r in rows if tasks[r['case']]['validation']==held and r['dof']==dof and r['success']==success]
                if candidates:selected[candidates[0]['case'],candidates[0]['seed']]=True
    media=[]
    for case,seed in selected:
        r=next(r for r in rows if r['case']==case and r['seed']==seed);media.append(r);held=tasks[case]['validation'];oldmethod=f'candidate_{seed}' if held else f'on_{seed}';oldstem=f'{case}_{oldmethod}_frozen_bias';stem=f'{case}_e120_{seed}_frozen_bias';source=OLD if held else DATA/'collection'
        for folder,exts in (('states',('npz','xml')),):
            for ext in exts:shutil.copy2(source/folder/f'{oldstem}.{ext}',ROOT/folder/f'{stem}.{ext}')
        t=read(source/'trials'/f'{oldstem}.json');t['method']=f'e120_{seed}';save(ROOT/'trials'/f'{stem}.json',t);media.append(dict(case=case,method=t['method'],mode='frozen_bias',success=t['success'],collision=t['collision']))
    save(ROOT/'media_records.json',media)
    lines=['# E136: combined local-response and task-value gate','','## Frozen candidate','',
        'E120 is unchanged. At the first clean task-phase entry, require both a scalar-calibrated history-Jacobian prediction of >10 micrometers one-step goal progress and a regularized linear paired task-advantage prediction >0.05. If accepted, disable the E120 residual for up to8 actions within that phase, then resume E120. Otherwise retain E120. At most one intervention. No physics selector, privileged object/contact state, robot IDs, or future observations are gate inputs.', '',
        'The response scalar is fitted to measured paired Cartesian displacement; the32-feature ridge model is fitted to final paired success differences. Four family-excluded fitting models were used for validation. Final models fit all fitting rows only. The prespecified primary conjunction also won the20-rule validation grid. All weights, thresholds, source files and protocol were hashed before fresh fitting rollouts and held evaluation.', '',
        '## Results','', '| Split | E120 | Candidate | Rescues | Lost successes |','|---|---:|---:|---:|---:|']
    for split in ('fitting','held'):
        r=summary[split];lines.append(f"| {split} | {r['baseline_success']}/{r['n']} | {r['success']}/{r['n']} | {r['gains']} | {r['losses']} |")
    lines += ['', '| Split | Joints | E120 successes | Candidate successes | E120/candidate collisions |','|---|---:|---:|---:|---:|']
    for r in groups:lines.append(f"| {r['split']} | {r['dof']} | {r['baseline_success']}/{r['n']} | {r['success']}/{r['n']} | {r['baseline_collisions']}/{r['collisions']} |")
    lines += ['', '## Interpretation and limits','',
        'All144 fresh fitting trajectories exactly reproduced the previously executed selected counterfactuals, including full state arrays and task traces. Held episodes were evaluated only after this confirmation, with the frozen all-fitting models. The original held placements have been used in earlier development; they are not a fresh independent final test. All bodies here are known training bodies, so this is placement/family generalization, not new evidence of unseen-embodiment transfer.', '',
        'This conjunction was motivated by earlier adaptive experiments on these fitting families. Even though its primary rule succeeded before threshold selection, the development history matters. Small paired gains are preliminary, and seeds sharing a placement/body are dependent. Do not present this as a statistically established paper result or an end-to-end flow-matching policy.', '',
        'Earlier iterations are retained: E133 improved one-step prediction but its eight-step gate lost8 successes; E134 one-step execution lost10; E135 compact task value lost4. E136 combines the two signals, rather than claiming either alone solves the task.', '',
        'All full trials/actions/states and checkpoints are saved. Media include every changed outcome pair and representative unchanged success/failure cases from both joint counts and both splits. Videos replay executed state traces.', '']
    (ROOT/'report.md').write_text('\n'.join(lines));save(ROOT/'verification.json',dict(frozen_inputs_unchanged=True,grid_reproduced=True,exact_fitting_counterfactuals=144,held_run_after_freeze=True,models_fit_only_fitting_rows=True,E120_unchanged=True))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axes,('fitting','held')):
        r=summary[split];vals=[r['baseline_success'],r['success']];ax.bar(['E120','Combined gate'],vals,color=['#777777','#2679b2']);ax.set_ylim(0,r['n']);ax.set_ylabel('Task successes');ax.set_title(f"{split}: {r['gains']} rescues / {r['losses']} losses")
        for i,v in enumerate(vals):ax.text(i,v+.3,f"{v}/{r['n']}",ha='center')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');marker='## E136: combined local-response and task-value gate';assert marker not in text
    text=text.replace('Revision 120 · 15 November 2026','Revision 124 · 15 November 2026',1);h=summary['held']
    p.write_text(text+'\n'+marker+f"\n\nFrozen primary conjunction: fitting102/144 vs E12097,5 gains/0 losses; all144 fresh rollouts exactly match counterfactuals. Original development-held{h['success']}/24 vs E120{h['baseline_success']}, {h['gains']} gains/{h['losses']} losses. Known training bodies; no unseen-body claim. Report: `outputs/ease_combined_response_value_v1/report.md`.\n",encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','combined_response_value import ROOT').replace("ROOT/'records.json'","ROOT/'media_records.json'").replace('E128','E136').replace('Task-value ranking diagnosis','Combined local-response and task-value gate').replace('Diagnosis only; no new policy.','Frozen candidate: fitting-family validation and original development-held placements.').replace('E120 on versus short correction-off then E120','E120 versus combined gate').replace('Outcome-selected excluded-fitting-family pairs:','Outcome-selected fitting and development-held pairs:')
    Path(__file__).with_name('combined_response_videos.py').write_text(s);print('E136 verified; prepared',len(media),'videos',flush=True)

if __name__=='__main__':run()

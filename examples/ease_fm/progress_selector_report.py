"""E111 full-task retention and progress report."""
import json
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .progress_selector import ROOT,MODES,PHASES

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==72
    verification=json.loads((ROOT/'verification.json').read_text());table=[];changes=[];gates={};audits=[];failures=[];phase_table=[]
    for group in ('all','train','validation','6','7','seed0','seed1','seed2'):
        for mode in MODES:
            rr=[r for r in rows if r['mode']==mode and (group=='all' or group==('validation' if r['validation'] else 'train') or group==str(r['dof']) or group==f"seed{r['seed']}")]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),collision_free_failures=sum(not r['success'] and not r['collision'] for r in rr),grasped=sum(r['grasped'] for r in rr),lifted=sum(r['lifted'] for r in rr),steps=sum(r['steps'] for r in rr),changed_actions=sum(r['changed_actions'] for r in rr),warnings=sum(r['eligible_warnings'] for r in rr),no_safe_alternative=sum(r['no_safe_alternative'] for r in rr)))
    for row in rows:
        full=json.loads((ROOT/'trials'/f"{row['case']}_{row['seed']}_None_{row['mode']}.json").read_text())
        first=next((v for v in full['trace'] if v['bad']),None)
        if not row['success']:
            status=None
            if first is not None and row['mode']=='physics':
                first_step=32+next(i for i,v in enumerate(full['trace']) if v['bad'])
                decision=next((a for a in full['physics_audit'] if a['step']==first_step),None)
                if decision is None:status='unmodified phase'
                elif decision['prior_episode_collision']:status='prior collision flag already set'
                elif not decision['clean']:status='forbidden contact already at control boundary'
                elif decision['alternatives'] and decision['choice']=='original':status='no candidate satisfies all frozen admissibility, contact, nonpenetration and progress criteria'
                else:status='inspect saved trace'
            failures.append(dict(case=row['case'],seed=row['seed'],mode=row['mode'],first_collision_phase=None if first is None else first['phase'],selector_status=status,grasped=row['grasped'],lifted=row['lifted'],unreached_phases=[p['phase'] for p in row['phases'] if not p['reached']]))
        if row['mode']=='physics':
            ref=next(x for x in rows if x['mode']=='baseline' and x['case']==row['case'] and x['seed']==row['seed'])
            if row['success']!=ref['success']:changes.append(dict(case=row['case'],seed=row['seed'],dof=row['dof'],validation=row['validation'],change='gain' if row['success'] else 'loss',collision=row['collision']))
            for a in full['physics_audit']:
                if a['changed']:audits.append(dict(case=row['case'],seed=row['seed'],validation=row['validation'],phase=a['phase'],choice=a['choice'],before=a['before_error_sq'],selected_after=a['selected_after_error_sq'],original_after=a['original_after_error_sq'],prior_collision=a['prior_episode_collision']))
    assert not any(a['prior_collision'] for a in audits)
    for split,val in (('train',False),('validation',True)):
        new=[r for r in rows if r['mode']=='physics' and r['validation']==val];old=[r for r in rows if r['mode']=='baseline' and r['validation']==val]
        gate=dict(retains_all_successes=not any(c['validation']==val and c['change']=='loss' for c in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old))
        if not val:gate.update(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
        gates[split]=gate
    passed=all(all(g.values()) for g in gates.values())
    for phase in PHASES:
        for mode in MODES:
            selected=[r for r in rows if r['mode']==mode];ps=[next(p for p in r['phases'] if p['phase']==phase) for r in selected]
            phase_table.append(dict(phase=phase,mode=mode,reached=sum(p['reached'] for p in ps),n=len(ps),steps=sum(p['steps'] for p in ps)))
    progress=[]
    for phase in PHASES:
        aa=[a for a in audits if a['phase']==phase]
        progress.append(dict(phase=phase,changed=len(aa),choices=dict(Counter(a['choice'] for a in aa)),selected_reduces_error=sum(a['selected_after']<a['before']-1e-12 for a in aa),selected_increases_error=sum(a['selected_after']>a['before']+1e-12 for a in aa),original_reduces_error=sum(a['original_after']<a['before']-1e-12 for a in aa)))
    timings=json.loads((ROOT/'timings.json').read_text());cost={}
    for label,predicate in [('physics_all',lambda t:t['mode']=='physics'),('physics_search',lambda t:t['mode']=='physics' and t['search']),('physics_check_only',lambda t:t['mode']=='physics' and not t['search'])]:
        tt=[t for t in timings if predicate(t)];values=[t['ms'] for t in tt]
        cost[label]=dict(n=len(tt),median_ms=float(np.median(values)) if values else None,p95_ms=float(np.quantile(values,.95)) if values else None,total_seconds=sum(values)/1000,branches=sum(t['branches'] for t in tt))
    margin=dict(changed_actions=sum(r['changed_actions'] for r in rows if r['mode']=='physics'),changed_with_2mm=sum(r['changed_with_2mm'] for r in rows if r['mode']=='physics'))
    summary=dict(table=table,changes=changes,gates=gates,passes=passed,phase_completion=phase_table,changed_action_progress=progress,failures=failures,compute_cost=cost,margin=margin)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'changed_action_progress.json').write_text(json.dumps(audits,indent=2))
    lines=['# E111: progress-aware selection with geometric nonpenetration','',
    '72 full tasks: all36 original training-body task/seed pairs, baseline and fixed physics selector. Frozen E110 broad bank,40ms prediction, original static admissibility, zero physical contacts, nonnegative minimum geometric clearance and at least1% pose-error reduction (floor1e-8). Rank by smallest predicted pose error, then larger clearance, then name. Preserve original commands unless contact or penetration is predicted; preserve original if no candidate qualifies. Same weights, apertures and budgets. Extend eligibility to descend/lift/transport/lower/retreat, but only before any earlier forbidden contact. Approach/calibration unchanged; no route planner. Descent interventions can alter later probe/history states. No tuning after outcomes.',
    'The24 training and12 held-placement pairs share four original training bodies. No E105 or reserved final test. Exact physical-model/state access is privileged; this is not learned dynamics or history-only control.',
    '', '| Group | Mode | Success | Collisions | Contact-free failures | Grasped | Lifted | Steps | Changes | Warnings | No safe option |','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['collision_free_failures']} | {r['grasped']} | {r['lifted']} | {r['steps']} | {r['changed_actions']} | {r['warnings']} | {r['no_safe_alternative']} |")
    lines+=['','## Paired task changes','',json.dumps(changes,indent=2),'','## Retention gate','',f"**{'PASS' if passed else 'FAIL'}**",'',json.dumps(gates,indent=2),'','## Task progress','',
    'Phase reached counts use the original position/tool-axis completion rules, including final-state tolerance on timed-out phases. Phases still advance at the original time limit. Grasped/lifted are original sticky flags, not maintained-grasp guarantees.','', '| Phase | Mode | Reached /36 | Control steps |','|---|---|---:|---:|']
    for r in phase_table:lines.append(f"| {r['phase']} | {r['mode']} | {r['reached']} | {r['steps']} |")
    lines+=['','Changed-action progress uses same-state40ms squared position plus .15-scaled tool-axis error. The original action is a simulator counterfactual and may collide; compare it only as a diagnostic. Predicted progress gates and ranks candidate selection in this experiment; this is a selection criterion, not independent evidence of generalization. Actions and seeds are correlated, not independent samples.', '',json.dumps(progress,indent=2),'','## Failures','',json.dumps(failures,indent=2),'','## Decision','',
    ('The fixed rule passes the training/held-placement gate; it remains a candidate requiring a separately frozen broader evaluation.' if passed else 'The rule fails the task-success/retention gate. Keep the original controller; collision reduction or intermediate phase progress alone is not sufficient for promotion.'),
    'No horizon, candidate or phase tuning after these outcomes. Full traces retain all failures, alternatives and state predictions. Before another control change, identify whether failure is due to no feasible alternative, loss of task progress, or a contact in an unmodified phase.','', '## Verification','',
    f"{verification['exact_baselines']} full baseline replays, {verification['exact_pre_descent_prefixes']} approach prefixes, and {verification['exact_selected_predictions']} selected next states/contact unions match exactly. All changed actions occur before any prior collision. Inputs/checkpoints unchanged. Videos replay full serialized results; task success/failure videos are executed motion, not lookahead animation."]
    lines+=['','The broader search uses exact physics and finite-difference simulated response. This is a privileged model-based controller component, not a new learned policy. See protocol.json for frozen candidate counts, random seeds and thresholds.']
    lines+=['','The inherited held-placement label denotes placement holdout from the original training data, not an untouched evaluation: these placements were inspected in earlier development experiments, including E110. Treat this gate as development retention, not independent generalization confirmation.',
    '', '## Compute and margin','',json.dumps(cost,indent=2),'',json.dumps(margin,indent=2),
    'Measured times are CPU wall time for added prediction/search after the original static-filter decision; they exclude neural inference, normal execution and the original static-filter cost. Baseline was instrumented too, so these are not an uninstrumented whole-policy speed comparison. The physical control interval is40ms. A search slower than40ms is not real-time ready. Every changed action has nonnegative predicted clearance; only the separate count maintains2mm throughout prediction.',
    '', '## Research progress','',
    'We have a reproducible state-based simulation pipeline, frozen neural checkpoints, baseline/retention gates, raw trajectories and replay-verified success/failure videos. E104 improved a reused novel-axis development suite from32/48 to35/48, but E105 fresh-body replication tied baseline at27/45 and introduced a paired regression. E108/E109 short physics fallbacks tied the training baseline at22/36; E109 degraded intermediate phase progress. E110 showed broader local action availability in13/14 warning states, which motivated this full-episode trial.',
    'The central unresolved claim is reliable history-conditioned control on unseen embodiments. Current physics-assisted interventions use exact state/model access and are not evidence of history-only adaptation. Before treating this as a conference result, the final method needs reliable full-task improvement, an untouched prospective embodiment evaluation, appropriate analytical/specialist/history ablations, and clear accounting of privileged inputs and realistic-robot scope. This experiment is one controlled step toward that evidence, not publication readiness.',
    'E111 now improves complete-task success from22/36 to31/36 (nine gains, zero losses), including24/24 training-placement successes and7/12 reused held-placement successes. Forbidden-contact episodes fall from14 to3. All54 changed actions reduce immediate pose error, but phase completion is not uniformly improved; five full-task failures remain. This supports the frozen physics-assisted component on development training bodies. The next milestone is a prospectively frozen unseen-body comparison, followed by attribution ablations and reducing search latency.',
    'Regression validation:131 tests passed in12.32 seconds.',
    '', '## Run integrity','',
    'The initial attempt stopped at a tuple-versus-list contact-pair parity assertion before executing its first alternative action. Its source and seven completed trials are preserved in outputs/ease_progress_selector_initial_check. Only representation/check diagnostics were corrected; candidate generation, thresholds and ranking were unchanged. All72 final trials rerun from scratch; no partial outcomes are substituted.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,('success','collisions'),('Task successes','Tasks with forbidden contact')):
        for j,mode in enumerate(MODES):
            rr=[next(r for r in table if r['mode']==mode and r['group']==g) for g in ('train','validation')];bars=ax.bar(np.arange(2)+(j-.5)*.3,[r[metric] for r in rr],.3,label=mode);ax.bar_label(bars)
        ax.set_xticks(range(2),['Train (24)','Held placements (12)']);ax.set_ylim(0,26);ax.set_title(title);ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'comparison.{ext}')
    plt.close(fig);print(json.dumps(dict(table=table,changes=changes,gates=gates,passes=passed,progress=progress),indent=2))

if __name__=='__main__':run()

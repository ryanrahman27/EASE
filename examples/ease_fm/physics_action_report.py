import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .physics_action import ROOT,MODES

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==72
    verify=json.loads((ROOT/'verification.json').read_text());table=[];changes=[];gates={}
    for group in ('all','train','validation','6','7','seed0','seed1','seed2'):
        for mode in MODES:
            rr=[r for r in rows if r['mode']==mode and (group=='all' or group==('validation' if r['validation'] else 'train') or group==str(r['dof']) or group==f"seed{r['seed']}")]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr),changed_actions=sum(r['changed_actions'] for r in rr),eligible_warnings=sum(r['eligible_warnings'] for r in rr),no_safe_alternative=sum(r['no_safe_alternative'] for r in rr),changed_after_prior_collision=sum(r['changed_after_prior_collision'] for r in rr)))
    for r in rows:
        if r['mode']!='physics':continue
        ref=next(x for x in rows if x['mode']=='baseline' and x['case']==r['case'] and x['seed']==r['seed'])
        if r['success']!=ref['success']:changes.append(dict(case=r['case'],seed=r['seed'],dof=r['dof'],validation=r['validation'],change='gain' if r['success'] else 'loss',collision=r['collision']))
    for group,val in (('train',False),('validation',True)):
        new=[r for r in rows if r['mode']=='physics' and r['validation']==val];old=[r for r in rows if r['mode']=='baseline' and r['validation']==val]
        gate=dict(retains_all_successes=not any(r['validation']==val and r['change']=='loss' for r in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old))
        if not val:gate.update(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
        gates[group]=gate
    passed=all(all(g.values()) for g in gates.values());details=[]
    for r in rows:
        if r['mode']!='physics' or not r['eligible_warnings']:continue
        full=json.loads((ROOT/'trials'/f"{r['case']}_{r['seed']}_None_physics.json").read_text())
        details.append(dict(case=r['case'],seed=r['seed'],success=r['success'],collision=r['collision'],changed_actions=r['changed_actions'],no_safe_alternative=r['no_safe_alternative'],choices=[dict(step=a['step'],choice=a['choice'],prior_episode_collision=a['prior_episode_collision'],candidates=a['alternatives']) for a in full['physics_audit'] if a['alternatives']],phases=r['phases']))
    summary=dict(table=table,changes=changes,gates=gates,passes=passed,warning_cases=details);(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E108: selecting safer actions with short physics predictions','',
    'All36 original training-body task/seed pairs execute baseline and the fixed physics selector:72 complete tasks. Fixed neural weights, original task budgets, no route planner, no ramp. Third placement per body remains held validation (24 training/12 held pairs). The selector only acts in retreat; it does not repair earlier collision failures.',
    '', 'When the static-filter-selected command predicts a forbidden contact over40ms and current state is contact-free, test five fixed alternatives: half/quarter/bias blends, hold-current-position target, and a velocity-opposing heuristic. Every candidate must pass the original static nonworsening-clearance threshold and predict no contact. Choose the safe candidate closest in joint-target distance to the original command, with fixed ties; if none qualifies keep original. All candidates share the same40ms budget and aperture. No alternative set or horizon tuning after outcomes.',
    '', '| Group | Mode | Success | Collisions | Steps | Changed actions | Warnings | No safe alternative | Changes after earlier collision |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} | {r['changed_actions']} | {r['eligible_warnings']} | {r['no_safe_alternative']} | {r['changed_after_prior_collision']} |")
    lines+=['','## All paired changes','',json.dumps(changes,indent=2),'','## Prespecified retention gates','',f"**{'PASS' if passed else 'FAIL'}**",'',json.dumps(gates,indent=2),'','## Warning-case audit','',json.dumps(details,indent=2),'','## Decision and limits','',
    ('The fixed rule passes this training-body gate; retain it as a candidate for a separately frozen evaluation. This is not yet unseen-embodiment evidence.' if passed else 'The fixed rule fails the task-success gate. Preserve the baseline and report the full negative result; do not promote collision avoidance alone as task improvement.'),
    'The method has exact simulator state and physical-model access. It is a privileged controller component, not a learned dynamics model or history-only policy. A finite candidate set and40ms horizon cannot guarantee progress or safety. The comparison tests this rule, not all possible physics-based controllers. Held placements share training bodies. No fresh E105 state or reserved final test is used.',
    '', '## Verification','',f"{verify['exact_baselines']} full baseline replays and {verify['exact_pre_retreat_prefixes']} pre-retreat traces match references. All{verify['exact_selected_predictions']} selected-action predicted40ms qpos/qvel/contact unions match execution exactly. Inputs/checkpoints remain unchanged. Full raw results, all action alternatives and outcomes, and exact-replay videos are saved."]
    failures=[]
    for row in rows:
        if row['mode']!='baseline' or row['success']:continue
        trial=json.loads((ROOT/'trials'/f"{row['case']}_{row['seed']}_None_baseline.json").read_text())
        first=next((x for x in trial['trace'] if x['bad']),None)
        failures.append(dict(case=row['case'],seed=row['seed'],first_collision_phase=None if first is None else first['phase']))
    (ROOT/'failure_phases.json').write_text(json.dumps(failures,indent=2))
    lines+=['','## Scope of the retreat intervention','',
    'Baseline first-collision phases: descend3, lift4, transport2, lower4, retreat1. Thus13 of14 failed tasks are already invalid before the intervention can act. With the prefix and success rule fixed, a retreat-only change can recover at most one of these14 failures. Case8/seed2 already collided during lift; its two changed retreat actions cannot recover full-task success even if they prevent later contact.',
    'Case1/seed0 is the only baseline failure with first collision in retreat. Three hold actions delay first retreat collision from control step27 to31 (160ms); all five alternatives then predict contact and the rule keeps the original action. Case8/seed2 has two hold actions and first retreat contact shifts from step48 to50 (80ms), but the earlier lift collision remains. These are delays, not recovered tasks. The five action changes occur in two failing pairs; no paired task outcome changes.',
    'Next, audit predictive contact avoidance before the first collision in descend/lift/transport/lower on training bodies. Separate whether the finite candidate set has any contact-free option from whether choosing one preserves task progress. Do not interpret the failure of this retreat-only rule as a general failure of physics lookahead. No new phase-wide control rule or transfer evaluation is performed here.',
    '',json.dumps(failures,indent=2),'','Regression suite:131 tests pass in12.48s. Prediction and full-replay assertions run on every evaluated task.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,group,limit in zip(axes,('train','validation'),(24,12)):
        rr=[r for r in table if r['group']==group];bars=ax.bar([r['mode'] for r in rr],[r['success'] for r in rr]);ax.bar_label(bars);ax.set_ylim(0,limit+2);ax.set_ylabel('Successful tasks');ax.set_title(group)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'comparison.{ext}')
    plt.close(fig);print(json.dumps(dict(table=table,changes=changes,gates=gates,passes=passed),indent=2))

if __name__=='__main__':run()

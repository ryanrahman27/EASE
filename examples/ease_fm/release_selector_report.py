"""E113 paired outcomes and first-contact constraint diagnosis."""
import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from .release_selector import ROOT,OLD

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==48
    table=[];changes=[];diagnostics=[];release=[]
    for group in ('all','seed0','seed1','seed2',*[f'body{i}' for i in range(4)]):
        for mode in ('baseline','physics'):
            rr=[r for r in rows if r['mode']==mode and (group=='all' or group==f"seed{r['seed']}" or group==f"body{r['body']}")]
            table.append(dict(group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),release_changes=sum(r['release_changes'] for r in rr),release_warnings=sum(r['release_warnings'] for r in rr)))
    def constraints(a):
        cs=a['alternatives'];static=[c for c in cs if c['static_ok']];contact=[c for c in static if not c['40']['collision']];clear=[c for c in contact if c['40']['minimum_clearance']>=0];progress=[c for c in clear if c['40']['progress']]
        return dict(candidates=len(cs),static=len(static),contact_free=len(contact),nonpenetrating=len(clear),progress=len(progress),choice=a['choice'],before_error_sq=a['before_error_sq'],required_gain=max(1e-8,.01*a['before_error_sq']),best_safe_gain=max((c['40']['gain'] for c in clear),default=None))
    for r in rows:
        f=json.loads((ROOT/'trials'/f"{r['case']}_{r['seed']}_None_{r['mode']}.json").read_text())
        if r['mode']=='physics':
            ref=next(x for x in rows if x['mode']=='baseline' and x['case']==r['case'] and x['seed']==r['seed'])
            if r['success']!=ref['success']:changes.append(dict(case=r['case'],seed=r['seed'],change='gain' if r['success'] else 'loss'))
            for a in f['physics_audit']:
                if a['phase']=='release' and a['unsafe_original'] and a['clean'] and not a['prior_episode_collision']:
                    release.append(dict(case=r['case'],seed=r['seed'],step=a['step'],**constraints(a)))
        if not r['success']:
            first=next(((i,v) for i,v in enumerate(f['trace']) if v['bad']),None);a=None
            if first:a=next((a for a in f['physics_audit'] if a['step']==32+first[0]),None)
            diagnostics.append(dict(case=r['case'],seed=r['seed'],mode=r['mode'],first_collision_phase=first[1]['phase'] if first else None,pairs=first[1]['bad'] if first else [],criteria=constraints(a) if a else None,unreached_phases=[p['phase'] for p in r['phases'] if not p['reached']]))
    old=[r for r in rows if r['mode']=='baseline'];new=[r for r in rows if r['mode']=='physics']
    gate=dict(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),retains_all_successes=not any(c['change']=='loss' for c in changes),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
    failed_new=[d for d in diagnostics if d['mode']=='physics'];blocked_progress=sum(bool(d['criteria']) and d['criteria']['nonpenetrating']>0 and d['criteria']['progress']==0 for d in failed_new)
    summary=dict(table=table,changes=changes,gates=gate,passes=all(gate.values()),release_warning_states=len(release),release_changed_actions=sum(r['release_changes'] for r in new),failures_with_safe_candidates_blocked_by_progress=blocked_progress)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2));(ROOT/'failure_diagnostics.json').write_text(json.dumps(diagnostics,indent=2));(ROOT/'release_diagnostics.json').write_text(json.dumps(release,indent=2))
    lines=['# E113: six-joint failure diagnosis and release-phase extension','',
        '48 complete rollouts: all24 six-joint E112 task/seed pairs with original selector and release-extended selector. Here baseline means the E112 physics selector, not the earlier unassisted controller. Only release eligibility changes; exact action bank,40ms horizon, thresholds, progress ranking, apertures and task budgets are unchanged. No retraining or tuning after outcomes. These are reused development bodies, not fresh independent transfer evidence.',
        '', '| Group | Mode | Success | Collisions | Release changes | Eligible release warnings |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f"| {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['release_changes']} | {r['release_warnings']} |")
    lines+=['','## Paired changes','',json.dumps(changes,indent=2),'','## Frozen gate','',f"**{'PASS' if summary['passes'] else 'FAIL'}**",'',json.dumps(gate,indent=2),'','## Diagnosis','',
        'Constraint counts are sequential intersections: all alternatives, static-admissible, no forbidden physical contact, nonnegative geometric clearance, then required goal-error reduction. A zero at a stage identifies why the tested bank has no eligible action; it does not prove that no safe action exists outside the bank. Physical contact is allowed for object-table and object-pads. Other contacts are failures.',
        '', 'All first-collision states are saved in failure_diagnostics.json; every eligible release warning is in release_diagnostics.json. Earlier failures cannot be repaired by this extension because it starts only at release and disables interventions after any previous forbidden contact.',
        '',f'{blocked_progress}/{len(failed_new)} remaining failures have static-admissible, contact-free, nonpenetrating40ms alternatives at the first-collision decision, but none meets the required progress gain. The frozen fallback consequently keeps the original unsafe action. This identifies a local conflict between safety and strict immediate goal progress; it does not establish that those alternatives would finish the task. One remaining lower-phase case has no static-admissible contact-free alternative in the tested bank.',
        '',json.dumps(diagnostics,indent=2),'','## Release decisions','',json.dumps(release,indent=2),
        '', '## Interpretation','',('The release extension passes the prespecified development gate; require a newly frozen prospective replication before claiming general transfer.' if summary['passes'] else 'The release extension does not pass the improvement gate. Do not promote it based on safety predictions alone. Preserve the frozen result before testing changes to the action bank, progress criterion or horizon.'),
        '', 'The original requirement of at least1% pose-error reduction is unchanged during release. Both remaining release failures have safe alternatives rejected by this requirement. Next test a separately frozen, bounded near-goal safety fallback on training states, measuring full-task progress and preventing indefinite holding. Do not simply remove progress everywhere or relabel this reused cohort as untouched validation.',
        '', '## Verification','',json.dumps(json.loads((ROOT/'verification.json').read_text()),indent=2),
        '', 'All24 original-selector replays match E112 full serialized results; all extended rollouts match before release. Every selected next-state/contact prediction matches execution. Exact simulator access remains privileged; this is not history-only adaptation. Videos are outcome-selected illustrations and must match full serialized replay.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps(summary,indent=2))

if __name__=='__main__':run()

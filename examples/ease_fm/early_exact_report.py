"""Save E167 as explicitly privileged diagnostic evidence, never a learned result."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
from .early_exact_diagnostic import ROOT,CONDITIONS
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    rows=read(ROOT/'records.json');assert len(rows)==96 and read(ROOT/'verification.json')['privileged_diagnostic_only'];summary={};selection=set()
    for condition in CONDITIONS:
        rr=[r for r in rows if r['condition']==condition];contacts=Counter()
        for r in rr:
            t=read(ROOT/condition/'trials'/f"{r['case']}_privileged_exact_map_frozen_bias.json");first=next((x for x in t['trace'] if x['bad']),None)
            if first:contacts[first['phase']]+=1
        summary[condition]=dict(n=48,success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),first_contact_phases=dict(contacts))
        for success in (True,False):
            options=[r for r in rr if r['success']==success]
            if options:selection.add(options[0]['case'])
    baseline={r['case']:r for r in rows if r['condition']=='release_only'};changes=[dict(case=r['case'],body=r['body'],before=baseline[r['case']]['success'],after=r['success']) for r in rows if r['condition']=='conditional_high' and r['success']!=baseline[r['case']]['success']]
    s=dict(methods=summary,gains=sum(x['after'] for x in changes),losses=sum(x['before'] for x in changes),changes=changes,privileged_diagnostic_only=True,no_learned_policy_claim=True,no_held_execution=True,E120_unchanged=True);save(ROOT/'summary.json',s)
    selection.update((13,26,72))
    for gain in (True,False):
        rr=[r for r in changes if r['after']==gain]
        if rr:selection.add(rr[0]['case'])
    for folder in ('trials','states'):(ROOT/folder).mkdir()
    videos=[]
    for case in sorted(selection):
        for condition in CONDITIONS:
            source=ROOT/condition;old=f'{case}_privileged_exact_map_frozen_bias';method='privileged_'+condition;new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
            for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
            videos.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(ROOT/'video_records.json',videos)
    lines=['# E167: privileged exact-motion route diagnostic','','**PRIVILEGED DIAGNOSTIC ONLY.** The controller reads the exact current simulator tool-motion Jacobian. It is not E120, a learned estimator result, an unseen-body result, or a mathematical task-success upper bound. No diagnostic labels are supplied to training.','', '| Exact-map controller | Six-joint successes | Collision episodes | First-contact phases |','|---|---:|---:|---|']
    for condition,v in summary.items():lines.append(f"| {condition} | {v['success']}/48 | {v['collisions']} | {v['first_contact_phases']} |")
    lines+=['',f"Conditional higher route versus matched exact-map release-only reference: **{s['gains']} rescues, {s['losses']} regressions**.",'','Both variants retain the265mm release fix, original measured32-step calibration, frozen measured bias, close probe, DLS gain/damping, phase budgets and strict contact/success scoring. The route rule raises approach330→400mm only if initial measured lateral travel exceeds8mm. No geometry-distance query, collision-aware route planner, future physics or success filtering is used. The original E120 learned correction is absent in both exact-map conditions, so only their within-diagnostic comparison cleanly isolates the route effect.','', 'All48 six-joint fitting placements were included for each condition. Release-only trajectories before lowering exactly reproduce the prior E149 exact-map reference. Inactive conditional routes exactly match the new release-only full states/traces. Calibration and frozen bias are identical, and E120 checkpoint hashes remain unchanged. This diagnostic was registered after fitting E164/E165 outcomes; it is adaptive diagnosis rather than independent validation.','', 'Residual contacts with exact current motion maps indicate that accurate instantaneous kinematics alone does not solve this fixed controller/route problem. This does not prove collision-free paths are impossible or that every learned controller must fail. A clearance-aware feasibility study would be needed to separate path feasibility from controller limitations.','', 'Videos are explicitly labeled privileged executed-state replays, including successes, failures and matched changed cases. Every saved state is checked against the task trace and decoded frame counts.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,('success','collisions'),('Six-joint task successes /48','Six-joint collision episodes /48')):
        vals=[summary[c][metric] for c in CONDITIONS];ax.bar(['Exact + release','Exact + high route'],vals);ax.set_ylim(0,48);ax.set_title(title)
        for i,x in enumerate(vals):ax.text(i,x+.5,str(x),ha='center')
    fig.suptitle('PRIVILEGED exact-current-motion diagnostic — fitting only');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','early_exact_diagnostic import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E167').replace('Task-value ranking diagnosis','PRIVILEGED exact-motion route diagnostic').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','PRIVILEGED exact current Jacobian, fitting cases only. No learned-policy claim. Paired release-only and conditional-high routes.').replace('Measured-history control | No physics selector or geometry filter','PRIVILEGED exact Jacobian | Diagnostic only | No learned-policy claim')
    Path(__file__).with_name('early_exact_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E167:' not in text;p.write_text(text+f"\n## E167: privileged exact-current-motion route diagnostic\n\nExact-map release-only {summary['release_only']['success']}/48, conditional high route {summary['conditional_high']['success']}/48; {s['gains']} rescues/{s['losses']} regressions. Simulator Jacobian is privileged and not a learned-policy result; no training labels or held execution. Report: `outputs/ease_early_exact_diagnostic_v1/report.md`.\n",encoding='utf-8');save(ROOT/'status.json',dict(status='verified_videos_pending',privileged_episodes=96,videos_planned=len(videos)));print(s,flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    save(ROOT/'status.json',dict(status='complete',privileged_episodes=96,videos=len(videos),privileged_diagnostic_only=True,release_fix_preserved=True,E120_unchanged=True,held_evaluated=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E167 privileged diagnostic artifacts complete',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]]()

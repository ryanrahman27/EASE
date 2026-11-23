"""Common reports/artifacts for fitting-only conditional approach follow-ups."""
from pathlib import Path
import shutil
import numpy as np
from .conditional_approach_control import ROOT as CONDITIONAL,BASE
from .descent_reference_control import ROOT as DESCENT
from .early_clearance_report import contact_counts
from .task_contact_report import stats
from .abstaining_task_value_gate import read,save,fingerprint

def root_for(name):return CONDITIONAL if name=='conditional' else DESCENT

def run(name):
    root=root_for(name);experiment='E165' if name=='conditional' else 'E166';rows=read(root/'records.json');v=read(root/'verification.json');assert len(rows)==96 and v['new_six_joint_episodes']==48;baseline=read(BASE/'records.json');lookup={r['case']:r for r in baseline};changes=[];active=0;inactive=0
    for r in rows:
        if r['dof']==7:assert r['success']==lookup[r['case']]['success'] and r['collision']==lookup[r['case']]['collision'];continue
        d=read(root/'decisions'/f"{r['case']}.json");distance=float(np.linalg.norm(np.array(d['start_position'])[:2]-d['xy']));assert distance==d['lateral_distance'] and d['active']==(distance>.008);active+=d['active'];inactive+=not d['active']
        for goal in d['goals']:
            if goal['phase']=='approach':assert goal['goal'][2]==(.4 if d['active'] else .33)
            if goal['phase'] in ('lower','release'):assert goal['goal'][2]==.265
            if goal['phase']=='descend':assert goal['goal'][2]==.24 and goal['budget']==100
        if r['success']!=lookup[r['case']]['success']:changes.append(dict(case=r['case'],body=r['body'],before=lookup[r['case']]['success'],after=r['success']))
    a,b=stats(rows),stats(baseline);contacts=contact_counts(root,rows);bc=contact_counts(BASE,baseline);early=sum(contacts.get(k,0) for k in ('approach','descend'));passed=a['6']['success']>b['6']['success'] and early<10 and a['6']['collisions']<=11 and not any(x['before'] for x in changes);s=dict(candidate=a,baseline=b,gains=sum(x['after'] for x in changes),losses=sum(x['before'] for x in changes),changes=changes,active_six_joint_episodes=active,inactive_exact_six_joint_replays=inactive,first_six_joint_contact_phases=contacts,baseline_first_six_joint_contact_phases=bc,qualifies=passed,E120_unchanged=True,release_fix_preserved=True,held_evaluated=False)
    if name=='descent':
        other=read(CONDITIONAL/'records.json');other_lookup={r['case']:r for r in other};s['versus_conditional_high']=dict(gains=sum(r['success'] and not other_lookup[r['case']]['success'] for r in rows if r['dof']==6),losses=sum(not r['success'] and other_lookup[r['case']]['success'] for r in rows if r['dof']==6))
    save(root/'summary.json',s)
    for p,h in read(root/'frozen.json').items():assert fingerprint(p)==h
    selected={1,13,26,72}
    for success in (True,False):
        rr=[r for r in rows if r['dof']==6 and r['success']==success]
        if rr:selected.add(rr[0]['case'])
    for gain in (True,False):
        rr=[r for r in changes if r['after']==gain]
        if rr:selected.add(rr[0]['case'])
    videos=[]
    for case in sorted(selected):
        for source,method in ((root,name),(BASE,'baseline_release')):
            old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(root/'trials'/f'{new}.json',t)
            for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',root/'states'/f'{new}.{ext}')
            videos.append(dict(case=case,method=method,mode='frozen_bias',success=t['success'],dof=6))
    save(root/'video_records.json',videos)
    module='conditional_approach_control' if name=='conditional' else 'descent_reference_control';video_module='conditional_approach_videos' if name=='conditional' else 'descent_reference_videos'
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT',module+' import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128',experiment).replace('Task-value ranking diagnosis','Fitting conditional approach comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Paired fitting six-joint successes and failures. E120 and elevated release retained. No geometry input or held evaluation. Includes diagnosed cases1,13,26,72.')
    Path(__file__).with_name(video_module+'.py').write_text(template)
    lines=[f'# {experiment}: '+('conditional higher approach' if name=='conditional' else 'conditional higher approach with descent reference cap'),'','E120 ensemble plus the E161265mm release target is the unchanged reference. All48 six-joint fitting placements were executed, plus4 seven-joint no-op replays;44 seven-joint results were reused unchanged. No held placement or unseen body was evaluated.','', '| Metric | Release-fixed E120 | Candidate |','|---|---:|---:|',f"| Six-joint successes | {b['6']['success']}/48 | {a['6']['success']}/48 |",f"| Six-joint collision episodes | {b['6']['collisions']} | {a['6']['collisions']} |",f"| First approach/descent contacts | 10 | {early} |",f"| All fitting successes, including unchanged seven-joint reference | {b['all']['success']}/96 | {a['all']['success']}/96 |",'',f"Paired six-joint outcomes: **{s['gains']} rescues, {s['losses']} regressions**. Meets fixed retention/contact screen: **{passed}**.",'','The approach target is raised330→400mm only when initial measured tool-to-target lateral distance exceeds8mm, the existing task tolerance. The decision is made once after calibration, without body IDs, geometry or contact signals. This rule was introduced after inspecting fitting E164 outcomes; it is adaptive development, not independent validation. No threshold sweep was performed.','']
    if name=='descent':lines+=['During descent in an active episode, position error used by the DLS command is capped at30mm; orientation error is unchanged. The exact original E120 bounded residual is preserved around the new DLS base. Completion uses the original final target, and the100-step budget is unchanged. All original/new DLS calculations and executed commands were independently recomputed from logged values. Other phases retain the original controller. This was registered after E164 outcomes, not selected by a held result.','']
    lines += [f"There were{active} active and{inactive} inactive six-joint episodes. Every inactive episode exactly reproduced its full baseline state trajectory and task trace. Calibration and frozen bias were verified for all cases; checkpoint/source hashes are unchanged. The original release fix and strict contact/success criteria remain in force. Elevated release permits a short drop, not gentle-contact placement.",'',f"First-contact phases: {contacts}. These count first failures; already failed trajectories may collide again later.",'','This is a task-controller development result on two known six-joint procedural bodies. It does not establish learned-estimator superiority, unseen morphology transfer or a safety guarantee. E120 is not replaced. All changed placements are retained in summary.json. Videos include outcome-selected baseline/candidate pairs, including the earlier rescue and regression cases.']
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,('success','collisions'),('Six-joint successes /48','Six-joint collision episodes /48')):
        vals=[b['6'][metric],a['6'][metric]];ax.bar(['E120 + release',name],vals,color=['#777777','#2379b5']);ax.set_ylim(0,48);ax.set_title(title)
        for i,value in enumerate(vals):ax.text(i,value+.5,str(value),ha='center')
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert f'## {experiment}:' not in text;p.write_text(text+f"\n## {experiment}: {name} early six-joint control\n\n{a['6']['success']}/48 versus37/48 baseline; {s['gains']} rescues/{s['losses']} regressions; {a['6']['collisions']} collision episodes, {early} first approach/descent contacts. Fixed screen passed: {passed}. Release fix and E120 weights preserved, fitting-only. Report: `{root}/report.md`.\n",encoding='utf-8');shutil.copy2(__file__,root/'source'/Path(__file__).name);save(root/'status.json',dict(status='verified_videos_pending',new_six_joint_episodes=48,seven_joint_exact_replays=4,videos_planned=len(videos)));print(s,flush=True)

def finish(name):
    root=root_for(name);videos=read(root/'videos/manifest.json')['videos'];assert len(videos)==len(read(root/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(root/'videos'/v['file'])==v['sha256']
    save(root/'status.json',dict(status='complete',new_six_joint_episodes=48,seven_joint_exact_replays=4,videos=len(videos),E120_unchanged=True,release_fix_preserved=True,held_evaluated=False));save(root/'artifact_hashes.json',{str(p.relative_to(root)):fingerprint(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print(name,'artifacts complete',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]](sys.argv[2])

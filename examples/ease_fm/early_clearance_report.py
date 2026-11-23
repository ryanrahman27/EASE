"""E163/E164 fitting geometry diagnosis, full outcomes, and paired replay artifacts."""
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from .early_clearance_control import ROOT,BASE,VARIANTS
from .early_contact_diagnosis import ROOT as DIAG
from .abstaining_task_value_gate import read,save,fingerprint
from .task_contact_report import stats

def diagnosis():
    rows=read(DIAG/'diagnosis.json');lines=['# E163: offline first-contact audit','','All 11 six-joint collision episodes under E120 plus the E161 release fix occur on body84000. Ten first contacts occur in approach/descent, and one in lift. The other six-joint body has no collision episodes.','', '**Geometry-label correction:** the trace strings `3` and `4` are unnamed MuJoCo geometry IDs. In these saved models they attach to `link1` and `link2`. Earlier informal descriptions calling them link3/link4 should be read as geometry IDs, not anatomical link indices. The task scores and contact criteria are unchanged.','', 'The inspected first saved task states have only millimetres of elbow/table clearance, and the supplied posture places joints near their ±2.5-rad limits. This motivates a higher approach waypoint and joint-limit centering rather than another home-posture attractor. It does not prove either will succeed.','', 'The audit uses privileged signed geometry distances at saved states, only for offline diagnosis. Neither E164 controller receives these distances, a kinematic model, a body ID, a link label, or simulator lookahead. Saved states occur after40ms control steps; distances can miss within-step transient contact and can reflect a newly formed contact before its next integration-step log.','', '| Case | First contact phase | Step index | Clearance at first saved task state (mm) |','|---|---|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['phase']} | {r['first_contact_step']} | {1000*r['snapshots'][0]['link_table_distance_m']['link1:geom3']:.3f} |")
    lines+=['','Complete snapshots, joint states, supplied home postures and all six link distances are in diagnosis.json. Source failure videos: [E161 gallery](../ease_release_clearance_control_v1/videos/index.html). No new rollouts were executed for this audit.'];(DIAG/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');save(DIAG/'artifact_hashes.json',{str(p.relative_to(DIAG)):fingerprint(p) for p in DIAG.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E163:' not in text;p.write_text(text+'\n## E163: early six-joint contact audit and geometry-ID correction\n\nAll11 six-joint E161 collisions occur on body84000,10 during approach/descent and1 during lift. Contact labels3/4 refer to geometry IDs attached to link1/link2, not link3/link4. Signed geometry inspected offline only, with no policy input. Millimetre-scale initial task clearance and near-limit posture motivate E164. Report: `outputs/ease_early_contact_diagnosis_v1/report.md`.\n',encoding='utf-8')

def contact_counts(root,rows):
    out=Counter()
    for r in rows:
        if r['dof']!=6:continue
        t=read(root/'trials'/f"{r['case']}_e120_ensemble_frozen_bias.json");first=next((v for v in t['trace'] if v['bad']),None)
        if first:out[first['phase']]+=1
    return dict(out)

def run():
    base=read(BASE/'records.json');baseline=stats(base);basecontacts=contact_counts(BASE,base);summaries={};selected=[];video_rows=[]
    for name in VARIANTS:
        root=ROOT/name;assert read(root/'verification.json')['new_six_joint_episodes']==48;rows=read(root/'records.json');assert len(rows)==96;before={r['case']:r for r in base};changes=[]
        for r in rows:
            if r['dof']==7:assert r['success']==before[r['case']]['success'] and r['collision']==before[r['case']]['collision'];continue
            stem=f"{r['case']}_e120_ensemble_frozen_bias";logs=read(root/'control'/f'{stem}.json');actions=read(root/'actions'/f'{stem}.json');trial=read(root/'trials'/f'{stem}.json');assert len(logs)==len(actions)==len(trial['trace'])
            for l,a in zip(logs,actions):
                np.testing.assert_array_equal(l['q_before'],a['q_before']);assert l['phase']==a['phase']
                if l['active']:
                    q=np.array(l['q_before']);np.testing.assert_array_equal(l['gradient'],-np.sign(q)*np.maximum(np.abs(q)-2.,0.));assert l['phase'] in ('approach','descend') and name=='centered_early';assert np.max(np.abs(l['delta']))<=.005000000001;np.testing.assert_array_equal(l['final_action'],np.clip(np.array(l['base_action'])+l['delta'],-.04,.04))
                else:np.testing.assert_array_equal(l['base_action'],l['final_action'])
                if l['phase']!='probe':np.testing.assert_array_equal(np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before'],a['executed_command'])
                if l['phase'] in ('lower','release'):assert l['goal'][2]==.265
                if l['phase']=='approach':assert l['goal'][2]==(.4 if name=='high_approach' else .33)
            if r['success']!=before[r['case']]['success']:changes.append(dict(case=r['case'],body=r['body'],before=before[r['case']]['success'],after=r['success']))
        candidate=stats(rows);contacts=contact_counts(root,rows);early=sum(contacts.get(x,0) for x in ('approach','descend'));passed=candidate['6']['success']>baseline['6']['success'] and candidate['6']['collisions']<=baseline['6']['collisions'] and early<sum(basecontacts.get(x,0) for x in ('approach','descend')) and not any(x['before'] for x in changes)
        summaries[name]=dict(candidate=candidate,baseline=baseline,first_six_joint_contact_phases=contacts,baseline_first_six_joint_contact_phases=basecontacts,gains=sum(x['after'] for x in changes),losses=sum(x['before'] for x in changes),changes=changes,qualifies=passed,bodies={str(b):stats([r for r in rows if r['body']==b]) for b in sorted({r['body'] for r in rows if r['dof']==6})})
        if passed:selected.append(name)
        chosen={}
        for success in (True,False):
            rr=[r for r in rows if r['dof']==6 and r['success']==success]
            if rr:chosen[rr[0]['case']]=True
        for gain in (True,False):
            rr=[r for r in changes if r['after']==gain]
            if rr:chosen[rr[0]['case']]=True
        # Preserve a direct view of both previously diagnosed early phases.
        chosen[13]=True;chosen[26]=True
        for case in chosen:
            for source,method in ((root,name),(BASE,'baseline_release')):
                if any(v['case']==case and v['method']==method for v in video_rows):continue
                stem=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';trial=read(source/'trials'/f'{stem}.json');trial['method']=method;save(ROOT/'trials'/f'{new}.json',trial)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{stem}.{ext}',ROOT/'states'/f'{new}.{ext}')
                video_rows.append(dict(case=case,method=method,mode='frozen_bias',success=trial['success'],dof=6))
    preferred=selected[0] if selected else None;s=dict(variants=summaries,selected=preferred,held_evaluated=False,E120_unchanged=True,release_fix_preserved=True,new_intervention_episodes=96,exact_noop_replays=8);save(ROOT/'summary.json',s);save(ROOT/'video_records.json',video_rows)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(new_intervention_episodes=96,exact_noop_replays=8,seven_joint_noop_results_preserved=True,actual_commands_verified=True,release_fix_verified=True,source_hashes_unchanged=True,no_geometry_policy_input=True,no_held_execution=True))
    lines=['# E164: early six-joint clearance interventions','','E120 ensemble plus the fixed265mm E161 release height is the baseline. Both controller changes are restricted to six-joint bodies. All48 six-joint fitting placements are evaluated per variant; seven-joint results are unchanged references, with four exact no-op replays per variant. This totals96 new intervention episodes and8 exact replays, not192 new independent episodes.','', '| Variant | Six-joint success | Six-joint collisions | First approach/descent contacts | Rescues / regressions |','|---|---:|---:|---:|---:|',f"| E120 + release baseline | {baseline['6']['success']}/48 | {baseline['6']['collisions']} | {sum(basecontacts.get(x,0) for x in ('approach','descend'))} | — |"]
    for name,v in summaries.items():lines.append(f"| {name} | {v['candidate']['6']['success']}/48 | {v['candidate']['6']['collisions']} | {sum(v['first_six_joint_contact_phases'].get(x,0) for x in ('approach','descend'))} | {v['gains']} / {v['losses']} |")
    lines+=['',f'Variant meeting the fixed retention/contact screen: **{preferred}**. Every regression remains recorded; no baseline replacement or held test is performed.','', '**High approach:** raise only the approach target330→400mm; descent remains240mm with its original100-step budget. **Centered early:** add a measured-Jacobian projected joint-limit centering term during approach/descent only, with comfort band ±2rad and maximum0.005rad correction. The original E120 bounded action correction remains present. These are separate interventions, never combined.','', 'Both preserve the original calibration, local probe, frozen measured bias, contact/success criteria and all phase budgets. The elevated release allows a short drop; it is not gentle-contact placement. Estimator weights are frozen. Neither variant uses simulator geometry, exact Jacobians, body IDs, contact signals or future physics. E163 geometry measurements were offline diagnosis only.','', 'The approach and centering changes alter earlier histories, so later task differences are downstream effects of those interventions, unlike E161’s exact pre-lowering intervention. All actual final commands are checked against the saved command log. Seven-joint no-op replay states/traces match E161 exactly.','', 'These are fitting-only studies on two procedural six-joint bodies. All original six-joint baseline failures lie on body84000. Results do not establish unseen-body transfer, learned-estimator superiority, or a general safety guarantee. First-contact counts do not imply that subsequent contact disappears in already failed episodes.','', 'The geometry IDs3/4 in old traces attach to link1/link2; see [E163 diagnosis](../ease_early_contact_diagnosis_v1/report.md). All outcomes, ordered histories, actions, controls and states are saved. Videos show actual saved-state replays, selected to include successes, failures, rescues/regressions and diagnosed cases13/26.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['E120 + release','Higher approach','Early centering'];fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric,title in zip(axes,('success','collisions'),('Six-joint task successes /48','Six-joint collision episodes /48')):
        vals=[baseline['6'][metric]]+[summaries[n]['candidate']['6'][metric] for n in VARIANTS];ax.bar(labels,vals,color=['#777777','#2379b5','#d18b35']);ax.set_ylim(0,48);ax.set_title(title)
        for i,v in enumerate(vals):ax.text(i,v+.5,str(v),ha='center')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','early_clearance_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E164').replace('Task-value ranking diagnosis','Early six-joint clearance comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Fitting six-joint trajectories: frozen E120 with release fix, higher approach, or early joint centering. Successes and failures retained. No geometry input or held test.')
    Path(__file__).with_name('early_clearance_videos.py').write_text(template);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E164:' not in text;p.write_text(text+'\n## E164: early six-joint task control with release fix retained\n\n'+'; '.join(f"{n}: {v['candidate']['6']['success']}/48 six-joint successes, {v['gains']} rescues/{v['losses']} regressions, {v['candidate']['6']['collisions']} collisions" for n,v in summaries.items())+f'. Baseline37/48 with11 collisions. Qualified variant: {preferred}. Fitting only, no privileged policy inputs or held evaluation. Report: `outputs/ease_early_clearance_control_v1/report.md`.\n',encoding='utf-8');save(ROOT/'status.json',dict(status='verified_videos_pending',new_intervention_episodes=96,exact_noop_replays=8,videos_planned=len(video_rows)));print(s,flush=True)

def finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    save(ROOT/'status.json',dict(status='complete',new_intervention_episodes=96,exact_noop_replays=8,videos=len(videos),release_fix_preserved=True,E120_unchanged=True,held_evaluated=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E164 artifacts complete',flush=True)

if __name__=='__main__':
    import sys
    if sys.argv[1]=='run':
        for f in ('trials','states'):(ROOT/f).mkdir()
    {'diagnosis':diagnosis,'run':run,'finish':finish}[sys.argv[1]]()

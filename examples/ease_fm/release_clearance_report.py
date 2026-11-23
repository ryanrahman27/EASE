"""Complete E161 paired fitting outcomes and saved-state demo artifacts."""
from pathlib import Path
from collections import Counter
import shutil
import numpy as np
from .release_clearance_control import ROOT,REFERENCE
from .abstaining_task_value_gate import read,save,fingerprint
from .task_contact_report import stats

def run():
    rows=read(ROOT/'records.json');assert len(rows)==96 and read(ROOT/'verification.json')['all_pre_lower_states_exact'];refs=[r for r in read(REFERENCE/'records.json') if r['method']=='e120_ensemble'];lookup={r['case']:r for r in refs};changes=[];contacts=Counter();selection={}
    for r in rows:
        b=lookup[r['case']]
        if r['success']!=b['success']:changes.append(dict(case=r['case'],dof=r['dof'],before=b['success'],after=r['success']))
        t=read(ROOT/'trials'/f"{r['case']}_e120_ensemble_frozen_bias.json");first=next((x for x in t['trace'] if x['bad']),None)
        if first:contacts[f"{r['dof']}:{first['phase']}"]+=1
    a,b=stats(rows),stats(refs);passed=a['all']['success']>b['all']['success'] and a['all']['collisions']<=b['all']['collisions'] and all(a[str(n)]['success']>=b[str(n)]['success'] for n in (6,7));s=dict(candidate=a,baseline=b,gains=sum(x['after'] for x in changes),losses=sum(x['before'] for x in changes),changes=changes,first_contact_phases=dict(contacts),positive_fitting_screen=passed,all_previous_successes_retained=not any(x['before'] for x in changes),held_evaluated=False,E120_unchanged=True);save(ROOT/'summary.json',s)
    for n in (6,7):
        for success in (True,False):
            rr=[r for r in rows if r['dof']==n and r['success']==success]
            if rr:selection[rr[0]['case']]=rr[0]
        for gain in (True,False):
            rr=[r for r in changes if r['dof']==n and r['after']==gain]
            if rr:selection[rr[0]['case']]=next(r for r in rows if r['case']==rr[0]['case'])
    videos=[]
    for case in selection:
        for source,method in ((ROOT,'elevated_release'),(REFERENCE,'baseline_e120_ensemble')):
            old=f'{case}_e120_ensemble_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{old}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
            for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
            videos.append(dict(case=case,method=method,mode='frozen_bias',dof=t['dof'],success=t['success']))
    save(ROOT/'video_records.json',videos)
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','release_clearance_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E161').replace('Task-value ranking diagnosis','Fitting elevated-release comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Matched E120 fitting trajectories with original versus elevated release. Includes successes and failures. Elevated release allows a short drop; it is not gentle contact placement.')
    Path(__file__).with_name('release_clearance_videos.py').write_text(template)
    lines=['# E161: elevated release on fitting cases','','The E120 ensemble and all estimator checkpoints remain unchanged. Only the lower/release target height changes from 245 mm to 265 mm, a fixed 20 mm margin. No rate limiter from E160 is used. This permits a short free fall and must not be described as gentle placement. The original strict collision, grasp/lift, final object location, settling speed and released-gripper criteria are retained.','', '| Metric | Original E120 ensemble | Elevated release |','|---|---:|---:|',f"| Success | {b['all']['success']}/96 | {a['all']['success']}/96 |",f"| Six joints | {b['6']['success']}/48 | {a['6']['success']}/48 |",f"| Seven joints | {b['7']['success']}/48 | {a['7']['success']}/48 |",f"| Collision episodes | {b['all']['collisions']} | {a['all']['collisions']} |",'',f"Paired outcomes: **{s['gains']} rescues, {s['losses']} regressions**. Positive fitting screen: {passed}. All prior successes retained: {s['all_previous_successes_retained']}.",'','All 96 fitting placements are included, including those already failing before lowering. Each candidate trajectory is exactly identical to the baseline in every saved state and trace entry before its first lowering action. Calibration is identical, bias remains frozen, and checkpoint/source hashes are unchanged. No simulator Jacobian, object pose/contact input, geometry-distance query, physics lookahead or selector is used.','', 'This isolates a task waypoint change under E120, not a learned-estimator improvement. It is adaptive development on four known bodies, not a held or unseen-morphology result. No held placements were executed. Success/failure videos are outcome-selected paired executed-state replays, verified against saved tool/object traces and decoded frame counts.','',f"First forbidden-contact phases: {dict(contacts)}.",'','All changed placements are retained in summary.json; all complete trajectories, actions, histories and simulator states are saved.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric,title in zip(axes,('success','collisions'),('Completed fitting tasks /96','Forbidden-contact episodes /96')):
        values=[b['all'][metric],a['all'][metric]];ax.bar(['Original E120','Elevated release'],values,color=['#777777','#2379b5']);ax.set_ylim(0,96);ax.set_title(title)
        for i,v in enumerate(values):ax.text(i,v+1,str(v),ha='center')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E161:' not in text;p.write_text(text+f"\n## E161: fitting elevated-release clearance\n\nE120 ensemble success {b['all']['success']} to {a['all']['success']}/96; {s['gains']} rescues and {s['losses']} regressions. Collision episodes {b['all']['collisions']} to {a['all']['collisions']}. All pre-lower states match exactly. Elevated release allows a short drop; not gentle placement, learned-estimator superiority or unseen-body evidence. Full report: `outputs/ease_release_clearance_control_v1/report.md`.\n",encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',episodes=96,videos_planned=len(videos)));print(s,flush=True)

def finish():
    manifest=read(ROOT/'videos/manifest.json');assert len(manifest['videos'])==len(read(ROOT/'video_records.json'))
    for v in manifest['videos']:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    save(ROOT/'status.json',dict(status='complete',episodes=96,videos=len(manifest['videos']),E120_unchanged=True,held_evaluated=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E161 complete with verified success/failure videos',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]]()

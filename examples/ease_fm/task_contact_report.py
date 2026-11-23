"""Audit and report fitting-only task-rate limiting, including every regression."""
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
from .task_contact_control import ROOT,REFERENCE
from .abstaining_task_value_gate import read,save,fingerprint

METHODS=('e120_ensemble','uniform32')

def stats(rows):
    return {str(n):dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr)) for n in ('all',6,7) if (rr:=[r for r in rows if n=='all' or r['dof']==n])}

def run():
    records=sum([read(ROOT/f'records_{m}.json') for m in METHODS],[]);assert len(records)==192
    refs=[r for r in read(REFERENCE/'records.json') if r['method'] in METHODS];tasks=read(ROOT/'cases.json');summaries={};selection={}
    for method in METHODS:
        rr=[r for r in records if r['method']==method];bb=[r for r in refs if r['method']==method];lookup={r['case']:r for r in bb};changes=[];contacts=Counter();budget=Counter();scales=[]
        for r in rr:
            case=r['case'];assert not tasks[case]['validation'];stem=f'{case}_{method}_frozen_bias';t=read(ROOT/'trials'/f'{stem}.json');b=read(REFERENCE/'trials'/f'{stem}.json');log=read(ROOT/'limits'/f'{stem}.json');actions=read(ROOT/'actions'/f'{stem}.json')
            assert t['calibration']==b['calibration'] and t['initial_estimator']==b['initial_estimator'];assert t['initial_estimator']['bias']==t['final_estimator']['bias'];assert len(log)==len(actions)==len(t['trace'])
            for l,a in zip(log,actions):
                assert l['phase']==a['phase'];np.testing.assert_array_equal(l['q_before'],a['q_before'])
                if l['phase']=='probe':np.testing.assert_array_equal(l['base_action'],l['final_action'])
                else:
                    expected=np.clip(np.array(l['q_before'])+np.clip(l['final_action'],-.04,.04),-2.5,2.5)-l['q_before']
                    np.testing.assert_array_equal(expected,a['executed_command']);scales.append(l['scale'])
            first=next((x for x in t['trace'] if x['bad']),None)
            if first:contacts[f"{r['dof']}:{first['phase']}"]+=1
            for phase in t['phases']:
                if not phase['reached']:budget[phase['phase']]+=1
            if r['success']!=lookup[case]['success']:changes.append(dict(case=case,dof=r['dof'],before=lookup[case]['success'],after=r['success']))
        candidate,baseline=stats(rr),stats(bb)
        passed=candidate['all']['success']>baseline['all']['success'] and candidate['all']['collisions']<=baseline['all']['collisions'] and all(candidate[str(n)]['success']>=baseline[str(n)]['success'] for n in (6,7))
        summaries[method]=dict(candidate=candidate,baseline=baseline,gains=sum(x['after'] for x in changes),losses=sum(x['before'] for x in changes),changes=changes,positive_fitting_screen=passed,first_contact_phases=dict(contacts),unreached_phase_counts=dict(budget),fraction_scaled=float(np.mean(np.array(scales)<1)),mean_scale=float(np.mean(scales)))
        for n in (6,7):
            for success in (True,False):
                candidates=[r for r in rr if r['dof']==n and r['success']==success]
                if candidates:selection[candidates[0]['case'],method]=candidates[0]
        for gain in (True,False):
            changed=[x for x in changes if x['after']==gain]
            if changed:
                case=changed[0]['case'];selection[case,method]=next(r for r in rr if r['case']==case)
    baseline_e120={r['case']:r for r in refs if r['method']=='e120_ensemble'}
    for method in METHODS:
        rr=[r for r in records if r['method']==method];summaries[method]['versus_original_E120']=dict(gains=sum(r['success'] and not baseline_e120[r['case']]['success'] for r in rr),losses=sum(not r['success'] and baseline_e120[r['case']]['success'] for r in rr))
    save(ROOT/'records.json',records);save(ROOT/'summary.json',summaries)
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    save(ROOT/'verification.json',dict(fitting_episodes=192,baseline_exact_replays=16,calibration_identical=True,frozen_bias=True,actual_executed_commands_verified=True,probes_unchanged=True,source_hashes_unchanged=True,no_held_execution=True))
    videos=[]
    for (case,method),r in selection.items():
        for source,label in ((ROOT,'limited'),(REFERENCE,'baseline')):
            old=f'{case}_{method}_frozen_bias';newmethod=f'{label}_{method}';new=f'{case}_{newmethod}_frozen_bias';trial=read(source/'trials'/f'{old}.json');trial['method']=newmethod;save(ROOT/'trials'/f'{new}.json',trial)
            for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
            videos.append(dict(case=case,method=newmethod,mode='frozen_bias',success=trial['success'],dof=trial['dof']))
    save(ROOT/'video_records.json',videos)
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','task_contact_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E160').replace('Task-value ranking diagnosis','Fitting task-control rate limits').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Matched baseline and rate-limited fitting trajectories. Outcome-selected successes and failures; no held test or safety guarantee.')
    Path(__file__).with_name('task_contact_videos.py').write_text(template)
    lines=['# E160: task-control rate limits on fitting cases','','All 96 fitting placements were evaluated for each controller. E120 checkpoints, reference results, calibration, fixed probes, task targets, time budgets and success/contact criteria remain unchanged. No held placements or unseen bodies were evaluated.','', '| Controller | Original success | Limited success | Six joints (original → limited) | Seven joints | Collision episodes | Rescues / regressions |','|---|---:|---:|---:|---:|---:|---:|']
    for method,s in summaries.items():
        a,b=s['baseline'],s['candidate'];lines.append(f"| {method} | {a['all']['success']}/96 | {b['all']['success']}/96 | {a['6']['success']} → {b['6']['success']} /48 | {a['7']['success']} → {b['7']['success']} /48 | {a['all']['collisions']} → {b['all']['collisions']} | {s['gains']} / {s['losses']} |")
    lines+=['','The intervention scales the complete proposed joint-motion vector after removing frozen measured bias, then restores bias. Limits per 40 ms step: 4 mm predicted position displacement, 0.0045 scaled tool-axis displacement, and 0.02 rad maximum joint displacement. The estimate comes from measured history. No exact kinematics, geometry clearance, contact signal, future simulator prediction or physics selector enters the controller. Thus this is measured-motion rate limiting, not collision avoidance or a safety guarantee.','', 'The baseline audit found arm-link/table contacts during approach/descent and pad/table contacts during lowering. These diagnoses motivated testing whether aggressive motion contributes to failures. Contacts remain scored with the original strict definition, including transient contacts; no failure is filtered out. Equal budgets intentionally expose lost progress from slowing down.','', 'All 16 prespecified baseline replays matched saved states and traces exactly before intervention. All 192 intervention episodes have verified calibration, frozen bias, actual executed commands and unchanged probe commands. Every candidate success/failure clip is paired with the same original reference placement; clips are outcome-selected and do not estimate success rates.','', 'This is fitting-only controller development, not evidence of new-morphology transfer or of a learned estimator outperforming analytical estimation. E120 remains the baseline; any gains and all regressions are reported separately. No automatic held evaluation is authorized by the fitting screen.','']
    for method,s in summaries.items():lines += [f"- {method}: positive fitting screen = {s['positive_fitting_screen']}; first forbidden-contact counts = {s['first_contact_phases']}; unreached phase counts = {s['unreached_phase_counts']}. These counts include later phases after an earlier failure."]
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4));x=np.arange(2)
    for ax,metric,title in zip(axes,('success','collisions'),('Completed fitting tasks /96','Forbidden-contact episodes /96')):
        ax.bar(x-.18,[summaries[m]['baseline']['all'][metric] for m in METHODS],.36,label='Original');ax.bar(x+.18,[summaries[m]['candidate']['all'][metric] for m in METHODS],.36,label='Rate limited');ax.set_xticks(x,['E120 ensemble','Uniform analytical']);ax.set_title(title);ax.legend();ax.set_ylim(0,96)
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    report=Path(__file__).with_name('TECHNICAL_REPORT.md');text=report.read_text(encoding='utf-8');assert '## E160:' not in text
    report.write_text(text+'\n## E160: fitting-only task-control rate limiting\n\n'+'; '.join(f"{m}: {s['baseline']['all']['success']} to {s['candidate']['all']['success']}/96, {s['gains']} rescues and {s['losses']} regressions" for m,s in summaries.items())+'. All criteria and E120 weights unchanged; no held execution. Full report: `outputs/ease_task_contact_control_v1/report.md`.\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',episodes=192,videos_planned=len(videos)));print(summaries,flush=True)

def finish():
    manifest=read(ROOT/'videos/manifest.json');assert len(manifest['videos'])==len(read(ROOT/'video_records.json'))
    for v in manifest['videos']:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    save(ROOT/'status.json',dict(status='complete',episodes=192,baseline_replays=16,videos=len(manifest['videos']),E120_unchanged=True,held_evaluated=False))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    print('E160 complete with verified paired success/failure videos',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]]()

"""Report the estimator-by-task-controller fitting comparison, with honest pairings."""
from pathlib import Path
import shutil
import numpy as np
from .release_clearance_transfer import ROOT,UNIFORM,LEARNED,E120_ROOT
from .abstaining_task_value_gate import read,save,fingerprint
from .task_contact_report import stats

def paired(a,b):
    lookup={r['case']:r for r in b};changes=[dict(case=r['case'],dof=r['dof'],before=lookup[r['case']]['success'],after=r['success']) for r in a if r['success']!=lookup[r['case']]['success']]
    return dict(gains=sum(x['after'] for x in changes),losses=sum(x['before'] for x in changes),changes=changes)

def run():
    roots={'e120':E120_ROOT,'uniform32':ROOT/'uniform32','learned':ROOT/'learned'};sources={'e120':UNIFORM,'uniform32':UNIFORM,'learned':LEARNED};methods={'e120':'e120_ensemble','uniform32':'uniform32','learned':'learned_ensemble'};data={};base={};summary={};video_rows=[]
    for name,root in roots.items():
        assert read(root/'verification.json')['all_pre_lower_states_exact'];data[name]=read(root/'records.json');base[name]=[r for r in read(sources[name]/'records.json') if r['method']==methods[name]];assert len(data[name])==len(base[name])==96
        summary[name]=dict(candidate=stats(data[name]),baseline=stats(base[name]),paired=paired(data[name],base[name]))
    s=dict(methods=summary,learned_vs_e120=paired(data['learned'],data['e120']),learned_vs_uniform=paired(data['learned'],data['uniform32']),new_episodes=192,reused_e120_episodes=96,no_held_execution=True,E120_unchanged=True)
    save(ROOT/'summary.json',s)
    for folder in ('trials','states'):(ROOT/folder).mkdir()
    # E120 clips are already saved in E161. Here save matched pairs for both additional estimators.
    for name in ('uniform32','learned'):
        chosen={}
        for n in (6,7):
            for success in (True,False):
                rr=[r for r in data[name] if r['dof']==n and r['success']==success]
                if rr:chosen[rr[0]['case']]=True
        for gain in (True,False):
            rr=[r for r in summary[name]['paired']['changes'] if r['after']==gain]
            if rr:chosen[rr[0]['case']]=True
        for case in chosen:
            for source,label in ((roots[name],'elevated'),(sources[name],'original')):
                method=label+'_'+name;stem=f'{case}_{methods[name]}_frozen_bias';new=f'{case}_{method}_frozen_bias';t=read(source/'trials'/f'{stem}.json');t['method']=method;save(ROOT/'trials'/f'{new}.json',t)
                for ext in ('xml','npz'):shutil.copy2(source/'states'/f'{stem}.{ext}',ROOT/'states'/f'{new}.{ext}')
                video_rows.append(dict(case=case,method=method,mode='frozen_bias',dof=t['dof'],success=t['success']))
    save(ROOT/'video_records.json',video_rows)
    template=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','release_clearance_transfer import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E162').replace('Task-value ranking diagnosis','Matched fitting release-margin comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Matched fitting trajectories: original versus elevated release with analytical and learned estimators. Successes and failures retained. No held evaluation; short-drop release, not gentle placement.')
    Path(__file__).with_name('release_clearance_transfer_videos.py').write_text(template)
    lines=['# E162: matched release-margin comparison','','This fitting-only extension was registered after partial E161 outcomes, with the release height unchanged at 265 mm. All 96 fitting placements are included for each estimator. The 96 E120 elevated-release episodes are reused from E161; 192 analytical/learned episodes are new. These are development results, not independent held validation.','', '| Estimator | Original success | Elevated release | Six joints (original → elevated) | Seven joints | Collisions | Rescues / regressions |','|---|---:|---:|---:|---:|---:|---:|']
    for name,v in summary.items():
        a,b=v['baseline'],v['candidate'];p=v['paired'];lines.append(f"| {name} | {a['all']['success']}/96 | {b['all']['success']}/96 | {a['6']['success']} → {b['6']['success']} /48 | {a['7']['success']} → {b['7']['success']} /48 | {a['all']['collisions']} → {b['all']['collisions']} | {p['gains']} / {p['losses']} |")
    lines+=['',f"With the same elevated-release controller, learned versus E120: {s['learned_vs_e120']['gains']} rescues / {s['learned_vs_e120']['losses']} regressions. Learned versus uniform analytical: {s['learned_vs_uniform']['gains']} rescues / {s['learned_vs_uniform']['losses']} regressions.",'','The learned model is the unchanged E153/E154 ensemble, using all three seeds and the same four family-excluded fitting models. No retraining, checkpoint selection, online weight update, new model inputs, object/contact signal or physics selector was introduced. The analytical condition retains its 32-transition, .98-recency, 1e-6-ridge estimate. E120 retains its original bounded correction and checkpoints.','', 'Only lower/release target height changes, by 20 mm. This permits short-drop placement and is not a gentle-contact manipulation result. All success criteria, phase budgets, probe motions and measured bias rules remain unchanged. Every saved state and trace before lowering matches its own original estimator/controller reference exactly. Thus within-estimator changes isolate the release-stage controller modification; cross-estimator comparisons still contain different earlier trajectories.','', 'E159 is not retested or used for fitting. No unseen bodies are evaluated. The original E120 results remain immutable and separately reported. Any controller gains should not be attributed to learned adaptation without the matched analytical comparison.','', '[E120 paired videos](../ease_release_clearance_control_v1/videos/index.html). This study saves additional paired analytical/learned successes and failures. Clips are selected by outcome for diagnosis, not for estimating rates. All changed placements and losses are retained in summary.json.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));names=list(summary);x=np.arange(3)
    for ax,metric,title in zip(axes,('success','collisions'),('Completed fitting tasks /96','Forbidden-contact episodes /96')):
        ax.bar(x-.18,[summary[n]['baseline']['all'][metric] for n in names],.36,label='Original');ax.bar(x+.18,[summary[n]['candidate']['all'][metric] for n in names],.36,label='Elevated release');ax.set_xticks(x,['E120 ensemble','Uniform analytical','Learned weights']);ax.set_ylim(0,96);ax.set_title(title);ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E162:' not in text;p.write_text(text+'\n## E162: matched fitting release-margin transfer\n\n'+'; '.join(f"{n}: {v['baseline']['all']['success']} to {v['candidate']['all']['success']}/96, {v['paired']['gains']} rescues/{v['paired']['losses']} regressions" for n,v in summary.items())+'. Original estimators unchanged, all pre-lower states exactly matched. Fitting only, no held evaluation. Report: `outputs/ease_release_clearance_transfer_v1/report.md`.\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='verified_videos_pending',episodes=192,reused_e120_episodes=96,videos_planned=len(video_rows)));print(s,flush=True)

def finish():
    manifest=read(ROOT/'videos/manifest.json');assert len(manifest['videos'])==len(read(ROOT/'video_records.json'))
    for v in manifest['videos']:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    save(ROOT/'status.json',dict(status='complete',episodes=192,reused_e120_episodes=96,videos=len(manifest['videos']),E120_unchanged=True,held_evaluated=False));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E162 complete with paired success/failure videos',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]]()

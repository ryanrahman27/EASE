"""E151/E152 full comparisons, action verification and success/failure replays."""
import shutil
from pathlib import Path
import numpy as np
from .home_posture_control import ROOT,WORKERS
from .prior_posture_reference import ROOT as PRIOR
from .history_estimator_control import ROOT as OLD
from .learned_calibration_control import ROOT as OLD_LEARNED
from .exact_map_control import ROOT as OLD_EXACT
from .calibration_prior_control import ROOT as OLD_PRIOR
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    summary=read(ROOT/'summary.json');rows=read(ROOT/'records.json')+read(PRIOR/'records.json');roots={m:ROOT/m for m in WORKERS};roots['calibration_prior']=PRIOR;previous={m:OLD for m in ('uniform32','e120_ensemble')};previous.update(learned_prior=OLD_LEARNED,privileged_exact_map=OLD_EXACT,calibration_prior=OLD_PRIOR);comparisons={};selected={}
    for r in rows:
        case,method=r['case'],r['method'];source=roots[method];stem=f'{case}_{method}_frozen_bias';trial=read(source/'trials'/f'{stem}.json');actions=read(source/'actions'/f'{stem}.json');posture=read(source/'posture'/f'case_{case}.json');assert len(actions)==len(posture)==len(trial['trace']);assert trial['initial_estimator']['bias']==trial['final_estimator']['bias']
        for a,p in zip(actions,posture):
            q=np.array(a['q_before']);expected=np.clip(q+np.clip(p['final_action'],-.04,.04),-2.5,2.5)-q;np.testing.assert_allclose(expected,a['executed_command'],atol=1e-12,rtol=0);assert np.max(np.abs(np.array(p['final_action'])-p['base_action']))<=.010000000001
            if p['phase']=='probe':np.testing.assert_array_equal(p['final_action'],p['base_action']);np.testing.assert_array_equal(p['posture_delta'],np.zeros(len(q)))
    for method in roots:
        rr=[r for r in rows if r['method']==method];old=[r for r in read(previous[method]/'records.json') if r['method']==method];pairs=[dict(case=r['case'],dof=r['dof'],before=next(v['success'] for v in old if v['case']==r['case']),after=r['success']) for r in rr];comparisons[method]=dict(old_success=sum(r['success'] for r in old),posture_success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),gains=sum(r['after'] and not r['before'] for r in pairs),losses=sum(r['before'] and not r['after'] for r in pairs),changes=[r for r in pairs if r['before']!=r['after']])
        if method in ('learned_prior','calibration_prior'):
            for gain in (True,False):
                changes=[r for r in pairs if r['after']==gain and r['before']!=r['after']]
                if changes:
                    case=changes[0]['case'];selected[case,method,'posture']=roots[method];selected[case,method,'standard']=previous[method]
    for method in ('learned_prior','calibration_prior','uniform32','e120_ensemble','privileged_exact_map'):
        for n in (6,7):
            for success in (True,False):
                rr=[r for r in rows if r['method']==method and r['dof']==n and r['success']==success]
                if rr:selected[rr[0]['case'],method,'posture']=roots[method]
    save(ROOT/'posture_changes.json',comparisons);save(ROOT/'execution_verification.json',dict(episodes=len(rows),actual_commands_match_logged_posture_actions=True,probe_actions_unchanged=True,bounded_delta=True,frozen_bias=True));lines=['# E151/E152: fixed home-posture objective','','All controllers receive the same observation-only damped projection toward their supplied starting posture. Gain.05, maximum per-joint correction.01rad, task damping.015. No scene-geometry lookup or physics selector. Exact-map entries remain privileged diagnostics. E152 adds the matched analytical-prior reference after partial E151 outcomes, without changing coefficients or models.','', '| Estimator | Original successes | Posture successes | Rescues | Lost successes | Posture collisions |','|---|---:|---:|---:|---:|---:|']
    for method,v in comparisons.items():lines.append(f"| {method} | {v['old_success']}/96 | {v['posture_success']}/96 | {v['gains']} | {v['losses']} | {v['collisions']} |")
    lines+=['',f"E151 frozen fresh-test screen: **{summary['qualifies_for_fresh_test']}**. The learned-prior controller recovers persistent case13 but loses other successes. It completes70/96, versus matched analytical-prior68 and uniform76. This does not establish a stronger overall learned controller. E120 weights and original baseline records remain unchanged; no fresh/held task was executed.",'','All480 executed fitting trajectories and posture corrections are retained. Every actual command was independently checked against the logged final action and joint limits; probe actions and measured frozen bias are unchanged. Videos are explicitly outcome-selected comparisons of successes and failures.',''];(ROOT/'report.md').write_text('\n'.join(lines));(PRIOR/'report.md').write_text('# E152: matched analytical-prior posture reference\n\nAnalytical calibration prior plus the unchanged E151 posture rule completes68/96 (34/48 each joint count), with20 collisions. Learned prior plus the same rule completes70/96. This additional control was registered after partial E151 results; no model or coefficient was changed. All96 outcomes, actions and states are retained.\n\n[Full matched comparison and videos](../ease_home_posture_control_v1/report.md).\n')
    report=Path(__file__).with_name('TECHNICAL_REPORT.md');text=report.read_text(encoding='utf-8');assert '## E151:' not in text and '## E152:' not in text;report.write_text(text+'\n## E151: fixed home-posture control\n\n'+lines[-5]+f' Report: `{ROOT}/report.md`.\n\n## E152: matched analytical-prior posture reference\n\n68/96 versus learned70/96; same posture objective. Added after partial E151 outcomes, no parameter changes. No fresh test.\n',encoding='utf-8')
    for folder in ('trials','states'):(ROOT/folder).mkdir(exist_ok=True)
    video=[]
    for (case,method,context),source in selected.items():
        oldstem=f'{case}_{method}_frozen_bias';alias=f'{method}_{context}';stem=f'{case}_{alias}_frozen_bias';trial=read(source/'trials'/f'{oldstem}.json');trial['source_method']=trial['method'];trial['method']=alias;save(ROOT/'trials'/f'{stem}.json',trial)
        for ext in ('npz','xml'):shutil.copy2(source/'states'/f'{oldstem}.{ext}',ROOT/'states'/f'{stem}.{ext}')
        video.append(dict(case=case,method=alias,mode='frozen_bias',success=trial['success'],collision=trial['collision']))
    save(ROOT/'video_records.json',video)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=list(comparisons);x=np.arange(len(keys));fig,ax=plt.subplots(figsize=(11,4));ax.bar(x-.18,[comparisons[k]['old_success'] for k in keys],.36,label='Original');ax.bar(x+.18,[comparisons[k]['posture_success'] for k in keys],.36,label='Posture objective');ax.set_xticks(x,keys,rotation=15);ax.set_ylim(0,96);ax.set_ylabel('Completed fitting tasks /96');ax.legend();fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    st=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','home_posture_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E151/E152').replace('Task-value ranking diagnosis','Home-posture comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Outcome-selected fitting successes and failures. Posture versus standard labels identify controller context; exact-map entries are privileged references.').replace('Measured-history control | No physics selector or geometry filter','No physics selector | exact-map entries are privileged');Path(__file__).with_name('posture_control_videos.py').write_text(st);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('Verified480 trajectories; prepared',len(video),'videos',flush=True)

if __name__=='__main__':run()

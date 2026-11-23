"""Verify and publish complete fitting-controller outcomes, including regressions."""
import shutil
from pathlib import Path
import numpy as np
from .history_estimator_control import ROOT,METHODS
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    rows=read(ROOT/'records.json');s=read(ROOT/'summary.json');assert len(rows)==480
    for p,h in read(ROOT/'frozen.json').items():assert fingerprint(p)==h
    for p,h in read(ROOT/'protocol.json')['checkpoints'].items():assert fingerprint(p)==h
    for r in rows:
        stem=f"{r['case']}_{r['method']}_frozen_bias";t=read(ROOT/'trials'/f'{stem}.json');archive=np.load(ROOT/'histories'/f'{stem}.npz');d={k:archive[k] for k in archive.files};archive.close();actions=read(ROOT/'actions'/f'{stem}.json');states=np.load(ROOT/'states'/f'{stem}.npz')['states'];assert len(actions)==len(states)==len(t['trace'])==len(d['query']);assert t['initial_estimator']['bias']==t['final_estimator']['bias'];assert np.isfinite(d['target']).all();assert t['success']==r['success']
        for i in range(1,len(actions)):
            np.testing.assert_allclose(d['query'][i-1,:r['dof']],np.array(actions[i]['q_before'])-actions[i-1]['q_before'],rtol=1e-5,atol=1e-8)
            np.testing.assert_array_equal(d['dq_history'][i,-1],d['query'][i-1]);np.testing.assert_array_equal(d['dx_history'][i,-1],d['target'][i-1])
    save(ROOT/'verification.json',dict(episodes=480,history_order_and_actual_joint_queries_verified=True,frozen_bias=True,source_and_E120_hashes_unchanged=True))
    lines=['# E144: complete fitting episodes for history estimation','','All five methods execute the same96 fitting placements with identical calibration, task schedule and collision definitions. Neural/kernel models exclude each placement ancestry family. This is development validation on known bodies, not a new embodiment test.','', '| Controller | All successes | Six-joint successes | Seven-joint successes | Collision episodes |','|---|---:|---:|---:|---:|']
    for k,v in s['methods'].items():lines.append(f"| {k} | {v['all']['success']}/{v['all']['n']} | {v['6']['success']}/{v['6']['n']} | {v['7']['success']}/{v['7']['n']} | {v['all']['collisions']} |")
    lines += ['', 'E120 ensemble is a new matched three-model reference, not the original195/288 individual-seed baseline. Learned estimator uses a fixed average of all three history-weight models inside an explicit analytical solve. No physics selector, exact Jacobian, object/contact input, task-success gate or online gradient update.', '', '| Learned versus | Rescues | Lost successes |','|---|---:|---:|']
    for k,v in s['paired'].items():lines.append(f"| {k} | {v['gains']} | {v['losses']} |")
    lines += ['',f"Passed frozen advancement screen: **{s['qualifies_for_fresh_test']}**. E120 fitting gain placement retention: {s['E120_fitting_gain_placements']}. All per-case outcomes and regressions are in summary.json. No fresh/held placements were executed. E120 remains unchanged.",'','All480 full state/action/history records are saved. History ordering, actual motion-query alignment, frozen bias, source and baseline hashes were checked. Gallery is outcome-selected and includes both successes and failures, not an unbiased sample of rates.','']
    (ROOT/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E144:' not in text;p.write_text(text+'\n## E144: complete history-estimator control\n\n'+str(s['methods'])+f"\n\nAdvancement: {s['qualifies_for_fresh_test']}. All regressions retained; E120 unchanged. Report: `{ROOT}/report.md`.\n",encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,4));ax.bar(METHODS,[s['methods'][m]['all']['success'] for m in METHODS]);ax.set_ylim(0,96);ax.set_ylabel('Completed tasks /96');ax.set_title('E144 | complete fitting episodes | known bodies');ax.tick_params(axis='x',rotation=15);fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    selected={}
    for m in METHODS:
        for dof in (6,7):
            for success in (True,False):
                rr=[r for r in rows if r['method']==m and r['dof']==dof and r['success']==success]
                if rr:r=rr[0];selected[r['case'],m]=r
    for k,v in s['paired'].items():
        for gain in (True,False):
            changes=[r for r in v['changes'] if r['after']==gain]
            if changes:
                case=changes[0]['case']
                for m in (k,'learned_ensemble'):selected[case,m]=next(r for r in rows if r['case']==case and r['method']==m)
    save(ROOT/'video_records.json',list(selected.values()));source=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','history_estimator_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E144').replace('Task-value ranking diagnosis','Complete history-estimator control').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Outcome-selected complete fitting episodes, including rescues and regressions. Known bodies; no fresh test.');Path(__file__).with_name('history_control_videos.py').write_text(source);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('Verified480 episodes; prepared',len(selected),'videos',flush=True)

if __name__=='__main__':run()

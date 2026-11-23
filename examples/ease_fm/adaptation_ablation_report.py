"""Aggregate all E114 outcomes, keeping analytic trials distinct from model seeds."""
import json,shutil
from collections import defaultdict
from pathlib import Path
import numpy as np
from .adaptation_ablation import ROOT,MODES
from .benchmark import fingerprint

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==192
    lookup={(r['case'],r['method'],r['mode']):r for r in rows};assert len(lookup)==192
    protocol=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**protocol['inputs'],**protocol['checkpoints']}.items():assert fingerprint(p)==h
    groups=defaultdict(list);initial={};residuals={};phase_rows=[]
    for r in rows:
        method=r['method'];family='analytical' if method=='analytical' else 'learned'
        for split in ('all',str(r['dof'])):
            groups[(family,r['mode'],split)].append(r)
        full=json.loads((ROOT/'trials'/f"{r['case']}_{method}_{r['mode']}.json").read_text())
        if r['case'] not in initial:initial[r['case']]=full['initial_estimator']
        assert full['initial_estimator']==initial[r['case']]
        assert len(full['trace'])==len(full['telemetry'])==r['steps']-32
        if r['mode']!='online':
            for key in ('gram','cross','history_count'):assert full['initial_estimator'][key]==full['final_estimator'][key]
        if r['mode']=='frozen_all':assert full['initial_estimator']==full['final_estimator']
        residuals[(r['case'],method,r['mode'])]=float(np.mean([v['position_prediction_error'] for v in full['telemetry']])*1000)
        first_contact=next((v['phase'] for v in full['trace'] if v['bad']),None)
        for phase in full['phases']:
            phase_rows.append(dict(case=r['case'],method=method,mode=r['mode'],dof=r['dof'],first_collision_phase=first_contact,**phase))
    summary=[]
    for (method,mode,dof),group in groups.items():
        summary.append(dict(method=method,mode=mode,dof=dof,n=len(group),success=sum(r['success'] for r in group),collision=sum(r['collision'] for r in group),grasped=sum(r['grasped'] for r in group),lifted=sum(r['lifted'] for r in group),mean_steps=float(np.mean([r['steps'] for r in group])),mean_episode_prediction_error_mm=float(np.mean([residuals[(r['case'],r['method'],mode)] for r in group]))))
    paired=[]
    for method in ('analytical','learned_0','learned_1','learned_2'):
        for control in ('frozen_response','frozen_all','analytical_online'):
            if method=='analytical' and control=='analytical_online':continue
            for dof in ('all',6,7):
                changes=[];n=0
                for case in range(16):
                    a=lookup[case,method,'online'];b=lookup[case,'analytical','online'] if control=='analytical_online' else lookup[case,method,control]
                    if dof!='all' and a['dof']!=dof:continue
                    n+=1
                    if a['success']!=b['success']:changes.append(dict(case=case,body=a['body'],dof=a['dof'],gain=bool(a['success'])))
                paired.append(dict(method=method,control=control,dof=str(dof),n=n,gains=sum(c['gain'] for c in changes),losses=sum(not c['gain'] for c in changes),changes=changes))
    bybody=[]
    for body in sorted(set(r['body'] for r in rows)):
        for method in ('analytical','learned_0','learned_1','learned_2'):
            for mode in MODES:
                group=[r for r in rows if r['body']==body and r['method']==method and r['mode']==mode]
                bybody.append(dict(body=body,dof=group[0]['dof'],method=method,mode=mode,success=sum(r['success'] for r in group),n=len(group)))
    (ROOT/'summary.json').write_text(json.dumps(dict(aggregate=summary,paired=paired,by_body=bybody),indent=2))
    (ROOT/'phase_diagnostics.json').write_text(json.dumps(phase_rows,indent=2))
    overall=lambda method,mode:next(a for a in summary if a['method']==method and a['mode']==mode and a['dof']=='all')
    live=overall('learned','online');fr=overall('learned','frozen_response');fa=overall('learned','frozen_all');analytic=overall('analytical','online')
    result_text=f"Learned online succeeds in {live['success']}/48 checkpoint-task runs, versus {fr['success']}/48 with response frozen and {fa['success']}/48 with all post-calibration adaptation frozen. Analytical online succeeds in {analytic['success']}/16 distinct tasks. Learned counts pool three checkpoint seeds on the same16 tasks; they are not48 independent body/task samples."
    if live['success']<=fr['success'] and live['success']<=fa['success']:
        result_text+=' This comparison does not show an aggregate task-success benefit from online learned adaptation over the calibrated frozen controls.'
    elif live['success']>fr['success'] and live['success']>fa['success']:
        result_text+=' Online learned adaptation improves aggregate success over both calibrated frozen controls on this development set; seed-level gains/losses and the analytical reference still determine how narrowly to interpret it.'
    else:result_text+=' The result depends on which component is frozen; it does not support a blanket benefit from all online adaptation.'
    if live['success']/48<=analytic['success']/16:result_text+=' The learned online estimator does not exceed analytical online task success.'
    lines=['# E114: Learned adaptation without privileged action selection','',result_text,'',
        '16 reused unseen-body development tasks: eight bodies, four previously studied axis families, two placements per body. Three frozen learned checkpoints and one analytical estimator, each in three update conditions, produce 192 complete episodes. The analytical episodes are not replicated as independent seed trials.','',
        'Every condition receives the same 32 measured calibration transitions. Online updates both response and command bias; frozen response updates only bias; frozen all updates neither. The same 32-step local probe is executed before grasp, but frozen conditions do not incorporate its motion observations. Online switches to the existing 32-transition recent-history estimator at that point. All learned conditions use a single E99 retained checkpoint throughout the episode, eliminating the earlier parent/student switch.','',
        'No geometry filter, physics selector, planner, privileged response Jacobian or gravity feedforward is used. Estimator initialization is 1e-6 I / zero cross statistics, not a model-derived Jacobian. The common simulator reset pose and phase goals are supplied; observations include measured joint positions and tool pose. Physics only executes the commanded actions and scores the outcome.','',
        '| Estimator | Updates | DOF | Successful | Collision episodes |','|---|---|---|---:|---:|']
    for a in summary:lines.append(f"| {a['method']} | {a['mode']} | {a['dof']} | {a['success']}/{a['n']} | {a['collision']} |")
    lines+=['','## Paired attribution','', '| Estimator online | Compared with | DOF | Gains | Losses |','|---|---|---|---:|---:|']
    for p in paired:lines.append(f"| {p['method']} | {p['control']} | {p['dof']} | {p['gains']} | {p['losses']} |")
    lines+=['','## Interpretation and scope','',
        'Online versus frozen response isolates the value of updating the motion response/history pipeline while retaining measured command-bias adaptation in both conditions. Frozen response versus frozen all exposes bias adaptation. Learned versus analytical online compares estimators under matched observations and controller structure. Online versus frozen all bundles both response and bias adaptation; it cannot alone establish a learned-estimator advantage.',
        '', 'Frozen controls retain a calibrated estimate; these are not zero-history or untrained controls. Later trajectories differ, so prediction residuals are own-trajectory diagnostics, not matched-transition prediction benchmarks. The local probe actions also depend on each controller’s current bias; interaction counts/budgets are matched, not post-divergence state trajectories.',
        '', 'Bodies are excluded from estimator training, but these tasks and axis families have already been examined during development. This is an attribution diagnostic, not a fresh confirmatory generalization test. Scores are not directly comparable to E112 because this removes its geometry filter and phase-dependent checkpoint handoff. No flow-matching policy was evaluated. No checkpoint is promoted or tuned from these results.',
        '', '## Verification and artifacts','',
        'All 192 trials have identical per-case initial sufficient statistics and calibration, frozen-state assertions, saved state trajectories and exact scene XML. Checkpoint/input hashes are unchanged. Task Jacobian queries are blocked during calibration and execution. Full traces, per-step prediction diagnostics, phase results and body/seed breakdowns are retained. Videos are selected diagnostics rather than a representative success-rate sample.']
    text='\n'.join(lines)+'\n';(ROOT/'report.md').write_text(text)
    # A compact figure is part of the technical report, not a promotional reel.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    for ax,dof in zip(axes,('6','7')):
        for i,family in enumerate(('learned','analytical')):
            vals=[next(a for a in summary if a['method']==family and a['mode']==mode and a['dof']==dof) for mode in MODES]
            ax.bar(np.arange(3)+(i-.5)*.34,[a['success']/a['n']*100 for a in vals],width=.32,label=family)
            for x,a in zip(np.arange(3)+(i-.5)*.34,vals):ax.text(x,a['success']/a['n']*100+1,f"{a['success']}/{a['n']}",ha='center',fontsize=9)
        ax.set_title(f'{dof}-joint unseen development bodies');ax.set_xticks(range(3),['Online','Response frozen','All frozen']);ax.set_ylim(0,110);ax.grid(axis='y',alpha=.2);ax.legend()
    axes[0].set_ylabel('Complete task success (%)');fig.suptitle('E114 · No physics selector or geometry filter');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    report=Path(__file__).with_name('TECHNICAL_REPORT.md');s=report.read_text(encoding='utf-8');marker='## E114: observation-only learned-adaptation ablations'
    assert marker not in s
    s=s.replace('Revision 101 · 14 November 2026','Revision 102 · 14 November 2026',1)
    s+='\n'+marker+'\n\n'+text.split('## Paired attribution')[0].split('\n',1)[1]+'\nComplete paired gains/losses, per-body breakdown and scope in `outputs/ease_adaptation_ablation_v1/report.md`. Reused development bodies; no new final-test, flow baseline or promotion claim.\n'
    report.write_text(s,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps([r for r in summary if r['dof']=='all'],indent=2))

if __name__=='__main__':run()

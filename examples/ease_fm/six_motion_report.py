"""Same-transition diagnostics and paired intervention outcomes for E116."""
import json,shutil
from pathlib import Path
import numpy as np
from .six_motion_core import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==80
    lookup={(r['case'],r['method'],r['condition']):r for r in rows}
    aggregate=[];paired=[];episodes=[];phase_stats={}
    for condition in ('baseline','probe','lift'):
        group=[r for r in rows if r['method']!='analytical' and r['condition']==condition]
        aggregate.append(dict(condition=condition,n=len(group),success=sum(r['success'] for r in group),collision=sum(r['collision'] for r in group),grasped=sum(r['grasped'] for r in group),lifted=sum(r['lifted'] for r in group)))
        if condition!='baseline':
            changes=[dict(case=r['case'],method=r['method'],gain=r['success']) for r in group if r['success']!=lookup[r['case'],r['method'],'baseline']['success']]
            blocked=[]
            for r in group:
                base=json.loads((ROOT/'trials'/f"{r['case']}_{r['method']}_baseline.json").read_text())
                idx=next(i for i,v in enumerate(base['trace']) if v['phase']==condition)
                if any(v['bad'] for v in base['trace'][:idx]):blocked.append(dict(case=r['case'],method=r['method']))
            paired.append(dict(condition=condition,gains=sum(c['gain'] for c in changes),losses=sum(not c['gain'] for c in changes),already_collided_before_handoff=blocked,changes=changes))
    for r in rows:
        if r['condition']!='baseline':continue
        p=ROOT/'trials'/f"{r['case']}_{r['method']}_baseline.json";full=json.loads(p.read_text());clean=[];first=None
        for i,(trace,d) in enumerate(zip(full['trace'],full['diagnostics'],strict=True)):
            if first is None and trace['bad']:first=dict(step=i,phase=trace['phase'])
            if first is None:
                clean.append(d)
                if r['method']!='analytical':phase_stats.setdefault(d['phase'],[]).append(d)
        keys=('learned_xyz_error','analytical_xyz_error','learned_axis_error','analytical_axis_error','relative_matrix_difference','action_difference','learned_saturated','analytical_saturated','gram_condition','gram_min_eigenvalue')
        means={k:float(np.mean([d[k] for d in clean])) if clean else None for k in keys}
        old=json.loads((Path('outputs/ease_frozen_bias_v1/trials')/f"{r['case']}_{r['method']}_frozen_all.json").read_text())
        episodes.append(dict(case=r['case'],method=r['method'],success=r['success'],frozen_all_success=old['success'],first_collision=first,clean_transitions=len(clean),**means))
    phase=[]
    for name,values in phase_stats.items():
        phase.append(dict(phase=name,n=len(values),learned_xyz_mm=float(np.mean([d['learned_xyz_error'] for d in values])*1000),analytical_xyz_mm=float(np.mean([d['analytical_xyz_error'] for d in values])*1000),learned_scaled_axis_mm=float(np.mean([d['learned_axis_error'] for d in values])*1000),analytical_scaled_axis_mm=float(np.mean([d['analytical_axis_error'] for d in values])*1000),median_gram_condition=float(np.median([d['gram_condition'] for d in values])),median_relative_matrix_difference=float(np.median([d['relative_matrix_difference'] for d in values])),mean_action_difference=float(np.mean([d['action_difference'] for d in values]))))
    neural=[e for e in episodes if e['method']!='analytical'];analytic=[r for r in rows if r['method']=='analytical']
    summary=dict(aggregate=aggregate,paired=paired,episodes=episodes,clean_phase_diagnostics=phase,analytical_success=sum(r['success'] for r in analytic),analytical_n=8,mean_episode_clean_learned_xyz_mm=float(np.mean([e['learned_xyz_error'] for e in neural])*1000),mean_episode_clean_analytical_xyz_mm=float(np.mean([e['analytical_xyz_error'] for e in neural])*1000),episodes_analytical_prediction_better=sum(e['analytical_xyz_error']<e['learned_xyz_error'] for e in neural))
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E116: Diagnosing six-joint learned motion adaptation','',
        'All eight six-joint E115 tasks and all three learned checkpoints are retained, including successes.32 baseline executions reproduce E115 exactly (24 learned,8 analytical);48 additional complete episodes substitute the analytical response estimator from either probe or lift onward. Bias stays frozen at the same post-calibration value. No physics selector, geometry filter, planner or exact Jacobian is used.','',
        '## Same-history response diagnostic','',
        f"Before first forbidden collision, the episode-macro mean XYZ prediction error is {summary['mean_episode_clean_learned_xyz_mm']:.4f} mm for the learned estimator and {summary['mean_episode_clean_analytical_xyz_mm']:.4f} mm for analytical ridge on the SAME measured transitions and pre-update statistics. Analytical prediction is better in {summary['episodes_analytical_prediction_better']}/24 learned-driven episodes.",
        '', 'The comparator receives the identical gram/cross history and predicts the actual next joint/tool transition before its update. No analytic driver trajectory is substituted for this metric. The first colliding transition and every subsequent transition are excluded from clean diagnostics. Phase aggregates weight transitions; the headline averages per-episode means. Scaled tool-axis errors use the existing0.15 response scale, not angular degrees.','',
        '| Phase | Clean steps | Learned XYZ mm | Analytical XYZ mm | Median history condition number |','|---|---:|---:|---:|---:|']
    for p in phase:lines.append(f"| {p['phase']} | {p['n']} | {p['learned_xyz_mm']:.4f} | {p['analytical_xyz_mm']:.4f} | {p['median_gram_condition']:.1f} |")
    lines+=['','## Controlled estimator substitution','', '| Condition | Success | Collision episodes |','|---|---:|---:|']
    for r in aggregate:lines.append(f"| {r['condition']} | {r['success']}/{r['n']} | {r['collision']} |")
    lines.append(f"| Analytical throughout (distinct tasks) | {summary['analytical_success']}/8 | {sum(r['collision'] for r in analytic)} |")
    for p in paired:lines+=['',f"Switch at {p['condition']}: {p['gains']} paired gains and {p['losses']} losses relative to learned baseline. {len(p['already_collided_before_handoff'])}/24 prefixes already contain a disqualifying collision before handoff."]
    lines+=['','## Interpretation boundaries','',
        'An analytical substitution keeps all observations, sufficient statistics, history updates, frozen bias and downstream controller unchanged at handoff. Improvements therefore support an estimator-output contribution to failure. They do not identify a unique training defect. Poor excitation/conditioning and action differences are diagnostic associations; this experiment does not manipulate history excitation or retrain the transformer. Saved action-disagreement/saturation fields compare nominal DLS commands; during the explicit probe these are not the executed probe commands.',
        '', 'The probe switch occurs after the existing recent32-history rebuild, before local probing. The lift switch retains the complete learned descent, probe and grasp. Once a baseline has already collided or dropped the object, later substitution may be unable to recover success; lack of recovery then does not exonerate earlier prediction errors. All selected switch points are fixed before outcomes. No best-state or best-seed selection.',
        '', 'These are reused development bodies and task placements. Scores are not fresh generalization estimates; no hardware or flow-matching comparison. A hybrid used for diagnosis is not promoted as a learned-adaptation result.',
        '', '## Artifacts and verification','',
        'Full observations, predicted-motion diagnostics, state sequences and exact scene XML are saved for all80 episodes.32 baselines match every E115 trace/state exactly;48 substitutions match the full pre-handoff trace. Input and checkpoint hashes remain unchanged. Outcome-labeled success/failure videos show diagnostic examples, not a representative success-rate sample.']
    text='\n'.join(lines)+'\n';(ROOT/'report.md').write_text(text)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    axes[0].bar(['Learned','Analytical\nsame history'],[summary['mean_episode_clean_learned_xyz_mm'],summary['mean_episode_clean_analytical_xyz_mm']]);axes[0].set_ylabel('Pre-collision XYZ prediction error (mm)');axes[0].set_title('Identical observed transitions')
    axes[1].bar(['Learned','Switch at probe','Switch at lift'],[r['success']/24*100 for r in aggregate]);axes[1].set_ylim(0,105);axes[1].set_ylabel('Complete-task success (%)')
    for i,r in enumerate(aggregate):axes[1].text(i,r['success']/24*100+2,f"{r['success']}/24",ha='center')
    fig.suptitle('E116 · Six-joint estimator diagnosis');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='## E116: six-joint learned estimator diagnosis';assert marker not in s
    s=s.replace('Revision 103 · 14 November 2026','Revision 104 · 14 November 2026',1)
    s+='\n'+marker+'\n\n32 exact baseline replays and48 full-task analytical-substitution interventions; frozen calibrated bias, no selector.\n\n'+lines[6]+'\n\n'
    for r in aggregate:s+=f"- {r['condition']}: {r['success']}/24 success, {r['collision']} collisions.\n"
    s+='\nSame-history errors, phase/conditioning diagnostics and paired outcomes: `outputs/ease_six_motion_diagnosis_v1/report.md`. Reused development tasks; no training or promotion.\n';p.write_text(s,encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps({k:v for k,v in summary.items() if k not in ('episodes','clean_phase_diagnostics')},indent=2))

if __name__=='__main__':run()

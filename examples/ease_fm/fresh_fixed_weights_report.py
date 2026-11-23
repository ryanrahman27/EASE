"""Report the one-time prospective comparison, including amendment and regressions."""
import shutil
from pathlib import Path
import numpy as np
from .fresh_fixed_weights import ROOT
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    s=read(ROOT/'summary.json');assert read(ROOT/'verification.json')['prospective_episodes']==192;rows=read(ROOT/'records.json');order=('learned_final','uniform32','e120_ensemble','analytical');m=s['methods'];lines=['# E159: one-time fresh-placement validation','','This is the first evaluation on the48 reserved E137 placements: new placements on four known training bodies, not unseen-embodiment transfer. All48 placements and all192 episodes are retained, including collisions and failures. One shared frozen learned ensemble controls both joint counts without body/family input or target-body weight training.','', '| Frozen controller | Success | Six joints | Seven joints | Collision episodes |','|---|---:|---:|---:|---:|']
    for k in order:
        v=m[k];lines.append(f"| {k} | {v['all']['success']}/48 ({100*v['all']['success']/48:.1f}%) | {v['6']['success']}/24 | {v['7']['success']}/24 | {v['all']['collisions']} |")
    lines+=['','| Learned compared with | Rescued tasks | Lost successes | Net tasks | No regressions? |','|---|---:|---:|---:|---|']
    for k,p in s['paired'].items():lines.append(f"| {k} | {p['gains']} | {p['losses']} | {p['net']:+d} | {p['no_regressions']} |")
    conclusion='The frozen model meets the prospective net-improvement pattern: higher overall success than uniform and E120 ensemble, no lower joint-count success, and no higher collision count.' if s['positive_net_pattern'] else 'The fitting improvement does not meet the prospective net-improvement pattern. The fresh outcomes remain a negative or mixed validation result; no model was retuned or substituted.'
    lines+=['',conclusion,'','The earlier E137/E154 retention gates were **not passed**. Before any fresh outcome, E159 explicitly amended the question from loss-free retention to prospective net improvement. E154 standard had77/96 fitting successes versus uniform74/E12072, but lost two uniform successes and failed legacy individual-seed E120 case13. Its matched E120 ensemble also failed that legacy case. All original failed-gate flags remain unchanged. This amended test does not retroactively satisfy them or authorize replacing E120. See [the frozen pre-test protocol](protocol.json).','', 'The selected model is the full-strength E153 relative-weight model with fixed.98recency and1e-6zero-centered ridge, refit once on all6706 fitting transitions/96 placements. All three predetermined final checkpoints are averaged. No calibration-prior term, posture correction, conservative weight attenuation or physics selector is used. Task calibration, fixed close probe, frozen measured bias, DLS, waypoint schedule, budgets and success/contact criteria match the references. The architecture is a learned/analytical hybrid, not a flow-matching action policy; no conventional flow-matching baseline was evaluated here.','', 'The fitting77/96 score came from family-excluded development models; E159 uses the prespecified full-fitting refit for prospective evaluation. Training normalization and weights were frozen before test execution. All measured calibration trajectories and initial statistics match exactly across methods, and all executed-history/query alignments, frozen bias and checkpoint/source hashes were verified.','', 'There are only four known body clusters. Descriptive paired counts and exploratory body-cluster intervals are not evidence of broad unseen-morphology generalization or a conference-ready result. All interval assumptions and changed placements are in summary.json. The held placements are not added to training, and this candidate is not retuned or repeatedly tested on them.','', '| Body | Learned | Uniform | E120 ensemble | Analytical |','|---|---:|---:|---:|---:|']
    for body in m['learned_final']['bodies']:lines.append('| '+body+' | '+' | '.join(f"{m[k]['bodies'][body]['success']}/{m[k]['bodies'][body]['n']}" for k in order)+' |')
    lines+=['', 'Videos are outcome-selected executed-state replays, not estimates of success rates. Both successes and failures are included, with matched baseline clips for representative rescues and regressions. Every rendered state is checked against the saved task trace and decoded frame count.',''];(ROOT/'report.md').write_text('\n'.join(lines));report=Path(__file__).with_name('TECHNICAL_REPORT.md');text=report.read_text(encoding='utf-8');assert '## E159:' not in text;short=f"Fresh known-body placements: learned{m['learned_final']['all']['success']}/48, uniform{m['uniform32']['all']['success']}/48, E120ensemble{m['e120_ensemble']['all']['success']}/48, analytical{m['analytical']['all']['success']}/48. "+conclusion;report.write_text(text+'\n## E159: amended prospective fresh-placement validation\n\n'+short+' Amendment registered before fresh outcomes; earlier retention gates remain failed. No unseen-body claim or retuning. Report: `outputs/ease_fresh_fixed_weights_v1/report.md`.\n',encoding='utf-8')
    selected={}
    for n in (6,7):
        for success in (True,False):
            rr=[r for r in rows if r['method']=='learned_final' and r['dof']==n and r['success']==success]
            if rr:selected[rr[0]['case'],'learned_final']=rr[0]
    for method,p in s['paired'].items():
        for n in (6,7):
            for gain in (True,False):
                rr=[r for r in p['changes'] if r['dof']==n and r['after']==gain]
                if rr:
                    case=rr[0]['case']
                    for name in ('learned_final',method):selected[case,name]=next(r for r in rows if r['case']==case and r['method']==name)
    # Include an honest baseline example even when there are no changed outcomes.
    for method in order[1:]:
        for success in (True,False):
            rr=[r for r in rows if r['method']==method and r['success']==success]
            if rr:selected[rr[0]['case'],method]=rr[0]
    save(ROOT/'video_records.json',list(selected.values()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4));labels=['Learned','Uniform32','E120 ensemble','Analytical'];values=[m[k]['all']['success'] for k in order];axes[0].bar(labels,values,color=['#2379b5','#888888','#d18b35','#6b9b6a']);axes[0].set_ylim(0,48);axes[0].set_ylabel('Completed fresh tasks /48');axes[0].tick_params(axis='x',rotation=15)
    for i,v in enumerate(values):axes[0].text(i,v+.4,str(v),ha='center')
    x=np.arange(2)
    for i,k in enumerate(order[:3]):axes[1].bar(x+(i-1)*.24,[m[k][str(n)]['success'] for n in (6,7)],.24,label=labels[i])
    axes[1].set_xticks(x,['Six joints','Seven joints']);axes[1].set_ylim(0,24);axes[1].set_ylabel('Completed fresh tasks /24');axes[1].legend(fontsize=8);fig.suptitle('E159 | frozen prospective comparison | known bodies, new placements');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    st=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','fresh_fixed_weights import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E159').replace('Task-value ranking diagnosis','Frozen fresh-placement comparison').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Outcome-selected successes, failures, rescues and regressions from the one-time frozen fresh-placement comparison. Known bodies; no unseen-morphology claim.').replace('case {case}','fresh case {case-10000}');Path(__file__).with_name('fresh_fixed_weights_videos.py').write_text(st);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'status.json',dict(status='evaluation_verified_videos_pending',fresh_episodes=192,positive_net_pattern=s['positive_net_pattern'],prior_retention_gate_passed=False,amendment_registered_before_outcomes=True,E120_unchanged=True));print(short,'Prepared',len(selected),'videos',flush=True)

def finish():
    manifest=read(ROOT/'videos/manifest.json');selected=read(ROOT/'video_records.json');assert len(manifest['videos'])==len(selected);assert all(v['all_saved_states_match_trace'] for v in manifest['videos'])
    for v in manifest['videos']:assert fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    from .fresh_fixed_weights_control import verify_frozen
    verify_frozen();s=read(ROOT/'summary.json');save(ROOT/'status.json',dict(status='complete',fresh_episodes=192,videos=len(selected),positive_net_pattern=s['positive_net_pattern'],prior_retention_gate_passed=False,amendment_registered_before_outcomes=True,E120_unchanged=True));save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E159 complete, verified',len(selected),'success/failure videos',flush=True)

if __name__=='__main__':
    import sys
    {'run':run,'finish':finish}[sys.argv[1]]()

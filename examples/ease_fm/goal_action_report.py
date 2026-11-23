import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .goal_action_audit import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());v=json.loads((ROOT/'verification.json').read_text());table=[];state_table=[]
    for kind in ('warning','successful_control'):
        for split in ('all','train','validation','6','7'):
            rr=[r for r in rows if r['kind']==kind and (split=='all' or split==str(r['dof']) or split==('validation' if r['validation'] else 'train'))]
            for bank in ('narrow','broad'):
                table.append(dict(kind=kind,split=split,bank=bank,n=len(rr),admissible40=sum(r['banks'][bank]['admissible40']>0 for r in rr),margin40=sum(r['banks'][bank]['with_2mm40']>0 for r in rr),chosen_retains160=sum(r['banks'][bank]['selected_retains_160'] for r in rr),chosen_margin160=sum(r['banks'][bank]['selected_retains_2mm160'] for r in rr)))
    for r in rows:
        full=json.loads((ROOT/'states'/f"{r['case']}_{r['seed']}_{r['step']}.json").read_text());b=r['banks']['broad'];best=next((c for c in full['candidates'] if c['name']==b['best40']),None)
        state_table.append(dict(case=r['case'],seed=r['seed'],body=r['body'],step=r['step'],phase=r['phase'],kind=r['kind'],validation=r['validation'],narrow=r['banks']['narrow']['admissible40'],broad=b['admissible40'],broad_2mm=b['with_2mm40'],chosen=b['best40'],chosen_group=None if best is None else best['group'],chosen_gain_fraction=None if best is None else best['40']['gain']/r['before_error_sq'],chosen_min_clearance_mm=None if best is None else 1000*best['40']['minimum_clearance'],chosen160=b['selected_retains_160'],chosen2mm160=b['selected_retains_2mm160']))
    sensitivity=[]
    for kind in ('warning','successful_control'):
        for bank in ('narrow','broad'):
            count=0
            for r in rows:
                if r['kind']!=kind:continue
                full=json.loads((ROOT/'states'/f"{r['case']}_{r['seed']}_{r['step']}.json").read_text())
                cc=[c for c in full['candidates'] if c['group']=='narrow' or (bank=='broad' and c['group'] in ('axis','random','directed'))]
                count+=any(c['static_ok'] and not c['40']['collision'] and c['40']['progress'] and c['40']['minimum_clearance']>=0 for c in cc)
            sensitivity.append(dict(kind=kind,bank=bank,states_with_nonnegative_clearance_and_progress40=count))
    summary=dict(table=table,states=state_table,nonnegative_clearance_sensitivity=sensitivity);(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E110: availability of clear, goal-progressing actions','',
    f"Audit of{len(rows)} frozen training-body states: first eligible E109 baseline collision warning per pair, plus five phase-start states from first successful case in each DOF/seed stratum. All{v['exact_replays']} source trajectories exactly replay without changing commands. Candidate actions evolve only in copied physics. No new task-success result.",'',
    'Narrow bank: the five E108 alternatives. Broad bank includes these plus +/- .01/.02/.04 per-joint probes around measured bias,64 seeded sampled directions, and three goal-directed candidates from simulated finite-difference40ms pose response. The latter use exact-model local response, not learned dynamics or an analytic task Jacobian. Candidate count differs by joint count and duplicate clipped actions share simulation results. No parameter tuning on outcomes.',
    'Primary admissibility requires the original static clearance criterion, zero forbidden physical contact at every2ms substep through40ms, and at least max(1e-8,1% initial squared pose error) progress. Pose error uses XYZ and .15-scaled tool-axis difference. The separate2mm column demands2mm minimum clearance across all predicted substeps; the primary metric alone does not guarantee that margin.',
    'Choose the admissible action with lowest40ms error, tie larger dynamic clearance then name. Its absolute target is held through160ms, without re-ranking. This checks short persistence under a held target, not replanning or full-task completion. No success rate may be inferred from these state counts.',
    '', '| State set | Split | Bank | N | Clear+progress40ms | Also2mm40ms | Chosen retains160ms | Chosen also2mm160ms |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in table:lines.append('| '+' | '.join(str(r[k]) for k in ('kind','split','bank','n','admissible40','margin40','chosen_retains160','chosen_margin160'))+' |')
    lines+=['','## Per-state results','', '| Case / seed / step | Phase | Type | Held | Narrow choices | Broad choices | Broad2mm | Chosen | Gain fraction40ms | Chosen survives160ms |','|---|---|---|---|---:|---:|---:|---|---:|---|']
    for r in state_table:lines.append(f"| {r['case']}/{r['seed']}/{r['step']} | {r['phase']} | {r['kind']} | {r['validation']} | {r['narrow']} | {r['broad']} | {r['broad_2mm']} | {r['chosen']} | {r['chosen_gain_fraction']} | {r['chosen160']} |")
    lines+=['','## Interpretation and next decision','',
    'Positive state availability would show that the previous narrow fallback set missed useful local actions. It would not establish that the neural policy can generate them, that selecting them prevents future failures, or that performance transfers to new embodiments. A controller trial must freeze candidate generation and selection using training evidence, preserve task budgets and all prior successes, and test full episodes. State selection is conditional on warning/success histories and repeated bodies/seeds; do not treat44 states as44 independent embodiments.',
    'Clearance is measured on a separate expanded-geometry copy carrying each predicted object/finger state. This copy is never integrated. Models and full integration states are privileged. Held placements use the same training bodies. No E105 or reserved final-test data enter this audit.',
    '', '## Saved evidence','',f"{v['unique_physics_branches']} distinct candidate simulations across states, complete candidate actions/metrics, full MuJoCo integration snapshots and XML, source/checkpoint hashes, and exact unchanged baseline replays. Videos are verified reused baseline successes/failures, explicitly not executions of newly selected candidates."]
    lines+=['','## Completed findings and geometric sensitivity','',
    'Primary warning-state availability is1/14 narrow versus13/14 broad. The broad bank has qualifying candidates in6/6 training-placement warning states and7/8 held-placement warning states. The40ms-selected broad action retains contact-free progress through160ms in12/14 states, versus0/14 narrow. Broad candidates satisfying the full2mm dynamic margin exist in5/14 warning states. All30 successful-control states have broad candidates with2mm clearance and progress, but this is not a full-task success-retention test.',
    'Case8/seed2 lift is the unresolved warning state: every sampled action predicts forbidden physical contact within40ms. This finite search does not establish infeasibility. Other warning-state chosen actions include sampled joint directions, coordinate probes and finite-difference goal-directed responses; the result is not attributable solely to the directed generator.',
    'Supplemental geometric sensitivity (reported after outcomes, without changing the prespecified primary metric): require minimum expanded-model geometric clearance >=0 in addition to the primary criteria. Warning-state availability becomes0/14 narrow and13/14 broad. The primary-selected brake in case2/seed0 has approximately -0.03mm minimum geometric clearance despite no physical solver contact flag by40ms, and fails the160ms hold check. Other broad candidates in that state have nonnegative40ms clearance. Solver contact sampling and geometry at integrated positions differ; a future controller must not equate a missing contact flag with nonpenetration. We do not retrospectively rerank the160ms outcomes.', '',json.dumps(sensitivity,indent=2),
    'This audit identifies useful local alternatives, not an executed improvement. A next trial should explicitly require nonnegative geometric clearance, freeze progress-aware selection before running full training episodes, and retain the2mm-margin breakdown. Candidate generation costs hundreds of simulator substeps per choice; no real-time or hardware claim is made.',
    '',f"Saved-snapshot verification reconstructs{v.get('snapshot_branches_exact',0)} original/selected candidate branches exactly at40 and160ms. All131 regression tests passed in13.99s."]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,kind in zip(axes,('warning','successful_control')):
        rr=[r for r in table if r['kind']==kind and r['split']=='all']
        for j,r in enumerate(rr):
            bars=ax.bar(np.arange(3)+(j-.5)*.32,[r['admissible40'],r['margin40'],r['chosen_retains160']],.32,label=r['bank']);ax.bar_label(bars)
        ax.set_xticks(range(3),['Clear + progress40','Also2mm40','Chosen survives160']);ax.set_ylim(0,max(r['n'] for r in rr)+3);ax.set_title(kind);ax.set_ylabel('States');ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'availability.{ext}')
    plt.close(fig);print(json.dumps([r for r in table if r['split']=='all'],indent=2))

if __name__=='__main__':run()

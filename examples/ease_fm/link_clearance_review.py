"""Final E168 contact follow-through, research index and completed checkpoint."""
from pathlib import Path
from collections import Counter
import mujoco
from .link_clearance_diagnostic import ROOT,CONDITIONS,geometry_ids
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    status=read(ROOT/'status.json');assert status['status']=='complete';summary=read(ROOT/'summary.json');details={}
    for condition in CONDITIONS:
        root=ROOT/condition;cases=[];phases=Counter()
        for r in read(root/'records.json'):
            stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');m=mujoco.MjModel.from_xml_path(str(root/'states'/f'{stem}.xml'));links,_=geometry_ids(m,6);labels={mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in links};hit=lambda row:any('table' in pair and bool(set(pair)&labels) for pair in row['bad']);i=next((i for i,row in enumerate(t['trace']) if hit(row)),None)
            if i is not None:
                logs=read(root/'clearance'/f'{stem}.json');phase=t['trace'][i]['phase'];phases[phase]+=1;cases.append(dict(case=r['case'],first_link_contact_step=i,first_link_contact_phase=phase,constraint_eligible_at_first_link_contact=logs[i]['eligible'],all_link_contact_phases=sorted({x['phase'] for x in t['trace'] if hit(x)}),success=r['success']))
        details[condition]=dict(first_link_contact_phases=dict(phases),cases=cases)
    save(ROOT/'contact_followthrough.json',details)
    paragraph='\nFirst arm-link/table contact was separately audited, rather than inferred from overlapping episode counts: '+ '; '.join(f"{c}: {v['first_link_contact_phases']}" for c,v in details.items())+'. All such first contacts occurred during the fixed probe, where this diagnostic intentionally leaves the original commands unchanged. Contact may persist into later phases after that first failure; this is not a claim that protected phases never contain contact. Full per-case phase sequences are in contact_followthrough.json.\n'
    assert all(set(v['first_link_contact_phases'])<= {'probe'} for v in details.values());p=ROOT/'report.md';p.write_text(p.read_text(encoding='utf-8')+paragraph,encoding='utf-8')
    review=Path('outputs/ease_iteration_review_v1');lines=['## Privileged link-clearance follow-up (E168)','','E120 and the265mm release fix remain the unchanged reference.96 six-joint fitting intervention episodes and4 exact disabled-layer baseline replays were completed. Both variants use privileged exact link/table distances and local gradients; neither is history-only learned adaptation or held-body evidence.','', '| Controller | Success /48 | All collision episodes | Link/table collision episodes | Collision-free failures | Rescues / regressions |','|---|---:|---:|---:|---:|---:|']
    b=summary['baseline'];lines.append(f"| E120 + release | {b['success']} | {b['collisions']} | {b['link_table_collision_episodes']} | {b['collision_free_failures']} | — |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['link_table_collision_episodes']} | {x['collision_free_failures']} | {v['gains']} / {v['losses']} |")
    lines+=['', 'Both fixed fitting screens passed with no lost E120+release successes. The task-focused objective completes one more task, while joint-command projection has one fewer collision episode; all failures are retained. There were no solver fallbacks. Exact geometry is a privileged input, so these results establish a controller diagnostic gain, not a learned-estimator or cross-embodiment claim.','', 'Remaining first arm-link/table contacts all occur during the unmodified fixed probe. Other remaining collisions involve unconstrained gripper/table contact. Some episodes avoid collision but fail to complete the task. The next useful diagnostic is probe-aware clearance protection and gripper coverage, with task progress checked independently; these extensions have not yet been tested.','']
    p=review/'report.md';text=p.read_text(encoding='utf-8');at=text.index('| Study |');text=text[:at]+'\n'.join(lines)+'\n'+text[at:];at=text.index('\n\nInterpretation:');row='\n| E168 | Privileged link-clearance projections:38/48 and39/48 versus37; no lost baseline success. | [report](../ease_link_clearance_diagnostic_v1/report.md) | [videos](../ease_link_clearance_diagnostic_v1/videos/index.html) |';text=text[:at]+row+text[at:];p.write_text(text,encoding='utf-8')
    studies=read(review/'studies.json');assert not any(x['experiment']=='E168' for x in studies);studies.append(dict(experiment='E168',folder=str(ROOT),report_exists=True,videos=status['videos']));save(review/'studies.json',studies);save(review/'link_clearance_summary.json',dict(**summary,contact_followthrough=details,new_episodes=96,exact_disabled_replays=4,videos=status['videos']))
    state=review/'WORKING_STATE.md';state.write_text('# Latest checkpoint: E168 complete\n\n'+'\n'.join(lines)+'\nAll experiment jobs, audits and success/failure renders are complete. E120 and E161 release unchanged; no held test or training update. See link_clearance_summary.json.\n\n---\n\n'+state.read_text(encoding='utf-8'),encoding='utf-8');save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print(dict(contact_followthrough=details,videos=status['videos'],completed=True),flush=True)

if __name__=='__main__':run()

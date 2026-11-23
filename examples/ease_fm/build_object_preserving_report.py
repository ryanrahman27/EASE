"""Adapt the E173 independent audit to strict-object planning and grasp guards."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('posture_route_report.py').read_text().replace('from .posture_route import','from .object_preserving import').replace('from .posture_route_planner import forbidden','from .object_preserving_plan import forbidden').replace('E173','E174')
    s=s.replace('attempted=False;route=None;index=0;checked=False;trigger_step=None','attempted=False;route=None;index=0;checked=False;trigger_step=None;pick=None;last_phase=None;ready=False;good=0;aborted=False;close_anchor=None;abort_anchor=None')
    s=s.replace("np.testing.assert_array_equal(plan['goal'],l['goal'])", "np.testing.assert_allclose(plan['goal'],l['goal'],atol=1e-12,rtol=0)")
    s=s.replace("route=np.array(plan['dense_path']);", "route=np.array(plan['dense_path']);pick=np.array(plan['pick_goal']);")
    s=s.replace("target=np.clip(.35*(waypoint-l['q_before'])+np.array(l['bias']),-.04,.04)","target=np.clip(waypoint-l['q_before']+np.array(l['bias']),-.04,.04)")
    old="                assert l['route_index']==index;np.testing.assert_array_equal(target,l['optimization_target'])"
    new="""                handoff=False;aperture=l['requested_aperture']
                if condition=='guarded_handoff' and route is not None:
                    if l['phase']=='close' and last_phase!='close':
                        close_anchor=np.array(l['q_before']);gap=d.site_xpos[site]-pick;obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object');drift=np.linalg.norm(d.xpos[obj,:2]-pick[:2]);axis=np.linalg.norm(d.site_xmat[site].reshape(3,3)[:,2]-[0,0,1]);ready=bool(np.linalg.norm(gap[:2])<=.008 and abs(gap[2])<=.012 and axis<.08 and drift<=.008)
                    if l['phase']=='lift' and last_phase=='close' and (not ready or good<5):aborted=True;abort_anchor=np.array(l['q_before'])
                    if l['phase']=='close':target=np.clip(close_anchor-l['q_before']+np.array(l['bias']),-.04,.04);aperture=aperture if ready else 0.;handoff=True
                    if aborted:target=np.clip(abort_anchor-l['q_before']+np.array(l['bias']),-.04,.04);aperture=0.;handoff=True
                assert l['handoff_control']==handoff and l['close_ready']==ready and l['aborted']==aborted and l['aperture']==aperture
                if close_anchor is not None:np.testing.assert_array_equal(close_anchor,l['close_anchor'])
                if abort_anchor is not None:np.testing.assert_array_equal(abort_anchor,l['abort_anchor'])
                assert l['route_index']==index;np.testing.assert_array_equal(target,l['optimization_target'])"""
    assert old in s;s=s.replace(old,new)
    s=s.replace("bool(active or np.min(effective)<=prior.ACTIVATION)","bool(active or handoff or np.min(effective)<=prior.ACTIVATION)")
    old="np.testing.assert_allclose(distances(m,d,ids,table),l['distance_after'],atol=1e-12,rtol=0);post_checks+=1"
    new="""np.testing.assert_allclose(distances(m,d,ids,table),l['distance_after'],atol=1e-12,rtol=0);post_checks+=1
                cube=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,'cube');pad_ids={mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,n) for n in ('gripper_left_pad','gripper_right_pad')};touch=set()
                for contact in d.contact[:d.ncon]:
                    pair={int(contact.geom1),int(contact.geom2)}
                    if cube in pair:touch|=pair&pad_ids
                assert len(touch)==l['end_pads']
                if l['phase']=='close':good=good+1 if len(touch)==2 else 0
                assert l['close_good']==good;last_phase=l['phase']"""
    assert old in s;s=s.replace(old,new)
    s=s.replace('            if not attempted:', '            if route is None:')
    old="desc=next(p for p in t['phases'] if p['phase']=='descend');assert desc['steps']==len(es) and len(es)<=(300 if condition=='route_extended' and route is not None else 100)"
    new="""if condition=='guarded_handoff':
                close_step=next(i for i,l in enumerate(logs) if l['phase']=='close');other=ROOT/'object_route';ot=read(other/'trials'/f'{stem}.json');assert t['trace'][:close_step]==ot['trace'][:close_step];np.testing.assert_array_equal(states[:close_step],np.load(other/'states'/f'{stem}.npz')['states'][:close_step])
            desc=next(p for p in t['phases'] if p['phase']=='descend');assert desc['steps']==len(es) and len(es)<=(300 if route is not None else 100)"""
    assert old in s;s=s.replace(old,new)
    s=s.replace("route_cases.append(dict(case=case,", "route_cases.append(dict(case=case,close_ready=ready,verified_final_close_steps=good,aborted=aborted,")
    s=s.replace('budgets_verified=True)', 'budgets_verified=True,grasp_guard_and_actual_end_contacts_verified=True,matched_preclose_prefix=True)')
    start=s.index("    lines=['# E174:");end=s.index("    (ROOT/'report.md').write_text",start)
    s=s[:start]+'''    lines=['# E174: object-preserving routes and guarded grasp handoff','','Two matched privileged fitting interventions,48 cases each, with4 exact E170 replays. Original E120 weights/frozen bias, probe protection and265mm release are unchanged.','','| Controller | Success /48 | Collision episodes | Collision-free failures | Rescues / losses vs E170 | Extra descent steps |','|---|---:|---:|---:|---|---:|']
    for label,x in summary['references'].items():lines.append(f"| {label} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | — | 0 |")
    for c,v in summary['conditions'].items():
        x=v['outcomes'];p=v['paired']['E170_protected'];lines.append(f"| {c} | {x['success']} | {x['collisions']} | {x['collision_free_failures']} | {p['rescues']} / {p['losses']} | {v['extra_descent_steps']} |")
    lines+=['','The planner rejects all robot/object contacts in static checks. It samples grasp configurations at current simulator object XY and240mm height, reverses a vertical approach to320mm, then connects to that pregrasp posture. This preserves the grasp/approach IK branch. The initial pregrasp-first search failed and is archived; this correction occurred before physical intervention outcomes. Case26 has an existing robot/object contact at the gate, so no strict route is accepted and the original controller is retained.','', 'Both variants use the same route, gain1 joint tracking, protected32-step probe and up to300 descent steps after accepted planning. Pick/close/lift XY follows the object location observed at planning, while final placement and release height remain unchanged. object_route retains the ordinary close/lift transition. guarded_handoff checks close-entry XY/Z/axis alignment and object drift, holds arm posture during closure, and requires both pads contacting at the END of the final5 close steps before lift. Otherwise it holds with fingers open through remaining scheduled phases and is scored normally as a failure.','', 'Geometry, simulator object pose and contact signals are privileged. These interventions do not train E120 or demonstrate learned-only cross-embodiment adaptation. Static contact-free sampling does not guarantee actual object preservation. Actual object displacement, grasp/lift and complete-task outcomes are retained. No held test is executed.','', 'Independent audit reconstructs gates, current-state plans, strict path contacts, waypoints, clearance allowances, all actual commands and apertures, close readiness, end-step pad contacts, sustained-contact counters and aborts. All saved post-state clearances and representative gradients are checked; accepted linear constraints and real phase budgets are verified. Nonexecuted-route episodes reproduce full E170 traces/states exactly. Both variants have identical trajectories before close. Four exact reference replays passed after correcting and archiving a calibration-time phase-bookkeeping bug before interventions.','', 'See object_handoff_diagnosis.json for object displacement and each remaining failure. Saved videos include every original failure and a retained success; replayed tool/object traces and decoded frame counts are verified. Contact scoring occurs at2ms;80ms video sampling may miss brief contacts. literature_review.md maps established planning/control methods to these failure modes, distinguishes source findings from proposed applications, and states which ideas were actually implemented.']
''' +s[end:]
    s=s.replace('posture_route_videos.py','object_preserving_videos.py').replace('E174 privileged route assistance | fitting only | budgets differ','E174 privileged object-preserving route / grasp handoff').replace('posture/route comparison','object-preserving route / handoff comparison')
    s=s.replace('PRIVILEGED route execution, original vs extended descent budget. Matched E170 and E120+release; fitting-only successes and failures.','PRIVILEGED object-avoiding routes and guarded grasp handoff; matched E170 references; fitting cases only.')
    start=s.index("    s=read(ROOT/'summary.json');paragraph=");end=s.index("    review=Path('outputs/ease_iteration_review_v1')",start)
    s=s[:start]+'''    s=read(ROOT/'summary.json');paragraph='; '.join(f"{c}: {v['outcomes']['success']}/48 success, {v['outcomes']['collisions']} collisions, {v['paired']['E170_protected']} versus E170; {v['extra_descent_steps']} extra descent steps" for c,v in s['conditions'].items());section='\\n## E174: object-preserving routes and grasp handoff\\n\\n'+paragraph+'.96 fitting episodes and4 exact reference replays. E120, probe protection and release preserved. Strict grasp-first planning finds routes for3 failures; case26 retains the baseline because it starts in object contact. Object pose/contact inputs privileged; no held test. Report: `outputs/ease_object_preserving_v1/report.md`; videos: `videos/index.html`; targeted review: `literature_review.md`.\\n'
''' +s[end:]
    Path(__file__).with_name('object_preserving_report.py').write_text(s)

if __name__=='__main__':run()

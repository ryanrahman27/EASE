"""E174 staged object-avoiding routes and a contact-verified grasp handoff."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('posture_route.py').read_text().replace('E173','E174').replace('from .posture_route_planner import plan','from .object_preserving_plan import plan').replace("ROOT=Path('outputs/ease_posture_route_v1')","ROOT=Path('outputs/ease_object_preserving_v1')").replace("CONDITIONS=('route_fixed','route_extended')","CONDITIONS=('object_route','guarded_handoff')")
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    assert len(read(ROOT/'planning_summary.json'))==4;(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E174',scope='Two fixed privileged fitting variants x48 cases,96 episodes and4 exact E170 replays. Preserve all44 successes, E120, probe protection and265mm release. No held test or training.',route='Strict robot/object contact rejection in planning; current simulator object XY determines grasp at240mm and pregrasp320mm. Grasp-first inverse approach on the same IK branch, then geometric connection. Up to300 descent steps after accepted planning; phase reach scoring unchanged. Track route with joint gain1.0, lookahead3, original+/- .04 clipping and E170 safety QP. All other phases use E120; pick/lift XY uses the planned object XY. Object pose is privileged.',conditions='object_route: staged route then usual probe/close/lift. guarded_handoff: same route/probe; on close entry require tool XY within8mm and Z within12mm of pick goal, axis error<.08 and object XY drift<=8mm. Hold current arm posture while closing if ready; otherwise keep fingers open. Require both pads contacting object at the END of each of the final5 close steps before lifting. Otherwise abort by holding current arm pose with fingers open for remaining scheduled phases. Aborts are failures under original scoring.',isolation='Neither variant changes E120 weights or calibrated bias.32-step initial calibration/probe schedule, all non-descent budgets, original final placement target and strict success/contact scoring preserved. No new initial-state reset. Only activated route episodes can alter targets/apertures or abort; unavailable plans fall back exactly. Simulator geometry, object position and end-step pad contacts explicitly privileged.',screen='More than44 successes, zero forbidden-contact episodes, no lost reference success; separately report object displacement, grasp/lift, aborts, extra steps and exact unchanged episodes. Static path validity does not ensure dynamic object preservation. No automatic promotion.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',ROOT/'planning_protocol.json',ROOT/'planning_summary.json',Path(__file__).with_name('object_preserving_plan.py'))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    for name in ('object_preserving.py','object_preserving_plan.py','object_preserving_planner.py','posture_route.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_object_preserving.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
''' +s[end:]
    s=s.replace("if planned['available']:self.route=np.array(planned['dense_path']);self.route_index=0","if planned['available']:self.route=np.array(planned['dense_path']);self.route_index=0;self.pick_goal=np.array(planned['pick_goal'])")
    s=s.replace('optimization_target=np.clip(.35*(waypoint-before[\'q\'])+c.bias,-.04,.04)','optimization_target=np.clip(waypoint-before[\'q\']+c.bias,-.04,.04)')
    old='                eligible=bool(route_active or np.min(effective)<=prior.ACTIVATION)'
    new="""                requested_aperture=float(aperture);handoff_control=False
                if condition=='guarded_handoff' and self.route is not None:
                    if self.phase=='close' and self.last_phase!='close':
                        self.close_anchor=before['q'].copy();gap=before['position']-self.pick_goal;drift=np.linalg.norm(self.data.xpos[self.obj,:2]-self.pick_goal[:2]);axis=np.linalg.norm(before['rotation'][:,2]-[0,0,1])
                        self.close_ready=bool(np.linalg.norm(gap[:2])<=.008 and abs(gap[2])<=.012 and axis<.08 and drift<=.008)
                    if self.phase=='lift' and self.last_phase=='close' and (not self.close_ready or self.close_good<5):self.aborted=True;self.abort_anchor=before['q'].copy()
                    if self.phase=='close':
                        optimization_target=np.clip(self.close_anchor-before['q']+c.bias,-.04,.04);aperture=aperture if self.close_ready else 0.;handoff_control=True
                    if self.aborted:optimization_target=np.clip(self.abort_anchor-before['q']+c.bias,-.04,.04);aperture=0.;handoff_control=True
                eligible=bool(route_active or handoff_control or np.min(effective)<=prior.ACTIVATION)"""
    assert old in s;s=s.replace(old,new)
    old="                logs.append(dict(route_active=route_active,"
    new="""                end_pads=len(super().contacts()[0])
                if self.phase=='close':self.close_good=self.close_good+1 if end_pads==2 else 0
                logs.append(dict(requested_aperture=requested_aperture,handoff_control=handoff_control,close_ready=self.close_ready,close_good=self.close_good,end_pads=end_pads,aborted=self.aborted,close_anchor=None if self.close_anchor is None else self.close_anchor.tolist(),abort_anchor=None if self.abort_anchor is None else self.abort_anchor.tolist(),pick_goal=None if self.pick_goal is None else self.pick_goal.tolist(),route_active=route_active,"""
    assert old in s;s=s.replace(old,new)
    s=s.replace('                return result\n        def targets','                self.last_phase=self.phase\n                return result\n        def targets')
    s=s.replace("env.extended_route=condition=='route_extended';","env.extended_route=condition!='replay';env.pick_goal=None;env.close_anchor=None;env.close_ready=False;env.close_good=0;env.last_phase=None;env.aborted=False;env.abort_anchor=None;")
    s=s.replace("            error=np.r_[goal-obs['position']", "            if env.pick_goal is not None and phase in ('descend','close','lift'):\n                goal=goal.copy();goal[:2]=env.pick_goal[:2];env.goal=goal.copy()\n            error=np.r_[goal-obs['position']")
    Path(__file__).with_name('object_preserving.py').write_text(s)

if __name__=='__main__':run()

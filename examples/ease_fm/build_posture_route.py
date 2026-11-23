"""Build E173 route execution around the immutable E170 protection."""
from pathlib import Path

def run():
    s=Path(__file__).with_name('descent_stall.py').read_text().replace('E172','E173').replace("ROOT=Path('outputs/ease_descent_stall_v1')","ROOT=Path('outputs/ease_posture_route_v1')").replace("CONDITIONS=('stall_recovery',)","CONDITIONS=('route_fixed','route_extended')")
    s=s.replace('from .arm_pick_place import PickPlace','from .arm_pick_place import PickPlace,ROT_SCALE,rotation_error,targets\nfrom .posture_route_planner import plan')
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    assert len(read(ROOT/'planning_summary.json'))==4;(ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E173',scope='Privileged alternative-posture route execution on48 six-joint fitting cases per variant,96 episodes plus4 exact E170 replays. No held execution or training. E17044/48 zero collisions remains protected reference and E120+release primary baseline.',planning='Use frozen posture_route_planner at the E172 measured local late-stall gate. Scratch IK/path queries only; exact current simulator geometry/object/jaw state privileged. All4 prior failure states have alternate static postures and sampled collision-free direct joint paths. Recompute online at first activation, never use an outcome or case lookup. Verify live state unchanged.',control='Track dense route by nearest remaining node plus3 lookahead nodes, joint command clip(.35*(waypoint-q)+frozen_bias,+/-.04), then unchanged E170 clearance/error-allowance QP. E120 still estimates its metric and handles all other phases; route execution is explicitly privileged assistance, not learned-only control. No modified probe or gripper schedule.',budgets='route_fixed preserves100 descent steps. route_extended allows up to300 descent steps only after a route is accepted, up to200 extra interactions; original reach termination still applies. Nonactivated episodes must reproduce complete reference traces/states exactly. All other budgets and265mm release fix unchanged.',gate='At least60 descent observations,25mm<position error<75mm, less than1mm progress over previous19 intervals, any abs joint>=2.4. Attempt planning once. No IDs in control; parameters frozen before execution outcomes.',screen='More than44 successes, zero collisions and no lost reference success. Separate static feasibility, task completion and interaction budget. Every failure and fallback retained.',inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',ROOT/'planning_protocol.json',ROOT/'planning_summary.json',Path(__file__).with_name('posture_route_planner.py'))}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    for name in ('posture_route.py','posture_route_planner.py','descent_stall.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_posture_route.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
''' +s[end:]
    old="if condition!='replay' and len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4:self.recovery_active=True"
    new="""if condition!='replay' and not self.planning_attempted and len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4:
                        self.planning_attempted=True;ctrl_before=self.data.ctrl.copy();time_before=self.data.time
                        planned=plan(self.model,qpos,qvel,self.goal);save(root/'planning'/f'{case}.json',planned)
                        np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);np.testing.assert_array_equal(self.data.ctrl,ctrl_before);assert self.data.time==time_before
                        if planned['available']:self.route=np.array(planned['dense_path']);self.route_index=0"""
    assert old in s;s=s.replace(old,new)
    start=s.index("                direct=bool(self.phase=='descend'");end=s.index('                result=super().step',start)
    s=s[:start]+'''                route_active=bool(self.phase=='descend' and self.route is not None);waypoint=None;optimization_target=nominal.copy()
                if route_active:
                    nearest=self.route_index+int(np.argmin(np.linalg.norm(self.route[self.route_index:]-before['q'],axis=1)));self.route_index=max(self.route_index,nearest);waypoint=self.route[min(self.route_index+3,len(self.route)-1)]
                    optimization_target=np.clip(.35*(waypoint-before['q'])+c.bias,-.04,.04)
                eligible=bool(route_active or np.min(effective)<=prior.ACTIVATION)
                if eligible:
                    g=distance_gradient(self.model,self.data,self.scratch,self.geom_ids,self.table_id,self.n);np.testing.assert_array_equal(self.data.qpos,qpos);np.testing.assert_array_equal(self.data.qvel,qvel);j,_=c.predict();j=np.asarray(j,dtype=np.float64);metric=j.T@j+.015**2*np.eye(self.n)
                    action,solve,rhs=prior.project(optimization_target,c.bias,before['q'],effective,g,metric)
''' +s[end:]
    old="logs.append(dict(posture=posture.tolist(),descent_observations=len(self.descent_errors),direct_task=direct,"
    new="logs.append(dict(route_active=route_active,route_index=self.route_index,waypoint=None if waypoint is None else waypoint.tolist(),planning_attempted=self.planning_attempted,descent_observations=len(self.descent_errors),"
    assert old in s;s=s.replace(old,new)
    s=s.replace('env.descent_errors=[];env.recovery_active=False;','env.descent_errors=[];env.route=None;env.route_index=0;env.planning_attempted=False;env.extended_route=condition==\'route_extended\';')
    s=s.replace('return original_rollout(env,home,c,**kwargs)','return route_rollout(env,home,c,**kwargs)')
    # Preserve controller wrappers but replace only their bottom rollout implementation.
    # original_rollout includes estimator/history observer; patch its captured bottom via core wrapper below.
    s=s.replace("with patch.object(old,'ROOT',root),patch.object(core,'PickPlace',ProtectedArm),patch.object(core,'targets',targets),patch.object(core,'probe_rollout',rollout):", "with patch.object(old,'ROOT',root),patch.object(core,'PickPlace',ProtectedArm),patch.object(core,'targets',targets),patch.object(core,'probe_rollout',rollout):")
    text=Path(__file__).with_name('local_probe.py').read_text();a=text.index('def probe_rollout(');b=text.index('\n\ndef evaluate(',a);func=text[a:b].replace('def probe_rollout(','def route_rollout(')
    func=func.replace('        for t in range(budget):',"        maximum=300 if phase=='descend' and getattr(env,'extended_route',False) else budget\n        for t in range(maximum):\n            if t>=budget and getattr(env,'route',None) is None:\n                t-=1;break")
    s=s.replace("if __name__=='__main__':",func+"\n\nif __name__=='__main__':")
    Path(__file__).with_name('posture_route.py').write_text(s)

if __name__=='__main__':run()

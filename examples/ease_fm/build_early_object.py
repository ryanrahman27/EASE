"""Derive E175 while retaining archived E174 sources/results unchanged."""
from pathlib import Path

def run():
    base=Path(__file__).parent;s=(base/'object_preserving.py').read_text()
    s=s.replace('E174','E175').replace("REFERENCE=Path('outputs/ease_descent_execution_v1/history_allowance')","REFERENCE=Path('outputs/ease_object_preserving_v1/guarded_handoff')").replace("ROOT=Path('outputs/ease_object_preserving_v1')","ROOT=Path('outputs/ease_early_object_recovery_v1')").replace("CONDITIONS=('object_route','guarded_handoff')","CONDITIONS=('early_guard',)").replace('REPLAYS=(0,26,62,67)','REPLAYS=(0,26,43,62,67)')
    s=s.replace('from .object_preserving_plan import plan','from .object_preserving_plan import plan\nfrom .early_object_plan import early_gate,observation')
    start=s.index('def prepare():');end=s.index('\ndef run(condition):',start)
    s=s[:start]+'''def prepare():
    (ROOT/'source').mkdir();shutil.copy2(BASE/'cases.json',ROOT/'cases.json')
    save(ROOT/'protocol.json',dict(experiment='E175',scope='48 six-joint fitting episodes and5 exact E174 guarded-handoff replays. Preserve47 successes. No training or held execution.',change='Add causal early object/tilt guard described in planning_protocol.json. Retain original late gate, strict planner, E120, frozen bias, probe protection, guarded handoff, release fix and all budgets/scoring. Replay disables only the new early guard.',screen='48 successes, zero collision episodes, no lost E174 successes. Report all outcomes even if rejected. No automatic promotion.',privileged=True,inputs={str(p):fingerprint(p) for p in (REFERENCE/'records.json',ROOT/'planning_protocol.json',ROOT/'planning_summary.json')}))
    for name in (*CONDITIONS,'replay'):
        root=ROOT/name;root.mkdir()
        for folder in ('trials','states','actions','histories','clearance','planning'):(root/folder).mkdir()
    for name in ('early_object.py','early_object_plan.py','early_object_diagnosis.py','object_preserving.py','object_preserving_plan.py','object_preserving_planner.py','descent_execution.py','probe_gripper_clearance.py','link_clearance_diagnostic.py','build_early_object.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',*(ROOT/'source').glob('*.py')]})
''' + s[end:]
    s=s.replace("['exact_replays']==4","['exact_replays']==5").replace("(4 if condition=='replay' else 48)","(5 if condition=='replay' else 48)").replace("exact_replays=4 if condition=='replay' else 0","exact_replays=5 if condition=='replay' else 0")
    old="if condition!='replay' and not self.planning_attempted and len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4:"
    new="self.object_history.append(observation(self.model,self.data,self.goal))\n                    early=condition!='replay' and early_gate(self.object_history)\n                    late=len(self.descent_errors)>=60 and .025<pe<.075 and self.descent_errors[-20]-pe<.001 and np.max(np.abs(before['q']))>=2.4\n                    if not self.planning_attempted and (early or late):\n                        self.trigger_reason='early_object_tilt' if early else 'late_stall'"
    assert old in s;s=s.replace(old,new)
    s=s.replace("if condition=='guarded_handoff' and self.route is not None:","if self.route is not None:")
    s=s.replace('logs.append(dict(requested_aperture=',"logs.append(dict(trigger_reason=self.trigger_reason,object_guard_observation=self.object_history[-1] if self.phase=='descend' else None,requested_aperture=")
    s=s.replace('env.descent_errors=[];env.route=None','env.object_history=[];env.trigger_reason=None;env.descent_errors=[];env.route=None').replace("env.extended_route=condition!='replay'","env.extended_route=True")
    (base/'early_object.py').write_text(s)

if __name__=='__main__':run()

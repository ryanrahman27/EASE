"""Reuse the E174 independent physical audit, adapting the new causal gate."""
from pathlib import Path

def run():
    base=Path(__file__).parent;s=(base/'object_preserving_report.py').read_text()
    s=s.replace('from .object_preserving import','from .early_object import').replace('from .object_preserving_plan import forbidden','from .object_preserving_plan import forbidden\nfrom .early_object_plan import early_gate,observation')
    s=s.replace("'E170_protected':(REFERENCE", "'E174_guarded':(REFERENCE").replace("['E170_protected']","['E174_guarded']")
    s=s.replace('es=[];errors=[];',"object_history=[];trigger_reason=None;es=[];errors=[];")
    old="if not attempted and len(es)>=60 and .025<pe<.075 and es[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4:"
    new="object_history.append(observation(m,d,l['goal']));assert l['object_guard_observation']==object_history[-1]\n                    early=early_gate(object_history);late=len(es)>=60 and .025<pe<.075 and es[-20]-pe<.001 and np.max(np.abs(l['q_before']))>=2.4\n                    if not attempted and (early or late):\n                        trigger_reason='early_object_tilt' if early else 'late_stall'"
    assert old in s;s=s.replace(old,new)
    s=s.replace("offline=read(ROOT/'plans'/f'{case}.json')", "offline=read((ROOT/'plans' if early else REFERENCE/'planning')/f'{case}.json')")
    s=s.replace("active=bool(l['phase']=='descend' and route is not None);", "assert l['trigger_reason']==trigger_reason\n                active=bool(l['phase']=='descend' and route is not None);")
    s=s.replace("if condition=='guarded_handoff' and route is not None:","if route is not None:")
    s=s.replace('if route is None:\n                assert t', "if trigger_reason!='early_object_tilt':\n                assert t")
    start=s.index("            if condition=='guarded_handoff':");end=s.index('            desc=next',start);s=s[:start]+s[end:]
    s=s.replace('dict(case=case,close_ready=', 'dict(case=case,trigger_reason=trigger_reason,close_ready=')
    s=s.replace("counts['success']>44", "counts['success']>47").replace('matched_preclose_prefix=True','matched_preintervention_prefix=True')
    s=s.replace('for case in (0,26,62,65,67):','for case in (0,26,43,62,65,67):').replace("'privileged_E170'","'privileged_E174'")
    s=s.replace('new_episodes=96,exact_replays=4','new_episodes=48,exact_replays=5')
    start=s.index("    lines=['# E174:");end=s.index("    import matplotlib",start)
    s=s[:start]+'''    lines=['# E175: earlier object-safe recovery','','48 fitting episodes and5 exact E174 guarded-handoff replays. E120, bias, probe protection, release fix, guarded handoff, late recovery and scoring remain unchanged. Simulator object pose and geometry remain privileged. No held cases or training.','','| Controller | Success /48 | Collisions |','|---|---:|---:|']
    for name,v in summary['references'].items():lines.append(f"| {name} | {v['success']} | {v['collisions']} |")
    for name,v in summary['conditions'].items():
        lines.append(f"| {name} | {v['outcomes']['success']} | {v['outcomes']['collisions']} |")
        lines+=['',f"Paired against E174: {v['paired']['E174_guarded']}. Exact unchanged episodes: {checks[name]['exact_nonactivated_episodes']}. Total descent steps above100 per episode: {v['extra_descent_steps']} ({v['extra_simulated_seconds']:.2f} simulated seconds)."]
    lines+=['','## Intervention and evidence','','The additional causal guard observes the last5 measured descent states. It triggers when the pad/object separating-axis gap is below12mm, tool-axis error exceeds.15, positional error exceeds15mm, the gap has closed by more than15mm and axis error has increased by more than.05 over4 control steps (160ms). SAT gaps provide a conservative lower bound on box separation, avoiding native box-distance numerical zeros. Thresholds were designed from these fitting reference histories; this is not a generalization test. No case identifiers enter the controller. The reference-state screen activates on cases26 and43; both have strict grasp-first paths before physical intervention.','','The planner is unchanged from E174: choose a feasible grasp, reverse a vertical approach on the same IK branch, then connect the current posture to that pregrasp while excluding all robot/object contact. The original300-step routed descent budget and guarded grasp transition remain. This applies the task-and-motion planning principle of jointly checking grasp and approach feasibility, and reactive task-state guards, rather than integrating or reproducing PDDLStream, cuRobo or STORM. See literature_application.md.','','The audit independently reconstructs causal gates, paths, executed commands, safety margins, actual pad contacts and guard decisions. Every saved post-state geometry is checked, and non-early episodes must exactly reproduce complete E174 traces/states, including its three rescued cases. Calibration and frozen bias match the reference. Static path checks alone do not establish dynamic safety; task outcomes and full physical contact scoring are retained. Videos retain the prior failure and all newly affected cases.']
    (ROOT/'report.md').write_text('\\n'.join(lines)+'\\n',encoding='utf-8')
''' + s[end:]
    s=s.replace("labels=['E120','E170',*CONDITIONS]", "labels=['E120','E174',*CONDITIONS]")
    s=s.replace('E174 privileged object-preserving route / grasp handoff','E175 privileged early object-safe recovery')
    s=s.replace("'object_preserving import ROOT'", "'early_object import ROOT'").replace(".replace('E168','E174')", ".replace('E168','E175')").replace('object_preserving_videos.py','early_object_videos.py')
    start=s.index('\ndef finish():');end=s.index("\nif __name__=='__main__':",start)
    s=s[:start]+'''\ndef finish():
    videos=read(ROOT/'videos/manifest.json')['videos'];assert len(videos)==len(read(ROOT/'video_records.json'))
    for v in videos:assert v['all_saved_states_match_trace'] and fingerprint(ROOT/'videos'/v['file'])==v['sha256']
    s=read(ROOT/'summary.json')['conditions']['early_guard'];v=read(ROOT/'verification.json')['conditions']['early_guard'];section=f"\\n## E175: earlier object-safe recovery\\n\\n{s['outcomes']['success']}/48 successes, {s['outcomes']['collisions']} collisions. Paired E174 comparison: {s['paired']['E174_guarded']}. {v['exact_nonactivated_episodes']} complete trajectories unchanged.48 fitting episodes and5 exact E174 replays. E120, probe protection and release fix unchanged. Early guard designed on fitting histories; privileged geometry/object pose/contact inputs, no held test. Report: `outputs/ease_early_object_recovery_v1/report.md`; verified success/failure videos: `videos/index.html`.\\n"
    review=Path('outputs/ease_iteration_review_v1')
    for p in (Path(__file__).with_name('TECHNICAL_REPORT.md'),review/'report.md'):
        text=p.read_text(encoding='utf-8')
        if '## E175:' not in text:p.write_text(text+section,encoding='utf-8')
    p=review/'WORKING_STATE.md';text=p.read_text(encoding='utf-8')
    if not text.startswith('# Latest checkpoint: E175'):p.write_text('# Latest checkpoint: E175 complete\\n'+section+'\\nAll runs, audits and videos complete.\\n\\n---\\n'+text,encoding='utf-8')
    studies=read(review/'studies.json')
    if not any(x['experiment']=='E175' for x in studies):studies.append(dict(experiment='E175',folder=str(ROOT),report_exists=True,videos=len(videos)));save(review/'studies.json',studies)
    save(ROOT/'status.json',dict(status='complete',new_episodes=48,exact_replays=5,videos=len(videos),held_evaluated=False,E120_unchanged=True,probe_protection_preserved=True))
    save(ROOT/'artifact_hashes.json',{str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});print('E175 complete',len(videos),'verified videos',flush=True)
''' +s[end:]
    (base/'early_object_report.py').write_text(s)

if __name__=='__main__':run()

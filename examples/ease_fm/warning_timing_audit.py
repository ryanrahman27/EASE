"""E77: read-only warning timing and six-joint contact coverage diagnosis."""
import json
import shutil
from pathlib import Path
from .benchmark import fingerprint

ROOT=Path('outputs/ease_warning_timing_v2')
SOURCE=Path('outputs/ease_response_validation_v1')


def run():
    ROOT.mkdir(exist_ok=False)
    inputs={};warnings=[];contacts=[];runs=json.loads((SOURCE/'records.json').read_text())
    def read(p):
        inputs[str(p)]=fingerprint(p)
        return json.loads(p.read_text())
    read(SOURCE/'records.json')
    spec=dict(experiment='E77',design='Read-only post-hoc E76 timing and first-contact coverage audit. All 96 learned trajectories, no new selection, fitting, threshold changes or actions.',
        scoring='Preserve E76 one-step false-alarm labels. For every warning on a noncollision step, report exact steps until first collision on the unchanged trajectory. Do not select a new horizon or relabel the E76 gate.',
        limits='Observed future collision does not establish warning usefulness under an intervention. Repeated checkpoints on a placement are dependent. Geometry identities are resolved from the original model; no new rollouts or task-success claims.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    tasks=[]
    from .mixed_sim import AXES,Arm
    from .arm_pick_place import PickPlace
    import mujoco
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;tasks.extend(read(p/'cases.json'));AXES.update(read(p/'axis_catalog.json'))
    for run in runs:
        if run['method']=='analytical':continue
        p=SOURCE/'trials'/f"{run['case']}_{run['method']}.json";rr=read(p)['records']
        old=read(Path('outputs/ease_geometry_filter_v2/trials')/f"{run['case']}_{run['method']}_geometry.json")
        first=next((r for r in rr if r['collision']),None)
        assert len(rr)==len(old['trace'])
        if first:
            i=first['step'];trace=old['trace'];phase=first['phase'];start=next(j for j,x in enumerate(trace) if x['phase']==phase)
            t=tasks[run['case']];env=PickPlace(Arm(**t['arm']),t['xy'],t['target'])
            def resolve(name):
                gid=int(name) if name.isdigit() else mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_GEOM,name)
                return dict(geom=name,body=mujoco.mj_id2name(env.model,mujoco.mjtObj.mjOBJ_BODY,int(env.model.geom_bodyid[gid])))
            contacts.append(dict(case=run['case'],method=run['method'],dof=run['dof'],step=i,phase=phase,phase_step=i-start,
                scored=first['evaluated'],pairs=trace[i]['bad'],resolved_pairs=[[resolve(n) for n in pair] for pair in trace[i]['bad']],
                position_error=trace[i]['position_error'],rotation_error=trace[i]['rotation_error'],pads=trace[i]['pads'],
                preceding_phases=[x for x in old['phases'] if x['phase'] in {r['phase'] for r in trace[:start]}]))
        for r in rr:
            if not r['evaluated'] or r['collision']:continue
            flags={m:r['clearances'][m]<.002-1e-6 for m in ('command','response_linear','velocity_linear','velocity_curve')}
            if not any(flags.values()):continue
            warnings.append(dict(case=run['case'],method=run['method'],dof=run['dof'],step=r['step'],phase=r['phase'],task_success=run['success'],
                warnings=flags,clearances=r['clearances'],steps_to_first_contact=first['step']-r['step'] if first else None,
                first_contact_phase=first['phase'] if first else None))
    extra=[r for r in warnings if r['warnings']['response_linear'] and not r['warnings']['command']]
    six=[r for r in contacts if r['dof']==6];response=[r for r in warnings if r['warnings']['response_linear']]
    result=dict(learned_trials=96,first_contacts=len(contacts),six_joint_contacts=six,response_warnings=len(response),
        response_only_warnings=extra,response_warning_lead_steps=[r['steps_to_first_contact'] for r in response],
        success_warnings=sum(r['task_success'] for r in response),original_E76_gate='FAIL; unchanged')
    assert len(contacts)==26 and len(six)==5 and len(response)==8 and len(extra)==4
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'warnings.json').write_text(json.dumps(warnings,indent=2));(ROOT/'contacts.json').write_text(json.dumps(contacts,indent=2))
    lines=['# E77: warning timing and six-joint contact coverage','',
        'Read-only diagnosis of all 96 learned E76 trajectories. No policy, fitting rule, threshold, action or original scoring label changes. E76 remains a failed prediction gate.','',
        'The response predictor has eight one-step false alarms. Seven occur one or two control steps before first contact; one occurs 83 steps before first contact. Four response-only warnings occur one or two steps (40-80 ms) before contact; one command-only warning offsets these to a net increase of three, all on placement 15 with different checkpoints. None occurs on a successful task. These are correlated observations on unchanged trajectories, not demonstrated useful interventions. A longer-horizon metric chosen after seeing this pattern would be exploratory.','',
        '| Six-joint case | Estimator | First-contact phase | Step within phase (zero-based) | Contact pairs |','|---|---|---|---:|---|']
    for r in six:lines.append(f"| {r['case']} | {r['method']} | {r['phase']} | {r['phase_step']} | {r['pairs']} |")
    lines+=['',
        'Only one of five six-joint first contacts is in the scored descent/lowering phases. Of the four omitted contacts, three involve arm-link/table contact during lift or transport; one involves a finger/table contact during lift. Exact geometry-to-body mappings and preceding phase completion flags are saved in contacts.json. End-effector clearance alone cannot cover these failures.','',
        'Next: run the frozen predictor across every pre-contact phase with successful-run controls. Keep the original one-step metric and fixed threshold, report phase-specific coverage and full warning lead times separately, and require whole-arm geometry. This is a new development diagnostic, not a revision of the failed E76 gate. Only after that audit should a bounded action intervention be tested for full-task recoveries and regressions.','',
        'No new simulation or training occurred. All 112 E76 trajectories had previously replayed exactly; this audit reads all 96 learned traces and verifies source hashes. It neither improves success rates nor establishes independent generalization.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    for p,h in inputs.items():assert fingerprint(p)==h
    (ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,hashes_unchanged=True,test_evaluated=False),indent=2))
    shutil.copy2(__file__,ROOT/'source.py')
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()

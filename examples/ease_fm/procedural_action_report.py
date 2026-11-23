"""E54/E55 score validation with model-generated procedural histories."""
import argparse
import json
from pathlib import Path
from .action_objective_report import agreement
from .benchmark import fingerprint


def group(name):return name.rsplit('_',1)[0] if name!='analytical' else name


def run(output):
    root=Path(output);rows=json.loads((root/'records.json').read_text());drivers=json.loads((root/'drivers.json').read_text())
    p=json.loads((root/'protocol.json').read_text());audit=json.loads((root/'audit.json').read_text())
    pose=p['experiment']=='E55'
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    assert len(drivers)==audit['driver_replays']==80
    assert all(r['exact_replay'] for r in drivers)
    assert len(rows)==audit['states']==audit['exact_first_steps']
    assert len({(r['case'],r['driver'],r['index']) for r in rows})==len(rows)
    for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
    assert all(len(r['candidates'])==10 for r in rows)
    lines=['# E54: action-score validity on procedural training histories','',
        'Frozen analytical and nine original/E52 models each drive all eight training bodies (four six-joint, four seven-joint). Each driver cold-starts with empty statistics and tracks alternating first/last training goals, switching every 50 steps with a 2 mm reference ramp. Driver histories are separate. The same ten candidates receive each sampled driver history and branch for five steps, continuing the moving-target schedule and online estimator/bias updates.',
        '', 'Position-only reaching with the existing procedural actuators, gain 0.5, damping 0.015, command clip 0.04 and joint limits +/-2.5. This is not a pose/contact manipulation benchmark. Native data, validation bodies and final-test bodies are excluded; no training is performed.',
        '',f"All {len(drivers)} driver trajectories replay exactly without branches. All {len(rows)} sampled first-step driver branches match the actual next step exactly. There are {10*len(rows)} counterfactual branches. Checkpoint hashes are unchanged.",
        '', '## History coverage', '', '| Driver group | Trajectories | Driver collisions | Sampled states | Missing planned states |','|---|---:|---:|---:|---:|']
    if pose:
        lines[0]='# E55: position/tool-direction score validity on procedural histories'
        lines[2]='Frozen analytical and nine original/E52 models each drive all eight training bodies. Two target poses per body come from seeded +/-0.15-radian home perturbations and forward kinematics, with no outcome/contact filtering. Each driver cold-starts with empty statistics and switches targets every 50 steps, ramping position by at most 2 mm and tool-Z direction by approximately 0.01 radians per step. All ten candidates branch from each measured driver history for five continuing control steps.'
        lines[4]='Position plus 0.15-scaled unit tool-Z tracking, with free yaw, gain 0.35, damping 0.015, command clip 0.04 and joint limits +/-2.5. This matches the task-coordinate representation of native manipulation but has no object or gripper. Both frozen model calls and measured bias updates are shared. Training-split procedural arms only; no native or held-out geometry and no optimization.'
    for g in ['analytical','original','prediction','control']:
        dd=[r for r in drivers if group(r['driver'])==g];ss=[r for r in rows if group(r['driver'])==g]
        lines.append(f"| {g} | {len(dd)} | {sum(r['collision'] for r in dd)} | {len(ss)} | {9*len(dd)-len(ss)} |")
    lines+=['','Drivers stop on collision; later missing states are retained as missing, not replaced. The 150-step moving-target runs ignore the old static-goal success flag and do not define a new task-success metric.',
        '', '## Case-macro ordering agreement', '',
        'At each common state, pairwise proxy order is compared with measured one/five-step error order. Values average within each body then across bodies; ties within 1e-12 are excluded. The eight bodies, not branches or candidate pairs, define clusters. Scores use the privileged instantaneous Jacobian only for diagnostics. All-candidate comparisons include analytical. All results are descriptive, not independent-sample significance claims.',
        '', '| DoF | Driver group | Horizon | Operator | Local unclipped | Local executed |','|---|---|---:|---:|---:|---:|']
    summary=[]
    for dof in ['6dof','7dof']:
        for g in ['all','analytical','original','prediction','control']:
            rr=[r for r in rows if r['robot']==dof and (g=='all' or group(r['driver'])==g)]
            for h in [1,5]:
                values=[]
                for score in ['operator_mse','local_unclipped','local_executed']:
                    result=agreement(rr,score,h);values.append(result['case_mean'])
                    summary.append(dict(dof=dof,driver_group=g,horizon=h,proxy=score,neural_only=False,collision_free=False,**result))
                    for neural,free in [(True,False),(False,True)]:
                        summary.append(dict(dof=dof,driver_group=g,horizon=h,proxy=score,neural_only=neural,collision_free=free,**agreement(rr,score,h,neural,free)))
                lines.append(f'| {dof} | {g} | {h} | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines+=['','## Target-switch sensitivity (all drivers)','','| DoF | Samples | Horizon | Operator | Local executed |','|---|---|---:|---:|---:|']
    for dof in ['6dof','7dof']:
        for category in ['switch','other']:
            rr=[r for r in rows if r['robot']==dof and ((r['index'] in [49,50,99,100])==(category=='switch'))]
            for h in [1,5]:
                a=agreement(rr,'operator_mse',h);b=agreement(rr,'local_executed',h)
                lines.append(f"| {dof} | {category} | {h} | {a['case_mean']:.3f} | {b['case_mean']:.3f} |")
    lines+=['','## Branch contacts and saturation','','| DoF | Candidate group | Branches | Five-step contact branches | First-command clipped branches |','|---|---|---:|---:|---:|']
    for dof in ['6dof','7dof']:
        for g in ['analytical','original','prediction','control']:
            rr=[x for r in rows if r['robot']==dof for x in r['candidates'] if group(x['method'])==g]
            lines.append(f"| {dof} | {g} | {len(rr)} | {sum(x['branch']['5']['collision'] for x in rr)} | {sum(x['clipped_joints']>0 for x in rr)} |")
    if pose:
        lines+=['','## Driver pose tracking (descriptive, not a success gate)','','| DoF | Driver group | Mean position error (m) | Mean tool-direction chord error |','|---|---|---:|---:|']
        for dof in [6,7]:
            for g in ['analytical','original','prediction','control']:
                rr=[t for r in drivers if r['dof']==dof and group(r['driver'])==g for t in r['trace']]
                lines.append(f"| {dof} | {g} | {sum(r['position_error'] for r in rr)/len(rr):.5f} | {sum(r['direction_error'] for r in rr)/len(rr):.5f} |")
    scope='position/tool-direction tracking without a gripper or object' if pose else 'position tracking without tool-orientation control'
    lines+=['','## Interpretation limits','',f'This is a frozen ranking audit, not a new trained controller. It tests {scope} on eight known training bodies and continuing candidate-generated histories; it does not validate grasp success or unseen embodiments. The local executed score assumes measured bias approximates actuator response; it does not differentiate through the simulator. Five-step rankings also include subsequent feedback updates and target changes that a one-step proxy does not forecast. Scores selected from these development results require a separate protocol before optimization. Keep original E35 checkpoints; no objective/model is automatically promoted. Per-body ranking rates, collision-free and neural-only comparisons are retained in summary.json.']
    (root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'report_audit.json').write_text(json.dumps(dict(exact_driver_replays=80,exact_driver_branches=len(rows),states=len(rows),branches=10*len(rows),records_sha256=fingerprint(root/'records.json'),unchanged_weights=True,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))

"""E30–E32 audited replication and neural-estimator isolation."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .benchmark import fingerprint
from .paired_data import register
from .paired_experiment import gate
from .motion_transformer import MotionTransformer, evaluate


def read(path):return json.loads(Path(path).read_text())


def report(data, output, coldstart, device='cuda', api_check=None):
    torch.set_num_threads(4)
    root=Path(output); manifest=register(data); protocol=read(root/'protocol.json')
    assert read(root/'status.json')['status']=='complete'
    assert fingerprint(Path(data)/'manifest.json')==protocol['manifest_sha256']
    pairs={p['id']:p['bodies'] for p in manifest['pairs']}
    expected={(b,g['id']) for g in manifest['goals']['validation'] for b in pairs[g['pair']]}
    goals={g['id']:g for g in manifest['goals']['validation']}
    dofs={a['seed']:len(a['lengths']) for a in manifest['arms']}
    summaries=[]; training=[]; evidence={}; replay_cases=0
    original_solve=np.linalg.solve
    def control_solve(a,b):
        if a.shape[-2:]!=(3,3):raise AssertionError('Neural mode attempted analytical joint-model solve')
        return original_solve(a,b)
    index=read(root/'index.json');assert {e['seed'] for e in index}=={0,1,2}
    for entry in index:
        seed=entry['seed'];folder=Path(entry['folder']);meta=read(folder/'training.json')
        assert fingerprint(folder/'policy.pt')==entry['policy_sha256']==meta['policy_sha256']
        assert meta['steps']==15000 and meta['seed']==seed and meta['profile_count']==128
        assert meta['manifest_sha256']==protocol['manifest_sha256']
        records=read(root/f'validation_seed_{seed}.json');assert len(records)==6144
        evidence[str(root/f'validation_seed_{seed}.json')]=fingerprint(root/f'validation_seed_{seed}.json')
        model=MotionTransformer().to(device).eval()
        model.load_state_dict(torch.load(folder/'policy.pt',map_location=device,weights_only=True))
        # Replaying both primary neural modes while forbidding hidden analytic J solves.
        with patch.object(mujoco,'mj_jacSite',side_effect=AssertionError('Privileged Jacobian forbidden')), \
             patch.object(np.linalg,'solve',side_effect=control_solve), \
             patch.object(torch.linalg,'solve',side_effect=AssertionError('Neural inverse forbidden')):
            for mode in ('neural_online','empty_neural_online'):
                replay=evaluate(manifest,model,'validation',mode,device)
                prior=[r for r in records if r['mode']==mode and r['probe_seed']==930000]
                assert replay==prior
                replay_cases+=len(replay)
        for mode in protocol['modes']+['random_neural_online']:
            probes=[930000] if mode=='random_neural_online' else protocol['probes']
            if mode=='empty_neural_online':
                strips=[[{k:v for k,v in r.items() if k!='probe_seed'} for r in records if r['mode']==mode and r['probe_seed']==p] for p in probes]
                assert strips[0]==strips[1]==strips[2]
            for probe in probes:
                selected=[r for r in records if r['mode']==mode and r['probe_seed']==probe]
                assert len(selected)==384 and {(r['body_seed'],r['goal_id']) for r in selected}==expected
                assert all(r['dof']==dofs[r['body_seed']] and r['condition']==goals[r['goal_id']]['condition'] for r in selected)
                assert all(r['success'] or r['steps']==150 for r in selected)
                assert all(not r['success'] or (r['error']<.03 and not r['collision']) for r in selected)
                cost=0 if mode=='empty_neural_online' else 32
                summaries.append(dict(seed=seed,mode=mode,probe_seed=probe,successes=sum(r['success'] for r in selected),
                    collisions=sum(r['collision'] for r in selected),mean_task_steps=float(np.mean([r['steps'] for r in selected])),
                    mean_total_steps=float(np.mean([r['steps']+cost for r in selected])),
                    success_by_50=sum(r['success'] and r['steps']+cost<=50 for r in selected),
                    conditions={c:{str(n):sum(r['success'] for r in selected if r['condition']==c and r['dof']==n) for n in (6,7)} for c in sorted({r['condition'] for r in selected})}))
        for mode in ('neural_frozen','neural_online','hybrid_online'):
            path=folder/f'train_{mode}.json'; rows=read(path)
            assert len(rows)==256
            assert {(r['body_seed'],r['goal_id']) for r in rows}=={(b,g['id']) for g in manifest['goals']['train'] for b in pairs[g['pair']]}
            evidence[str(path)]=fingerprint(path)
            training.append(dict(seed=seed,mode=mode,**gate(rows)))
    cold=Path(coldstart);assert read(cold/'status.json')['status']=='complete'
    assert read(cold/'protocol.json')['manifest_sha256']==protocol['manifest_sha256']
    cold_rows=read(cold/'records.json');assert len(cold_rows)==1536
    evidence[str(cold/'records.json')]=fingerprint(cold/'records.json')
    cold_summary=[]
    for mode in sorted({r['mode'] for r in cold_rows}):
        rows=[r for r in cold_rows if r['mode']==mode]
        assert {(r['body_seed'],r['goal_id']) for r in rows}==expected
        cold_summary.append(dict(mode=mode,successes=sum(r['success'] for r in rows),mean_steps=float(np.mean([r['steps'] for r in rows])),collisions=sum(r['collision'] for r in rows)))
    (root/'summary.json').write_text(json.dumps(dict(validation=summaries,training=training,coldstart=cold_summary),indent=2))
    api=None
    if api_check:
        directory=Path(api_check);api=read(directory/'audit.json');rows=read(directory/'records.json')
        assert api['manifest_sha256']==protocol['manifest_sha256']
        assert api['checkpoint_sha256']==next(e['policy_sha256'] for e in index if e['seed']==0)
        assert api['records_sha256']==fingerprint(directory/'records.json')
        assert api['controller_source_sha256']==fingerprint(Path(__file__).with_name('motion_transformer.py'))
        assert len(rows)==384 and {(r['body_seed'],r['goal_id']) for r in rows}==expected
        assert api['successes']==sum(r['success'] for r in rows)
        evidence[str(directory/'records.json')]=fingerprint(directory/'records.json')
    (root/'audit.json').write_text(json.dumps(dict(neural_inverse_guard_replay_cases=replay_cases,
        stored_validation_records=18432,empty_probe_duplicates=2304,evidence_sha256=evidence,
        test_evaluated=False,report_source_sha256=fingerprint(Path(__file__))),indent=2))
    lines=['# E30–E32: explicit motion Transformer and shared control','',
        'The redesigned model is a 235,011-parameter, three-layer joint-token Transformer (width 96, four attention heads). Measured displacement Gram/cross statistics become joint tokens; the network predicts one 3D Jacobian column per joint. A shared damped Cartesian controller and measured command-tracking bias produce actions. This replaces direct action-chunk prediction; it is not a fine-tuned E27 checkpoint. Model weights stay frozen during evaluation while motion statistics update from task transitions.','',
        'Each model trains for 15,000 updates × 128 synthetic linear systems. Supervision distills ridge estimates from random Jacobians and excitation profiles drawn only from 128 training-robot probes, mixed with Gaussian excitation. No validation geometry or expert actions are used. Data, objective, representation, controller and online adaptation all change from E27; this is not an isolated architecture ablation.','',
        'Seed 0 was observed before the unchanged seeds 1/2 replication. Every trained model is evaluated with three probes. Empty mode has no probe and its three identical repetitions are explicitly deduplicated in interpretation. Hybrid uses the network only on step one, then analytical identification; it is not neural-only estimation.','',
        '## Validation success','',
        'Each entry lists three probe results out of 384 for a training seed. Empty and random controls have one meaningful result per seed.','',
        '| Mode | Seed 0 | Seed 1 | Seed 2 |','|---|---|---|---|']
    for mode in protocol['modes']+['random_neural_online']:
        cells=[]
        for seed in (0,1,2):
            rows=[r for r in summaries if r['seed']==seed and r['mode']==mode and (r['probe_seed']==930000 or mode not in ('empty_neural_online','random_neural_online'))]
            cells.append(' / '.join(str(r['successes']) for r in rows))
        lines.append('| '+mode+' | '+' | '.join(cells)+' |')
    lines+=['','## Unseen six-joint cases','','Each entry is successes out of 48, across probe seeds. Empty has no probe.','',
        '| Mode | Training seed | Familiar posture | Shifted posture |','|---|---:|---|---|']
    for mode in ('neural_frozen','neural_online','empty_neural_online'):
        for seed in (0,1,2):
            rows=[r for r in summaries if r['mode']==mode and r['seed']==seed and (r['probe_seed']==930000 or mode!='empty_neural_online')]
            cells=[' / '.join(str(r['conditions'][c]['6']) for r in rows) for c in ('new_axes_seen_posture','new_axes_shifted_posture')]
            lines.append(f'| {mode} | {seed} | '+' | '.join(cells)+' |')
    lines+=['','## Interaction cost','','Means average all validation cases, including full-horizon failures. Calibrated modes pay 32 steps per task; empty mode pays zero separate calibration. Reset and computation time are excluded.','',
        '| Mode | Mean task steps | Mean including probe | Mean successes by 50 total steps / 384 |','|---|---:|---:|---:|']
    for mode in protocol['modes']+['random_neural_online']:
        rows=[r for r in summaries if r['mode']==mode and (r['probe_seed']==930000 or mode not in ('empty_neural_online','random_neural_online'))]
        values=[np.mean([r[k] for r in rows]) for k in ('mean_task_steps','mean_total_steps','success_by_50')]
        lines.append('| '+mode+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
    lines+=['','## Training control','','| Seed | Mode | Success / 256 | Gate |','|---:|---|---:|---|']
    for r in training:lines.append(f"| {r['seed']} | {r['mode']} | {r['overall'][0]} | {r['passed']} |")
    lines+=['','## Cold-start attribution controls','','E32 was added after seeing seed-0 empty-history success. All rows have no separate calibration. Neural bootstrap/constant use the seed-0 model; the fixed first action is +0.01 rad for every joint, counted as task step one.','',
        '| Controller | Success / 384 | Collisions | Mean steps |','|---|---:|---:|---:|']
    for r in cold_summary:lines.append(f"| {r['mode']} | {r['successes']} | {r['collisions']} | {r['mean_steps']:.2f} |")
    lines+=['',
        'The trained network must be distinguished from the analytical controller around it. Random-network and constant-output controls test whether useful learned predictions and task-time updates matter. A successful fixed-probe analytical controller prevents claiming learned superiority merely from cold-start success. Matching this benchmark is a functional milestone, not evidence of novelty or broad morphology generalization.','',
        f'All {replay_cases} primary neural/empty replay cases match exactly while simulator Jacobian access, all torch.linalg.solve calls, and NumPy solves larger than the 3×3 Cartesian control system are forbidden. Thus these neural modes do not secretly substitute an analytical joint-model inverse. Other analytical operations, measured tracking-bias updates, and the 3×3 control solve remain intentional components.','',
        'The same fixed procedural validation families and expert-filtered reaching goals are used throughout. No final-test policy evaluation, hardware, vision, orientation control, or manipulation claim is made. Further claims require broader morphology families and tasks plus a demonstrated benefit over analytical identification.','']
    if api:
        lines += [f"The reusable single-robot MotionController interface independently achieves {api['successes']}/384 without calibration using the seed-0 checkpoint, with the same inverse/API guards. One model instance is shared across six/seven-joint controller states. This separately checks the deployment interface rather than relying only on the batched evaluator.",'']
    (root/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    methods=['neural_frozen','neural_online','empty_neural_online']
    for ax,c,title in zip(axes,('new_axes_seen_posture','new_axes_shifted_posture'),('New 6-DoF axes / familiar posture','New 6-DoF axes / shifted posture')):
        for i,mode in enumerate(methods):
            rows=[r for r in summaries if r['mode']==mode and (r['probe_seed']==930000 or mode!='empty_neural_online')]
            values=[r['conditions'][c]['6']/48*100 for r in rows]
            ax.bar(i,np.mean(values),color=['#8396ad','#17877e','#cf9b40'][i])
            ax.scatter(np.linspace(i-.1,i+.1,len(values)),values,color='black',s=12,zorder=3)
        ax.set(title=title,ylim=(0,105),xticks=range(3),xticklabels=['Frozen neural','Online neural','Cold-start neural'])
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Success (%)');fig.suptitle('Motion Transformer: learned Jacobian + shared controller')
    fig.tight_layout();fig.savefig(root/'six_joint_transformer.png',dpi=160);plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('data','output','coldstart'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--api-check')
    report(**vars(parser.parse_args()))

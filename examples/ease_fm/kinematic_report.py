"""Audit E28/E29 measured-kinematics control, mechanism, and interaction cost."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .paired_data import register


def read(path): return json.loads(Path(path).read_text())


def report(data, identified, ablation):
    manifest = register(data); root = Path(ablation); initial = Path(identified)
    pairs = {p['id']: p['bodies'] for p in manifest['pairs']}
    summaries = []; evidence = {}; datasets = {}
    for directory in (initial, root):
        protocol = read(directory/'protocol.json')
        assert read(directory/'status.json')['status'] == 'complete'
        assert protocol['manifest_sha256'] == fingerprint(Path(data)/'manifest.json')
        records = read(directory/'records.json'); datasets[protocol['experiment']] = records
        evidence[str(directory/'records.json')] = fingerprint(directory/'records.json')
        assert len(records) == 4224
        assert all(r['success'] or r['steps'] == 150 for r in records)
        assert all(not r['success'] or (r['error'] < .03 and not r['collision']) for r in records)
        for split in ('train', 'validation'):
            expected = {(body, g['id']) for g in manifest['goals'][split] for body in pairs[g['pair']]}
            for probe in ([930000] if split == 'train' else protocol['calibration_seeds']):
                for mode in protocol['methods']:
                    selected = [r for r in records if r['split'] == split and r['probe_seed'] == probe and r['mode'] == mode]
                    assert len(selected) == len(expected)
                    assert {(r['body_seed'], r['goal_id']) for r in selected} == expected
                    for condition in sorted({r['condition'] for r in selected}):
                        for dof in (6, 7):
                            cell = [r for r in selected if r['condition'] == condition and r['dof'] == dof]
                            summaries.append(dict(experiment=protocol['experiment'], split=split, probe_seed=probe,
                                mode=mode, condition=condition, dof=dof, successes=sum(r['success'] for r in cell),
                                trials=len(cell), collisions=sum(r['collision'] for r in cell),
                                mean_task_steps=float(np.mean([r['steps'] for r in cell])),
                                mean_single_use_steps=float(np.mean([r['steps']+32 for r in cell])),
                                success_by_50=sum(r['success'] and r['steps']+32 <= 50 for r in cell)))
    key = lambda r: (r['split'], r['probe_seed'], r['body_seed'], r['goal_id'])
    left = {key(r): r for r in datasets['E28'] if r['mode'] == 'identified_online'}
    right = {key(r): r for r in datasets['E29'] if r['mode'] == 'identified_online'}
    assert left == right, 'Guarded replay must match all recorded results exactly'
    diagnostics = read(initial/'identification.json')
    unique = {(r['body_seed'], r['condition'], r['probe_seed']): r for r in diagnostics if r['split'] == 'validation' and r['mode'] == 'identified_frozen'}
    identification = dict(probes=len(unique), median_relative_jacobian_error=float(np.median([r['jacobian_relative_error'] for r in unique.values()])),
        maximum_relative_jacobian_error=max(r['jacobian_relative_error'] for r in unique.values()))
    audit = dict(manifest_sha256=fingerprint(Path(data)/'manifest.json'), evidence_sha256=evidence,
        guarded_online_replay_exact=True, replay_cases=len(left), test_evaluated=False,
        identification=identification, report_source_sha256=fingerprint(Path(__file__)))
    (root/'audit.json').write_text(json.dumps(audit, indent=2))
    (root/'summary.json').write_text(json.dumps(summaries, indent=2))
    lines = ['# E28–E29: observation-only adaptive kinematic control', '',
        'This is an analytical controller, not an improved Transformer checkpoint. It estimates a local Cartesian Jacobian from measured joint/end-effector displacements and a joint-command tracking bias from executed actions. Only measured observations enter the controller; no axis sequence, links, masses, gravity forces, simulator Jacobian, or target-arm expert actions are used.', '',
        'The 32-step calibration, goals, resets, action limit, 150-step task horizon, 3 cm / five-step success criterion, and collision rejection match E27. Online variants additionally update their estimates from task transitions. E28 compares frozen, online, and wrong-twin initialization. E29 is a post-E28 mechanism check with fixed gains, isolating Jacobian and bias updates and blocking mj_jacSite during every rollout.', '',
        '## Unseen six-joint validation', '',
        'Entries are successes for three calibration seeds, each out of 48 cases. These are probe variations on fixed bodies, not independent learned-policy training seeds.', '',
        '| Controller | Familiar posture | Shifted posture |', '|---|---|---|']
    methods = [('E28','identified_frozen'), ('E28','identified_online'), ('E28','twin_online'), ('E29','jacobian_online'), ('E29','bias_online')]
    for experiment, mode in methods:
        cells = []
        for condition in ('new_axes_seen_posture', 'new_axes_shifted_posture'):
            cells.append(' / '.join(str(next(r['successes'] for r in summaries if r['experiment'] == experiment and r['mode'] == mode and r['condition'] == condition and r['dof'] == 6 and r['probe_seed'] == seed)) for seed in (930000,940000,950000)))
        lines.append('| '+mode+' | '+' | '.join(cells)+' |')
    lines += ['', '## All validation and cost', '',
        '| Controller | Successes / 1152 | Collisions | Mean task steps | Including one probe per task | Success by 50 total steps |', '|---|---:|---:|---:|---:|---:|']
    for experiment, mode in methods:
        records = [r for r in datasets[experiment] if r['split'] == 'validation' and r['mode'] == mode]
        steps = np.mean([r['steps'] for r in records])
        lines.append(f"| {mode} | {sum(r['success'] for r in records)} | {sum(r['collision'] for r in records)} | {steps:.2f} | {steps+32:.2f} | {sum(r['success'] and r['steps']+32 <= 50 for r in records)} |")
    lines += ['', f"The initial Jacobian estimate has median relative Frobenius error {identification['median_relative_jacobian_error']:.3f} and maximum {identification['maximum_relative_jacobian_error']:.3f} across {identification['probes']} distinct validation probes. These privileged diagnostics are computed outside the controller. The complete online replay ({len(left)} training/validation cases) is exactly unchanged with simulator Jacobian access blocked.", '',
        'Probe cost is 32 control interactions, including for wrong-twin diagnostics. All failures cost 150 task steps. Reset and computation costs are excluded. Reusing calibration is possible only for the same robot/reset; the table charges a fresh probe for every task. The bias estimate is a measured command-minus-motion residual, not an identified gravity model.', '',
        'The result establishes working measured-feedback control for this fixed simulated reaching benchmark, including previously failing six-joint arrangements. It does not establish neural cross-embodiment transfer, novelty, real-world control, pose/orientation manipulation, or unseen final-test performance. Wrong-twin recovery suggests online feedback can correct much of an initially incorrect model; initial-history necessity is not established. Online task adaptation also changes the information protocol relative to E27 frozen-context Transformers.', '',
        'The next learned architecture should expose a kinematic/motion predictor and update it from observed transitions, retaining this controller as a strong baseline. A neural version must earn its value over this analytical solution; adding a Transformer around it is not evidence of a contribution.', '']
    (root/'report.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    labels = ['Frozen', 'Both online', 'Twin + online', 'Jacobian only', 'Bias only']
    for ax, condition, title in zip(axes, ('new_axes_seen_posture','new_axes_shifted_posture'), ('New 6-DoF axes / familiar posture','New 6-DoF axes / shifted posture')):
        values = [[r['successes']/48*100 for r in summaries if r['experiment']==exp and r['mode']==mode and r['condition']==condition and r['dof']==6] for exp,mode in methods]
        ax.bar(range(5), [np.mean(v) for v in values], color=['#8196ad','#187d79','#60a69d','#da9b43','#a980af'])
        for i, v in enumerate(values): ax.scatter(np.full(3,i),v,color='black',s=13,zorder=3)
        ax.set(title=title,ylim=(0,105),xticks=range(5),xticklabels=labels)
        ax.tick_params(axis='x',rotation=20); ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Success (%)'); fig.suptitle('Measured kinematic control — dots show calibration seeds')
    fig.tight_layout(); fig.savefig(root/'six_joint_control.png',dpi=160); plt.close(fig)
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for key in ('data','identified','ablation'): parser.add_argument('--'+key,required=True)
    report(**vars(parser.parse_args()))

"""Audit E43 inference diagnostics and E44 synthetic fine-tuning gates."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint


def run(runs,training,reach,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);summaries=[];audit=[]
    lines=['# E43–E44: native motion-estimation diagnostics','',
        'E43 compares unchanged E35 weights under baseline, direction rescaling, and Jacobian smoothing. The analytical shadow fit is diagnostic only and never supplies neural actions. E44 fine-tunes all three seeds for 2,000 fixed updates on synthetic histories; no native robot trajectory or geometry enters training. All task evaluations reuse observed development cases; final test remains reserved.','']
    baseline_rows=None
    for folder in map(Path,runs):
        protocol=json.loads((folder/'protocol.json').read_text());rows=json.loads((folder/'records.json').read_text())
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        assert not protocol['test_evaluated']
        assert fingerprint(Path(protocol['source'])/'cases.json')==protocol['cases_sha256']
        for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
        tasks=json.loads((Path(protocol['source'])/'cases.json').read_text());robots=protocol.get('robots','panda').split(',')
        methods=['analytical','neural_0','neural_1','neural_2'] if protocol['variant']=='baseline' else ['neural_0','neural_1','neural_2']
        expected={(m,i) for m in methods for i,t in enumerate(tasks) if t['robot'] in robots}
        assert len(rows)==len(expected) and {(r['method'],r['case']) for r in rows}==expected
        if protocol['variant']=='baseline':baseline_rows=rows
        panda_reference=json.loads((Path(protocol['reference'])/'records.json').read_text())
        piper_reference=json.loads((Path(protocol['source'])/'records.json').read_text())
        for row in rows:
            assert row['steps']==len(row['probe'])+len(row['trace'])
            assert len(row['predictions'])==len(row['trace'])
            previous=next(r for r in (panda_reference if row['robot']=='panda' else piper_reference)
                if r['method']==row['method'] and r['case']==row['case'])
            assert row['probe']==previous['probe']
            if row['success']:assert not row['collision'] and row['grasped'] and row['lifted']
        lines += [f'## {folder.name}','','| Robot | Method | Success /9 | Contacts | Failures |','|---|---|---:|---:|---|']
        for robot in robots:
            for method in methods:
                selected=[r for r in rows if r['robot']==robot and r['method']==method];assert len(selected)==9
                failures=Counter(r['failure'] for r in selected if not r['success'])
                summary=dict(run=folder.name,robot=robot,method=method,successes=sum(r['success'] for r in selected),
                    contacts=sum(r['collision'] for r in selected),failures=dict(failures))
                summaries.append(summary);reasons='; '.join(f'{k}: {v}' for k,v in sorted(failures.items())) or 'none'
                lines.append(f"| {robot} | {method} | {summary['successes']} | {summary['contacts']} | {reasons} |")
        lines+=[''];audit.append(dict(folder=str(folder),records_sha256=fingerprint(folder/'records.json'),protocol_sha256=fingerprint(folder/'protocol.json')))
    lines += ['## Matched one-step diagnostics on baseline neural trajectories','',
        'Errors below are computed from the next measured joint/tool displacement, using estimates made before that displacement. Each trial contributes equally. These describe directional prediction along executed motion, not full-Jacobian accuracy or proof of causal failure.',
        '', '| Seed | Mean XYZ error (mm) | Shadow XYZ error (mm) | Direction-error ratio, neural/shadow |', '|---|---:|---:|---:|']
    diagnostics=[]
    for seed in range(3):
        selected=[r for r in baseline_rows if r['method']==f'neural_{seed}']
        means={key:float(np.mean([np.mean([p[key] for p in r['predictions']]) for r in selected])) for key in ['position_error','shadow_position_error','direction_error','shadow_direction_error']}
        diagnostics.append(dict(seed=seed,**means))
        lines.append(f"| {seed} | {means['position_error']*1000:.3f} | {means['shadow_position_error']*1000:.3f} | {means['direction_error']/means['shadow_direction_error']:.2f} |")
    training=Path(training);p=json.loads((training/'protocol.json').read_text());assert p['steps']==2000
    assert json.loads((training/'status.json').read_text())['status']=='complete'
    assert fingerprint(Path(p['data'])/'manifest.json')==p['training_manifest_sha256']
    for path,sha in p['parents'].items():assert fingerprint(path)==sha
    lines += ['','## Fixed synthetic validation','','| Seed | Initial MSE | Final MSE |','|---|---:|---:|']
    for seed in range(3):
        folder=training/f'seed_{seed}';m=json.loads((folder/'training.json').read_text())
        assert m['steps']==p['steps'] and len(m['losses'])==p['steps'] and np.isfinite(m['losses']).all()
        assert fingerprint(folder/'policy.pt')==m['policy_sha256'] and m['parent_sha256'] in p['parents'].values()
        lines.append(f"| {seed} | {m['validation_initial']:.6f} | {m['validation_final']:.6f} |")
    reach=Path(reach);rr=json.loads((reach/'records.json').read_text());assert len(rr)==120
    assert json.loads((reach/'status.json').read_text())['status']=='complete'
    rp=json.loads((reach/'protocol.json').read_text());assert not rp['test_evaluated']
    for path,sha in rp['checkpoints'].items():assert fingerprint(path)==sha
    assert len({(r['robot'],r['method'],r['goal_id']) for r in rr})==120
    lines += ['','## Fine-tuned native reaching gate','','| Robot | Method | Success /12 |','|---|---|---:|']
    for robot in ['piper','panda']:
        for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
            chosen=[r for r in rr if r['robot']==robot and r['method']==method];assert len(chosen)==12
            lines.append(f"| {robot} | {method} | {sum(r['success'] for r in chosen)} |")
    lines += ['','The reused reaching runner has a stale prose label saying E35 weights in this completed run. Its recorded checkpoint paths/hashes correctly identify E44 weights; future runner wording is corrected. The completed protocol is preserved unchanged.',
        'All manipulation calibration records match their earlier per-robot reference exactly. Synthetic validation improvement is not a manipulation or reaching guarantee. Keep all model seeds and the original E35 checkpoints; native development scores do not establish generalization or publication readiness. No original procedural training-suite retention claim is made by this native-only gate.','']
    (root/'report.md').write_text('\n'.join(lines));(root/'summary.json').write_text(json.dumps(summaries,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(runs=audit,diagnostics=diagnostics,reach_sha256=fingerprint(reach/'records.json'),test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True)
    for key in ['training','reach','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

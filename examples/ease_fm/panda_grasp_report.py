"""Audit E42 paired Panda interventions and retain every negative variant."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .real_arms import verify_assets
from .benchmark import fingerprint


def run(assets,runs,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);manifest=verify_assets(assets)
    all_rows={};audits=[];summaries=[]
    lines=['# E42: Panda grasp and release diagnostics','',
        'All interventions reuse nine observed E41 Panda tasks, the same homes, calibration seeds, native mechanics and frozen E35 checkpoints. No new training, friction/force changes, hardware or final-test evaluation.',
        'Baseline: exact E41 revision-3 replay with tool/object orientation, gripper opening and actuator-force telemetry.',
        'Yaw: a third application of the same motion Transformer predicts tool-X motion, alongside XYZ/tool-Z, to align fingers with initial cube faces. This is an experimental nine-row task controller, not the default policy.',
        'Hold: keep native arm position targets fixed and pause estimator updates during close/release/settle.',
        'Settle: add a 2 mm / 0.03 direction-error pre-grasp gate to Hold.',
        'Supported: preserve the baseline controller and switch from lowering to release when measured cube center is 0.215–0.230 m and XY target error is below 20 mm. Release holds the current tool target. Despite the variant name, this permits a short drop and does not require measured table support. Final success criteria remain unchanged.',
        '','These sequentially developed variants are not independent confirmation. Full failures are retained.','']
    for folder in map(Path,runs):
        protocol=json.loads((folder/'protocol.json').read_text());rows=json.loads((folder/'records.json').read_text())
        assert json.loads((folder/'status.json').read_text())['status']=='complete'
        assert protocol['commit']==manifest['commit'] and not protocol['test_evaluated']
        assert fingerprint(Path(protocol['source'])/'cases.json')==protocol['cases_sha256']
        for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
        assert len(rows)==45 and len({(r['method'],r['case']) for r in rows})==45
        for row in rows:
            assert row['steps']==len(row['probe'])+len(row['trace'])==len(row['telemetry'])
            if row['success']:assert not row['collision'] and row['grasped'] and row['lifted'] and row['failure'] is None
        variant=protocol['variant'];all_rows[variant]=rows
        lines += [f'## {variant} ({folder.name})','','| Method | Success /9 | Contacts | Mean steps | Failures |',
            '|---|---:|---:|---:|---|']
        for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
            selected=[r for r in rows if r['method']==method];assert len(selected)==9
            failures=Counter(r['failure'] for r in selected if not r['success'])
            summary=dict(variant=variant,method=method,successes=sum(r['success'] for r in selected),
                contacts=sum(r['collision'] for r in selected),mean_steps=float(np.mean([r['steps'] for r in selected])),failures=dict(failures))
            summaries.append(summary)
            reasons='; '.join(f'{k}: {v}' for k,v in sorted(failures.items())) or 'none'
            lines.append(f"| {method} | {summary['successes']} | {summary['contacts']} | {summary['mean_steps']:.1f} | {reasons} |")
        lines+=[''];audits.append(dict(folder=str(folder),records_sha256=fingerprint(folder/'records.json'),
            protocol_sha256=fingerprint(folder/'protocol.json'),source_sha256={p.name:fingerprint(p) for p in (folder/'source').glob('*.py')}))
    baseline={(r['method'],r['case']):r for r in all_rows['baseline']}
    for variant,rows in all_rows.items():
        for row in rows:
            previous=baseline[row['method'],row['case']]
            assert row['probe']==previous['probe']
            if variant=='supported':
                prefix=lambda r:[t for t in r['trace'] if t['phase'] not in ('lower','release','retreat','settle')]
                assert prefix(row)==prefix(previous)
    lines += ['## Interpretation','',
        'Every calibration record matches across variants. The object-height release comparison also has exactly identical pre-lowering trajectories, isolating the later release intervention on these observed tasks. Grasp losses before lowering cannot be repaired by this intervention. The hold and stricter-settling results must not be pooled with the release variant.',
        'Mean steps include calibration and early termination; faster failed attempts do not establish efficiency. Small reused samples, ideal object-position observations and scripted sequencing preclude claims of reliable general manipulation or learned-controller superiority.','']
    (root/'report.md').write_text('\n'.join(lines));(root/'summary.json').write_text(json.dumps(summaries,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(runs=audits,assets_verified=True,calibration_exact=True,supported_prefix_exact=True,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True)
    run(**vars(p.parse_args()))

"""Audit pinned model provenance and native-arm integration experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
from .real_arms import verify_assets
from .benchmark import fingerprint


def run(assets,reach,pilots,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);manifest=verify_assets(assets);summary=[];hashes={}
    lines=['# E39–E40: native PiPER and Panda integration','',
        f"Source: [Google DeepMind MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/{manifest['commit']}), pinned commit `{manifest['commit']}`. All {len(manifest['files'])} downloaded XML/mesh/license/document files pass SHA-256 verification and were verified against upstream Git blob hashes at download. PiPER license: MIT; Panda: Apache-2.0. The original model folders remain unmodified.", '',
        'The adapter preserves native inertias, collision models, actuators, limits, gripper coupling and gravity compensation. PiPER has six arm joints plus two finger joints and seven actuators; Panda has seven arm joints plus two finger joints and eight actuators. PiPER gripper control is 0–0.035 m; Panda uses its native 0–255 mapping. Both expose normalized openness. PiPER includes body gravity compensation; Panda does not. A 25 Hz control period uses 20 native 2 ms substeps.','',
        'Tool sites are added to the derived scene at PiPER link6 z=0.105 m and Panda hand z=0.1034 m, with tool +Z pointing back from the fingers. These are geometry-based task frames, not hardware-calibrated tool centers.','',
        '## Reaching gate','',
        'Twelve unfiltered Cartesian offsets per model (±XYZ at 4 and 8 cm) from its native home, maintaining initial tool direction. All methods receive the same 32 measured probe steps; up to 200 task steps, with five holds within 1.5 cm and direction difference 0.08, and no contact. Frozen E35 model weights are unchanged.','',
        '| Robot | Controller | Success /12 | Contacts | Mean total steps |','|---|---|---:|---:|---:|']
    for label,folder in [('reach',reach)]+[(Path(p).name,p) for p in pilots]:
        folder=Path(folder);assert json.loads((folder/'status.json').read_text())['status']=='complete'
        protocol=json.loads((folder/'protocol.json').read_text());rows=json.loads((folder/'records.json').read_text())
        for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
        hashes[str(folder/'records.json')]=fingerprint(folder/'records.json')
        if label!='reach':
            screening=json.loads((folder/'screening.json').read_text())
            lines += ['',f'## Manipulation: {label}','',
                f"Base height {protocol.get('base_height',0)} m; compact fixture: {protocol.get('compact',False)}. Endpoint IK accepted {sum(r['accepted'] for r in screening)}/{len(screening)} candidates. All accepted tasks retained regardless of reference success.",'',
                '| Robot | Controller | Success | Grasped | Lifted | Contacts | Mean total steps |','|---|---|---:|---:|---:|---:|---:|']
        for robot in ['piper','panda']:
            for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
                selected=[r for r in rows if r['robot']==robot and r['method']==method]
                if not selected:continue
                key='goal_id' if label=='reach' else 'case'
                assert len({r[key] for r in selected})==len(selected)
                assert all(not r['success'] or not r['collision'] for r in selected)
                if label=='reach':assert all(r['total_steps']==32+r['task_steps'] for r in selected)
                else:assert all(r['steps']==32+len(r['trace']) for r in selected)
                result=dict(suite=label,robot=robot,method=method,count=len(selected),success=sum(r['success'] for r in selected),
                    contacts=sum(r['collision'] for r in selected),mean_steps=float(np.mean([r['total_steps'] if label=='reach' else r['steps'] for r in selected])))
                if label=='reach':lines.append(f"| {robot} | {method} | {result['success']} | {result['contacts']} | {result['mean_steps']:.1f} |")
                else:
                    result.update(grasped=sum(r['grasped'] for r in selected),lifted=sum(r['lifted'] for r in selected))
                    lines.append(f"| {robot} | {method} | {result['success']}/{result['count']} | {result['grasped']} | {result['lifted']} | {result['contacts']} | {result['mean_steps']:.1f} |")
                summary.append(result)
    lines += ['',
        'The initial table stands above the robot base. The second scene mounts both bases at tabletop height. A subsequent structural PiPER grid motivated a separate compact fixture (3 cube positions, 4 cm transfer, approach 0.30 m/lift 0.32 m) after the earlier 0.35 m lift failed IK. These are observed development revisions with different task geometry; do not pool versions or interpret the revisions as learning gains. IK failure does not prove physical impossibility.','',
        'Manipulation uses the native physical grippers and free cube, shared scripted phases, measured pose statistics, and no inference-time weight update. Calibration and forbidden contacts count toward task outcomes; no attachment or teleportation. Privileged Jacobian access is blocked for observation-only evaluation. Successful reaching and isolated manipulation trials do not establish hardware validity, reliable grasping, or superiority over analytical identification. No reserved final-test rollout occurred.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(commit=manifest['commit'],record_hashes=hashes,checkpoint_hashes_unchanged=True,asset_hashes_verified=True,test_evaluated=False),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--reach',required=True)
    p.add_argument('--pilots',nargs='+',required=True);p.add_argument('--output',required=True);run(**vars(p.parse_args()))

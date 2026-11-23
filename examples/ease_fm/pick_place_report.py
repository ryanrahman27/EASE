"""Audit and report all physical arm manipulation pilots without success filtering."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint


def run(training,unseen,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);summary=[];hashes={}
    lines=['# E37–E38: physical arm-driven pick-and-place','',
        'The same frozen E35 motion-estimator weights independently predict Cartesian position and scaled tool-Z-direction motion. A shared 6-row damped control solve regulates XYZ and downward tool direction, leaving yaw free. Grasp/phase decisions use a common scripted sequencer; this is a hybrid controller, not a learned end-to-end manipulation policy. No model retraining occurs.','',
        'Every observation-only trial begins with 32 home-relative joint-probe steps and continues without resetting. Probe interactions and contacts count toward the task. The physical cube is free, held by friction, and released onto the table; no attachment or teleportation is used. Success requires grasp and lift, final placement within 2.5 cm and settled tabletop height/speed, no final finger contact, and no forbidden contact at any time.','',
        'Development revisions: the mounting is flipped so fingers extend away from the terminal link. Full orientation IK failed on both original six-joint families; free yaw restored endpoint feasibility. Initial free-running calibration and low grasp clearance produced near-universal contact failures; V2 uses bounded home-relative joint probing and raises grasp/release waypoints by 1 cm for every controller. Earlier trials are retained.','',
        '| Suite | Controller | Success | Six-joint | Seven-joint | Grasped | Lifted | Forbidden contact | Mean total steps |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for suite,folder in [('Training families',training),('Unseen families',unseen)]:
        folder=Path(folder);assert json.loads((folder/'status.json').read_text())['status']=='complete'
        protocol=json.loads((folder/'protocol.json').read_text())
        for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
        rows=json.loads((folder/'records.json').read_text())+json.loads((folder/'privileged.json').read_text())
        for filename in ['records.json','privileged.json','protocol.json']:hashes[str(folder/filename)]=fingerprint(folder/filename)
        for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
            selected=[r for r in rows if r['method']==method]
            assert len({r['case'] for r in selected})==len(selected)
            for r in selected:
                assert len(r['probe'])==32 and r['steps']==32+len(r['trace'])
                assert all(np.isfinite(t['position_error']) and np.isfinite(t['rotation_error']) for t in r['trace'])
                assert not r['success'] or (r['grasped'] and r['lifted'] and not r['collision'])
            result=dict(suite=suite,method=method,count=len(selected),success=sum(r['success'] for r in selected),
                six_count=sum(r['dof']==6 for r in selected),six=sum(r['success'] and r['dof']==6 for r in selected),
                seven_count=sum(r['dof']==7 for r in selected),seven=sum(r['success'] and r['dof']==7 for r in selected),
                grasped=sum(r['grasped'] for r in selected),lifted=sum(r['lifted'] for r in selected),collisions=sum(r['collision'] for r in selected),
                mean_steps=float(np.mean([r['steps'] for r in selected])))
            summary.append(result)
            lines.append(f"| {suite} | {method} | {result['success']}/{result['count']} | {result['six']}/{result['six_count']} | {result['seven']}/{result['seven_count']} | {result['grasped']} | {result['lifted']} | {result['collisions']} | {result['mean_steps']:.1f} |")
    lines+=['','The training-family suite contains 12 IK-feasible cases across four families (two per joint count). The unseen pilot requested the same three placements on the first two E33 axis families per joint count, chosen by manifest order. IK found all required endpoints for only four of twelve candidates (one six-joint, three seven-joint cases); all four were evaluated, regardless of expert success. Eight IK failures are retained and are not proof of physical impossibility. No final-test family was evaluated.','',
        'This is a small, endpoint-screened development pilot on ideal state observations. Initial joints are placed at an IK-derived approach pose. It does not test arbitrary-start approach planning, vision, learned sequencing, unseen grippers, or broad manipulation transfer. Frozen networks receive measured joints, tool position/direction and goals through the controller, never axis/link metadata or simulator Jacobians. Privileged access is blocked in observation-only rollouts. Neural prediction is tested with both NumPy and Torch solves blocked; the shared 6×6 task-control solve is intentional.','',
        'Next expand structurally feasible placements, inspect contact trajectories and stage failures, and confirm on more unseen six-joint families before drawing a generalization or learning-advantage conclusion.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');(root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(record_hashes=hashes,checkpoint_hashes_unchanged=True,test_evaluated=False),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['training','unseen','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

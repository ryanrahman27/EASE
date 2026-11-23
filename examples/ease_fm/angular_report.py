"""Audit and summarize the E50 contact diagnostic and E51 orientation study."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint


def run(contact,angular,reference):
    contact=Path(contact);angular=Path(angular);reference=Path(reference)
    cr=json.loads((contact/'records.json').read_text());ar=json.loads((angular/'records.json').read_text());old=json.loads((reference/'records.json').read_text())
    cp=json.loads((contact/'protocol.json').read_text());ap=json.loads((angular/'protocol.json').read_text())
    assert len(cr)==48 and len(ar)==120;assert cp['checkpoints']==ap['checkpoints']
    for path,sha in ap['checkpoints'].items():assert fingerprint(path)==sha
    assert cp['cases_sha256']==ap['cases_sha256']==fingerprint(reference/'cases.json')
    for r in cr:
        expected=next(x for x in old if (x['case'],x['method'])==(r['case'],r['method'].replace('neural_','old_')))
        for key in ['trace','phases','probe','steps','success','failure']:assert r[key]==expected[key]
    for r in ar:
        expected=next(x for x in old if (x['case'],x['method'])==(r['case'],r['method'].replace('neural_','old_')))
        assert r['probe']==expected['probe'];assert r['steps']==32+len(r['trace'])
        assert len(r['telemetry'])==len(r['trace'])
        matrix=np.array(r['target']);np.testing.assert_allclose(matrix.T@matrix,np.eye(3),atol=1e-12)
        assert abs(np.linalg.det(matrix)-1)<1e-12
    lines=['# E50: measured contact diagnostic','',cp['scope'],'',
        '| Robot | Method | Case | Success | Failure | Close yaw mismatch (deg) | Final close normal forces (N) |',
        '|---|---|---:|---|---|---:|---|']
    for r in cr:
        close=[d for d in r['telemetry'] if d['phase']=='close']
        yaw=f"{np.degrees(close[-1]['yaw_error']):.2f}" if close else 'not reached'
        force=str([round(f,3) for f in close[-1]['normal_forces']]) if close else 'not reached'
        lines.append(f"| {r['robot']} | {r['method']} | {r['case']} | {r['success']} | {r['failure']} | {yaw} | {force} |")
    lines+=['','All 48 full traces reproduce E46 exactly. Normal forces are instantaneous final-physics-state measurements, not minima over physics substeps. Yaw mismatch is modulo cube symmetry (90 degrees); it is a diagnostic, not a success criterion. Corner-oriented grips can succeed, and face-aligned grips can fail; these observations motivate but do not prove an orientation cause.']
    (contact/'report.md').write_text('\n'.join(lines)+'\n')
    summary=json.loads((angular/'summary.json').read_text());effects=[];eligible=[]
    lines=['# E51: full angular control with frozen E35 estimators','',ap['hypothesis'],'',ap['control'],'',ap['fairness'],'',
        '| Condition | Robot | Privileged /6 | Analytical /6 | Neural seeds 0 / 1 / 2 (each /6) |',
        '|---|---|---:|---:|---|']
    for condition in ['baseline','hold','align']:
        for robot in ['piper','panda']:
            scores=[]
            for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
                if condition=='baseline':
                    score=sum(r['success'] for r in old if r['robot']==robot and r['method']==method.replace('neural_','old_'))
                else:score=next(s['success'] for s in summary if (s['condition'],s['robot'],s['method'])==(condition,robot,method))
                scores.append(score)
            lines.append(f"| {condition} | {robot} | {scores[0]} | {scores[1]} | {' / '.join(map(str,scores[2:]))} |")
    lines+=['','## Costs and failures','', '| Condition | Robot | Method | Collisions | Mean total steps | Failures |', '|---|---|---|---:|---:|---|']
    for s in summary:lines.append(f"| {s['condition']} | {s['robot']} | {s['method']} | {s['collisions']} | {s['mean_steps']:.1f} | {s['failures']} |")
    for condition in ['hold','align']:
        passes=True;improved=False
        for robot in ['piper','panda']:
            for seed in range(3):
                sub=[r for r in ar if (r['condition'],r['robot'],r['method'])==(condition,robot,f'neural_{seed}')]
                pairs=[(r,next(x for x in old if x['method']==f'old_{seed}' and x['case']==r['case'])) for r in sub]
                gained=[r['case'] for r,x in pairs if r['success'] and not x['success']]
                lost=[r['case'] for r,x in pairs if x['success'] and not r['success']]
                collision_change=sum(r['collision']-x['collision'] for r,x in pairs)
                passes &= len(gained)>=len(lost) and collision_change<=0
                improved |= len(gained)>len(lost)
                effects.append(dict(condition=condition,robot=robot,seed=seed,gained=gained,lost=lost,collision_change=collision_change))
        if passes and improved:eligible.append(condition)
    (angular/'effects.json').write_text(json.dumps(effects,indent=2))
    (angular/'decision.json').write_text(json.dumps(dict(promote=False,retention_candidates=eligible,
        reason='Observed development comparison. Require no per-seed/robot aggregate success or collision regression, then separate retention validation before adoption.',test_evaluated=False),indent=2))
    lines+=['','## Interpretation limits','',
        'Hold versus align uses the same angular representation and stricter full-orientation gate, changing the yaw target only. Either versus baseline changes both orientation representation and the controlled objective/gate; it is not a pure representation ablation. Final physical success, force limits, friction and object-release rules are unchanged. All homes remain even when the added orientation goal is hard to reach. The same frozen network predicts translational and angular Jacobians separately; no analytical fallback enters neural prediction.',
        '',f'Variants passing the preliminary per-robot/per-seed success and collision retention screen: {eligible}. No variant is automatically promoted. Full paired gains/losses are in effects.json. These are repeated development tasks, not independent embodiment validation.']
    (angular/'report.md').write_text('\n'.join(lines)+'\n')
    (angular/'report_audit.json').write_text(json.dumps(dict(exact_contact_replays=48,angular_trials=120,original_calibration_matches=120,proper_rotation_targets=True,unchanged_weights=True,test_evaluated=False),indent=2))
    print('\n'.join(lines[:17]))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['contact','angular','reference']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

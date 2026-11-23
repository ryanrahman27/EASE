"""Audit E45 retention attempt and E46 frozen fresh-placement comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .real_arms import verify_assets


def run(assets,training,gate,development,reach,fresh,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False);verify_assets(assets)
    train=Path(training);tp=json.loads((train/'protocol.json').read_text());assert tp['retention'] and tp['steps']==2000
    assert json.loads((train/'status.json').read_text())['status']=='complete'
    assert fingerprint(Path(tp['data'])/'manifest.json')==tp['training_manifest_sha256']
    for path,sha in tp['parents'].items():assert fingerprint(path)==sha
    lines=['# E45–E46: retention-regularized training and fresh native tasks','',
        'E45 starts from all three E35 checkpoints and alternates original short-history synthetic batches with E44 long-history synthetic batches for 2,000 updates per seed. Learning rate is 2e-5. A frozen corresponding E35 teacher penalizes prediction drift, weighted 2 for six joints and 0.5 for seven. No native robot data enters training. The changed learning rate and penalty form a combined intervention.',
        'E46 fixes six new placements per robot before candidate task results, with both transfer directions and no policy-success filtering. All old/new seeds and shared analytical/privileged references are evaluated. These are fresh placements on familiar robot models, not unseen embodiments or the reserved final test.','']
    hashes={};summary=[];audit={}
    lines += ['## Synthetic validation','','| Seed | Before MSE | After MSE |','|---|---:|---:|']
    for seed in range(3):
        folder=train/f'seed_{seed}';meta=json.loads((folder/'training.json').read_text());sha=fingerprint(folder/'policy.pt')
        assert sha==meta['policy_sha256'] and meta['parent_sha256'] in tp['parents'].values()
        assert len(meta['losses'])==meta['steps']==2000 and np.isfinite(meta['losses']).all()
        hashes[seed]=sha;lines.append(f"| {seed} | {meta['validation_initial']:.6f} | {meta['validation_final']:.6f} |")
    ga=json.loads((Path(gate)/'audit.json').read_text());assert ga['manifest_sha256']==tp['training_manifest_sha256']
    lines += ['','## Original training-control gate','','| Seed | Success |','|---|---:|']
    for s in ga['summary']:
        rows=json.loads((Path(gate)/f"seed_{s['seed']}.json").read_text())
        assert s['policy_sha256']==hashes[s['seed']] and s['count']==len(rows)==256
        assert s['success']==sum(r['success'] for r in rows)
        assert fingerprint(Path(gate)/f"seed_{s['seed']}.json")==s['records_sha256']
        lines.append(f"| {s['seed']} | {s['success']}/256 |")
    for title,folder,denominator in [('Observed manipulation',Path(development),9),('Native reaching',Path(reach),12)]:
        rows=json.loads((folder/'records.json').read_text());p=json.loads((folder/'protocol.json').read_text())
        assert json.loads((folder/'status.json').read_text())['status']=='complete' and not p['test_evaluated']
        for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
        lines += ['',f'## {title}','','| Robot | Method | Success | Contacts |','|---|---|---:|---:|']
        for robot in ['piper','panda']:
            methods=sorted({r['method'] for r in rows if r['robot']==robot})
            for method in methods:
                selected=[r for r in rows if r['robot']==robot and r['method']==method];assert len(selected)==denominator
                success=sum(r['success'] for r in selected);contacts=sum(r['collision'] for r in selected)
                lines.append(f'| {robot} | {method} | {success}/{denominator} | {contacts} |')
                summary.append(dict(set=title,robot=robot,method=method,success=success,count=denominator,contacts=contacts))
        audit[title]=dict(count=len(rows),sha256=fingerprint(folder/'records.json'))
    fresh=Path(fresh);fp=json.loads((fresh/'protocol.json').read_text());fe=json.loads((fresh/'evaluation.json').read_text())
    assert json.loads((fresh/'status.json').read_text())['status']=='complete'
    assert fingerprint(fresh/'cases.json')==fe['cases_sha256'] and not fp['test_evaluated']
    for path,sha in fe['checkpoints'].items():assert fingerprint(path)==sha
    cases=json.loads((fresh/'cases.json').read_text());screen=json.loads((fresh/'screening.json').read_text())
    rows=json.loads((fresh/'records.json').read_text());assert len(rows)==8*len(cases)
    methods=['privileged','analytical']+[f'{group}_{s}' for group in ['old','new'] for s in range(3)]
    assert {(r['method'],r['case']) for r in rows}=={(m,i) for m in methods for i in range(len(cases))}
    references={r['case']:r for r in rows if r['method']=='analytical'}
    for row in rows:
        assert row['probe']==references[row['case']]['probe']
        assert row['steps']==len(row['probe'])+len(row['trace'])
        if row['success']:assert not row['collision'] and row['grasped'] and row['lifted'] and row['failure'] is None
    lines += ['','## Fresh placements','',f"Nominal endpoint IK accepts {sum(s['accepted'] for s in screen)}/{len(screen)} candidates. Every accepted case is retained. All calibration traces match across methods.",
        '', '| Robot | Method | Success | Contacts | Failures |','|---|---|---:|---:|---|']
    for robot in ['piper','panda']:
        for method in methods:
            selected=[r for r in rows if r['robot']==robot and r['method']==method];success=sum(r['success'] for r in selected)
            failures=Counter(r['failure'] for r in selected if not r['success']);contacts=sum(r['collision'] for r in selected)
            reasons='; '.join(f'{k}: {v}' for k,v in sorted(failures.items())) or 'none'
            lines.append(f'| {robot} | {method} | {success}/{len(selected)} | {contacts} | {reasons} |')
            summary.append(dict(set='fresh',robot=robot,method=method,success=success,count=len(selected),contacts=contacts,failures=dict(failures)))
    lines += ['','Passing the original training gate is not sufficient to establish native manipulation retention. Report each seed and robot separately; fresh-placement performance does not erase regressions on observed tasks. The fresh set uses the same two robot models, ideal pose observations and scripted task phases. No claim of hardware reliability, unseen-embodiment generalization or analytical superiority is supported by these tests alone.','']
    audit['fresh']=dict(count=len(rows),sha256=fingerprint(fresh/'records.json'),matched_calibration=True)
    audit['training_gate']=ga;audit['test_evaluated']=False
    failures=[]
    for label,baseline in [('Observed manipulation',[9,7,8]),('Native reaching',[9,9,11])]:
        for seed,old_success in enumerate(baseline):
            candidate=next(s for s in summary if s['set']==label and s['robot']=='piper' and s['method']==f'neural_{seed}')
            if candidate['success']<old_success:
                failures.append(dict(task=label,seed=seed,old_success=old_success,new_success=candidate['success']))
    decision=dict(promote=False if failures else None,six_joint_retention_failures=failures,
        note='Retaining all original checkpoints. Nondecreasing observed six-joint manipulation and reaching per seed is a necessary retention condition, not sufficient evidence for promotion.')
    lines += ['## Retention decision','',
        'The candidate fails the necessary six-joint retention condition and is not promoted.' if failures else 'The necessary six-joint retention condition passes; this alone does not authorize promotion.','']
    (root/'report.md').write_text('\n'.join(lines));(root/'summary.json').write_text(json.dumps(summary,indent=2));(root/'audit.json').write_text(json.dumps(audit,indent=2))
    (root/'decision.json').write_text(json.dumps(decision,indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','training','gate','development','reach','fresh','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

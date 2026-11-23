"""Audit and report the matched E56 objective experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .benchmark import fingerprint


def run(training, evaluation, reference):
    root=Path(training);ev=Path(evaluation);ref=Path(reference)
    p=json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    assert json.loads((ev/'status.json').read_text())['status']=='complete'
    for path,sha in p['parents'].items():assert fingerprint(path)==sha
    ep=json.loads((ev/'protocol.json').read_text())
    assert fingerprint(ref/'cases.json')==ep['cases_sha256']
    assert fingerprint(ref/'records.json')==ep['reference_sha256']
    rows=json.loads((ev/'records.json').read_text());assert len(rows)==72
    reuse=json.loads((ev/'reuse.json').read_text())
    reused_source=Path(reuse['source'])/'records.json'
    assert fingerprint(reused_source)==reuse['records_sha256']
    assert [r for r in rows if r['condition']=='prediction']==[r for r in json.loads(reused_source.read_text()) if r['condition']=='prediction']
    reused_gate=json.loads((root/'prediction_reuse.json').read_text())
    assert all(r['identical_checkpoint'] and r['identical_batches'] and r['identical_fit_anchor_losses'] for r in reused_gate['matches'])
    assert len({(r['condition'],r['seed'],r['case']) for r in rows})==72
    old=json.loads((ref/'records.json').read_text())
    lines=['# E56: matched local-action estimator training','',
        'Six final checkpoints: three E35 initializations, each with prediction-only and local-action fine tuning on exactly shared synthetic batches and action contexts. Both use clean-ridge MSE plus the E35 anchor; control adds 0.01 times normalized local residual with gain .35, damping .015, clipping .04, bias compensation and joint limits +/-2.5. Each receives 2,000 updates at learning rate 2e-5, batch 128. No native data or geometry enters training. The state-batch distribution and RNG match E52; action contexts use a separate RNG. Native evaluation uses the original tool-Z controller.',
        '', 'The local residual is divided by squared task error plus .002^2, then averaged. Synthetic errors use J*dq; bias and joint positions are independently sampled. This models clipping and approximate bias compensation, but not evolving kinematics, contact or exact actuator dynamics. Score validation does not guarantee optimization improves control. The .01 coefficient is fixed, with no sweep.',
        '', '## Fixed synthetic validation', '',
        '| Seed | Condition | Initial prediction MSE | Final prediction MSE | Initial normalized action loss | Final normalized action loss |',
        '|---|---|---:|---:|---:|---:|']
    gates={};hashes={}
    for seed in range(3):
        metas=[]
        for condition in ['prediction','control']:
            folder=root/condition/f'seed_{seed}';meta=json.loads((folder/'training.json').read_text());metas.append(meta)
            assert meta['steps']==len(meta['losses'])==2000
            assert fingerprint(folder/'policy.pt')==meta['policy_sha256']
            hashes[f'{condition}_{seed}']=meta['policy_sha256']
            a,b=meta['validation_initial'],meta['validation_final']
            lines.append(f'| {seed} | {condition} | {a[0]:.6f} | {b[0]:.6f} | {a[2]:.4f} | {b[2]:.4f} |')
        assert metas[0]['batch_sha256']==metas[1]['batch_sha256']
        assert metas[0]['context_sha256']==metas[1]['context_sha256']
        assert metas[0]['parent_sha256']==metas[1]['parent_sha256']
    lines+=['','Prediction-only checkpoints, state-batch hashes and fit/anchor loss traces exactly reproduce E52 for all three seeds. Its 768 training-gate episodes and 36 native trials are reused with source/hash checks, not counted as new independent evaluations. The local-action arm receives 768 new training-gate episodes and 36 new native trials.','', '## Original training-control gate','','| Condition | Seed | Success /256 |','|---|---:|---:|']
    gate_manifest=None
    for condition in ['prediction','control']:
        audit=json.loads((root/f'gate_{condition}'/'audit.json').read_text())
        if gate_manifest is None:gate_manifest=audit['manifest_sha256']
        assert audit['manifest_sha256']==gate_manifest
        for x in audit['summary']:
            assert x['policy_sha256']==hashes[f"{condition}_{x['seed']}"]
            rr=root/f'gate_{condition}'/f"seed_{x['seed']}.json"
            assert fingerprint(rr)==x['records_sha256']
            records=json.loads(rr.read_text());assert len(records)==x['count']==256
            assert sum(r['success'] for r in records)==x['success']
            gates[(condition,x['seed'])]=x['success']==256
            lines.append(f"| {condition} | {x['seed']} | {x['success']} |")
    lines+=['','## Observed native manipulation','','| Condition | Robot | Seed 0 /6 | Seed 1 /6 | Seed 2 /6 | Collisions across seeds |','|---|---|---:|---:|---:|---:|']
    for condition in ['original','prediction','control']:
        for robot in ['piper','panda']:
            rr=[r for r in old if r['robot']==robot and r['method'].startswith('old_')] if condition=='original' else [r for r in rows if r['condition']==condition and r['robot']==robot]
            scores=[sum(r['success'] for r in rr if r['method']==(f'old_{s}' if condition=='original' else f'neural_{s}')) for s in range(3)]
            lines.append(f'| {condition} | {robot} | '+ ' | '.join(map(str,scores))+f" | {sum(r['collision'] for r in rr)} |")
    lines+=['','Analytical reference solves 6/6 on each robot. Original and analytical scores reuse the E46 records, already exactly replayed in E50; they are not new independent trials. All twelve placements are observed development cases.','', '## Retention and paired changes','']
    effects=[];eligible=[]
    for condition in ['prediction','control']:
        passes=all(gates[(condition,s)] for s in range(3));improved=False
        for robot in ['piper','panda']:
            for seed in range(3):
                rr=[r for r in rows if (r['condition'],r['robot'],r['seed'])==(condition,robot,seed)]
                gained=[];lost=[];collision_change=0
                for r in rr:
                    b=next(x for x in old if x['method']==f'old_{seed}' and x['case']==r['case'])
                    assert r['probe']==b['probe']
                    if r['success'] and not b['success']:gained.append(r['case'])
                    if b['success'] and not r['success']:lost.append(r['case'])
                    collision_change+=r['collision']-b['collision']
                passes &= len(gained)>=len(lost) and collision_change<=0
                improved |= len(gained)>len(lost)
                effects.append(dict(condition=condition,robot=robot,seed=seed,gained=gained,lost=lost,collision_change=collision_change))
        if passes and improved:eligible.append(condition)
    lines += [f'Candidates passing preliminary training/success/collision retention: {eligible}. No automatic promotion; any candidate still needs native reaching and broader retention, then independent validation.', '', '## Failures and interaction cost', '', '| Condition | Failures | Mean steps, including calibration |','|---|---|---:|']
    for condition in ['prediction','control']:
        rr=[r for r in rows if r['condition']==condition]
        lines.append(f"| {condition} | {dict(Counter(r['failure'] for r in rr if not r['success']))} | {sum(r['steps'] for r in rr)/len(rr):.1f} |")
    lines+=['','## Direct matched objective comparison','','| Robot | Seed | Control gains over prediction | Control losses versus prediction |','|---|---:|---|---|']
    for robot in ['piper','panda']:
        for seed in range(3):
            gained=[];lost=[]
            for a in [r for r in rows if (r['condition'],r['robot'],r['seed'])==('control',robot,seed)]:
                b=next(r for r in rows if (r['condition'],r['seed'],r['case'])==('prediction',seed,a['case']))
                if a['success'] and not b['success']:gained.append(a['case'])
                if b['success'] and not a['success']:lost.append(a['case'])
            lines.append(f'| {robot} | {seed} | {gained} | {lost} |')
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'effects.json').write_text(json.dumps(effects,indent=2))
    (root/'decision.json').write_text(json.dumps(dict(promote=False,retention_candidates=eligible,test_evaluated=False),indent=2))
    (root/'report_audit.json').write_text(json.dumps(dict(paired_batch_hashes_match=True,paired_context_hashes_match=True,unchanged_parents=True,new_training_gate_episodes=768,reused_training_gate_episodes=768,new_native_trials=36,reused_native_trials=36,calibration_matches=72,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['training','evaluation','reference']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

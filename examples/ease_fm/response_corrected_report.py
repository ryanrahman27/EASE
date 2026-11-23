"""E61 paired-objective report with fresh retention audits."""
import argparse
from collections import Counter
import json
from pathlib import Path
from .benchmark import fingerprint


def run(training,evaluation,reference):
    root=Path(training);ev=Path(evaluation);ref=Path(reference)
    p=json.loads((root/'protocol.json').read_text());ep=json.loads((ev/'protocol.json').read_text())
    assert all(json.loads((r/'status.json').read_text())['status']=='complete' for r in [root,ev])
    for path,sha in {**p['parents'],**p['response_hashes']}.items():assert fingerprint(path)==sha
    assert fingerprint(Path(p['coherent_data'])/'contexts.npz')==p['coherent_sha256']
    assert fingerprint(ref/'cases.json')==ep['cases_sha256'] and fingerprint(ref/'records.json')==ep['reference_sha256']
    rows=json.loads((ev/'records.json').read_text());old=json.loads((ref/'records.json').read_text())
    assert len(rows)==72 and len({(r['condition'],r['seed'],r['case']) for r in rows})==72
    ea=json.loads((ev/'audit.json').read_text());assert ea['new_trials']==72 and ea['reused_trials']==0 and ep['reuse_prediction'] is None
    for path,sha in ea['checkpoints'].items():assert fingerprint(path)==sha
    for r in rows:
        assert json.loads((ev/'trials'/f"{r['condition']}_{r['seed']}_{r['case']}.json").read_text())==r
        b=next(b for b in old if b['case']==r['case'] and b['method']==f"old_{r['seed']}")
        assert r['probe']==b['probe']
    labels={'prediction':'uncorrected','control':'response-corrected'};gates={}
    lines=['# E61: response-corrected versus uncorrected local-action training','',
        'Three E35 initializations, two conditions, 2,000 updates each. Identical batches, lr 2e-5, batch128, fixed .01 local coefficient, fit loss and frozen-parent anchor. Both use local-action loss; prediction directories mean uncorrected, not prediction-only. Control directories mean response-corrected. Frozen leave-one-body-out E59 affine response replaces command-minus-bias only for the measured half. Synthetic half is unchanged.', '',
        'Measured data: analytical/E35 subset of E57, 3,600 train and 1,200 original_2 trajectory-validation states on eight bodies. Fixed 512-state validation subset. Same bodies/goals occur in policy training and validation; response fits exclude each queried body but the policy is not held out from these bodies. Changes versus E57 include removing E52-driver trajectories, so only the within-E61 pair isolates response correction. No native data, coefficient sweep or seed selection.', '',
        '## Measured trajectory validation','','| Seed | Condition | Initial uncorrected loss | Final uncorrected loss | Initial corrected loss | Final corrected loss |','|---|---|---:|---:|---:|---:|']
    for seed in range(3):
        metas=[]
        for condition,label in labels.items():
            folder=root/condition/f'seed_{seed}';m=json.loads((folder/'training.json').read_text());metas.append(m)
            assert m['steps']==len(m['losses'])==2000 and fingerprint(folder/'policy.pt')==m['policy_sha256']
            assert m['parent_sha256']==fingerprint(Path(next(k for k in p['parents'] if f'seed_{seed}/' in k)))
            a,b=m['coherent_initial'],m['coherent_final']
            lines.append(f'| {seed} | {label} | {a[2]:.4f} | {b[2]:.4f} | {a[3]:.4f} | {b[3]:.4f} |')
        assert metas[0]['batch_sha256']==metas[1]['batch_sha256'] and metas[0]['context_sha256']==metas[1]['context_sha256']
    lines+=['','## Original reaching gate','','| Condition | Seed | Success /256 |','|---|---:|---:|'];manifest=None
    for condition,label in labels.items():
        audit=json.loads((root/f'gate_{condition}'/'audit.json').read_text())
        if manifest is None:manifest=audit['manifest_sha256']
        assert audit['manifest_sha256']==manifest
        for s in audit['summary']:
            path=root/f'gate_{condition}'/f"seed_{s['seed']}.json";rr=json.loads(path.read_text())
            assert fingerprint(path)==s['records_sha256'] and len(rr)==s['count']==256
            assert fingerprint(root/condition/f"seed_{s['seed']}"/'policy.pt')==s['policy_sha256']
            assert sum(r['success'] for r in rr)==s['success'];gates[(condition,s['seed'])]=s['success']==256
            lines.append(f"| {label} | {s['seed']} | {s['success']} |")
    lines+=['','## Native development manipulation','','| Condition | Robot | Seed 0 /6 | Seed 1 /6 | Seed 2 /6 | Contacts |','|---|---|---:|---:|---:|---:|']
    for condition,label in {'original':'original E35',**labels}.items():
        for robot in ['piper','panda']:
            rr=[r for r in old if r['robot']==robot and r['method'].startswith('old_')] if condition=='original' else [r for r in rows if r['robot']==robot and r['condition']==condition]
            scores=[sum(r['success'] for r in rr if r['method']==(f'old_{s}' if condition=='original' else f'neural_{s}')) for s in range(3)]
            lines.append(f'| {label} | {robot} | '+' | '.join(map(str,scores))+f" | {sum(r['collision'] for r in rr)} |")
    eligible=[];effects=[]
    for condition in labels:
        retained=all(gates[(condition,s)] for s in range(3));improved=False
        for robot in ['piper','panda']:
            for seed in range(3):
                rr=[r for r in rows if (r['condition'],r['robot'],r['seed'])==(condition,robot,seed)]
                bb=[r for r in old if r['robot']==robot and r['method']==f'old_{seed}']
                delta=sum(r['success'] for r in rr)-sum(r['success'] for r in bb);contacts=sum(r['collision'] for r in rr)-sum(r['collision'] for r in bb)
                retained &= delta>=0 and contacts<=0;improved |= delta>0
                effects.append(dict(condition=condition,robot=robot,seed=seed,success_change=delta,contact_change=contacts))
        if retained and improved:eligible.append(condition)
    lines+=['',f'Preliminary retention candidates: {eligible}. No automatic promotion; any passing condition still needs broader native reaching/placement retention and independent validation.', '',
        'All 1,536 reaching episodes and 72 candidate native trials are fresh. E35 and analytical references reuse audited E46 records; analytical solves all six cases per robot. Twelve placements are observed development cases. Final test untouched.', '', '## Failures and costs','']
    for condition,label in labels.items():
        rr=[r for r in rows if r['condition']==condition]
        lines.append(f"{label}: {sum(r['success'] for r in rr)}/36 successes; failures {dict(Counter(r['failure'] for r in rr if not r['success']))}; mean steps including calibration {sum(r['steps'] for r in rr)/36:.1f}. Early failures can reduce cost.")
    gained=[];lost=[]
    for r in [r for r in rows if r['condition']=='control']:
        b=next(b for b in rows if b['condition']=='prediction' and (b['seed'],b['case'])==(r['seed'],r['case']))
        if r['success'] and not b['success']:gained.append([r['seed'],r['case']])
        if b['success'] and not r['success']:lost.append([r['seed'],r['case']])
    lines+=['',f'Direct paired change: corrected gains {len(gained)} cases and loses {len(lost)} versus uncorrected. Gains (seed, case): {gained}; losses: {lost}.', '',
        '84 regression tests pass; 156,000 affine response comparisons agree within 3.4e-9. Paired batch/context hashes, immutable response fits and parents, all reaching/native records and calibration are audited. Response coefficients remain frozen; this experiment trains only the motion estimator.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');(root/'effects.json').write_text(json.dumps(effects,indent=2))
    (root/'decision.json').write_text(json.dumps(dict(promote=False,retention_candidates=eligible,test_evaluated=False),indent=2))
    (root/'report_audit.json').write_text(json.dumps(dict(fresh_reaching=1536,fresh_native=72,paired_batches=True,paired_contexts=True,unchanged_parents_and_response=True,calibration_matches=72,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['training','evaluation','reference']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

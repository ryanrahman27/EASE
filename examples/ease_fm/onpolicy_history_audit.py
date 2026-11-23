"""Verify data separation and unchanged collection prefixes before fitting E101."""
import json
import shutil
from pathlib import Path
import numpy as np
from .onpolicy_history import ROOT,TASKS,e99,prior
from .benchmark import fingerprint


def run():
    spec=json.loads((ROOT/'protocol.json').read_text());audit=json.loads((ROOT/'data_audit.json').read_text())
    assert fingerprint(ROOT/'task_contexts.npz')==audit['contexts_sha256']
    assert fingerprint(ROOT/'provenance.json')==audit['provenance_sha256']
    tasks=json.loads((TASKS/'cases.json').read_text());arms=json.loads(Path('outputs/ease_paired_factorial_v2/manifest.json').read_text())['arms']
    for t in tasks:assert t['arm']==next(a for a in arms if a['seed']==t['arm']['seed'])
    with np.load(ROOT/'task_contexts.npz') as f:validation=f['validation'];eligible=f['eligible']
    prov=json.loads((ROOT/'provenance.json').read_text());splits=[set(),set()]
    for i,r in enumerate(prov):
        assert bool(validation[i])==r['validation']==(r['case']%3==2)
        assert bool(eligible[i])==r['eligible'];splits[int(validation[i])].add((r['case'],r['driver']))
    assert not splits[0]&splits[1] and len(splits[0])==24 and len(splits[1])==12
    outcomes={}
    for split,val in [('train',False),('validation',True)]:
        rr=[json.loads(p.read_text()) for p in (ROOT/'training_trials').glob('*.json')];rr=[r for r in rr if r['validation']==val]
        outcomes[split]=dict(n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr))
        for r in rr:
            old=json.loads((e99.ROOT/'training_trials'/f"{r['case']}_{r['driver']}.json").read_text())
            assert r['calibration']==old['calibration']
            prefix=lambda x:[t for t in x['trace'] if t['phase'] in ('approach','descend','probe')]
            assert prefix(r)==prefix(old)
    hashes={}
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copy2(p,ROOT/'source'/p.name);hashes[p.name]=fingerprint(p)
    (ROOT/'source/dependency_hashes.json').write_text(json.dumps(hashes,indent=2))
    result=dict(exact_collection_prefixes=36,exact_manifest_arms=True,training_trajectories=24,validation_trajectories=12,disjoint_placements=True,outcomes=outcomes,test_evaluated=False)
    (ROOT/'collection_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':run()

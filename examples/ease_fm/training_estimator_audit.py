"""Audit row provenance and numerical targets before training."""
import json
import numpy as np
import torch
from .training_estimator_repair import ROOT
from .estimator_distillation import analytical_targets
from .benchmark import fingerprint

f=np.load(ROOT/'contexts.npz');rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==48
ranges=[];start=0
for r in rows:
    end=start+r['steps']-32;assert np.all(f['case'][start:end]==r['case'])
    ranges.append(dict(start=start,end_exclusive=end,case=r['case'],driver=r['method'],body=r['body'],dof=r['dof'],validation=r['case']%3==2));start=end
assert start==len(f['g'])
assert set(f['case'][~f['validation']]).isdisjoint(set(f['case'][f['validation']]))
ids=np.linspace(0,start-1,min(start,256),dtype=int)
g=torch.tensor(f['g'][ids]);c=torch.tensor(f['c'][ids]);mask=torch.tensor(f['mask'][ids]);target=analytical_targets(g,c,mask)
error=0.
for k,i in enumerate(ids):
    n=int(f['mask'][i].sum());expected=np.linalg.solve(f['g'][i,:n,:n].astype(float),f['c'][i,:n].astype(float));actual=target[k,:n].numpy()
    np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=1e-6);error=max(error,float(np.max(np.abs(actual-expected))))
(ROOT/'context_provenance.json').write_text(json.dumps(ranges,indent=2))
p=ROOT/'data_audit.json';a=json.loads(p.read_text());a.update(exact_row_coverage=True,placement_split_disjoint=True,independent_target_checks=len(ids),max_target_error=error,provenance_sha256=fingerprint(ROOT/'context_provenance.json'));p.write_text(json.dumps(a,indent=2));print(json.dumps(a,indent=2))

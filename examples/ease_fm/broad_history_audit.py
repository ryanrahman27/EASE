import json,shutil
from pathlib import Path
import numpy as np
import torch
from .broad_history_repair import ROOT,OLD,analytical_targets
torch.set_num_threads(4)
f=np.load(ROOT/'contexts.npz');old=np.load(OLD/'contexts.npz');audit=json.loads((ROOT/'data_audit.json').read_text());tasks=json.loads((ROOT/'cases.json').read_text())
for k in f.files:np.testing.assert_array_equal(f[k][:len(old['g'])],old[k])
offset=len(old['g']);ranges=json.loads((ROOT/'new_context_provenance.json').read_text());end=0
for r in ranges:
    assert r['start']==end;end=r['end_exclusive'];a=offset+r['start'];b=offset+end
    assert np.all(f['case'][a:b]==r['case']) and np.all(f['validation'][a:b]==r['validation'])
    full=json.loads((ROOT/'trials'/f"{r['case']}_{r['driver']}_frozen_bias.json").read_text());assert b-a==len(full['trace'])
assert offset+end==len(f['g'])
for i,c in enumerate(f['case']):assert bool(f['validation'][i])==tasks[int(c)]['validation']
valid=f['clean']&~f['validation'];special=np.isin(f['phase'],['descend','lower']);pools=[valid&(f['case']<12),valid&(f['case']>=12)&special,valid&(f['case']>=12)&~special]
assert all(p.any() and not f['validation'][p].any() for p in pools)
ids=np.linspace(0,len(f['g'])-1,256,dtype=int);target=analytical_targets(torch.tensor(f['g'][ids]),torch.tensor(f['c'][ids]),torch.tensor(f['mask'][ids])).numpy();maxerr=0.
for j,i in enumerate(ids):
    n=int(f['mask'][i].sum());expected=np.linalg.solve(f['g'][i,:n,:n].astype(float),f['c'][i,:n].astype(float));np.testing.assert_allclose(target[j,:n],expected,rtol=1e-5,atol=1e-6);maxerr=max(maxerr,float(np.max(np.abs(target[j,:n]-expected))))
audit.update(original_contexts_exact=True,new_row_provenance_exact=True,all_pool_rows_fitting_only=True,pool_sizes=[int(p.sum()) for p in pools],independent_targets=256,max_target_error=maxerr)
(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps(audit,indent=2))

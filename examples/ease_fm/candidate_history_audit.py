import json,shutil
from pathlib import Path
import numpy as np
import torch
from .candidate_history_repair import ROOT,OLD,analytical_targets
torch.set_num_threads(4)
with np.load(ROOT/'contexts.npz') as archive:f={k:archive[k] for k in archive.files}
with np.load(OLD/'contexts.npz') as archive:old={k:archive[k] for k in archive.files}
offset=len(old['g']);audit=json.loads((ROOT/'data_audit.json').read_text());provenance=json.loads((ROOT/'context_provenance.json').read_text());tasks=json.loads((ROOT/'cases.json').read_text())
for k in old:np.testing.assert_array_equal(f[k][:offset],old[k])
assert np.all(f['driver'][:offset]==-1);end=0
for p in provenance:
    assert p['start']==end and not tasks[p['case']]['validation'];end=p['end_exclusive'];a=offset+p['start'];b=offset+end
    assert np.all(f['case'][a:b]==p['case']) and np.all(f['driver'][a:b]==p['seed']) and not f['validation'][a:b].any()
assert offset+end==len(f['g']) and len(audit['exact_candidate_replays'])==48
rows=json.loads((ROOT/'records.json').read_text());lookup={(r['case'],r['method']):r for r in rows};pools={}
for seed in range(3):
    own=f['clean']&~f['validation']&(f['driver']==seed);success=np.array([lookup[int(c),f'reference_{seed}']['success'] for c in f['case']]);pools[str(seed)]=dict(failed_rows=int(np.sum(own&~success)),successful_rows=int(np.sum(own&success)),failing_fitting_cases=sorted({int(c) for c in f['case'][own&~success]}));assert pools[str(seed)]['failed_rows'] and pools[str(seed)]['successful_rows']
ids=np.linspace(0,len(f['g'])-1,256,dtype=int);target=analytical_targets(torch.tensor(f['g'][ids]),torch.tensor(f['c'][ids]),torch.tensor(f['mask'][ids])).numpy();err=0.
for j,i in enumerate(ids):
    n=int(f['mask'][i].sum());wanted=np.linalg.solve(f['g'][i,:n,:n].astype(float),f['c'][i,:n].astype(float));np.testing.assert_allclose(target[j,:n],wanted,rtol=1e-5,atol=1e-6);err=max(err,float(np.max(np.abs(target[j,:n]-wanted))))
audit.update(original_contexts_exact=True,driver_provenance_exact=True,all_new_histories_fitting_only=True,sampling_pools=pools,independent_ridge_checks=256,max_target_error=err)
(ROOT/'data_audit.json').write_text(json.dumps(audit,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps({k:v for k,v in audit.items() if k!='exact_candidate_replays'},indent=2))

import json,shutil
from pathlib import Path
import numpy as np
from .candidate_history_repair import ROOT,fingerprint
with np.load(ROOT/'contexts.npz') as archive:f={k:archive[k] for k in archive.files}
checks=[]
for seed in range(3):
    p=ROOT/f'targets_seed_{seed}.npz'
    with np.load(p) as archive:t={k:archive[k] for k in archive.files}
    ids=t['row_ids'];assert f['clean'][ids].all() and not f['validation'][ids].any();assert np.isfinite(t['matrix']).all() and np.isfinite(t['action']).all()
    maxerr=0.
    for k in np.linspace(0,len(ids)-1,128,dtype=int):
        i=ids[k];n=int(f['mask'][i].sum());j=t['matrix'][k,:n].T.astype(float);wanted=np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*f['error'][i].astype(float))+f['bias'][i,:n],-.04,.04)
        np.testing.assert_allclose(t['action'][k,:n],wanted,rtol=1e-4,atol=1e-5);assert np.all(t['action'][k,n:]==0);maxerr=max(maxerr,float(np.max(np.abs(t['action'][k,:n]-wanted))))
    checks.append(dict(seed=seed,rows=len(ids),protected=int(t['protected'].sum()),numpy_action_checks=128,max_action_error=maxerr,targets_sha256=fingerprint(p)))
(ROOT/'target_audit.json').write_text(json.dumps(checks,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(json.dumps(checks,indent=2))

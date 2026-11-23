"""Independent new-condition replay and artifact completion."""
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .frozen_bias_core import ROOT,load,evaluate
from .geometry_filter import GeometryArm
from .benchmark import fingerprint

def run():
    torch.set_num_threads(4);tasks,models,hashes=load();rows=json.loads((ROOT/'records.json').read_text())
    selected=[(0,m) for m in models]
    for dof in (6,7):
        for outcome in (False,True):
            candidates=[r for r in rows if r['mode']=='frozen_bias' and r['dof']==dof and r['success']==outcome]
            if candidates:selected.append((candidates[0]['case'],candidates[0]['method']))
    checked=[]
    with patch.object(GeometryArm,'clearance',side_effect=RuntimeError('No geometry access')),patch.object(GeometryArm,'select_action',side_effect=RuntimeError('No selector')):
        for case,method in dict.fromkeys(selected):
            states=[]
            def capture(env):states.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
            r=evaluate(case,tasks[case],method,models[method],'frozen_bias',capture=capture)
            stem=f'{case}_{method}_frozen_bias';assert json.loads(json.dumps(r))==json.loads((ROOT/'trials'/f'{stem}.json').read_text())
            np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);checked.append(stem);print('exact',stem,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    v=json.loads((ROOT/'verification.json').read_text());v['exact_new_replays']=checked;v['selector_geometry_access_guarded']=True
    (ROOT/'verification.json').write_text(json.dumps(v,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def finish():
    v=json.loads((ROOT/'verification.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text());assert v['exact_new_replays']
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    p=ROOT/'report.md';s=p.read_text();s+=f"\nFinal audit: {len(v['exact_old_controls'])} exact old-control replays and {len(v['exact_new_replays'])} exact new-condition replays; {len(m['videos'])} trace-verified success/failure videos saved.\n";p.write_text(s)
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',new_episodes=64,reused_episodes=192,videos=len(m['videos']),exact_new_replays=len(v['exact_new_replays']),test_evaluated=False),indent=2))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
    print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    if len(sys.argv)>1:finish()
    else:run()

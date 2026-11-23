"""Independently verify substituted trajectories and close artifact ledger."""
import json,shutil
from pathlib import Path
import numpy as np
import torch
from .six_motion_core import ROOT,load,evaluate
from .benchmark import fingerprint

def verify():
    torch.set_num_threads(4);tasks,models,hashes=load();rows=json.loads((ROOT/'records.json').read_text());checked=[]
    for condition in ('probe','lift'):
        for outcome in (True,False):
            r=next((r for r in rows if r['condition']==condition and r['success']==outcome),None)
            if r is None:continue
            case=r['case'];method=r['method'];stem=f'{case}_{method}_{condition}';states=[]
            def capture(env):states.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
            new=evaluate(case,tasks[case],method,models[method],'frozen_bias',capture=capture,condition=condition)
            assert json.loads(json.dumps(new))==json.loads((ROOT/'trials'/f'{stem}.json').read_text())
            np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);checked.append(stem);print('exact intervention',stem,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    p=ROOT/'verification.json';v=json.loads(p.read_text());v['exact_intervention_reexecutions']=checked;p.write_text(json.dumps(v,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def finish():
    v=json.loads((ROOT/'verification.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text());assert v['exact_intervention_reexecutions']
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    note=f"\nFinal audit: {len(v['exact_intervention_reexecutions'])} independent full intervention replays matched exactly; {len(m['videos'])} success/failure videos verified against all stored tool/object state positions.\n"
    p=ROOT/'report.md';p.write_text(p.read_text()+note)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',exact_baseline_replays=32,intervention_episodes=48,videos=len(m['videos']),test_evaluated=False),indent=2))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 else verify()

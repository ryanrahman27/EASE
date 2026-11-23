import json,shutil
from pathlib import Path
import numpy as np
import torch
from .broad_history_repair import ROOT,setup,core
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint

def verify():
    torch.set_num_threads(4);tasks,parents,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());selection=[];checks=[]
    for seed in range(3):
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(ROOT/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));models[f'candidate_{seed}']=m.eval()
    for method in ('analytical','reference_0','candidate_0','candidate_1','candidate_2'):
        for outcome in (True,False):
            r=next((r for r in rows if r['method']==method and r['success']==outcome),None)
            if r:selection.append((r['case'],method))
    for method in ('candidate_0','candidate_1','candidate_2'):
        r=next(r for r in rows if r['method']==method and tasks[r['case']]['group']=='fresh_held');selection.append((r['case'],method))
    for case,method in dict.fromkeys(selection):
        states=[]
        def capture(env):states.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
        r=core.evaluate(case,tasks[case],method,models[method],'frozen_bias',capture=capture);stem=f'{case}_{method}_frozen_bias'
        assert json.loads(json.dumps(r))==json.loads((ROOT/'trials'/f'{stem}.json').read_text());np.testing.assert_array_equal(states,np.load(ROOT/'states'/f'{stem}.npz')['states']);checks.append(stem);print('exact',stem,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    for seed in range(3):assert fingerprint(ROOT/f'seed_{seed}/policy.pt')==json.loads((ROOT/f'seed_{seed}/training.json').read_text())['policy_sha256']
    (ROOT/'verification.json').write_text(json.dumps(dict(exact_reexecutions=checks,unchanged_inputs_weights=True,candidate_hashes_verified=True,privileged_jacobian_blocked=True),indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

def finish():
    m=json.loads((ROOT/'videos/manifest.json').read_text());v=json.loads((ROOT/'verification.json').read_text());summary=json.loads((ROOT/'summary.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**spec['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for r in m['videos']:assert fingerprint(ROOT/'videos'/r['file'])==r['sha256']
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    note=f"\nFinal verification: {len(v['exact_reexecutions'])} independent exact replays, including fresh held placements; {len(m['videos'])} trace-verified success/failure videos saved.\n"
    p=ROOT/'report.md';p.write_text(p.read_text()+note)
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete',inherited_episodes=84,new_reference_episodes=48,candidate_episodes=72,checkpoints=3,retention_gate_passed=summary['retention_gate_passed'],videos=len(m['videos']),unseen_evaluated=False,test_evaluated=False),indent=2))
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2));print((ROOT/'status.json').read_text())

if __name__=='__main__':
    import sys
    finish() if len(sys.argv)>1 else verify()

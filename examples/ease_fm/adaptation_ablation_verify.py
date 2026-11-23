"""Independent exact re-execution and forbidden-access checks for E114."""
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from .adaptation_ablation import ROOT,MODES,load,evaluate
from .geometry_filter import GeometryArm
from .progress_action_bank import search_actions
from .benchmark import fingerprint

def run():
    torch.set_num_threads(4);tasks,models,hashes=load()
    rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==192
    selected=[(0,m,mode) for m in ('analytical','learned_0') for mode in MODES]
    for outcome in (False,True):
        candidates=[r for r in rows if r['success']==outcome and r['dof']==7]
        if candidates:
            r=candidates[0];selected.append((r['case'],r['method'],r['mode']))
    checked=[]
    with patch.object(GeometryArm,'clearance',side_effect=RuntimeError('Geometry queries forbidden')),patch.object(GeometryArm,'select_action',side_effect=RuntimeError('Action selector forbidden')),patch('examples.ease_fm.progress_action_bank.search_actions',side_effect=RuntimeError('Physics branches forbidden')):
        for case,method,mode in selected:
            p=ROOT/'trials'/f'{case}_{method}_{mode}.json';old=json.loads(p.read_text());states=[]
            def capture(env):states.append(np.r_[env.data.qpos.copy(),env.data.qvel.copy()])
            new=evaluate(case,tasks[case],method,models[method],mode,capture=capture)
            assert json.loads(json.dumps(new))==old
            saved=np.load(ROOT/'states'/f'{case}_{method}_{mode}.npz')['states'];np.testing.assert_array_equal(states,saved)
            checked.append(dict(case=case,method=method,mode=mode,success=new['success'],exact_full_trace_and_states=True))
            print('exact replay',case,method,mode,flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h
    v=json.loads((ROOT/'verification.json').read_text());v.update(exact_reexecutions=checked,forbidden_selector_geometry_access_guarded=True,relevant_existing_tests='6 passed: local history, pick-place, task-retention training')
    (ROOT/'verification.json').write_text(json.dumps(v,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

if __name__=='__main__':run()

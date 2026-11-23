"""Finalize E125 workspace coverage before any new collection occurs."""
import json,shutil
from pathlib import Path
import numpy as np
from .broad_task_value_gate import ROOT,COLLECT,OLD

def run():
    assert not list(COLLECT.glob('records_seed_*.json'))
    history=ROOT/'design_history';history.mkdir(exist_ok=False)
    for name in ('cases.json','protocol.json','placement_audit.json','placement_coverage.png'):shutil.copy2(ROOT/name,history/name)
    tasks=json.loads((OLD/'cases.json').read_text());rng=np.random.default_rng(125000);expanded=[]
    for case,t in enumerate(tasks):
        if t['validation']:continue
        for variant in range(2):
            new=json.loads(json.dumps(t));xy=rng.uniform([.275,-.10],[.425,.10]);delta=xy-np.array(t['xy']);target_delta=delta+rng.uniform([-.020,-.020],[.020,.020]);new['xy']=xy.tolist();new['target']=(np.array(t['target'])+target_delta).tolist();new.update(validation=False,group='expanded_train',source_case=case,variant=variant,object_offset=delta.tolist(),target_offset=target_delta.tolist());expanded.append(new)
    tasks+=expanded;assert len(tasks)==56;keys=[(t['arm']['seed'],*np.round(t['xy'],12),*np.round(t['target'],12)) for t in tasks];assert len(set(keys))==56
    (ROOT/'cases.json').write_text(json.dumps(tasks,indent=2));p=json.loads((ROOT/'protocol.json').read_text());p['placements']='Object XY uniform in[0.275,0.425] x[-0.10,0.10] meters; destination preserves source displacement plus independent +/-20mm jitter. Two variants per16 fitting source cases. Same home and new calibration seeds. Geometry coverage revised after plotting the existing split, before collecting any new outcomes. No held labels, outcome or IK screening. Exact held cases unchanged/excluded from training; negative-Y spatial region is now included in fitting support.';(ROOT/'protocol.json').write_text(json.dumps(p,indent=2))
    (history/'README.txt').write_text('Initial local-jitter catalog was replaced before collecting any new episode. Coverage plot exposed a missing negative-Y region. Final workspace sampling uses no held outcomes; exact held placements remain excluded. Archived preliminary placements were never used for fitting.\n')
    for name in ('broad_task_value_gate.py','build_broad_task_value_gate.py','broaden_gate_workspace.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)

if __name__=='__main__':run()

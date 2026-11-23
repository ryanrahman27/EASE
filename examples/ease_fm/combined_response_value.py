"""E136: conjunction of structured local response and compact task value."""
import shutil
from pathlib import Path
import numpy as np
from .short_horizon_paired import data,score
from .structured_response import ROOT as MOTION
from .compact_response_value import ROOT as VALUE
from .abstaining_task_value_gate import read,save,fingerprint

ROOT=Path('outputs/ease_combined_response_value_v1')
MOTION_MARGINS=(5e-6,1e-5,2.5e-5,5e-5,1e-4)
VALUE_MARGINS=(.025,.05,.1,.2)

def run():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','actions'):(ROOT/folder).mkdir()
    inputs=[MOTION/'oof.npz',MOTION/'verification.json',VALUE/'oof.npz',VALUE/'verification.json',VALUE/'dataset.npz',VALUE/'primary.json']
    save(ROOT/'protocol.json',dict(experiment='E136',inputs={str(p):fingerprint(p) for p in inputs},design='Require E133 scalar predicted1-step progress AND E135 lambda1 predicted final paired task advantage. Same family-excluded OOF predictions; measured inputs only. Up to8 off actions in first accepted clean phase, then resume E120. No physics selector. No retraining or alternative penalties before selection.',
        primary=dict(motion_margin=1e-5,value_margin=.05),motion_margins=list(MOTION_MARGINS),value_margins=list(VALUE_MARGINS),selection='20 fixed conjunction thresholds plus full abstention. Require zero E120 losses, maximize gains, fewer switches, larger task margin then motion margin. Adaptive fitting-validation selection after E133-E135; not unbiased performance.',
        held='Evaluate only qualifying nontrivial no-loss candidate after freezing all thresholds and all-fitting model fits. Previously used original held placements are development validation, not an independent final benchmark.',E120_unchanged=True))
    _,_,_,_,meta,tasks=data();motion=np.load(MOTION/'oof.npz')['scalar'];value=np.load(VALUE/'oof.npz')['1.0']
    rules=[dict(motion_margin=m,value_margin=v) for m in MOTION_MARGINS for v in VALUE_MARGINS]+[dict(motion_margin=1e9,value_margin=1e9)]
    grid=[score((motion>r['motion_margin'])&(value>r['value_margin']),r,meta,tasks) for r in rules]
    chosen=max([r for r in grid if r['losses']==0],key=lambda r:(r['gains'],-r['switches'],r['rule']['value_margin'],r['rule']['motion_margin']))
    primary=next(r for r in grid if r['rule']==dict(motion_margin=1e-5,value_margin=.05))
    save(ROOT/'cases.json',tasks);save(ROOT/'grid.json',grid);save(ROOT/'primary.json',primary);save(ROOT/'selected.json',chosen)
    summary=dict(primary={k:v for k,v in primary.items() if k!='episodes'},selected={k:v for k,v in chosen.items() if k!='episodes'},held_evaluated=False,E120_unchanged=True);save(ROOT/'summary.json',summary);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(summary,flush=True)

if __name__=='__main__':run()

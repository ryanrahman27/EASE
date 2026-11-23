"""Read-only evaluation of a measured-progress activation rule on E170 fitting traces."""
from pathlib import Path
import numpy as np
from .abstaining_task_value_gate import read,save
REF=Path('outputs/ease_descent_execution_v1/history_allowance')
ROOT=Path('outputs/ease_descent_stall_v1')

def run(minimum_steps=20,upper=float('inf')):
    ROOT.mkdir(exist_ok=True);result=[]
    for r in read(REF/'records.json'):
        stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(REF/'trials'/f'{stem}.json');logs=read(REF/'clearance'/f'{stem}.json');errors=[];trigger=None
        for i,l in enumerate(logs):
            if l['phase']!='descend':continue
            # Previous post-state is the current observation before action.
            e=float(np.linalg.norm(np.array(l['goal'])-t['trace'][i-1]['position']));errors.append(e)
            if len(errors)>=minimum_steps and .025<e<upper and errors[-20]-e<.001 and np.max(np.abs(l['q_before']))>=2.4:
                trigger=dict(step=i,descent_step=len(errors),error=e,improvement=errors[-20]-e);break
        result.append(dict(case=r['case'],success=r['success'],trigger=trigger))
    suffix='_local' if np.isfinite(upper) else ''
    save(ROOT/f'diagnosis_{minimum_steps}{suffix}.json',dict(minimum_descent_steps=minimum_steps,upper_error_m=upper if np.isfinite(upper) else None,rule='20 observed descent errors; error>25mm; improvement over19 intervals<1mm; any abs joint>=2.4. Latch until descent ends. Uses measured task error only; fitting-only diagnostic.',reference_cases=result));print([r for r in result if r['trigger']],flush=True)

if __name__=='__main__':
    import sys
    run(int(sys.argv[1]) if len(sys.argv)>1 else 20,float(sys.argv[2]) if len(sys.argv)>2 else float('inf'))

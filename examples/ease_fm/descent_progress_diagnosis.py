"""Saved-state fitting diagnosis of E170 descent stalls, before new control tests."""
from pathlib import Path
import numpy as np
from .abstaining_task_value_gate import read,save
REFERENCE=Path('outputs/ease_descent_execution_v1/history_allowance')
ROOT=Path('outputs/ease_descent_progress_v1')

def run():
    ROOT.mkdir(exist_ok=True);rows=[]
    for r in read(REFERENCE/'records.json'):
        stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(REFERENCE/'trials'/f'{stem}.json');logs=read(REFERENCE/'clearance'/f'{stem}.json');indices=[i for i,l in enumerate(logs) if l['phase']=='descend'];last=indices[-1];tail=indices[-20:];l=logs[last]
        x=dict(case=r['case'],success=r['success'],descent_steps=len(indices),terminal_error_xyz=(np.array(l['goal'])-t['trace'][last]['position']).tolist(),rotation_error=t['trace'][last]['rotation_error'],last20_position_error_reduction=t['trace'][tail[0]]['position_error']-t['trace'][last]['position_error'],terminal_q=l['q_before'],terminal_clearance=l['distance_after'],terminal_allowance=l['allowance'],terminal_slack=l['solve'].get('linear_slack'),last20_nominal_norm=float(np.mean([np.linalg.norm(logs[i]['nominal_action']) for i in tail])),last20_executed_norm=float(np.mean([np.linalg.norm(logs[i]['executed_command']) for i in tail])))
        rows.append(x)
    save(ROOT/'diagnosis.json',dict(reference=str(REFERENCE),all48_fitting_cases=rows,failures=[r for r in rows if not r['success']]))
    print([r for r in rows if not r['success']],flush=True)

if __name__=='__main__':run()

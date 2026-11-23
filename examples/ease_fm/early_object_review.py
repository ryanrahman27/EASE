"""Actual object movement and matched interaction costs for E175."""
from pathlib import Path
import shutil
import numpy as np
from .early_object import ROOT,REFERENCE
from .abstaining_task_value_gate import read,save

def run():
    rows=[]
    for case in (26,43,62,65,67):
        for label,root in (('E174',REFERENCE),('early_guard',ROOT/'early_guard')):
            stem=f'{case}_e120_ensemble_frozen_bias';t=read(root/'trials'/f'{stem}.json');logs=read(root/'clearance'/f'{stem}.json');desc=[i for i,l in enumerate(logs) if l['phase']=='descend'];close=[i for i,l in enumerate(logs) if l['phase']=='close'];positions=np.array([t['trace'][i]['object'][:2] for i in desc]);initial=positions[0];active=next((i for i,l in enumerate(logs) if l['planning_attempted']),None)
            rows.append(dict(case=case,condition=label,success=t['success'],collision=t['collision'],grasped=t['grasped'],lifted=t['lifted'],steps=len(t['trace']),descent_steps=len(desc),terminal_descent_error_mm=1000*t['trace'][desc[-1]]['position_error'],max_object_descent_displacement_mm=1000*float(np.max(np.linalg.norm(positions-initial,axis=1))),final_object_descent_displacement_mm=1000*float(np.linalg.norm(positions[-1]-initial)),close_ready=logs[close[-1]]['close_ready'],final_two_pad_close_steps=logs[close[-1]]['close_good'],aborted=logs[-1]['aborted'],trigger_step=active,trigger_reason=None if active is None else logs[active].get('trigger_reason','E174_late_stall')))
    save(ROOT/'object_handoff_diagnosis.json',rows)
    lines=['','## Physical execution and interaction cost','','| Case | Controller | Success | Descent steps | Max cube XY movement during descent (mm) | Final hand error (mm) | Final two-pad close steps |','|---|---|---|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['case']} | {r['condition']} | {r['success']} | {r['descent_steps']} | {r['max_object_descent_displacement_mm']:.3f} | {r['terminal_descent_error_mm']:.3f} | {r['final_two_pad_close_steps']} |")
    old_steps=sum(len(read(REFERENCE/'trials'/f"{r['case']}_e120_ensemble_frozen_bias.json")['trace']) for r in read(REFERENCE/'records.json'));new_steps=sum(len(read(ROOT/'early_guard/trials'/f"{r['case']}_e120_ensemble_frozen_bias.json")['trace']) for r in read(ROOT/'early_guard/records.json'))
    save(ROOT/'interaction_cost.json',dict(E174_control_steps=old_steps,E175_control_steps=new_steps,added_control_steps=new_steps-old_steps,added_simulated_seconds=.04*(new_steps-old_steps)))
    lines+=['',f'Total executed control steps across48 cases: E174={old_steps}, E175={new_steps}; difference={new_steps-old_steps} steps ({.04*(new_steps-old_steps):.2f} simulated seconds). This includes all task phases, excluding the identical initial calibration. Planning wall-clock latency is not benchmarked.','', 'Cube displacement is measured over saved40ms descent states. Original contact scoring is evaluated at2ms physics substeps. Video playback samples every80ms and can miss brief contacts. No frame selection changes the task score.']
    p=ROOT/'report.md';s=p.read_text(encoding='utf-8')
    if '## Physical execution and interaction cost' not in s:p.write_text(s+'\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(rows,flush=True)

if __name__=='__main__':run()

"""Cross-study progress diagnosis without equating lower error to task success."""
import shutil
from pathlib import Path
from .descent_stall import ROOT,REFERENCE
from .abstaining_task_value_gate import read,save

def run():
    sources={'E170':REFERENCE,'E171_direct':Path('outputs/ease_descent_progress_v1/direct_task'),'E171_gated':Path('outputs/ease_descent_progress_v1/limit_gated'),'E172':ROOT/'stall_recovery'};rows=[]
    for case in (26,62,65,67):
        item={'case':case,'conditions':{}}
        for label,root in sources.items():
            stem=f'{case}_e120_ensemble_frozen_bias';t=read(root/'trials'/f'{stem}.json');trace=[x for x in t['trace'] if x['phase']=='descend'];logs=read(root/'clearance'/f'{stem}.json');descent=[x for x in logs if x['phase']=='descend'];first=next((i+1 for i,l in enumerate(descent) if l.get('direct_task',False)),None)
            item['conditions'][label]=dict(success=t['success'],collision=t['collision'],terminal_error_mm=1000*trace[-1]['position_error'],rotation_error=trace[-1]['rotation_error'],last20_progress_mm=1000*(trace[-20]['position_error']-trace[-1]['position_error']),terminal_joint5=descent[-1]['q_before'][4],recovery_first_descent_step=first,descent_steps=len(trace))
        rows.append(item)
    save(ROOT/'progress_comparison.json',rows)
    summary=read(ROOT/'summary.json')['conditions']['stall_recovery'];audit=read(ROOT/'verification.json')['conditions']['stall_recovery'];lines=['','## Progress versus completed tasks','','| Case | Reference error (mm) | Recovery error (mm) | Reference orientation error | Recovery orientation error | Last20-step recovery progress (mm) |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        a=r['conditions']['E170'];b=r['conditions']['E172'];lines.append(f"| {r['case']} | {a['terminal_error_mm']:.2f} | {b['terminal_error_mm']:.2f} | {a['rotation_error']:.4f} | {b['rotation_error']:.4f} | {b['last20_progress_mm']:.2f} |")
    lines+=['',f"All {audit['exact_nonactivated_episodes']} nonactivated episodes reproduce the protected reference's complete states and traces exactly. Full-task outcome: {summary['outcomes']['success']}/48 successes, {summary['outcomes']['collisions']} collisions; rescues/losses versus E170: {summary['paired']['E170_protected']}. No new completion is inferred from a smaller descent error.",'', 'The broader E171 variants each lost one success and are rejected. E172 preserves the existing successes by avoiding intervention on their observed reference trajectories. This activation rule was developed using fitting diagnostics, so retention here does not establish gate generalization. The E120 baseline, probe protection and release fix remain unchanged.','', 'The remaining errors and late progress distinguish a shorter approach to a constrained equilibrium from successful grasping. Orientation can trade off against position, and the nonlinear execution/model mismatch remains. Merely increasing the time budget is not established as a solution. Next work should inspect whether a different collision-free arm posture or earlier route can reach the grasp pose, with a clearly separated privileged feasibility diagnostic before changing the learned controller.']
    p=ROOT/'report.md';text=p.read_text(encoding='utf-8')
    if '## Progress versus completed tasks' not in text:p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(rows,flush=True)

if __name__=='__main__':run()

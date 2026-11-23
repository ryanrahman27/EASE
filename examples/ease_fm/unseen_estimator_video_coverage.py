"""Add every shared failure to the prespecified diagnostic video selection."""
from pathlib import Path
import shutil
from .unseen_estimator import ROOT,CONDITIONS
from .abstaining_task_value_gate import read,save

def run():
    rows=read(ROOT/'video_records.json');existing={(r['case'],r['method']) for r in rows};failed={p['case'] for p in read(ROOT/'paired.json')['cases'] if not p['e120_success'] or not p['analytical_success']};added=[]
    for case in sorted(failed):
        for method in CONDITIONS:
            label='privileged_'+method
            if (case,label) in existing:continue
            old=f'{case}_{method}_frozen_bias';new=f'{case}_{label}_frozen_bias';t=read(ROOT/method/'trials'/f'{old}.json');t['method']=label;save(ROOT/'trials'/f'{new}.json',t)
            for ext in ('npz','xml'):shutil.copy2(ROOT/method/'states'/f'{old}.{ext}',ROOT/'states'/f'{new}.{ext}')
            rows.append(dict(case=case,method=label,mode='frozen_bias',success=t['success'],dof=6));added.append(dict(case=case,method=label))
    assert all(any(r['case']==c and r['method']=='privileged_'+m for r in rows) for c in failed for m in CONDITIONS)
    save(ROOT/'video_records.json',rows);save(ROOT/'video_coverage.json',dict(all_failure_cases=sorted(failed),additional_shared_failure_videos=added,total=len(rows),selection='Prespecified first task of every body and all discordant outcomes, extended after evaluation to cover every failed case in both methods. Diagnostic illustrations only; all outcomes remain in denominators.'))
    p=ROOT/'report.md';text=p.read_text(encoding='utf-8');text+='\nVideo coverage was extended after evaluation to include every failed case for both estimators, including shared failures. This changes illustrations only; all raw outcomes and comparison denominators are unchanged. See video_coverage.json.\n';p.write_text(text,encoding='utf-8');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(len(rows),'videos; all failure cases',sorted(failed),flush=True)

if __name__=='__main__':run()

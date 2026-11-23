"""Categorize prospective failures without tuning the tested controller."""
from pathlib import Path
from collections import Counter
import shutil
from .unseen_estimator import ROOT,CONDITIONS
from .abstaining_task_value_gate import read,save

def run():
    details=read(ROOT/'case_diagnosis.json');summaries={};failures=[]
    for method in CONDITIONS:
        phases=Counter();categories=Counter();contacts=Counter()
        for r in read(ROOT/method/'records.json'):
            case=r['case'];t=read(ROOT/method/'trials'/f'{case}_{method}_frozen_bias.json');d=details[method][str(case)];cal=next((dict(step=i,pairs=x['bad']) for i,x in enumerate(t['calibration']) if x['bad']),None);contact=next((dict(step=i,phase=x['phase'],pairs=x['bad']) for i,x in enumerate(t['trace']) if x['bad']),None)
            if t['success']:continue
            if t['collision']:
                assert cal is not None or contact is not None
                phase='calibration' if cal else contact['phase'];phases[phase]+=1;categories['collision']+=1;pairs=cal['pairs'] if cal else contact['pairs']
                for pair in {tuple(sorted(p)) for p in pairs}:contacts[' / '.join(pair)]+=1
            else:
                category='guard_abort' if d['aborted'] else 'no_grasp' if not t['grasped'] else 'no_lift' if not t['lifted'] else 'placement_or_release';categories[category]+=1
            failures.append(dict(method=method,case=case,body=r['body'],family=r['family'],calibration_contact=cal,first_task_contact=contact,**{k:d[k] for k in ('success','collision','grasped','lifted','routed','trigger','trigger_descent_observation','close_ready','final_two_pad_close_steps','max_cube_descent_displacement_mm','aborted','failed_phases')}))
        summaries[method]=dict(categories=dict(categories),first_collision_phase=dict(phases),first_contact_pairs=dict(contacts))
    save(ROOT/'failure_categories.json',dict(summary=summaries,failures=failures))
    lines=['','## Failure categories','','Contact episodes are classified by their first forbidden contact, including calibration. Non-contact failures are classified by guard abort, missing grasp, missing lift, or final placement/release. These categories describe outcomes; they do not identify a unique causal estimator defect.','','| Estimator | Failure categories | First collision phases |','|---|---|---|']
    for method,x in summaries.items():lines.append(f"| {method} | {x['categories']} | {x['first_collision_phase']} |")
    lines+=['','Detailed contacts, grasp guards, recovery triggers and failed phases are in failure_categories.json. Initial calibration is identical between estimators and retains the original E176 behavior. This prospective cohort will not be retuned and relabeled as untouched.']
    p=ROOT/'report.md';text=p.read_text(encoding='utf-8')
    if '## Failure categories' not in text:p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(summaries,flush=True)

if __name__=='__main__':run()

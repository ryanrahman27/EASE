"""Summarize E169 measured probe motion and remaining execution errors."""
import numpy as np
from .probe_gripper_clearance import ROOT
from .abstaining_task_value_gate import read,save

def run():
    s=read(ROOT/'summary.json');details={}
    for c,v in s['conditions'].items():
        changed=[x for x in v['probe_excitation'] if x['changed_probe_steps']]
        ratios=[x['gram_eigenvalues'][0]/max(x['reference_gram_eigenvalues'][0],1e-30) for x in changed]
        details[c]=dict(changed_probe_episodes=len(changed),changed_probe_commands=sum(x['changed_probe_steps'] for x in changed),minimum_eigenvalue_ratio_median=None if not ratios else float(np.median(ratios)),minimum_eigenvalue_ratio_min=None if not ratios else float(min(ratios)),first_contact_cases=[x['case'] for x in v['first_contact_prediction_diagnostics']])
    contact=next(x for x in s['conditions']['both']['first_contact_prediction_diagnostics'] if x['case']==26)
    save(ROOT/'followthrough.json',dict(conditions=details,case26_first_contact=contact))
    lines=['','## Interpretation and execution-error follow-through','','Probe-only and combined coverage rescue cases27 and63, retain both E168 gains13 and66 and lose no E168 or E120+release success. Arm-link/table collision episodes fall from3 to0; four gripper/table collision episodes remain, all beginning in descent. Three episodes are collision-free failures rather than successes. Gripper coverage alone changes neither task successes nor collision counts. The simpler probe-only extension therefore accounts for the measured task gain; added gripper coverage has no demonstrated aggregate benefit here.','','| Coverage | Episodes with modified probe | Modified probe commands | Median smallest Gram-eigenvalue ratio to E168 (modified episodes) |','|---|---:|---:|---:|']
    for c,x in details.items():lines.append(f"| {c} | {x['changed_probe_episodes']} | {x['changed_probe_commands']} | {x['minimum_eigenvalue_ratio_median']} |")
    lines+=['','These are measured arm-motion excitation statistics, not estimator accuracy or safety guarantees. Per-case spectra remain in summary.json. Reduced probe excitation can accompany a successful contact-avoidance intervention; task success and motion information must be judged separately.','',f"In combined-coverage case26, the first pad/table contact is step{contact['step']} during descent. The left-pad row predicted {contact['predicted_distance'][7]*1000:.3f}mm clearance, but the saved actual clearance was {contact['actual_distance'][7]*1000:.3f}mm. The QP passed and did not fall back. This directly identifies a command-to-motion prediction discrepancy; it does not identify its complete dynamical cause. A useful next fitting diagnostic is a conservative execution-error allowance or short-horizon dynamics treatment, holding this probe protection fixed.",'','Audit implementation note: independent gripper-gradient reconstruction uses an eligible step with a saved preceding full state. The initial post-calibration jaw state was not stored separately, so substituting the first post-action jaw state produced a small audit-only mismatch. The checker was corrected to use the next eligible saved state; tolerances and all controller outcomes were unchanged. Every post-action geometry state was still checked.']
    p=ROOT/'report.md';text=p.read_text(encoding='utf-8')
    if '## Interpretation and execution-error follow-through' not in text:p.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
    print(details,flush=True)

if __name__=='__main__':run()

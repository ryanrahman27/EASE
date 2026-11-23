"""Refresh the research index after the frozen prospective evaluation."""
from pathlib import Path
import json

ROOT = Path('outputs/ease_iteration_review_v1')

def run():
    p = ROOT/'report.md'
    text = p.read_text(encoding='utf-8')
    start = text.index('E120 remains immutable.')
    end = text.index('| Study |')
    text = text[:start] + '''E120 remains immutable. The reserved 48-placement test has now been evaluated once, after freezing E159. It contains new placements on four known bodies, not unseen embodiments. No fresh outcomes were used to fit or substitute the candidate.

The strongest development task result was E154: 77/96 versus uniform analytical 74/96 and E120 ensemble 72/96. It did not retain every prior success. In the prospective E159 comparison, the final learned model completed 41/48, uniform analytical 41/48, E120 ensemble 39/48, and the original analytical condition 39/48. Learned versus uniform had two rescues and two regressions, with seven collision episodes versus six. Thus the prospective improvement criterion failed; the learned estimator is not promoted over the analytical baseline. The original retention gates remain failed, and the pre-test amendment is preserved in E159's protocol.

Prediction improvements are reproducible, but task gains over the strongest analytical reference are not established. The exact-map diagnostic E149 also retained substantial task failures, indicating that improving local motion estimates alone does not resolve the controller/contact problem. Any next structural controller study must begin on fitting cases and use a newly reserved final test; E159 is no longer an untouched test set.

''' + text[end:]
    additions = [
      ('E151','ease_home_posture_control_v1','Posture objective: uniform76/96, learned prior70, E12070, exact77. No learned promotion.'),
      ('E152','ease_prior_posture_reference_v1','Matched analytical-prior posture reference68/96 versus learned70; still below uniform76.'),
      ('E153','ease_fixed_uniform_learning_v1','Fixed analytical kernel with learned relative weights: position RMSE0.388mm versus0.415; all seeds improve.'),
      ('E154','ease_fixed_uniform_control_v1','Standard learned77/96 versus uniform74 and E12072; five uniform rescues, two regressions. Retention gate fails.'),
      ('E155','ease_prior_projector_control_v1','Separate prior-map posture projection: targeted fitting diagnostic fails retention; no full promotion.'),
      ('E156','ease_lift_posture_control_v1','Lift-only projection fails targeted retention despite verified identical pre-lift states.'),
      ('E157','ease_conservative_history_weights_v1','Fixed weight attenuation retains modest prediction improvement; no task claim.'),
      ('E158','ease_conservative_lift_control_v1','Conservative lift variant retains case6 but fails legacy13; rejected.'),
      ('E159','ease_fresh_fixed_weights_v1','Frozen fresh comparison: learned41/48, uniform41, E12039, analytical39. Prospective improvement criterion fails.'),
    ]
    lines = text.splitlines()
    lines = [l for l in lines if not l.startswith('| E151 |') and not l.startswith('| E152 |')]
    at = next(i for i,l in enumerate(lines) if l.startswith('Interpretation:'))
    extra = []
    for e,folder,finding in additions:
        video = f'[videos](../{folder}/videos/index.html)' if Path('outputs',folder,'videos/index.html').exists() else 'See report for shared/source-context videos'
        extra.append(f'| {e} | {finding} | [report](../{folder}/report.md) | {video} |')
    while at > 0 and not lines[at-1].strip():
        at -= 1
    lines[at:at] = extra
    text = '\n'.join(lines).replace('Interpretation: these are adaptive fitting-family development splits on known bodies, not independent final-test estimates.', 'Interpretation: E133–E158 are adaptive fitting-family development studies on known bodies, not independent final-test estimates. E159 is a one-time prospective placement test, with its amendment registered before outcomes.')
    text = text.replace('source-context videos or pending', 'See report for source-context videos')
    p.write_text(text+'\n',encoding='utf-8')
    studies = json.loads((ROOT/'studies.json').read_text())
    existing = {r['experiment'] for r in studies}
    studies += [dict(experiment=e,folder='outputs/'+f) for e,f,_ in additions if e not in existing]
    for row in studies:
        folder=Path(row['folder']);row['report_exists']=(folder/'report.md').exists()
        row['videos']=len(list((folder/'videos').glob('*.mp4')))
    (ROOT/'studies.json').write_text(json.dumps(studies,indent=2)+'\n')
    state=ROOT/'WORKING_STATE.md'
    state.write_text('''# Latest checkpoint: E159 complete

The following older checkpoint is retained for provenance, but its running-session and unopened-test statements are superseded. All E159 workers, audits, report and 24 video renders completed. E159 learned41/48 ties uniform41 and exceeds E120 ensemble39; two uniform rescues/two regressions, seven versus six collision episodes. The prespecified positive pattern is false. E120 unchanged. Do not retune on E159 or call these unseen-body results. Main technical report and compact index updated. No experiment jobs remain active. The research objective of a validated learned task advantage is not achieved.

---

'''+state.read_text(encoding='utf-8'),encoding='utf-8')

if __name__ == '__main__':
    run()

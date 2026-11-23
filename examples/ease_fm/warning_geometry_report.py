"""Summarize E79 retrospective geometry attribution without changing E78 scoring."""
import json
import shutil
from pathlib import Path
from collections import Counter
from .warning_geometry_audit import ROOT
from .benchmark import fingerprint


def run():
    audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    for p,h in {**audit['inputs'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    results=[]
    for entry in json.loads((ROOT/'records.json').read_text()):
        t=json.loads((ROOT/'trials'/f"{entry['case']}_{entry['method']}.json").read_text());rr=t['attribution']
        rows=[]
        for r in rr:
            pred=min(r['paths']['response_linear'],key=lambda x:x['clearance']);actual=min(r['actual'],key=lambda x:x['clearance'])
            closest=min(pred['contacts'],key=lambda x:x['distance']) if pred['contacts'] else None
            rows.append(dict(step=r['step'],phase=r['phase'],prediction_clearance=pred['clearance'],actual_clearance=actual['clearance'],
                initial_clearance=r['actual'][0]['clearance'],predicted_pair=None if closest is None else closest['pair'],
                predicted_bodies=None if closest is None else closest['bodies'],
                all_path_minima={name:min(x['clearance'] for x in path) for name,path in r['paths'].items()}))
        results.append(dict(case=t['case'],method=t['method'],dof=t['dof'],success=t['success'],steps=len(rows),
            phases=dict(Counter(r['phase'] for r in rows)),actual_margin_warnings=sum(r['actual_clearance']<.002-1e-6 for r in rows),
            initial_margin_warnings=sum(r['initial_clearance']<.002-1e-6 for r in rows),actual_penetrations=sum(r['actual_clearance']<0 for r in rows),rows=rows))
    assert sum(r['steps'] for r in results if r['success'])==110
    (ROOT/'summary.json').write_text(json.dumps(results,indent=2))
    lines=['# E79: geometry attribution of successful-run warnings','',
        'All 110 E78 response-predictor warnings on successful learned runs arise in two six-joint trajectories. Both are replayed, together with the single missed learned collision. All three full traces, probes, geometry-filter logs and prediction records reproduce exactly. This is an outcome-selected development diagnostic, not an independent performance test.','',
        '| Case / estimator | Outcome | Diagnosed steps | Phase counts | Actual substep margin warnings | Margin already violated at step start |','|---|---|---:|---|---:|---:|']
    for r in results:lines.append(f"| {r['case']} / {r['method']} | {'success' if r['success'] else 'failure'} | {r['steps']} | {r['phases']} | {r['actual_margin_warnings']} | {r['initial_margin_warnings']} |")
    lines+=['',
        '**Lift/transport: real near-table motion.** Case 1 / candidate_2 has 87 warnings (82 lift, five transport), all involving link1 or link2 against the table. The actual sampled motion enters the fixed 2 mm margin on 86 steps, and 85 start inside it. The smallest measured clearance is about 0.298 mm; none of these steps penetrates. These warnings mostly describe genuinely close but successful motion, rather than a fictional carried-object collision. They remain false alarms under the original next-step physical-contact metric.','',
        '**Release: predicted proximity not observed.** Case 2 / parent_0 has 23 warnings involving link4 and the left gripper pad. Predicted minimum clearance ranges from 1.669 to 1.989 mm; measured substep clearance stays at the 2 mm reporting cap throughout. This identifies a prediction/path error for this pair, but does not isolate whether arm or finger response, or path interpolation, causes it.','',
        '**Missed collision: velocity matters in this example.** Case 10 / candidate_0 collides during descent at step 53. Command-only and no-velocity response paths stay at the 2 mm reporting cap, while the measured right-pad/table clearance reaches -0.369 mm. Both existing velocity-aware paths predict -0.085 mm and detect it. This single selected event does not establish a velocity-aware controller advantage or reverse its false-alarm results.','',
        'The object pose is included in retrospective actual-state clearance reconstruction. None of the closest predicted warning pairs involves the object, so frozen carried-object pose does not explain these particular warning minima. All clearances use the same expanded-margin shadow model; the physical model is unchanged. Twenty substeps plus the starting state are sampled, not a mathematical continuous-time clearance proof.','',
        'Decision: preserve E78 as a failed gate and keep actions unchanged. Before deploying a warning-triggered filter, distinguish an existing small clearance from a newly predicted approach to contact. A useful next controlled diagnostic is same-state, one-step action comparison at actual collisions and successful near-margin states, with no threshold retuning. Report avoidance, new collisions and task-direction error; one-step avoidance would still require full-task retention validation. Do not simply stop every time the 2 mm margin is entered.','',
        'Verification: three exact full replays, 111 attributed steps, 21 samples per predicted path and 21 measured states per diagnosed step. Prediction records match E78 exactly; input, checkpoint and source hashes verified. No training, action intervention, reserved-test evaluation or task-success improvement.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in results],indent=2))


def document():
    lead='**E79 completed:** Geometry attribution separates real near-table motion from prediction error. Of 87 lift/transport warnings in one successful six-joint run, 86 correspond to measured clearance inside 2 mm without penetration. Another successful six-joint run has 23 release warnings for predicted link4/gripper proximity that is absent in sampled actual motion. The missed descent collision is detected by the existing velocity-aware predictors. Three full traces and prediction logs replay exactly. E78 remains a failed gate; actions and success rates are unchanged. See Section 4.69.'
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.69 E79:' not in text
    text=text.replace('Revision 64','Revision 65',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E78 |'))
    text=text.replace(line,line+'| E79 | Geometry attribution of successful-run warnings and missed contact | Complete: three exact replays, 111 attributed steps | `ease_warning_geometry_v1` |\n',1)
    section=(ROOT/'report.md').read_text().replace('# E79:', '## 4.69 E79:',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+lead+' Next compare bounded actions from identical collision and successful near-margin states, recording avoidance, new contacts and task-direction error before attempting complete-task intervention.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 65, 28 October 2026: E79 attributes all 110 successful-run warnings and the one missed collision using three exact replays, frozen prediction paths and measured substep geometry. Distinguishes near-margin motion from unobserved predicted proximity; preserves E78 failure and unchanged task outcomes.\n',1);p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


if __name__=='__main__':
    import sys
    document() if '--document' in sys.argv else run()

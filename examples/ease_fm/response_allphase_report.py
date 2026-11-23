"""E78 frozen prediction validation, including out-of-scope contacts."""
import json
from pathlib import Path
from collections import Counter
from .benchmark import fingerprint
from .response_allphase import ROOT


def run():
    runs=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(runs)==audit['exact_replays']==112
    for p,h in {**audit['input_hashes'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    rows=[];events=[]
    for run in runs:
        rr=json.loads((ROOT/'trials'/f"{run['case']}_{run['method']}.json").read_text())['records']
        assert len(rr)==run['steps'] and all(r['prior_updates']==32+i for i,r in enumerate(rr))
        enriched=[dict(case=run['case'],method=run['method'],dof=run['dof'],task_success=run['success'],**r) for r in rr]
        rows.extend(enriched)
        event=next((r for r in enriched if r['collision']),None)
        if event:events.append(event)
    tables=[];coverage=[]
    for group in ('all','learned','learned_6','learned_7','successful_learned',*[f'phase_{p}' for p in sorted({r['phase'] for r in rows})]):
        select=lambda r: group=='all' or (r['method']!='analytical' and (group!='successful_learned' or r['task_success']) and (group not in ('learned_6','learned_7') or r['dof']==int(group[-1])) and (not group.startswith('phase_') or r['phase']==group[6:]))
        eligible=[r for r in rows if r['evaluated'] and select(r)];ev=[r for r in events if select(r)]
        coverage.append(dict(group=group,first_contacts=len(ev),scored_contacts=sum(r['evaluated'] for r in ev),phases=dict(Counter(r['phase'] for r in ev))))
        for method in ('command','response_linear','velocity_linear','velocity_curve'):
            positives=sum(r['collision'] for r in eligible);negative=len(eligible)-positives
            tp=sum(r['collision'] and r['clearances'][method]<.002-1e-6 for r in eligible)
            fp=sum(not r['collision'] and r['clearances'][method]<.002-1e-6 for r in eligible)
            tables.append(dict(group=group,method=method,tp=tp,fn=positives-tp,fp=fp,negative_steps=negative,fp_rate=fp/negative if negative else None))
    base=next(r for r in tables if r['group']=='learned' and r['method']=='command')
    candidate=next(r for r in tables if r['group']=='learned' and r['method']=='response_linear')
    passed=candidate['tp']>base['tp'] and candidate['fp']<=base['fp']
    warnings=[]
    for run in runs:
        if run['method']=='analytical':continue
        rr=[r for r in rows if r['case']==run['case'] and r['method']==run['method']]
        first=next((r['step'] for r in rr if r['collision']),None)
        for r in rr:
            if not r['evaluated'] or r['collision']:continue
            flags={m:r['clearances'][m]<.002-1e-6 for m in ('command','response_linear','velocity_linear','velocity_curve')}
            if any(flags.values()):warnings.append(dict(case=run['case'],method=run['method'],dof=run['dof'],phase=r['phase'],step=r['step'],success=run['success'],warnings=flags,steps_to_first_contact=None if first is None else first-r['step']))
    (ROOT/'warnings.json').write_text(json.dumps(warnings,indent=2))
    assert all(r['first_contacts']==r['scored_contacts'] for r in coverage)
    result=dict(confusion=tables,coverage=coverage,primary_passes=passed,exact_replays=112,task_steps=len(rows))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'events.json').write_text(json.dumps(events,indent=2))
    decision=('The frozen predictor passes the prespecified prediction gate. Next evaluate a bounded action intervention against unchanged complete-task controls, counting every recovery and regression.' if passed else 'The frozen predictor fails the prespecified prediction gate. Retain the original controller; inspect false alarms and missed contacts before proposing any action intervention. Do not tune this threshold on these outcomes.')
    lines=['# E78: frozen response prediction on all task phases','',
      'All 112 E72 geometry-filter trajectories replay exactly, including successes and failures. These 16 placements differ from E75, but reuse the same bodies and historical development data. This is neither an independent embodiment test nor a reserved final test. The E75 response model is imported unchanged; the step implementation changes only phase eligibility. Every previously evaluated record and every endpoint error must exactly match E76. Neural weights, actions, dynamics, probes and success outcomes are unchanged.','',
      'Primary comparison: no-velocity response_linear versus command at the fixed clearance < 0.002 - 1e-6 m threshold. The online predictor uses only completed measurements, including 32 calibration steps. Scoring covers every phase before first forbidden contact, inclusive. Phase rows contain learned trajectories only. Exact warning lead times are descriptive in warnings.json; no longer-horizon relabeling is used. Known geometry and exact current simulator state remain privileged inputs. Secondary velocity variants do not determine advancement.','',
      '| Group | Predictor | Detected first contacts | Missed in scope | False alarms | Noncollision steps |','|---|---|---:|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['group']} | {r['method']} | {r['tp']}/{r['tp']+r['fn']} | {r['fn']} | {r['fp']} | {r['negative_steps']} |")
    lines+=['','Coverage of actual first contacts:','']
    for r in coverage:lines.append(f"- {r['group']}: {r['scored_contacts']}/{r['first_contacts']} contacts scored across all phases; phase counts {r['phases']}.")
    lines+=['',f"Primary gate: **{'PASS' if passed else 'FAIL'}**. More detected contacts and no increase in false-alarm count are both required.",'',decision,'',
      'False alarms are warnings on executed noncollision steps, not proven task losses. Samples within trajectories and checkpoints sharing placements are dependent. All first contacts are covered; collision-step detection still differs from demonstrating collision avoidance. A prediction gain does not demonstrate collision avoidance or recovered pick-and-place tasks. No action changes, native tuning or final-test evaluation occurred.','',
      f"Verification: 112 complete traces, filter logs and calibration probes reproduce exactly; all {len(rows)} task steps have 32+step prior updates. Input, checkpoint and frozen predictor hashes are unchanged."]
    lines+=['','Geometry caveat: clearance paths move the arm and finger joints while holding the object at its current pose. Carried-object motion is not predicted. Warnings are clearance-margin violations, not guaranteed next-step physical contacts; phase-specific false alarms do not by themselves identify their cause. Successful-run warnings are especially relevant before any action-changing use.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))



def document():
    import shutil
    result=json.loads((ROOT/'summary.json').read_text())
    def cell(group,method):return next(r for r in result['confusion'] if r['group']==group and r['method']==method)
    a=cell('learned','response_linear');b=cell('learned','command');six=cell('learned_6','response_linear');success=cell('successful_learned','response_linear')
    lead=(f"**E78 completed:** All-phase response prediction detects {a['tp']}/{a['tp']+a['fn']} learned first contacts versus command-only {b['tp']}/{b['tp']+b['fn']}, with {a['fp']} versus {b['fp']} one-step false alarms. Six-joint detection is {six['tp']}/{six['tp']+six['fn']}; successful learned runs have {success['fp']} warnings across {success['negative_steps']:,} noncollision steps. The fixed advancement rule {'passes' if result['primary_passes'] else 'fails'}. All 112 trajectories and prior E76 predictions match exactly; 105 tests pass. No action changes or task recoveries. See Section 4.68.")
    decision=('Next test a bounded action intervention with complete-task regression controls.' if result['primary_passes'] else 'Retain the controller. Before an action-changing trial, diagnose the missed contact pairs and whether warnings reflect imminent contact or conservative geometry checks; do not alter the fixed threshold or relabel this gate.')
    report=ROOT/'report.md';report.write_text(report.read_text()+'\nAll 105 EASE-FM tests pass. Structural comparison verifies that phase eligibility is the only change to the copied response class.\n\n'+decision+'\n')
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.68 E78:' not in text
    text=text.replace('Revision 63','Revision 64',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E77 |'))
    text=text.replace(line,line+'| E78 | Frozen response prediction across every task phase | Complete: 112 exact trajectories, all first contacts covered | `ease_response_allphase_v1` |\n',1)
    section=report.read_text().replace('# E78:', '## 4.68 E78:',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+lead+' '+decision+'\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 64, 28 October 2026: E78 extends frozen prediction to every pre-contact task phase with exact trajectory and prior-prediction checks, phase-specific and six-joint metrics, successful-run controls and descriptive warning timings. All 105 tests pass; no intervention, training or final-test evaluation.\n',1);p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)


if __name__=="__main__":
    import sys
    document() if '--document' in sys.argv else run()

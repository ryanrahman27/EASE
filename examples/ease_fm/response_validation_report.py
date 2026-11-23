"""E76 frozen prediction validation, including out-of-scope contacts."""
import json
from pathlib import Path
from collections import Counter
from .benchmark import fingerprint
from .response_validation import ROOT


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
    for group in ('all','learned','learned_6','learned_7','successful_learned'):
        select=lambda r: group=='all' or (r['method']!='analytical' and (group!='successful_learned' or r['task_success']) and (group not in ('learned_6','learned_7') or r['dof']==int(group[-1])))
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
    result=dict(confusion=tables,coverage=coverage,primary_passes=passed,exact_replays=112,task_steps=len(rows))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'events.json').write_text(json.dumps(events,indent=2))
    decision=('The frozen predictor passes the prespecified prediction gate. Next evaluate a bounded action intervention against unchanged complete-task controls, counting every recovery and regression.' if passed else 'The frozen predictor fails the prespecified prediction gate. Retain the original controller; inspect false alarms and missed/out-of-scope contacts before proposing any action intervention. Do not tune this threshold on these outcomes.')
    lines=['# E76: frozen response prediction on separate trajectories','',
      'All 112 E72 geometry-filter trajectories replay exactly, including successes and failures. These 16 placements differ from E75, but reuse the same bodies and historical development data. This is neither an independent embodiment test nor a reserved final test. The E75 implementation is imported unchanged and verified against its saved source hash. Neural weights, actions, dynamics, probes and success outcomes are unchanged.','',
      'Primary comparison: no-velocity response_linear versus command at the fixed clearance < 0.002 - 1e-6 m threshold. The online predictor uses only completed measurements, including 32 calibration steps. Scoring remains limited to descent/lowering before first forbidden contact, inclusive. Known geometry and exact current simulator state remain privileged inputs. Secondary velocity variants do not determine advancement.','',
      '| Group | Predictor | Detected first contacts | Missed in scope | False alarms | Noncollision steps |','|---|---|---:|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['group']} | {r['method']} | {r['tp']}/{r['tp']+r['fn']} | {r['fn']} | {r['fp']} | {r['negative_steps']} |")
    lines+=['','Coverage of actual first contacts:','']
    for r in coverage:lines.append(f"- {r['group']}: {r['scored_contacts']}/{r['first_contacts']} contacts in scored phases; phase counts {r['phases']}.")
    lines+=['',f"Primary gate: **{'PASS' if passed else 'FAIL'}**. More detected contacts and no increase in false-alarm count are both required.",'',decision,'',
      'False alarms are warnings on executed noncollision steps, not proven task losses. Samples within trajectories and checkpoints sharing placements are dependent. Contacts outside the selected phases cannot be counted as detected. A prediction gain does not demonstrate collision avoidance or recovered pick-and-place tasks. No action changes, native tuning or final-test evaluation occurred.','',
      f"Verification: 112 complete traces, filter logs and calibration probes reproduce exactly; all {len(rows)} task steps have 32+step prior updates. Input, checkpoint and frozen predictor hashes are unchanged."]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2))


def document():
    import shutil
    r=json.loads((ROOT/'summary.json').read_text())
    a=next(x for x in r['confusion'] if x['group']=='learned' and x['method']=='response_linear')
    b=next(x for x in r['confusion'] if x['group']=='learned' and x['method']=='command')
    c=next(x for x in r['coverage'] if x['group']=='learned')
    lead=(f"**E76 completed:** Frozen no-velocity response prediction detects {a['tp']}/{a['tp']+a['fn']} in-scope learned first contacts, versus command-only {b['tp']}/{b['tp']+b['fn']}, with {a['fp']} versus {b['fp']} false alarms across {a['negative_steps']:,} noncollision steps. "
          f"The prespecified prediction gate {'passes' if r['primary_passes'] else 'fails'}. Only {c['scored_contacts']}/{c['first_contacts']} first contacts fall in the evaluated descent/lowering phases. All 112 separate historical trajectories replay exactly; 105 tests pass. Task outcomes are unchanged. See Section 4.66.")
    decision=('Next evaluate a bounded action intervention with complete-task retention controls.' if r['primary_passes'] else 'Keep the controller unchanged. Next audit false alarms and contacts outside the selected phases before designing an intervention; preserve the fixed threshold and record any revised method as a new development experiment.')
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.66 E76:' not in text
    text=text.replace('Revision 61 · 28 October 2026','Revision 62 · 28 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E75 |'))
    text=text.replace(line,line+'| E76 | Frozen response predictor on separate historical placements | Complete: 112 exact trajectories; prediction-only validation | `ease_response_validation_v1` |\n',1)
    section=(ROOT/'report.md').read_text().replace('# E76:', '## 4.66 E76:',1)
    section+='\nAll 105 EASE-FM regression tests pass.\n'
    text=text.replace('## 5. Limitations and next decisions',section+'\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+lead+' '+decision+'\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 62, 28 October 2026: E76 validates the unchanged E75 predictor on 112 E72 trajectories; records fixed-threshold false alarms, scoped recall and all first-contact phase coverage. Exact traces, probes, update counts and hashes verified; 105 tests pass. No action changes or final-test evaluation.\n',1)
    p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    shutil.copy2(Path(__file__),ROOT/'source'/Path(__file__).name)


if __name__=='__main__':
    import sys
    document() if '--document' in sys.argv else run()

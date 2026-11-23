"""E75 prediction accuracy and first-contact/false-alarm audit."""
import json
from pathlib import Path
import shutil
import numpy as np
from .response_prediction import ROOT
from .benchmark import fingerprint


def run():
    runs=json.loads((ROOT/'records.json').read_text());audit=json.loads((ROOT/'audit.json').read_text());spec=json.loads((ROOT/'protocol.json').read_text())
    assert len(runs)==119 and audit['exact_replays']==119
    for p,h in {**audit['input_hashes'],**spec['checkpoints']}.items():assert fingerprint(p)==h
    for name in ('response_prediction.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        assert fingerprint(Path(__file__).with_name(name))==fingerprint(ROOT/'source'/name)
    rows=[]
    for run in runs:
        full=json.loads((ROOT/'trials'/f"{run['case']}_{run['method']}.json").read_text())
        assert len(full['records'])==run['steps']
        for i,r in enumerate(full['records']):
            assert r['prior_updates']==32+i
            rows.append(dict(case=run['case'],method=run['method'],dof=run['dof'],task_success=run['success'],**r))
    methods=('command','response_linear','velocity_linear','velocity_curve');tables=[];errors=[]
    for group in ('all','learned'):
        eligible=[r for r in rows if r['evaluated'] and (group=='all' or r['method']!='analytical')]
        positives=sum(r['collision'] for r in eligible);negatives=len(eligible)-positives
        for threshold,label in [(0.,'penetration'),(.002-1e-6,'2mm_warning')]:
            for method in methods:
                tp=sum(r['collision'] and r['clearances'][method]<threshold for r in eligible)
                fp=sum(not r['collision'] and r['clearances'][method]<threshold for r in eligible)
                tables.append(dict(group=group,threshold=label,method=method,positive_steps=positives,negative_steps=negatives,tp=tp,fn=positives-tp,fp=fp,tn=negatives-fp))
        for method in methods[:3]:
            errors.append(dict(group=group,method=method,evaluated_steps=len(eligible),arm_rmse_rad=float(np.sqrt(np.mean([r['arm_endpoint_mse'][method] for r in eligible])))))
    gates=[]
    for label in ('penetration','2mm_warning'):
        base=next(r for r in tables if r['group']=='learned' and r['threshold']==label and r['method']=='command')
        for method in ('velocity_linear','velocity_curve'):
            candidate=next(r for r in tables if r['group']=='learned' and r['threshold']==label and r['method']==method)
            gates.append(dict(method=method,threshold=label,passes=candidate['tp']>base['tp'] and candidate['fp']<=base['fp']))
    events=[{k:r[k] for k in ('case','method','dof','step','phase','clearances','arm_endpoint_mse')} for r in rows if r['evaluated'] and r['collision']]
    assert len(events)==10
    result=dict(confusion=tables,endpoint_errors=errors,gates=gates,task_steps=len(rows),event_count=len(events),any_gate_passes=any(r['passes'] for r in gates))
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2));(ROOT/'events.json').write_text(json.dumps(events,indent=2))
    lines=['# E75: measured joint-response and velocity prediction','',
        'All 119 E73 geometry-controller trajectories replay exactly, including successful tasks and every failure. Two analytical online ridge models are fitted separately per actuator from completed interactions: command/intercept and command/velocity/intercept. They update through the 32-step calibration and the subsequent trajectory. Predictions are made before observing the next displacement. Neural policy weights and actions remain unchanged.','',
        'Geometry evaluation covers every descent/lowering step before the first forbidden contact, including that contact step. These phases were selected from E74 evidence and are development diagnostics. Every path uses the same known geometry, object pose and 21 interpolation points; none uses future dynamic simulator rollout. Current joint velocity is an additional measurement. This is not a trained replacement for the motion Transformer.','',
        '| Group | Threshold | Predictor/path | Detected contacts | Missed contacts | False alarms | Noncollision steps |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in tables:lines.append(f"| {r['group']} | {r['threshold']} | {r['method']} | {r['tp']}/{r['positive_steps']} | {r['fn']} | {r['fp']} | {r['negative_steps']} |")
    lines+=['','| Group | Endpoint predictor | Arm displacement RMSE (rad) | Evaluated steps |','|---|---|---:|---:|']
    for r in errors:lines.append(f"| {r['group']} | {r['method']} | {r['arm_rmse_rad']:.6f} | {r['evaluated_steps']} |")
    lines+=['','Paths: command is the joint-target straight line. response_linear uses the no-velocity response endpoint. velocity_linear uses the velocity-conditioned endpoint. velocity_curve uses that same endpoint with a quadratic trajectory constrained to the measured initial velocity. Endpoint errors are step-weighted means of arm-joint squared errors; finger joints are also predicted for clearance queries.','',
        'The advancement rule requires more detected first contacts than command-only prediction without more false alarms at the same threshold, separately reported for each velocity path.']
    for r in gates:lines.append(f"- {r['method']} at {r['threshold']}: {'PASS' if r['passes'] else 'FAIL'}.")
    lines+=['','The no-velocity response comparator is promising: on learned trajectories it detects 7/9 contacts at penetration and 8/9 at the fixed 2 mm warning threshold, with zero false alarms among 4,920 noncollision steps at either threshold. Velocity models detect 8/9 but add one false alarm. The prespecified advancement rule targeted velocity-aware paths; selecting the simpler comparator for subsequent work is an exploratory inference, not a passed prespecified gate or a task-success result. Freeze it before further validation rather than adjusting thresholds on these events.','',
        'A false alarm here is a predicted collision/margin violation on an executed noncollision step. It is not a demonstrated lost task, but it measures how often an action-changing filter would potentially interfere. Adjacent samples and repeated checkpoints are dependent, and only ten actual first-contact events are available. Higher recall alone is insufficient to justify a new controller.','',
        'Verification: all 119 complete traces, geometry logs and probes reproduce exactly; every prediction has exactly 32+step prior updates. Input, model and source hashes remain unchanged. All 105 EASE-FM tests pass, including a read-only prediction check and synthetic held-out identification test. No action changes, task recoveries, native tuning or reserved final-test evaluation occurred.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(11,4));x=np.arange(4)
    for i,label in enumerate(('penetration','2mm_warning')):
        selected=[next(r for r in tables if r['group']=='learned' and r['threshold']==label and r['method']==m) for m in methods]
        for ax,key in zip(axs,('tp','fp')):
            bars=ax.bar(x+(i-.5)*.35,[r[key] for r in selected],.35,label=label);ax.bar_label(bars,fontsize=8)
    for ax in axs:ax.set_xticks(x,methods,rotation=20);ax.legend(fontsize=8)
    axs[0].set(ylabel='First contacts detected /9',ylim=(0,10));axs[1].set(ylabel='False alarms on noncollision steps')
    fig.suptitle('E75: causal prediction on unchanged learned-controller trajectories');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(learned_confusion=[r for r in tables if r['group']=='learned'],endpoint_errors=errors,gates=gates,task_steps=len(rows)),indent=2))


def document():
    r=json.loads((ROOT/'summary.json').read_text());passed=r['any_gate_passes']
    decision=('A prespecified prediction gate passes; preserve the fixed predictor for a subsequent complete-task retention experiment, without claiming a task gain here.' if passed else 'No velocity-aware path passes the prediction/false-alarm gate. Keep the original controller and do not insert this predictor into action selection.')
    lead='**E75 completed:** On unchanged learned-controller trajectories, measured response without velocity detects 8/9 first contacts at the fixed 2 mm threshold with zero false alarms in 4,920 noncollision steps, versus command-only 1/9. Velocity models detect 8/9 with one false alarm, failing their stricter prespecified gate. The simpler comparator is an exploratory candidate requiring separate validation, not a recovered-task result. All 119 traces replay exactly; 105 tests pass. See Section 4.65.'
    next_step=('Next compare an action-changing version against the original and analytical controls on complete tasks, retaining every successful six-joint case and all query/interaction costs.' if passed else 'The no-velocity comparator detects 8/9 learned first contacts with no false alarms at the already fixed 2 mm threshold. Freeze that simpler model as an exploratory candidate and validate its prediction/false-alarm tradeoff on separate procedural trajectories before enabling action changes. Do not choose new thresholds from these ten events; better mean endpoint prediction alone is insufficient for collision avoidance.')
    rp=ROOT/'report.md';rp.write_text(rp.read_text()+'\n### Decision and next step\n\n'+decision+' '+next_step+' Geometry and velocity access must remain explicit; none of these diagnostics establishes learned superiority over analytical control.\n')
    folder=Path(__file__).parent;p=folder/'TECHNICAL_REPORT.md';text=p.read_text();assert '## 4.65 E75:' not in text
    text=text.replace('Revision 60 · 28 October 2026','Revision 61 · 28 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E74 |'))
    text=text.replace(line,line+'| E75 | Causal joint-response prediction and false-alarm controls | Complete: 119 exact trajectories; prediction-only audit | `ease_response_prediction_v1` |\n',1)
    section=rp.read_text().replace('# E75: measured joint-response and velocity prediction','## 4.65 E75: measured joint-response and velocity prediction',1)
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\n'+decision+' '+next_step+' The original policy and success rates are unchanged by E75.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 61, 28 October 2026: Completed E75 causal response-prediction audit on 119 exactly reproduced E73 trajectories, including successful trajectories. Recorded velocity/no-velocity endpoint errors and collision recall/false alarms at fixed thresholds, verified prior-update counts and unchanged hashes; inspected figure; 105 tests pass. No action changes, task-recovery claims, native tuning or final-test evaluation.\n',1)
    p.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    shutil.copy2(Path(__file__),ROOT/'source'/Path(__file__).name)


if __name__=='__main__':
    import sys
    document() if '--document' in sys.argv else run()

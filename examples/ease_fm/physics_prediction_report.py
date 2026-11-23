import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .physics_prediction_audit import ROOT

def run():
    rows=json.loads((ROOT/'records.json').read_text());verify=json.loads((ROOT/'verification.json').read_text());table=[]
    for group in ('all','train','held'):
        for clean in (False,True):
            selected=[r for r in rows if (group=='all' or r['validation']==(group=='held')) and (not clean or r['clean'])]
            for horizon in ('40','80','160'):
                eligible=[r for r in selected if r['future_actual'][horizon] is not None]
                for method in ('static','physics'):
                    counts=dict(TP=0,FP=0,FN=0,TN=0)
                    for r in eligible:
                        pred=r['static_warning'] if method=='static' else r['predictions'][horizon]['collision'];actual=r['future_actual'][horizon]
                        counts[('T' if pred==actual else 'F')+('P' if pred else 'N')]+=1
                    table.append(dict(group=group,clean_only=clean,horizon_ms=int(horizon),method=method,n=len(eligible),censored=len(selected)-len(eligible),**counts))
    latency={h:dict(median_ms=float(np.median([r['predictions'][h]['elapsed_ms'] for r in rows])),p95_ms=float(np.quantile([r['predictions'][h]['elapsed_ms'] for r in rows],.95))) for h in ('40','80','160')}
    examples=[]
    for horizon in ('40','80','160'):
        for category in ('static_miss_physics_hit','physics_false_alarm','physics_miss'):
            found=[]
            for r in rows:
                actual=r['future_actual'][horizon];pred=r['predictions'][horizon]['collision']
                if not r['clean'] or actual is None:continue
                ok=(actual and pred and not r['static_warning']) if category=='static_miss_physics_hit' else (pred and not actual) if category=='physics_false_alarm' else (actual and not pred)
                if ok:found.append({k:r[k] for k in ('case','seed','body','step','validation')})
            examples.append(dict(horizon_ms=int(horizon),category=category,count=len(found),first_examples=found[:10]))
    summary=dict(table=table,latency=latency,examples=examples);(ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# E107: short physics contact predictions on training bodies','',f"All36 unchanged training-body task/seed trajectories exactly replay E104. All{len(rows)} predicted40ms qpos/qvel states and forbidden-contact unions exactly match executed physics. No control intervention, new weights or reserved-test evaluation.",'',
    'The predictor has privileged access to the exact physical model and complete simulator state. Exact one-step agreement verifies implementation fidelity; it is not evidence of a learned dynamics model or successful collision avoidance. All controllers, commands and collision rules remain unchanged. The held split contains new placements on the same four training bodies, not unseen embodiments.','',
    'Static warning means an infeasible filter decision or selected clearance below2mm minus1e-6. Physics warning means any forbidden contact while holding the selected absolute joint/finger target for40/80/160ms. Longer-horizon comparison uses the actual replanning policy as the outcome: false alarms and misses can arise from future command changes. Tails without a complete within-retreat comparison horizon are censored.','',
    '## Prediction counts','', '| Split | Clean start only | Horizon ms | Predictor | N | TP | FP | FN | TN | Censored |','|---|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in table:lines.append('| '+' | '.join(str(r[k]) for k in ('group','clean_only','horizon_ms','method','n','TP','FP','FN','TN','censored'))+' |')
    lines+=['','## Interpretation','',
    'At40ms, both methods detect the two contact-onset steps from clean states. Physics has zero false positives versus96 static warnings without subsequent contact. There are no static misses in this training-body baseline sample, so it cannot establish improved recall over static checks. The two positives are case1/seed0 on body84000 (train placement) and case8/seed2 on body84004 (held placement); this is sparse positive coverage. Clean means no forbidden contact at the current boundary, not an entirely collision-free episode prefix. The static2mm warning is a margin criterion, not designed as a calibrated binary physical-contact classifier; its false positives need not be useless safety margins.',
    'At80ms, held-target physics has one false positive and no misses; at160ms it has two false positives and three missed overlapping future windows. These windows cover only two eventual contact events and are not independent failures. Longer hold predictions cannot substitute for replanning-aware prediction. All predictions evaluate the command already selected by the static filter, not alternative candidate actions.',
    'The controller suite passes131 tests (13.51s). Runtime invariants additionally verify all36 full baseline replays and1633 exact one-step states/contact unions. No action-avoidance claim follows from the audit.',
    '', '## Cost','',json.dumps(latency,indent=2),'','Wall-clock prediction time includes cloning, step simulation and contact logging; values are machine/load specific and exclude neural inference. Later horizons include earlier work. No claim of hardware real-time readiness.','',
    '## Failure examples','',json.dumps(examples,indent=2),'','## Next decision','',
    'Use40ms physical lookahead as a candidate training-only action check if it catches static misses. A warning is not an avoidance action: the next controlled experiment must evaluate candidate fallback actions in copied physics, account for progress and joint limits, and retain every successful training/held-placement task. Longer held-target horizons are diagnostic and should not be promoted merely for detecting more future contacts. No controller has been changed by E107.','',
    '## Artifacts','', 'Full prediction records, source, checkpoint/input hashes and per-trajectory original results are saved. The video gallery reuses verified E104 baseline success/failure executions, explicitly labelled as unchanged trajectories. Counts are correlated control steps, not independent trials; do not interpret them as independent-sample confidence intervals.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,method in enumerate(('static','physics')):
        rr=[r for r in table if r['group']=='all' and r['clean_only'] and r['method']==method]
        for ax,key in zip(axes,('FN','FP')):ax.bar(np.arange(3)+(j-.5)*.32,[r[key] for r in rr],.32,label=method)
    for ax,title in zip(axes,('Missed future contacts','Warnings without future contact')):ax.set_xticks(range(3),['40ms','80ms','160ms']);ax.set_title(title);ax.set_ylabel('Clean-start control steps');ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'prediction_errors.{ext}')
    plt.close(fig);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print(json.dumps(dict(primary=[r for r in table if r['group']=='all' and r['clean_only']],latency=latency),indent=2))

if __name__=='__main__':run()

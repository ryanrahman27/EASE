"""Summarize paired local ranking validity without treating branches as tasks."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint


def agreement(records,score,horizon,neural_only=False,collision_free=False):
    cases={};pairs=0
    for r in records:
        values=[x for x in r['candidates'] if not neural_only or x['method']!='analytical']
        for a,b in combinations(values,2):
            aa=a['branch'][str(horizon)];bb=b['branch'][str(horizon)]
            if collision_free and (aa['collision'] or bb['collision']):continue
            predicted=a['scores'][score]-b['scores'][score]
            actual=aa['error_squared']-bb['error_squared']
            if abs(predicted)<=1e-12 or abs(actual)<=1e-12:continue
            cases.setdefault(r['case'],[]).append(float(predicted*actual>0));pairs+=1
    return dict(case_mean=float(np.mean([np.mean(v) for v in cases.values()])) if cases else None,
        case_rates={k:float(np.mean(v)) for k,v in cases.items()},non_tied_pairs=pairs,clusters=len(cases))


def run(output):
    root=Path(output);p=json.loads((root/'protocol.json').read_text());audit=json.loads((root/'audit.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    rows=json.loads((root/'records.json').read_text())
    assert len(rows)==audit['states']==audit['repeat_checks']
    assert len(audit['exact_replays'])==12 and all(r['exact'] for r in audit['exact_replays'])
    for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
    assert all(len(r['candidates'])==10 for r in rows)
    lines=['# E53: does the objective rank actual actions correctly?','',
        'No training. Frozen original/E52 models and analytical estimation receive identical measured histories from all twelve analytical E46 development trajectories. Samples are the first and twentieth step of each reached phase. At each sample, ten controllers branch from exact copied simulator data, with the current reference and aperture fixed for five steps. Online estimator/bias updates, actuator dynamics, clipping, joint limits and contacts remain active.',
        '',f"All twelve complete driver trajectories replay exactly. {len(rows)} sampled states produce {10*len(rows)} counterfactual branches plus {audit['repeat_checks']} exact analytical repeat checks. These are local probes, not independent tasks or success evaluations. No checkpoints changed; the final test is untouched.",
        '', 'The three proxies use a privileged instantaneous geometric Jacobian for diagnostic scoring only: E52 global response-operator MSE; local unclipped target residual; and local executed-command residual after clipping, joint limits and subtracting measured bias. None of these scores enters candidate commands. The executed-command proxy still assumes measured bias predicts actuator response, and omits contact effects.',
        '', '## Pairwise ordering agreement', '',
        'Higher is better. Each row averages agreement within each case and then across cases, avoiding domination by longer trajectories. Exact/numerical ties at 1e-12 are excluded. Cases, phases and branches remain dependent; no independent-sample significance claim is made. All-pair results include analytical; neural-only excludes it.', '',
        '| Robot | Horizon | Candidates | Proxy | Case-mean agreement | Non-tied pairs |','|---|---:|---|---|---:|---:|']
    summary=[]
    for robot in ['piper','panda']:
        rr=[r for r in rows if r['robot']==robot]
        for h in [1,5]:
            for neural in [False,True]:
                for score in ['operator_mse','local_unclipped','local_executed']:
                    result=agreement(rr,score,h,neural)
                    summary.append(dict(robot=robot,horizon=h,neural_only=neural,proxy=score,collision_free=False,**result))
                    lines.append(f"| {robot} | {h} | {'neural' if neural else 'all'} | {score} | {result['case_mean']:.3f} | {result['non_tied_pairs']} |")
                    summary.append(dict(robot=robot,horizon=h,neural_only=neural,proxy=score,collision_free=True,**agreement(rr,score,h,neural,True)))
    phase_summary=[]
    lines+=['','## Phase-specific agreement (all candidates)','','| Robot | Phase | States | Operator, one step | Executed, one step | Executed, five steps |','|---|---|---:|---:|---:|---:|']
    for robot in ['piper','panda']:
        for phase in dict.fromkeys(r['phase'] for r in rows):
            rr=[r for r in rows if (r['robot'],r['phase'])==(robot,phase)]
            metrics=[agreement(rr,'operator_mse',1),agreement(rr,'local_executed',1),agreement(rr,'local_executed',5)]
            phase_summary.append(dict(robot=robot,phase=phase,states=len(rr),operator_one=metrics[0],executed_one=metrics[1],executed_five=metrics[2]))
            lines.append(f"| {robot} | {phase} | {len(rr)} | "+' | '.join(f"{m['case_mean']:.3f}" for m in metrics)+' |')
    lines+=['','## Local error and contacts','','| Robot | Method | Branches | Mean one-step scaled error | Mean five-step scaled error | Five-step contact failures | First-command clipped branches |','|---|---|---:|---:|---:|---:|---:|']
    for robot in ['piper','panda']:
        for method in [x['method'] for x in rows[0]['candidates']]:
            rr=[x for r in rows if r['robot']==robot for x in r['candidates'] if x['method']==method]
            a=np.mean([np.sqrt(x['branch']['1']['error_squared']) for x in rr]);b=np.mean([np.sqrt(x['branch']['5']['error_squared']) for x in rr])
            lines.append(f"| {robot} | {method} | {len(rr)} | {a:.6f} | {b:.6f} | {sum(x['branch']['5']['collision'] for x in rr)} | {sum(x['clipped_joints']>0 for x in rr)} |")
    lines+=['','Scaled error is the norm of position residual in metres concatenated with 0.15 times tool-Z direction residual. These means combine phases and are not task-success or safety guarantees.','',
        '## Matched control-aware versus prediction-only rankings','','| Robot | Horizon | Seed | Operator favors control | Measured residual favors control | Comparable states |','|---|---:|---:|---:|---:|---:|']
    matched=[]
    for robot in ['piper','panda']:
        for h in [1,5]:
            for seed in range(3):
                proxy=actual=count=0
                for r in rows:
                    if r['robot']!=robot:continue
                    a=next(x for x in r['candidates'] if x['method']==f'control_{seed}')
                    b=next(x for x in r['candidates'] if x['method']==f'prediction_{seed}')
                    dp=a['scores']['operator_mse']-b['scores']['operator_mse'];da=a['branch'][str(h)]['error_squared']-b['branch'][str(h)]['error_squared']
                    if abs(dp)<=1e-12 or abs(da)<=1e-12:continue
                    count+=1;proxy+=dp<0;actual+=da<0
                matched.append(dict(robot=robot,horizon=h,seed=seed,proxy_favors_control=int(proxy),measured_favors_control=int(actual),states=count))
                lines.append(f'| {robot} | {h} | {seed} | {proxy} | {actual} | {count} |')
    lines+=['','## Limits and decision','','This audit changes neither training nor deployment. It tests local ranking validity on analytical-history development states; it does not reproduce candidate-specific histories, long-horizon grasp losses or final task scores. The five-step reference is held fixed, unlike the continuing sequencer. Operator and action-specific scores answer different questions, so weak agreement identifies objective mismatch but does not by itself prove the cause of E52 failures. Even stronger agreement here would require validation on procedurally generated training arms before defining a new learning objective. Keep E35; no model or loss is promoted. Collision-free ranking details and per-case rates are retained in summary.json.']
    (root/'summary.json').write_text(json.dumps(dict(rankings=summary,matched=matched,phases=phase_summary),indent=2))
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    (root/'report_audit.json').write_text(json.dumps(dict(records_sha256=fingerprint(root/'records.json'),exact_replays=12,states=len(rows),counterfactual_branches=10*len(rows),unchanged_weights=True,test_evaluated=False),indent=2))
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(**vars(p.parse_args()))

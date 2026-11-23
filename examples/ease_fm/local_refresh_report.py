"""Complete E47 paired effects and retention decision from immutable trial records."""
import argparse
import json
from pathlib import Path
from .benchmark import fingerprint
from .local_refresh import summarize


def run(output,reference):
    root=Path(output);summarize(root,reference)
    p=json.loads((root/'protocol.json').read_text())
    for path,sha in p['checkpoints'].items():assert fingerprint(path)==sha
    assert fingerprint(root/'cases.json')==p['cases_sha256']
    rows=json.loads((root/'records.json').read_text());lookup={(r['condition'],r['method'],r['case']):r for r in rows}
    for case in range(12):
        a=lookup[('probe','privileged',case)];b=lookup[('frozen_probe','privileged',case)]
        for key in ['trace','steps','success','failure']:assert a[key]==b[key],(case,key)
    effects=[]
    for robot in ['piper','panda']:
        for method in ['privileged','analytical','neural_0','neural_1','neural_2']:
            targets=[r for r in rows if (r['condition'],r['robot'],r['method'])==('probe',robot,method)]
            for control in ['baseline','wait','frozen_probe']:
                pairs=[(r,lookup[(control,method,r['case'])]) for r in targets]
                effects.append(dict(robot=robot,method=method,control=control,
                    gained=[r['case'] for r,c in pairs if r['success'] and not c['success']],
                    lost=[r['case'] for r,c in pairs if c['success'] and not r['success']],
                    mean_step_change=sum(r['steps']-c['steps'] for r,c in pairs)/len(pairs)))
    regressions=[x for x in effects if x['method'].startswith('neural') and x['control']=='baseline' and len(x['lost'])>len(x['gained'])]
    decision=dict(promote=False,neural_aggregate_regressions=regressions,
        reason='Diagnostic on repeatedly observed development placements. Retain original reference; no independent reliability validation or consistent learned superiority established.',
        test_evaluated=False)
    (root/'effects.json').write_text(json.dumps(effects,indent=2));(root/'decision.json').write_text(json.dumps(decision,indent=2))
    summary=json.loads((root/'summary.json').read_text())
    lines=['','## Neural outcomes by seed','',
        '| Condition | PiPER (each /6) | Panda (each /6) |', '|---|---|---|']
    for condition in ['baseline','wait','probe','frozen_probe']:
        scores=[' / '.join(str(next(s['success'] for s in summary if (s['condition'],s['robot'],s['method'])==(condition,robot,f'neural_{seed}'))) for seed in range(3)) for robot in ['piper','panda']]
        lines.append(f'| {condition} | {scores[0]} | {scores[1]} |')
    lines+=['','## Paired neural effects of updating during probing','',
        '| Robot | Seed | Versus | Gained cases | Lost cases | Mean step change |', '|---|---|---|---|---|---:|']
    for e in effects:
        if e['method'].startswith('neural'):
            lines.append(f"| {e['robot']} | {e['method']} | {e['control']} | {e['gained']} | {e['lost']} | {e['mean_step_change']:+.1f} |")
    lines+=['','No promotion. Per-task gains and losses are reported rather than treating multiple seeds on the same six placements as independent trials. Step differences include early termination, so a shorter failed episode is not an efficiency win. The frozen-probe comparison changes both motion-statistic and actuator-bias updates; it cannot attribute an effect exclusively to the Transformer’s Jacobian estimate. The baseline comparison includes the added hover waypoint. All 240 records and 60 exact baseline replays are retained.']
    with (root/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines[:9]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--reference',required=True)
    run(**vars(p.parse_args()))

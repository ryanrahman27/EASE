"""Audit and summarize all prespecified E34 conditions without selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .robustness import CONDITIONS


def run(output):
    root=Path(output);protocol=json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    assert protocol['conditions']==CONDITIONS
    for e in protocol['checkpoints']:assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
    summary=[];families=[];hashes={};allrows={}
    methods=['analytical','neural_0','neural_1','neural_2']
    for condition in CONDITIONS:
        name=condition['name']
        for method in methods:
            path=root/f'{name}__{method}.json';rows=json.loads(path.read_text());hashes[path.name]=fingerprint(path)
            assert len(rows)==1024 and {r['goal_id'] for r in rows}==set(range(1024))
            assert all(r['condition']==name and 1<=r['steps']<=150 and np.isfinite(r['error']) for r in rows)
            assert all(not r['success'] or (not r['collision'] and r['error']<.03) for r in rows)
            assert all(r['success'] or r['steps']==150 for r in rows)
            allrows[name,method]={r['goal_id']:r for r in rows}
            result=dict(condition=name,method=method,success=sum(r['success'] for r in rows),
                six=sum(r['success'] for r in rows if r['dof']==6),seven=sum(r['success'] for r in rows if r['dof']==7),
                collisions=sum(r['collision'] for r in rows),mean_steps=float(np.mean([r['steps'] for r in rows])),
                by50=sum(r['success'] and r['steps']<=50 for r in rows))
            baseline=allrows[name,'analytical']
            result['neural_only']=sum(r['success'] and not baseline[r['goal_id']]['success'] for r in rows)
            result['analytical_only']=sum(not r['success'] and baseline[r['goal_id']]['success'] for r in rows)
            summary.append(result)
            for family in range(16):
                selected=[r for r in rows if r['family_id']==family];assert len(selected)==64
                families.append(dict(condition=name,method=method,family=family,success=sum(r['success'] for r in selected),collisions=sum(r['collision'] for r in selected)))
    (root/'summary.json').write_text(json.dumps(dict(overall=summary,families=families),indent=2))
    (root/'audit.json').write_text(json.dumps(dict(records=28672,record_hashes=hashes,checkpoint_hashes_unchanged=True,test_evaluated=False),indent=2))
    lines=['# E34: frozen-controller robustness comparison','',
        'All 1,024 E33 development tasks are retained in each condition, with three unchanged neural checkpoints and the fixed-probe analytical baseline. No training or condition-specific tuning. Every method has 150 task interactions and no separate calibration. Clean success/contact/step counts reproduce E33 exactly. Final test remains reserved.','',
        'Noise is independent Gaussian joint-position noise in radians and end-effector noise in meters (sigma 0.001 or 0.005), with an exact target. One deterministic realization per task is shared across methods at equal time. Delay is 40 or 120 ms for both measured positions, with timestamp-aligned commands. History conditions rebuild statistics and tracking bias from only the last 4 or 8 transitions; they test memory restriction, not isolated physical excitation. Success and contact are evaluated on true simulator state.','',
        '| Condition | Controller | Success /1024 | Six /512 | Seven /512 | Contacts | Mean steps | Success by 50 | Neural-only / analytical-only |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"| {r['condition']} | {r['method']} | {r['success']} | {r['six']} | {r['seven']} | {r['collisions']} | {r['mean_steps']:.2f} | {r['by50']} | {r['neural_only']} / {r['analytical_only']} |")
    lines+=['','All 28,672 records pass case-key, finite-error, success/contact and horizon checks. Neural runs block simulator Jacobians and analytical joint-model inverses; the shared 3×3 Cartesian control solve remains intentional. Per-family results are in `summary.json`.','',
        'These are correlated tasks on 16 procedural families, using one disturbance realization per task and three learned model seeds. No significance or universal robustness claim follows from these counts. The analytical baseline has fixed clean-condition hyperparameters; this is not a comparison against a noise-tuned estimator.','',
        '![Robustness success](robustness.png)']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,denom,title in zip(axes,['success','six'],[1024,512],['All arms','Six-joint arms']):
        for method in methods:
            values=[next(r[key] for r in summary if r['condition']==c['name'] and r['method']==method)/denom*100 for c in CONDITIONS]
            ax.plot(range(7),values,'o-',label=method)
        ax.set_xticks(range(7),[c['name'] for c in CONDITIONS],rotation=35,ha='right')
        ax.set_ylim(0,105);ax.set_ylabel('Success (%)');ax.set_title(title);ax.grid(alpha=.25)
    axes[1].legend();fig.tight_layout();fig.savefig(root/'robustness.png',dpi=160);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);run(**vars(parser.parse_args()))

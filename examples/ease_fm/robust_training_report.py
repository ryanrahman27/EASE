"""E35 complete development comparison, retaining old models and controller controls."""
import argparse
import json
from pathlib import Path
import numpy as np
from .benchmark import fingerprint
from .robustness import CONDITIONS


def run(output,previous):
    root=Path(output);prior=Path(previous);protocol=json.loads((root/'protocol.json').read_text())
    assert json.loads((root/'status.json').read_text())['status']=='complete'
    assert fingerprint(Path(protocol['train'])/'manifest.json')==protocol['train_sha256']
    assert fingerprint(Path(protocol['evaluation'])/'manifest.json')==protocol['evaluation_sha256']
    assert json.loads((prior/'protocol.json').read_text())['manifest_sha256']==protocol['evaluation_sha256']
    hashes={};summary=[]
    for e in protocol['parents']:
        assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
        meta=json.loads((root/f'seed_{e["seed"]}'/'training.json').read_text())
        assert meta['parent_sha256']==e['policy_sha256'] and meta['steps']==protocol['steps']
        assert meta['policy_sha256']==fingerprint(root/f'seed_{e["seed"]}'/'policy.pt')
    for condition in CONDITIONS:
        name=condition['name'];delayed=bool(condition.get('delay'))
        baseline=json.loads(((root/f'{name}__analytical.json') if delayed else (prior/f'{name}__analytical.json')).read_text())
        baseline={r['goal_id']:r for r in baseline}
        methods=['analytical','old_0','old_1','old_2','new_0','new_1','new_2']
        for method in methods:
            if delayed or method.startswith('new'):path=root/f'{name}__{method}.json'
            else:path=prior/f'{name}__{method.replace("old_","neural_")}.json'
            rows=json.loads(path.read_text());hashes[str(path)]=fingerprint(path)
            assert len(rows)==1024 and {r['goal_id'] for r in rows}==set(range(1024))
            assert all(r['condition']==name and np.isfinite(r['error']) for r in rows)
            assert all(not r['success'] or (not r['collision'] and r['error']<.03) for r in rows)
            assert all(r['success'] or r['steps']==150 for r in rows)
            summary.append(dict(condition=name,method=method,success=sum(r['success'] for r in rows),
                six=sum(r['success'] for r in rows if r['dof']==6),collisions=sum(r['collision'] for r in rows),
                mean_steps=float(np.mean([r['steps'] for r in rows])),by50=sum(r['success'] and r['steps']<=50 for r in rows),
                method_only=sum(r['success'] and not baseline[r['goal_id']]['success'] for r in rows),
                analytical_only=sum(not r['success'] and baseline[r['goal_id']]['success'] for r in rows)))
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    (root/'audit.json').write_text(json.dumps(dict(record_hashes=hashes,new_rollouts=29*1024,comparison_rows=49*1024,
        inherited_rows=20*1024,all_parent_hashes_unchanged=True,test_evaluated=False),indent=2))
    lines=['# E35: noisy/short-history training and matched delay handling','',
        'Three original models each receive 5,000 additional AdamW updates (learning rate 0.0001, batch 128). Synthetic histories use training-only motion profiles mixed with Gaussian excitation, lengths 1–32, and correlated measurement differences from noisy positions. The target is the clean weighted ridge solution. Architecture and parameter count remain unchanged. This is a multi-change fine-tuning experiment with additional compute, not an isolated noise ablation.','',
        'All 1,024 previously observed E33/E34 development tasks are reused. No final test, new-family confirmation or post-result tuning. Delay-aware control uses gain 0.5/(1+delay steps) for the analytical baseline, old neural models and new neural models. This separates training effects from the shared controller change. Non-delay baseline rows are inherited from E34.','',
        '| Condition | Method | Success /1024 | Six /512 | Contacts | Mean steps | Success by 50 | Method-only / analytical-only |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['condition']} | {r['method']} | {r['success']} | {r['six']} | {r['collisions']} | {r['mean_steps']:.2f} | {r['by50']} | {r['method_only']} / {r['analytical_only']} |")
    lines+=['','29,696 new rollouts and 20,480 inherited rows comprise this comparison. Dataset and parent/new checkpoint hashes and all case keys are verified. Neural evaluation blocks simulator Jacobians and analytical joint-model solves; the intentional shared 3×3 control solve remains.','',
        '![E35 comparison](comparison.png)']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,denom,title in zip(axes,['success','six'],[1024,512],['All arms','Six-joint arms']):
        for method in ['analytical','old_0','old_1','old_2','new_0','new_1','new_2']:
            vals=[next(r[key] for r in summary if r['condition']==c['name'] and r['method']==method)/denom*100 for c in CONDITIONS]
            ax.plot(range(7),vals,'o--' if method.startswith('old') else 'o-',alpha=.5 if method.startswith('old') else 1,label=method)
        ax.set_xticks(range(7),[c['name'] for c in CONDITIONS],rotation=35,ha='right');ax.set_ylim(0,105)
        ax.set_title(title);ax.set_ylabel('Success (%)');ax.grid(alpha=.25)
    axes[1].legend(ncol=2);fig.tight_layout();fig.savefig(root/'comparison.png',dpi=160);plt.close(fig)
    for c in CONDITIONS:
        print(c['name'],[(r['method'],r['success'],r['six'],round(r['mean_steps'],2)) for r in summary if r['condition']==c['name']])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--previous',required=True);run(**vars(p.parse_args()))

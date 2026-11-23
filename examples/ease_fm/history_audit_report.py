"""Verify and plot frozen rollout interventions without importing the simulator."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def report(output,replication):
    root=Path(output); rows=json.loads((root/'summary.json').read_text())
    entries=json.loads((Path(replication)/'index.json').read_text())
    digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    checks=[]
    for entry in entries:
        seed=entry['seed']; result=json.loads((root/f'seed_{seed}.json').read_text())
        assert len(result['records'])==672 and not result['test_evaluated']
        assert result['checkpoint_sha256']==entry['student_sha256']
        for name in ['teacher','student']:
            assert digest(Path(entry[name].replace('\\','/'))/'policy.pt')==entry[name+'_sha256']
        key=lambda r:(r['body_seed'],r['goal_id'])
        base={key(r):r for r in result['records'] if r['mode']=='correct32' and r['split']=='validation_topology'}
        static=[r for r in result['records'] if r['mode']=='repeat_initial_static' and r['split']=='validation_topology']
        checks.append({'seed':seed,'novel_full_only':sum(base[key(r)]['success'] and not r['success'] for r in static),
                       'novel_static_only':sum(r['success'] and not base[key(r)]['success'] for r in static)})
    old=json.loads(Path('outputs/ease_history_audit_offline_v2/teacher_action_audit.json').read_text())
    new=json.loads((root/'teacher_action_audit.json').read_text())
    assert len(old)==len(new)
    for a,b in zip(old,new):
        assert all(a[k]==b[k] for k in ['seed','body_seed','source','band','states'])
    difference=max(abs(a['mae_rad']-b['mae_rad']) for a,b in zip(old,new))
    try: import matplotlib.pyplot as plt
    except ImportError:
        sys.path.insert(0,str(Path('.recovery-deps').resolve())); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split,title in zip(axes,['validation','validation_topology'],['New bodies, familiar axis families','Withheld axis arrangements']):
        for i,(mode,label) in enumerate([('correct32','Full history'),('repeat_initial_static','Repeated initial pose'),('dynamics_only','Pose features removed')]):
            cases=[r for r in rows if r['mode']==mode and r['split']==split]
            bars=ax.bar(np.arange(3)+(i-1)*.25,[100*r['successes']/r['trials'] for r in cases],.24,label=label)
            ax.bar_label(bars,labels=[f"{r['successes']}/{r['trials']}" for r in cases],fontsize=8,padding=2)
        ax.set_xticks([0,1,2],['Seed 0','Seed 1*','Seed 2']); ax.set_ylim(0,108)
        ax.set_ylabel('Task success (%)'); ax.set_title(title,fontsize=11); ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels(); fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.suptitle('E20: Initial pose cues nearly replace the history sequence')
    fig.text(.5,.01,'Frozen-policy interventions; not separately trained baselines. *Seed 1 failed training gates. No final-test evaluation.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.16,1,.94]); fig.savefig(root/'interventions.png',dpi=180); plt.close(fig)
    (root/'verification.json').write_text(json.dumps({'rollout_records':2016,'model_hashes_unchanged':True,
        'paired_novel_outcomes':checks,'teacher_audit_max_cross_platform_mae_difference_rad':difference,
        'report_source_sha256':digest(Path(__file__))},indent=2))
    print(json.dumps(checks,indent=2)); print('Maximum teacher MAE cross-platform difference',difference)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); parser.add_argument('--replication',required=True)
    report(**vars(parser.parse_args()))

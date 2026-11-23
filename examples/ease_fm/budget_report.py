"""Verify and plot the fixed-data training-budget experiment."""
import argparse
import json
from pathlib import Path
import sys

from .benchmark import fingerprint
from .budget_experiment import gate


def report(output):
    root=Path(output)
    rows=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert len(rows)==len(protocol['contexts'])*len(protocol['milestones'])
    for row in rows:
        folder=root/row['context']/str(row['steps'])
        meta=json.loads((folder/'training.json').read_text())
        mode='correct32' if row['context']=='history' else 'oracle'
        result=json.loads((folder/(mode+'.json')).read_text())
        assert fingerprint(folder/'policy.pt')==meta['policy_sha256']==result['checkpoint_sha256']
        assert fingerprint(folder/'continuation.pt')==meta['continuation_sha256']
        assert meta['manifest_sha256']==protocol['manifest_sha256']
        assert len(result['records'])==939 and not result['test_evaluated']
        assert meta['transitions']==protocol['training_rows'] and meta['steps']==row['steps']
        computed=gate(result['records'])
        assert computed['overall']==row['overall'] and computed['passed']==row['passed']
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.path.insert(0,str(Path('.recovery-deps').resolve()))
        import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for context,label in [('history','Measured history'),('privileged','True physical parameters')]:
        selected=[r for r in rows if r['context']==context]
        x=[r['steps']/1000 for r in selected]
        axes[0].plot(x,[100*r['overall'][0]/r['overall'][1] for r in selected],marker='o',label=label)
        axes[1].plot(x,[100*min(s/n for s,n in r['by_body'].values()) for r in selected],marker='o',label=label)
    for ax,title,threshold in zip(axes,['Overall training success','Lowest training-body success'],[95,90]):
        ax.axhline(threshold,color='gray',linestyle='--',label=f'{threshold}% gate')
        ax.set_title(title); ax.set_xlabel('Optimizer updates (thousands)'); ax.set_ylabel('Success (%)')
        ax.set_xticks([10,20,30]); ax.set_ylim(50,101); ax.spines[['top','right']].set_visible(False)
        ax.legend(loc='lower right',fontsize=8,frameon=False)
    fig.suptitle('E16: More training on the same recovery dataset')
    fig.text(.5,.015,'Seed 0; 939 training starts; 244,289 parameters. Privileged input is a diagnostic; no held-out evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.06,1,.96]); fig.savefig(root/'learning_curve.png',dpi=180); plt.close(fig)
    checks={'verified_checkpoints':len(rows),'held_out_evaluated':False,
            'report_source_sha256':fingerprint(Path(__file__)),
            'replay_checks':{context:json.loads((root/context/'replay_check.json').read_text()) for context in protocol['contexts']}}
    (root/'verification.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output'); report(**vars(parser.parse_args()))

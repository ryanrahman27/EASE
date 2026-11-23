"""Summarize training-context interventions and frozen unseen-arm validation."""
import argparse
import json
from pathlib import Path
import sys
from .benchmark import fingerprint


def report(training,validation):
    training=Path(training); validation=Path(validation)
    rows=json.loads((validation/'summary.json').read_text())
    protocol=json.loads((validation/'protocol.json').read_text())
    outcomes=json.loads((validation/'outcomes.json').read_text())
    assert fingerprint(training/'policy.pt')==protocol['checkpoint_sha256']==outcomes['checkpoint_sha256']
    assert len(outcomes['records'])==784 and not outcomes['test_evaluated']
    train_rows=json.loads((training/'summary.json').read_text())
    lines=['# E17: Frozen-controller history distillation','','| History | Training successes | Collisions | Gate |','|---|---:|---:|---|']
    for r in train_rows:
        s,n=r['overall']; lines.append(f"| {r['mode']} | {s}/{n} | {r['collisions']} | {r['passed']} |")
    (training/'summary.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.path.insert(0,str(Path('.recovery-deps').resolve()))
        import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split,title in zip(axes,['validation','validation_topology'],['New bodies, familiar axis families (6 arms)','Withheld axis arrangements (8 arms)']):
        subset={r['mode']:r for r in rows if r['split']==split}
        for offset,prefix,label,color in [(-.18,'correct','Correct history','#167a9e'),(.18,'wrong','Wrong-arrangement history','#b96831')]:
            cases=[subset[prefix+str(n)] for n in [8,16,32]]
            bars=ax.bar([i+offset for i in range(3)],[100*r['successes']/r['trials'] for r in cases],.35,label=label,color=color)
            ax.bar_label(bars,labels=[f"{r['successes']}/{r['trials']}" for r in cases],fontsize=9,padding=3)
        empty=subset['empty']; ax.axhline(100*empty['successes']/empty['trials'],color='gray',linestyle='--',label='Empty history')
        ax.set_xticks([0,1,2],['8','16','32']); ax.set_xlabel('Calibration transitions'); ax.set_ylim(0,108)
        ax.set_ylabel('Task success (%)'); ax.set_title(title,fontsize=11); ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.suptitle('E18: Frozen history-conditioned policy on unseen arms')
    fig.text(.5,.01,'One training seed; no target-arm training or weight updates. Development validation; final test reserved.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.15,1,.94]); fig.savefig(validation/'history_comparison.png',dpi=180); plt.close(fig)
    (validation/'verification.json').write_text(json.dumps({'checkpoint_unchanged':True,'case_records':784,
        'final_test_evaluated':False,'report_source_sha256':fingerprint(Path(__file__))},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--training',required=True); parser.add_argument('--validation',required=True)
    report(**vars(parser.parse_args()))

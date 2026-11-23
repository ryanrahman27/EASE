"""Summarize the full training-body gate without pooling historical simulator versions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from .benchmark import fingerprint


def report(output):
    root=Path(output)
    protocol=json.loads((root/'protocol.json').read_text())
    rows=[]; bodies=[]
    for phase in ['original','recovery']:
        for kind in ['flow','direct']:
            folder=root/(phase+'_'+kind)
            result=json.loads((folder/'correct32.json').read_text())
            meta=json.loads((folder/'training.json').read_text())
            if result['checkpoint_sha256']!=fingerprint(folder/'policy.pt') or meta['manifest_sha256']!=protocol['manifest_sha256']:
                raise ValueError('Artifact mismatch')
            records=result['records']
            row={'phase':phase,'model':kind,'successes':sum(r['success'] for r in records),'trials':len(records),
                'parameters':meta['parameters'],'training_rows':meta['transitions'],'mean_error_m':float(np.mean([r['error'] for r in records]))}
            for family in ['six','seven']:
                cases=[r for r in records if r['family'].startswith(family)]
                row[family]=[sum(r['success'] for r in cases),len(cases)]
            rows.append(row)
            for body in sorted({r['body_seed'] for r in records}):
                cases=[r for r in records if r['body_seed']==body]
                bodies.append({'phase':phase,'model':kind,'body_seed':body,'family':cases[0]['family'],
                    'successes':sum(r['success'] for r in cases),'trials':len(cases)})
    summary={'overall':rows,'by_body':bodies,'manifest_sha256':protocol['manifest_sha256'],
        'source_hashes':{p.name:fingerprint(p) for p in (root/'source').glob('*.py')}}
    (root/'control_report.json').write_text(json.dumps(summary,indent=2))
    lines=[f"# Full training-body control: {protocol.get('experiment', 'unspecified')}",'','Single training seed; synchronized observations; correct32 frozen calibration. No held-out policy evaluation.','',
        '| Data | Objective | Six-joint | Seven-joint | Total | Training examples |', '|---|---|---:|---:|---:|---:|']
    for r in rows:
        six='/'.join(map(str,r['six'])); seven='/'.join(map(str,r['seven']))
        lines.append(f"| {r['phase']} | {r['model']} | {six} | {seven} | {r['successes']}/{r['trials']} | {r['training_rows']} |")
    lines+=['','| Body | Original flow | Original direct | Recovery flow | Recovery direct |','|---|---:|---:|---:|---:|']
    for body in sorted({r['body_seed'] for r in bodies}):
        selected=[r for r in bodies if r['body_seed']==body]
        lines.append('| '+str(body)+' | '+' | '.join(f"{r['successes']}/{r['trials']}" for r in selected)+' |')
    (root/'control_report.md').write_text('\n'.join(lines)+'\n')
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        sys.path.insert(0,str(Path('.recovery-deps').resolve()))
        import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(max(11,len({r['body_seed'] for r in bodies})*1.1),5)); ids=sorted({r['body_seed'] for r in bodies}); x=np.arange(len(ids)); width=.19
    for index,(phase,kind) in enumerate([(p,k) for p in ['original','recovery'] for k in ['flow','direct']]):
        selected=[r for r in bodies if r['phase']==phase and r['model']==kind]
        ax.bar(x+(index-1.5)*width,[100*r['successes']/r['trials'] for r in selected],width,label=phase+' / '+kind)
    labels=[]; family_counts={}
    for body in ids:
        family=next(r['family'] for r in bodies if r['body_seed']==body)
        family_counts[family]=family_counts.get(family,0)+1
        labels.append(f"{family}\nbody {family_counts[family]}")
    ax.set_xticks(x,labels)
    ax.tick_params(axis='x',labelsize=8)
    ax.set_ylabel('Training-episode success (%)'); ax.set_ylim(0,105)
    ax.axhline(90,color='gray',linestyle='--',linewidth=1,label='Per-body gate (90%)')
    ax.spines[['top','right']].set_visible(False)
    ax.set_title('Full training-body control with synchronized observations')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
    fig.text(.5,.01,'One training seed; same original episodes and shared recovery data. No test-body policy evaluation.',ha='center',fontsize=9)
    fig.tight_layout(rect=[0,.13,1,1]); fig.savefig(root/'control_comparison.png',dpi=180); plt.close(fig)
    print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('output'); report(**vars(p.parse_args()))

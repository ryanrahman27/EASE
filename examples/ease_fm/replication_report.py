"""Audit and summarize all prespecified seeds without selecting successes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .benchmark import fingerprint
from .history_distill import encoder_key


def report(output):
    root=Path(output); entries=json.loads((root/'index.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    assert [r['seed'] for r in entries]==protocol['seeds']
    rows=[]; families=[]; verifications=[]
    for entry in entries:
        teacher_path=Path(entry['teacher'])/'policy.pt'; student_path=Path(entry['student'])/'policy.pt'
        assert fingerprint(teacher_path)==entry['teacher_sha256'] and fingerprint(student_path)==entry['student_sha256']
        teacher=torch.load(teacher_path,map_location='cpu',weights_only=True)
        student=torch.load(student_path,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,teacher[k]) for k,v in student.items() if not encoder_key(k))
        for model in ['teacher','student']:
            artifact=json.loads(Path(entry[model+'_validation']).read_text())
            assert artifact['checkpoint_sha256']==entry[model+'_sha256']
            assert artifact['manifest_sha256']==protocol['manifest_sha256'] and not artifact['test_evaluated']
            records=artifact['records']
            assert len(records)==(112 if model=='teacher' else 784)
            assert all(r['split'] in ['validation','validation_topology'] for r in records)
            if model=='teacher': assert artifact['privileged_inputs'] and all(r['calibration_actions']==0 for r in records)
            for mode in sorted({r['mode'] for r in records}):
                for split in ['validation','validation_topology']:
                    selected=[r for r in records if r['mode']==mode and r['split']==split]
                    row={'seed':entry['seed'],'model':model,'mode':mode,'split':split,
                        'successes':sum(r['success'] for r in selected),'trials':len(selected),'collisions':sum(r['collision'] for r in selected),
                        'diagnostic_due_to_failed_training_gate':entry['diagnostic_validation_due_to_failed_gate']}
                    rows.append(row)
                    for family in sorted({r['family'] for r in selected}):
                        cases=[r for r in selected if r['family']==family]
                        families.append({**row,'family':family,'successes':sum(r['success'] for r in cases),
                                         'trials':len(cases),'collisions':sum(r['collision'] for r in cases)})
        verifications.append({'seed':entry['seed'],'frozen_controller_exact':True,'hashes_verified':True})
    (root/'summary.json').write_text(json.dumps({'by_seed':rows,'by_family':families},indent=2))
    def count(seed,model,mode,split):
        return next(r for r in rows if (r['seed'],r['model'],r['mode'],r['split'])==(seed,model,mode,split))
    lines=['# E19: Three-seed replication','','Fixed dataset; seed 0 is the earlier pilot, seeds 1/2 are additional prespecified runs. No seed omitted.','',
        '| Seed | Teacher training | Student training | Both gates pass |','|---|---:|---:|---|']
    for e in entries:
        lines.append(f"| {e['seed']} | {'/'.join(map(str,e['teacher_gate']['overall']))} | {'/'.join(map(str,e['student_gate']['overall']))} | {not e['diagnostic_validation_due_to_failed_gate']} |")
    for split,label in [('validation','New bodies in familiar axis families'),('validation_topology','Withheld axis arrangements')]:
        lines+=['',f'## {label}','','| Seed | True parameters | Empty | Correct8 | Wrong8 | Correct16 | Wrong16 | Correct32 | Wrong32 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for e in entries:
            cases=[count(e['seed'],'teacher','oracle',split)]+[count(e['seed'],'student',m,split) for m in ['empty','correct8','wrong8','correct16','wrong16','correct32','wrong32']]
            lines.append('| '+str(e['seed'])+' | '+' | '.join(f"{r['successes']}/{r['trials']}" for r in cases)+' |')
    lines+=['','## Novel-family diagnosis','','| Family | Seed | True parameters | Correct32 student |','|---|---:|---:|---:|']
    for family in ['six_d','six_e','seven_d','seven_e']:
        for seed in protocol['seeds']:
            cases=[next(r for r in families if r['family']==family and r['seed']==seed and r['model']==model and r['mode']==mode)
                   for model,mode in [('teacher','oracle'),('student','correct32')]]
            lines.append('| '+family+' | '+str(seed)+' | '+' | '.join(f"{r['successes']}/{r['trials']}" for r in cases)+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    try: import matplotlib.pyplot as plt
    except ImportError:
        sys.path.insert(0,str(Path('.recovery-deps').resolve())); import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    conditions=[('student','correct32','Correct history'),('student','empty','Empty history'),
                ('student','wrong32','Wrong history'),('teacher','oracle','True parameters')]
    for ax,split,title in zip(axes,['validation','validation_topology'],['New bodies, familiar families','Withheld axis arrangements']):
        for i,(model,mode,label) in enumerate(conditions):
            cases=[count(s,model,mode,split) for s in protocol['seeds']]
            ax.bar(np.arange(3)+(i-1.5)*.19,[100*r['successes']/r['trials'] for r in cases],.19,label=label)
        ax.set_xticks([0,1,2],['Seed 0 (pilot)','Seed 1*','Seed 2']); ax.set_ylim(0,105)
        ax.set_ylabel('Task success (%)'); ax.set_title(title); ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.045),ncol=4,frameon=False)
    fig.suptitle('E19: Replication with fixed training data and evaluation arms')
    fig.text(.5,.01,'*Seed 1 failed training gates. History: 32 transitions. True parameters: privileged. No target-arm training; final test reserved.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.16,1,.94]); fig.savefig(root/'replication.png',dpi=180); plt.close(fig)
    (root/'verification.json').write_text(json.dumps({'seeds':verifications,'report_source_sha256':fingerprint(Path(__file__))},indent=2))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('output'); report(**vars(parser.parse_args()))

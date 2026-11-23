"""Frozen unseen-arm validation after the distilled policy passes training control."""
import argparse
import json
from pathlib import Path
import shutil
from .benchmark import fingerprint
from .diverse_validation import evaluate


def run(data,run,output,device='cuda'):
    run=Path(run); root=Path(output)
    gates=json.loads((run/'summary.json').read_text())
    assert next(r for r in gates if r['mode']=='correct32')['passed']
    root.mkdir(parents=True,exist_ok=False)
    checkpoint=run/'policy.pt'
    protocol={'experiment':'E18','scope':'Frozen distilled policy on unseen validation arms',
        'checkpoint_sha256':fingerprint(checkpoint),'manifest_sha256':fingerprint(Path(data)/'manifest.json'),
        'modes':['empty','correct8','wrong8','correct16','wrong16','correct32','wrong32'],
        'training_gate_passed':True,'weights_updated':False,'target_arm_training':False,
        'physical_parameters_at_inference':False,'test_evaluated':False,
        'calibration':'Separate excitation episode followed by task reset; frozen history and weights during reaching',
        'comparison':'Correct versus both empty and same-DoF different-axis training-arm history; familiar-family and novel-family validation reported separately',
        'limitations':'One model seed; 6 familiar-family bodies and 8 novel-family bodies; development validation, no target-arm specialist comparison'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    snapshot=root/'source'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path,snapshot/path.name)
    evaluate(data,checkpoint,root/'outcomes.json',device)
    records=json.loads((root/'outcomes.json').read_text())['records']
    rows=[]
    for mode in protocol['modes']:
        for split in ['validation','validation_topology']:
            subset=[r for r in records if r['mode']==mode and r['split']==split]
            rows.append({'mode':mode,'split':split,'successes':sum(r['success'] for r in subset),
                'trials':len(subset),'collisions':sum(r['collision'] for r in subset),
                'by_family':{f:[sum(r['success'] for r in subset if r['family']==f),sum(r['family']==f for r in subset)]
                             for f in sorted({r['family'] for r in subset})}})
    assert fingerprint(checkpoint)==protocol['checkpoint_sha256']
    assert len(records)==784 and not any(r['split']=='test' for r in records)
    (root/'summary.json').write_text(json.dumps(rows,indent=2))
    lines=['# E18: Frozen unseen-arm validation','','| History | Familiar families | Unseen axis arrangements |','|---|---:|---:|']
    for mode in protocol['modes']:
        values=[r for r in rows if r['mode']==mode]
        lines.append('| '+mode+' | '+' | '.join(f"{r['successes']}/{r['trials']}" for r in values)+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['data','run','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda'); run(**vars(parser.parse_args()))

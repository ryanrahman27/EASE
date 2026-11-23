"""Three-seed matched trained pose-only baseline; teachers/data/budgets fixed."""
import argparse
import json
from pathlib import Path
import shutil
from .benchmark import fingerprint
from .history_distill import run as distill
from .diverse_validation import evaluate


def run(data,source,replication,output,device='cuda'):
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    entries=json.loads((Path(replication)/'index.json').read_text())
    protocol={'experiment':'E21','seeds':[0,1,2],'steps':10000,'batch':32,'lr':.0003,
        'input':'First q and ee from sampled context, repeated; no velocity/actions/deltas',
        'matching':'Same seed, architecture, frozen teacher, training rows, sampled contexts, lengths, normalization and optimizer as E17/E19; changed encoder input during training and inference',
        'limitations':'First sampled training-window pose need not be an episode initial pose; evaluation uses initial calibration pose. Full-history baseline trained on Windows, pose model in WSL; intact policy replay verified in E20.',
        'gate_policy':'Keep all seeds and report failed training gates; no tuning on validation',
        'manifest_sha256':fingerprint(Path(data)/'manifest.json'),'final_test_evaluated':False}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2)); snapshot=root/'source'; snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p,snapshot/p.name)
    index=[]
    for entry in entries:
        seed=entry['seed']; teacher=Path(entry['teacher'].replace('\\','/'))/'policy.pt'
        folder=root/f'seed_{seed}'
        distill(data,source,teacher,folder,steps=10000,device=device,seed=seed,diagnostic=True,pose_only=True)
        path=root/f'validation_{seed}.json'; evaluate(data,folder/'policy.pt',path,device)
        index.append({'seed':seed,'pose':str(folder),'full_history':entry['student'].replace('\\','/'),
            'teacher':str(teacher),'teacher_sha256':fingerprint(teacher),'pose_sha256':fingerprint(folder/'policy.pt'),
            'validation':str(path),'full_validation':entry['student_validation'].replace('\\','/'),
            'full_training_gate':entry['student_gate'],
            'pose_training_gate':next(r for r in json.loads((folder/'summary.json').read_text()) if r['mode']=='correct32')})
        (root/'index.json').write_text(json.dumps(index,indent=2)); print('COMPLETED POSE SEED',seed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['data','source','replication','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda'); run(**vars(parser.parse_args()))

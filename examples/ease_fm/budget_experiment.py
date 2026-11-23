"""Fixed-data direct-policy learning curves with preserved optimizer state."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .benchmark import fingerprint
from .mixed_data import MixedData
from .diverse_oracle import OracleData, oracle_rollout
from .full_control import add_recovery, fit, rollout


def gate(records):
    bodies={b:[r for r in records if r['body_seed']==b] for b in sorted({r['body_seed'] for r in records})}
    counts={b:[sum(r['success'] for r in rows),len(rows)] for b,rows in bodies.items()}
    success=sum(r['success'] for r in records)
    return {'passed':success/len(records)>=.95 and all(s/n>=.9 for s,n in counts.values()),
            'overall':[success,len(records)],'by_body':counts}


def run(data,source,oracle_source,output,device='cuda'):
    torch.set_num_threads(4)
    root=Path(output); source=Path(source); oracle_source=Path(oracle_source)
    manifest_hash=fingerprint(Path(data)/'manifest.json')
    previous=json.loads((source/'protocol.json').read_text())
    assert previous['manifest_sha256']==manifest_hash
    shared=json.loads((source/'shared_recovery.json').read_text())
    for relative,digest in shared['hashes'].items():
        assert fingerprint(source/relative)==digest
    root.mkdir(parents=True,exist_ok=False)
    milestones=[10000,20000,30000]
    protocol={'experiment':'E16','scope':'Fixed-data training-budget diagnostic; no held-out evaluation',
        'milestones':milestones,'seed':0,'batch_size':32,'lr':.0003,'objective':'direct',
        'contexts':['history','privileged'],'parameters':244289,'manifest_sha256':manifest_hash,
        'recovery_hashes':shared['hashes'],'training_rows':shared['combined_transitions'],
        'training':'Replay from original initialization because E15 saved no optimizer state; uninterrupted optimizer and random streams through 30000 steps; checkpoint only during training',
        'selection':'Report all budgets; no early stopping or held-out tuning',
        'gate':'95 percent overall and 90 percent per training body',
        'test_evaluated':False,'validation_evaluated':False}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    snapshot=root/'source'; snapshot.mkdir()
    for path in Path(__file__).parent.glob('*.py'): shutil.copyfile(path,snapshot/path.name)
    rows=[]
    for context,cls,baseline in [('history',MixedData,source/'recovery_direct'),('privileged',OracleData,oracle_source/'direct')]:
        dataset=add_recovery(cls(data),[source/f'original_{kind}/recovery.npz' for kind in ['flow','direct']])
        assert len(dataset.row_episode)==shared['combined_transitions']
        assert dataset.manifest.get('synchronized_observations',False)
        folder=root/context; folder.mkdir()

        def save_checkpoint(step,model,opt,losses,rng):
            checkpoint=folder/str(step); checkpoint.mkdir()
            torch.save(model.state_dict(),checkpoint/'policy.pt')
            torch.save({'step':step,'optimizer':opt.state_dict(),'rng':rng},checkpoint/'continuation.pt')
            (checkpoint/'training.json').write_text(json.dumps({'kind':'direct','seed':0,'steps':step,
                'losses':losses,'transitions':len(dataset.row_episode),'parameters':sum(p.numel() for p in model.parameters()),
                'manifest_sha256':manifest_hash,'recovery_sha256':fingerprint(source/'shared_recovery.json'),
                'policy_sha256':fingerprint(checkpoint/'policy.pt'),'continuation_sha256':fingerprint(checkpoint/'continuation.pt'),
                'context':context,'torch':torch.__version__,'device':device},indent=2))

        model=fit(dataset,folder/'fit','direct',milestones[-1],device,manifest_hash,
                  fingerprint(source/'shared_recovery.json'),milestones,save_checkpoint)
        old=torch.load(baseline/'policy.pt',map_location='cpu',weights_only=True)
        replay=torch.load(folder/'10000/policy.pt',map_location='cpu',weights_only=True)
        comparison={'exact_tensor_match':all(torch.equal(old[k],replay[k]) for k in old),
            'max_absolute_difference':max(float((old[k].double()-replay[k].double()).abs().max()) for k in old),
            'baseline_sha256':fingerprint(baseline/'policy.pt')}
        (folder/'replay_check.json').write_text(json.dumps(comparison,indent=2))
        print(context,'10000-step replay',comparison,flush=True)
        for step in milestones:
            checkpoint=folder/str(step)
            model.load_state_dict(torch.load(checkpoint/'policy.pt',map_location=device,weights_only=True)); model.eval()
            records=(rollout if context=='history' else oracle_rollout)(dataset,model,'direct',checkpoint,device,
                     mode='correct32' if context=='history' else 'oracle')
            result=gate(records)
            (checkpoint/'gate.json').write_text(json.dumps(result,indent=2))
            failed=[r for r in records if not r['success']]
            row={'context':context,'steps':step,**result,'collisions':sum(r['collision'] for r in records),
                 'failed_median_error_m':float(np.median([r['error'] for r in failed])) if failed else None}
            rows.append(row)
            (root/'summary.json').write_text(json.dumps(rows,indent=2))
            print(context,step,result,flush=True)
        del model,dataset
    lines=['# E16: Fixed-data training budget','','Single seed, direct regression, all 939 training starts; no held-out policy evaluation.','',
           '| Context | Updates | Success | Gate |','|---|---:|---:|---|']
    for row in rows:
        s,n=row['overall']; lines.append(f"| {row['context']} | {row['steps']} | {s}/{n} ({100*s/n:.1f}%) | {row['passed']} |")
    (root/'summary.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['data','source','oracle_source','output']: parser.add_argument('--'+name.replace('_','-'),required=True)
    parser.add_argument('--device',default='cuda')
    run(**vars(parser.parse_args()))

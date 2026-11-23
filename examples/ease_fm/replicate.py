"""Prespecified seed replication and privileged unseen-arm diagnostic."""
import argparse
import json
from pathlib import Path
import shutil
import torch
from .benchmark import fingerprint
from .budget_experiment import gate
from .diverse_oracle import OracleData, oracle_rollout
from .diverse_validation import evaluate
from .full_control import add_recovery, fit
from .history_distill import run as distill


def run(data,source,output,device='cuda'):
    torch.set_num_threads(4)
    source=Path(source); root=Path(output)
    manifest_hash=fingerprint(Path(data)/'manifest.json')
    shared=json.loads((source/'shared_recovery.json').read_text())
    for name,digest in shared['hashes'].items(): assert fingerprint(source/name)==digest
    root.mkdir(parents=True,exist_ok=False)
    protocol={'experiment':'E19','seeds':[0,1,2],'new_seeds':[1,2],
        'teacher_steps':30000,'student_steps':10000,'batch':32,'lr':.0003,
        'manifest_sha256':manifest_hash,'recovery_hashes':shared['hashes'],
        'fixed_data':'Same seed-0 recovery union for all seeds; this tests optimization variability conditional on a fixed dataset',
        'seed_effect':'Teacher and student initialization plus training batch/context streams vary together; no isolated variance attribution',
        'selection':'No seed dropping, checkpoint selection, early stopping, or validation tuning; failed gates retained and validation labeled diagnostic',
        'validation':'Same 14 arms and 8 goals; student empty/correct/wrong 8/16/32; privileged teacher uses true parameters, no target-arm training',
        'primary':'Per-seed correct32 versus both empty and wrong32 on novel families; all history lengths and families reported',
        'diagnosis':'Compare frozen teacher and student on each withheld family; privileged reference is not a formal upper bound',
        'final_test_evaluated':False,'seed0':'Previously observed pilot; seeds 1 and 2 prespecified additional runs'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    snapshot=root/'source'; snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p,snapshot/p.name)
    index=[]
    for seed in protocol['seeds']:
        folder=root/f'seed_{seed}'; folder.mkdir()
        if seed==0:
            teacher=Path('outputs/ease_budget_v1/privileged/30000')
            student=Path('outputs/ease_distill_v1')
            student_validation=Path('outputs/ease_distill_validation_v1/outcomes.json')
            teacher_gate=json.loads((teacher/'gate.json').read_text())
        else:
            dataset=add_recovery(OracleData(data),[source/f'original_{kind}/recovery.npz' for kind in ['flow','direct']])
            teacher=folder/'teacher'
            def checkpoint(step,model,opt,losses,rng):
                torch.save({'step':step,'optimizer':opt.state_dict(),'rng':rng},folder/'teacher_continuation.pt')
            print('START teacher seed',seed,flush=True)
            model=fit(dataset,teacher,'direct',30000,device,manifest_hash,fingerprint(source/'shared_recovery.json'),
                      checkpoint_steps=[30000],on_checkpoint=checkpoint,seed=seed)
            records=oracle_rollout(dataset,model,'direct',teacher,device,mode='oracle')
            teacher_gate=gate(records)
            (teacher/'gate.json').write_text(json.dumps(teacher_gate,indent=2))
            del model,dataset
            student=folder/'student'
            print('START student seed',seed,'teacher gate',teacher_gate,flush=True)
            distill(data,source,teacher/'policy.pt',student,steps=10000,device=device,seed=seed,diagnostic=True)
            student_validation=folder/'student_validation.json'
            evaluate(data,student/'policy.pt',student_validation,device)
        teacher_validation=folder/'teacher_validation.json'
        evaluate(data,teacher/'policy.pt',teacher_validation,device,privileged=True)
        student_gate=next(r for r in json.loads((student/'summary.json').read_text()) if r['mode']=='correct32')
        entry={'seed':seed,'teacher':str(teacher),'student':str(student),
            'teacher_gate':teacher_gate,'student_gate':student_gate,
            'diagnostic_validation_due_to_failed_gate':not (teacher_gate['passed'] and student_gate['passed']),
            'teacher_validation':str(teacher_validation),'student_validation':str(student_validation),
            'teacher_sha256':fingerprint(teacher/'policy.pt'),'student_sha256':fingerprint(student/'policy.pt')}
        index.append(entry); (root/'index.json').write_text(json.dumps(index,indent=2))
        print('COMPLETED seed',seed,flush=True)
    print('All prespecified seeds completed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['data','source','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--device',default='cuda'); run(**vars(parser.parse_args()))

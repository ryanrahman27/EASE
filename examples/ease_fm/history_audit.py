"""Frozen feature interventions and teacher state-fitting audit."""
import argparse
import json
from pathlib import Path
import shutil
import hashlib
import sys
import platform
import numpy as np
import torch
from .joint_policy import JointFlowPolicy

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

MODES=['correct32','shuffle_actions','shuffle_transitions','static_only','repeat_initial_static','dynamics_only']


def intervene(contexts,mode,mask):
    result=contexts.copy()
    # Fixed permutations shared by all cases/seeds; avoid altering any RNG stream.
    order=np.random.default_rng(20261018).permutation(32)
    static=[0,5,6,7]
    dynamic=[1,2,3,4,8,9,10]
    if mode=='shuffle_actions': result[:,:,:,2]=contexts[:,:,:,2][:,order,:]
    elif mode=='shuffle_transitions': result=contexts[:,order].copy()
    elif mode=='static_only': result[:,:,:,dynamic]=0
    elif mode=='repeat_initial_static':
        result[:]=contexts[:,:1]; result[:,:,:,dynamic]=0
    elif mode=='dynamics_only': result[:,:,:,static]=0
    elif mode!='correct32': raise ValueError(mode)
    result*=mask[:,None,:,None]
    return result


@torch.no_grad()
def teacher_audit(data,source,entries,root,device):
    data=Path(data); source=Path(source)
    manifest=json.loads((data/'manifest.json').read_text())
    assert fingerprint(data/'manifest.json')==(data/'manifest.sha256').read_text()
    arms=[a for a in manifest['arms'] if a['split']=='train']; ids={a['seed']:i for i,a in enumerate(arms)}
    episodes=[e for e in manifest['episodes'] if e['split']=='train' and e['success'] and not e['collision']]
    qs=[]; tasks=[]; targets=[]; bs=[]
    for episode in episodes:
        path=data/episode['file']; assert fingerprint(path)==manifest['asset_hashes'][episode['file']]
        with np.load(path) as values:
            n,dof=values['joints'].shape[:2]
            padded=np.zeros((n,7,2),np.float32); padded[:,:dof]=values['joints']; qs.append(padded)
            action=np.zeros((n,7),np.float32); action[:,:dof]=values['action']/.04*.04; targets.append(action)
            tasks.append(values['task']); bs.append(np.full(n,ids[episode['body_seed']]))
    original_count=sum(len(x) for x in qs)
    shared=json.loads((source/'shared_recovery.json').read_text())
    for kind in ['flow','direct']:
        path=source/f'original_{kind}/recovery.npz'
        key=next(k for k in shared['hashes'] if Path(k.replace('\\','/')).name==path.name and f'original_{kind}' in k)
        assert fingerprint(path)==shared['hashes'][key]
        with np.load(path) as values:
            qs.append(values['joints']); tasks.append(values['task']); targets.append(values['action'][:,0]*.04)
            bs.append(np.asarray([ids[episodes[int(i)]['body_seed']] for i in values['episode']]))
    bodies=np.concatenate(bs); q=torch.from_numpy(np.concatenate(qs)); task=torch.from_numpy(np.concatenate(tasks))
    target=np.concatenate(targets)
    assert len(q)==shared['combined_transitions']
    masks=np.asarray([[j<len(a['lengths']) for j in range(7)] for a in arms])
    mask=torch.tensor(masks[bodies])
    physical=np.zeros((len(arms),32,7,11),np.float32)
    for i,a in enumerate(arms):
        for j,axis in enumerate(manifest['axis_catalog'][a['family']]):
            physical[i,:,j,:3]=[float(x) for x in axis.split()]
            physical[i,:,j,3:7]=[a['lengths'][j],a['masses'][j],a['strength'],a['damping']]
    rows=[]
    for entry in entries:
        folder=Path(entry['teacher']); model=JointFlowPolicy(history=True).to(device)
        model.load_state_dict(torch.load(folder/'policy.pt',map_location=device,weights_only=True)); model.eval()
        assert fingerprint(folder/'policy.pt')==entry['teacher_sha256']
        bodymask=torch.tensor(masks,device=device)
        latent=model.encode(torch.tensor(physical,device=device),
                            torch.full((len(arms),),32,device=device),bodymask)
        predictions=[]
        for start in range(0,len(q),128):
            end=min(start+128,len(q))
            pred=model(torch.zeros(end-start,16,7,device=device),torch.zeros(end-start,device=device),
                       q[start:end].to(device),task[start:end].to(device),mask[start:end].to(device),latent[bodies[start:end]])[:,0].clamp(-1,1)*.04
            predictions.append(pred.cpu())
        pred=torch.cat(predictions).numpy()
        distance=np.linalg.norm(task[:,:3].numpy()-task[:,3:].numpy(),axis=1)
        for i,arm in enumerate(arms):
            dof=len(arm['lengths'])
            for source_name,source_mask in [('original',np.arange(len(q))<original_count),('recovery',np.arange(len(q))>=original_count)]:
                for band,band_mask in [('near',distance<.06),('far',distance>=.06)]:
                    selected=(bodies==i)&source_mask&band_mask
                    if not selected.any(): continue
                    error=pred[selected,:dof]-target[selected,:dof]
                    rows.append({'seed':entry['seed'],'body_seed':arm['seed'],'family':arm['family'],'source':source_name,'band':band,
                        'states':int(selected.sum()),'mae_rad':float(np.abs(error).mean()),'rmse_rad':float(np.sqrt(np.square(error).mean())),
                        'expert_mean_abs_rad':float(np.abs(target[selected,:dof]).mean()),
                        'teacher_training_success':entry['teacher_gate']['by_body'][str(arm['seed'])]})
        assert fingerprint(folder/'policy.pt')==entry['teacher_sha256']
        print('teacher audit complete seed',entry['seed'],flush=True)
    (root/'teacher_action_audit.json').write_text(json.dumps(rows,indent=2))


@torch.no_grad()
def offline_history_probe(data,entries,root,device):
    data=Path(data); manifest=json.loads((data/'manifest.json').read_text())
    arms={a['seed']:a for a in manifest['arms'] if a['split']=='train'}
    histories=[]; physical=[]; masks=[]; calibrations=manifest['calibrations']
    for calibration in calibrations:
        assert calibration['split']=='train'
        a=arms[calibration['body_seed']]; dof=len(a['lengths'])
        path=data/calibration['file']; assert fingerprint(path)==manifest['asset_hashes'][calibration['file']]
        h=np.load(path); assert h.shape==(32,dof,11)
        histories.append(np.pad(h,((0,0),(0,7-dof),(0,0))))
        values=np.zeros((32,7,11),np.float32)
        for j,axis in enumerate(manifest['axis_catalog'][a['family']]):
            values[:,j,:3]=[float(x) for x in axis.split()]
            values[:,j,3:7]=[a['lengths'][j],a['masses'][j],a['strength'],a['damping']]
        physical.append(values); masks.append([j<dof for j in range(7)])
    histories=np.stack(histories); masks=np.asarray(masks); mask=torch.tensor(masks,device=device)
    lengths=torch.full((len(histories),),32,device=device); rows=[]
    for entry in entries:
        teacher=JointFlowPolicy(history=True).to(device).eval(); student=JointFlowPolicy(history=True).to(device).eval()
        teacher.load_state_dict(torch.load(Path(entry['teacher'])/'policy.pt',map_location=device,weights_only=True))
        student.load_state_dict(torch.load(Path(entry['student'])/'policy.pt',map_location=device,weights_only=True))
        target=teacher.encode(torch.tensor(np.stack(physical),device=device),lengths,mask)
        for mode in MODES:
            z=student.encode(torch.tensor(intervene(histories,mode,masks),device=device),lengths,mask)
            error=((z-target).square()*mask[...,None]).sum((1,2))/(mask.sum(1)*student.width)
            for i,c in enumerate(calibrations):
                rows.append({'seed':entry['seed'],'mode':mode,'file':c['file'],'body_seed':c['body_seed'],
                    'family':arms[c['body_seed']]['family'],'latent_mse':float(error[i])})
        assert fingerprint(Path(entry['student'])/'policy.pt')==entry['student_sha256']
    (root/'offline_latent_probe.json').write_text(json.dumps({'scope':'Existing calibration episodes used in training; not independent validation or rollout success','records':rows},indent=2))


def run(data,source,replication,output,device='cuda',offline_only=False):
    torch.set_num_threads(4)
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    entries=json.loads((Path(replication)/'index.json').read_text())
    for entry in entries:
        for name in ['teacher','student','teacher_validation','student_validation']:
            entry[name]=entry[name].replace('\\','/')
    protocol={'offline_only':offline_only,'simulation_blocker':'Windows Application Control prevents MuJoCo DLL loading' if offline_only else None,'experiment':'E20','date':'2026-10-18','seeds':[e['seed'] for e in entries],
        'manifest_sha256':fingerprint(Path(data)/'manifest.json'),'modes':MODES,
        'features':'q,dq,action/.04,delta_q,delta_dq,ee_xyz,delta_ee_xyz',
        'static_columns':[0,5,6,7],'dynamic_columns':[1,2,3,4,8,9,10],
        'shuffle_actions':'Fixed permutation of action column only; state and response columns unchanged',
        'shuffle_transitions':'Same permutation of complete transitions; preserves within-transition pairing',
        'static_only':'Keep q and ee across time, zero velocity/action/deltas; movement remains implicit in the pose sequence',
        'repeat_initial_static':'Repeat the first q and ee across 32 positions; zero other columns',
        'dynamics_only':'Zero q and ee; retain velocity/action/deltas',
        'interpretation':'Frozen feature interventions are potentially out of distribution; loss of performance is not a causal proof of necessity',
        'calibration':'All conditions reuse 32 collected transitions; repeated static input does not establish a zero-interaction trained baseline',
        'teacher_audit':'First-action error on all fixed original/recovery training states by body and near (<.06m)/far goal distance; state-weighted, not episode-balanced',
        'weights_updated':False,'final_test_evaluated':False,'seed1_failed_training_gate_retained':True,
        'runtime':{'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda},
        'replay':'Record cross-platform differences from E19; compare interventions against current-runtime correct32'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    snapshot=root/'source'; snapshot.mkdir()
    for p in Path(__file__).parent.glob('*.py'): shutil.copyfile(p,snapshot/p.name)
    rows=[]; replays=[]
    if not offline_only:
        from .diverse_validation import evaluate
    for entry in ([] if offline_only else entries):
        path=root/f"seed_{entry['seed']}.json"
        evaluate(data,Path(entry['student'])/'policy.pt',path,device,intervention=intervene,intervention_modes=MODES)
        records=json.loads(path.read_text())['records']
        reference=json.loads(Path(entry['student_validation']).read_text())['records']
        key=lambda r:(r['body_seed'],r['goal_id'])
        baseline={key(r):r for r in reference if r['mode']=='correct32'}
        current=[r for r in records if r['mode']=='correct32']
        replays.append({'seed':entry['seed'],'cases':len(current),
            'success_disagreements':sum(r['success']!=baseline[key(r)]['success'] for r in current),
            'collision_disagreements':sum(r['collision']!=baseline[key(r)]['collision'] for r in current),
            'max_final_error_difference_m':max(abs(r['error']-baseline[key(r)]['error']) for r in current)})
        (root/'replay_checks.json').write_text(json.dumps(replays,indent=2))
        for mode in MODES:
            for split in ['validation','validation_topology']:
                selected=[r for r in records if r['mode']==mode and r['split']==split]
                rows.append({'seed':entry['seed'],'mode':mode,'split':split,'successes':sum(r['success'] for r in selected),
                    'trials':len(selected),'collisions':sum(r['collision'] for r in selected),
                    'diagnostic_failed_training_gate':entry['diagnostic_validation_due_to_failed_gate']})
        assert fingerprint(Path(entry['student'])/'policy.pt')==entry['student_sha256']
        (root/'summary.json').write_text(json.dumps(rows,indent=2))
    teacher_audit(data,source,entries,root,device)
    if offline_only:
        offline_history_probe(data,entries,root,device)
        (root/'status.json').write_text(json.dumps({'teacher_audit_completed':True,'offline_training_calibration_probe_completed':True,'rollout_interventions_completed':False,'reason':protocol['simulation_blocker']},indent=2))
        return
    lines=['# E20: Frozen history interventions','','Interventions compared with intact history in this runtime; see replay_checks.json for cross-platform differences. Seed 1 failed the training gates.','']
    for split in ['validation','validation_topology']:
        lines += [f'## {split}','','| Mode | Seed 0 | Seed 1 | Seed 2 |','|---|---:|---:|---:|']
        for mode in MODES:
            cases=[r for r in rows if r['mode']==mode and r['split']==split]
            lines.append('| '+mode+' | '+' | '.join(f"{r['successes']}/{r['trials']}" for r in cases)+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n'); print('\n'.join(lines),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ['data','source','replication','output']: parser.add_argument('--'+name,required=True)
    parser.add_argument('--offline-only',action='store_true')
    parser.add_argument('--device',default='cuda'); run(**vars(parser.parse_args()))

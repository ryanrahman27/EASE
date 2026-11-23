"""E33: frozen-policy development suite beyond the paired axis families."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .benchmark import fingerprint
from .mixed_sim import Arm, AXES, MixedReach
from .paired_data import register
from .kinematic_control import IdentifiedController
from .motion_transformer import MotionTransformer, evaluate

VECTORS={'X':'1 0 0','Y':'0 1 0','Z':'0 0 1'}
VARIANTS=[(False,False),(True,False),(False,True),(True,True)]


def spatial_sequence(sequence):
    name='broad_rank_probe';n=len(sequence)
    AXES[name]=[VECTORS[c] for c in sequence]
    try:
        env=MixedReach(Arm(-1,'structural',name,tuple([.15]*n),tuple([1.]*n),1.,2.))
        for phase in (0.,.7,1.4):
            env.data.qpos[:]=.35*np.sin(np.arange(n)+phase)+.15
            mujoco.mj_forward(env.model,env.data)
            jac=np.zeros((3,n));mujoco.mj_jacSite(env.model,env.data,jac,None,0)
            if np.linalg.matrix_rank(jac,tol=1e-7)==3:return True
        return False
    finally:
        AXES.pop(name,None)


def axis_sequences(rng, excluded, per_dof=8, audit=None):
    used=set(excluded);result=[]
    for dof in (6,7):
        accepted=0
        while accepted<per_dof:
            sequence=''.join(rng.choice(list(VECTORS),dof))
            if sequence in used:continue
            valid=len(set(sequence))>=2 and spatial_sequence(sequence)
            if audit is not None:audit.append(dict(kind='axis',sequence=sequence,accepted=bool(valid)))
            if not valid:continue
            used.add(sequence);result.append(sequence);accepted+=1
    return result


def family_arms(sequence, fid, rng):
    n=len(sequence);name=f'broad_{n}_{fid}'
    nominal=np.array([.2,.22,.2,.16,.14,.12] if n==6 else [.16,.18,.16,.14,.12,.10,.08])
    lengths=nominal*rng.uniform(.8,1.2,n)
    shifted=nominal*rng.uniform(.55,1.55,n)
    masses=rng.uniform(.7,1.3,n);shifted_masses=rng.uniform(.5,1.8,n)
    strength=float(rng.uniform(.3,2.0));damping=float(rng.uniform(.3,5.0))
    AXES[name]=[VECTORS[c] for c in sequence]
    return [Arm(90000+fid*4+i,'development',name,tuple(shifted if geometry else lengths),
        tuple(shifted_masses if geometry else masses),strength if dynamics else 1.,damping if dynamics else 2.)
        for i,(geometry,dynamics) in enumerate(VARIANTS)]


def build(root, prior, seed=330000):
    root=Path(root);root.mkdir()
    prior_manifest=register(prior)
    reverse={v:k for k,v in VECTORS.items()}
    excluded={''.join(reverse[v] for v in axes) for axes in AXES.values()}
    rng=np.random.default_rng(seed)
    audits=[]
    sequences=axis_sequences(rng,excluded,audit=audits)
    m=dict(schema='ease_mixed_v1',benchmark='ease_broad_development_v1',experiment='E33',
        seed=seed,arms=[],pairs=[],axis_catalog={},families=[],episodes=[],calibrations=[],
        goals=dict(train=[],validation=[],test=[]),synchronized_observations=True,test_evaluated=False,
        excluded_axis_sequences=sorted(excluded),prior_manifest_sha256=fingerprint(Path(prior)/'manifest.json'))
    gid=0
    for fid,sequence in enumerate(sequences):
        arms=family_arms(sequence,fid,rng);n=len(sequence)
        m['axis_catalog'][arms[0].family]=AXES[arms[0].family]
        homes=[];origins=[]
        for posture,spread in [('near',.45),('wide',.85)]:
            for attempt in range(200):
                home=rng.uniform(-spread,spread,n);home[1]+=.4
                valid=True;initials=[];ranks=[]
                for arm in arms:
                    env=MixedReach(replace(arm,home=tuple(home)),synchronize=True)
                    try:
                        obs=env.reset(0,[0,0,.9],spread=0)
                        jac=np.zeros((3,n));mujoco.mj_jacSite(env.model,env.data,jac,None,0)
                        rank=int(np.linalg.matrix_rank(jac,tol=1e-7));ranks.append(rank)
                        valid &= rank==3;initials.append(obs['task'][:3].astype(float))
                    except ValueError:
                        valid=False;ranks.append(None)
                audits.append(dict(kind='home',family=fid,posture=posture,attempt=attempt,accepted=bool(valid),ranks=ranks,home=home.tolist()))
                if valid:break
            else:
                (root/'structural_audit.json').write_text(json.dumps(audits,indent=2))
                raise RuntimeError('No structurally valid home; preserve failed suite')
            homes.append(home);origins.append(initials)
        m['families'].append(dict(id=fid,sequence=sequence,dof=n,homes=[h.tolist() for h in homes],bodies=[a.seed for a in arms]))
        for i,arm in enumerate(arms):
            arm=replace(arm,home=tuple(homes[0]));arms[i]=arm
            m['arms'].append(arm.manifest());m['pairs'].append(dict(id=arm.seed,bodies=[arm.seed],split='development'))
            (root/f'body_{arm.seed}.xml').write_text(MixedReach(arm).xml)
        for pi,(posture,home) in enumerate(zip(('near','wide'),homes)):
            accepted=0
            for attempt in range(1600):
                direction=rng.normal(size=n);direction/=np.linalg.norm(direction)
                target=home+direction*rng.uniform(.2,.55)
                valid=bool(np.max(np.abs(target))<2.2);targets=[];distances=[];contacts=[]
                for vi,arm in enumerate(arms):
                    env=MixedReach(replace(arm,home=tuple(home)),synchronize=True)
                    env.reset(0,[0,0,.9],spread=0)
                    env.data.qpos[:]=target;mujoco.mj_forward(env.model,env.data)
                    xyz=env.data.site_xpos[0].copy();distance=float(np.linalg.norm(xyz-origins[pi][vi]))
                    contact=bool(env.data.ncon);valid &= not contact and .06<=distance<=.18
                    targets.append(xyz);distances.append(distance);contacts.append(contact)
                audits.append(dict(kind='goal',family=fid,posture=posture,attempt=attempt,accepted=bool(valid),
                    joint_target=target.tolist(),distances=distances,contacts=contacts))
                if not valid:continue
                for vi,(arm,xyz) in enumerate(zip(arms,targets)):
                    geometry,dynamics=VARIANTS[vi]
                    condition=f"geometry_{'shifted' if geometry else 'nominal'}__dynamics_{'shifted' if dynamics else 'nominal'}__posture_{posture}"
                    m['goals']['validation'].append(dict(id=gid,pair=arm.seed,xyz=xyz.tolist(),home=home.tolist(),
                        family_id=fid,target_id=accepted,joint_target=target.tolist(),condition=condition,seed=0))
                    gid+=1
                accepted+=1
                if accepted==8:break
            assert accepted==8,'Insufficient reachable endpoints; preserve failed suite'
        (root/'structural_audit.json').write_text(json.dumps(audits,indent=2))
        print('built family',fid,sequence,'cases',gid,flush=True)
    assert len(m['goals']['validation'])==1024
    (root/'structural_audit.json').write_text(json.dumps(audits,indent=2))
    m['asset_hashes']={p.name:fingerprint(p) for p in root.iterdir() if p.is_file()}
    (root/'manifest.json').write_text(json.dumps(m,indent=2))
    (root/'manifest.sha256').write_text(fingerprint(root/'manifest.json'))
    return m


def analytic(manifest, mode):
    arms={a['seed']:Arm(**a) for a in manifest['arms']};records=[]
    for goal in manifest['goals']['validation']:
        arm=replace(arms[goal['pair']],home=tuple(goal['home']))
        env=MixedReach(arm,synchronize=True);before=env.reset(0,goal['xyz'],spread=0)
        controller=IdentifiedController(np.zeros((32,arm.dof,11)),True)
        for step in range(150):
            if mode=='privileged_reference':action=env.expert()
            elif mode=='fixed_probe_analytical' and step==0:action=np.full(arm.dof,.01)
            else:action=controller.action(before)
            after,done,info=env.step(action)
            if mode!='privileged_reference':controller.update(before,after,info['executed_action'])
            before=after
            if done:break
        records.append(dict(mode=mode,body_seed=arm.seed,goal_id=goal['id'],dof=arm.dof,
            condition=goal['condition'],steps=env.steps,**{k:info[k] for k in ('success','error','collision')}))
    return records


def run(prior, replication, output, device='cuda'):
    torch.set_num_threads(4)
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    entries=json.loads((Path(replication)/'index.json').read_text())
    assert {e['seed'] for e in entries}=={0,1,2}
    for e in entries:assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
    protocol=dict(experiment='E33',generation_seed=330000,checkpoints=entries,
        prior_manifest_sha256=fingerprint(Path(prior)/'manifest.json'),test_evaluated=False,
        scope='Separate development suite. Frozen checkpoints; no fitting, validation tuning, or final-test use.',
        families='Eight new axis strings per DoF, all axes X/Y/Z sampled freely; exclude every prior registered arrangement and candidates that cannot attain rank3 at any of three fixed generic nominal poses. Structural exclusions recorded; no controller outcome filtering.',
        physical='Cross geometry and dynamics: nominal lengths .8..1.2 times nominal, masses .7..1.3, strength1 damping2; shifted lengths .55..1.55, masses .5..1.8; shifted strength .3..2 and damping .3..5.',
        postures='Two shared homes per family: uniform +/- .45 or .85 rad plus .4 rad at joint1. Reject only contact or rank<3.',
        goals='Eight shared target joint offsets per home across four physical variants. Target radius .2.. .55 rad in joint space; endpoint distance .06.. .18m, joint bounds +/-2.2, contact-free endpoints. No rollout-success filtering.',
        evaluation='1024 cases, same 150-step horizon, action limit and success rule. Cold-start neural seeds0/1/2 versus fixed +.01 rad first-step analytical, zero-initialized analytical, and privileged expert reference. Same task interaction cost, zero separate calibration. All outcomes retained.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    source=root/'source';source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):shutil.copyfile(p,source/p.name)
    manifest=build(root/'data',prior)
    records=analytic(manifest,'privileged_reference')
    (root/'privileged_reference.json').write_text(json.dumps(records,indent=2))
    print('privileged_reference',sum(r['success'] for r in records),'/',len(records),flush=True)
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        for mode in ('fixed_probe_analytical','analytical_cold'):
            records=analytic(manifest,mode)
            (root/f'{mode}.json').write_text(json.dumps(records,indent=2))
            print(mode,sum(r['success'] for r in records),'/',len(records),flush=True)
        original_solve=np.linalg.solve
        def control_solve(a,b):
            assert a.shape[-2:]==(3,3),'Analytical joint-model inverse forbidden'
            return original_solve(a,b)
        with patch.object(np.linalg,'solve',side_effect=control_solve),patch.object(torch.linalg,'solve',side_effect=RuntimeError('Neural inverse forbidden')):
            for e in entries:
                model=MotionTransformer().to(device)
                model.load_state_dict(torch.load(Path(e['folder'])/'policy.pt',map_location=device,weights_only=True));model.eval()
                records=evaluate(manifest,model,'validation','empty_neural_online',device)
                (root/f'neural_{e["seed"]}.json').write_text(json.dumps(records,indent=2))
                assert fingerprint(Path(e['folder'])/'policy.pt')==e['policy_sha256']
                print('neural',e['seed'],sum(r['success'] for r in records),'/',len(records),flush=True)
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('prior','replication','output'):parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda')
    run(**vars(parser.parse_args()))

"""E15 dataset: multiple axis families per joint count, disjoint held-out families."""
import argparse
import json
from pathlib import Path
import shutil
from dataclasses import replace
import numpy as np
from .benchmark import fingerprint
from .mixed_sim import AXES, MixedReach, sample_arm, explore
from .mixed_data import expert_rollout

TRAIN=['six','six_novel','six_c','seven','seven_b','seven_c']
NOVEL=['six_d','six_e','seven_d','seven_e']
TEST=['six_f','seven_f']


def build(output,seed=202610170,train_goals=40,eval_goals=8):
    root=Path(output); root.mkdir(parents=True,exist_ok=False)
    arms=[]
    for i,family in enumerate(TRAIN):
        arms.extend(sample_arm(seed+i*100+j,family,'train') for j in range(2))
        arms.append(sample_arm(seed+10000+i*100,family,'validation'))
    for i,family in enumerate(NOVEL):
        arms.extend(sample_arm(seed+20000+i*100+j,family,'validation_topology') for j in range(2))
    arms.extend(sample_arm(seed+30000+i*100,family,'test') for i,family in enumerate(TEST))
    arms=[replace(a,home=tuple([0,.5,-.9,0,.6,0] if a.dof==6 else [0,.5,0,-.9,0,.6,0])) for a in arms]
    manifest={'schema':'ease_mixed_v1','benchmark':'ease_axis_diverse_v2','seed':seed,'synchronized_observations':True,
        'reset_design':'Common home joint vector within each DoF; identical reset-noise seed across bodies for each task',
        'arms':[a.manifest() for a in arms],'axis_catalog':{k:AXES[k] for k in TRAIN+NOVEL+TEST},
        'training_families':TRAIN,'novel_validation_families':NOVEL,'final_test_families':TEST,
        'goals':{},'episodes':[],'calibrations':[],'feasibility_audit':[],
        'privileged_inputs':False,'scope':'New benchmark; six_novel from earlier pilots now explicitly a training family',
        'success_threshold_m':.03,'hold_steps':5,'control_hz':25,'action_limit_rad':.04,'history_steps':32}
    (root/'design.json').write_text(json.dumps(manifest,indent=2))
    source=root/'source'; source.mkdir()
    for p in Path(__file__).parent.glob('*.py'):
        shutil.copyfile(p,source/p.name)
    envs=[MixedReach(a,synchronize=True) for a in arms]
    for env in envs:
        (root/f'body_{env.arm.seed}.xml').write_text(env.xml)
    rng=np.random.default_rng(seed)
    for role,count,offset in [('train',train_goals,100000),('validation',eval_goals,200000),('test',eval_goals,300000)]:
        accepted=[]
        for attempt in range(count*100):
            xyz=rng.uniform([-.20,-.18,.78],[.25,.18,.95]); task_seed=seed+offset+attempt
            cache=[]; failed=[]
            for env in envs:
                trajectory,result=expert_rollout(env,task_seed,xyz)
                cache.append((env,trajectory,result))
                if not result['success']: failed.append(env.arm.seed)
            audit={'role':role,'candidate':attempt,'goal':xyz.tolist(),'seed':task_seed,'failed_arms':failed}
            manifest['feasibility_audit'].append(audit)
            with (root/'feasibility.jsonl').open('a') as f: f.write(json.dumps(audit)+'\n')
            if failed: continue
            gid=len(accepted); accepted.append({'id':gid,'xyz':xyz.tolist(),'seed':task_seed})
            if role=='train':
                for env,trajectory,result in cache:
                    if env.arm.split!='train': continue
                    for variant in ['nominal','broad_start']:
                        if variant=='broad_start': trajectory,result=expert_rollout(env,task_seed+400000,xyz,spread=.35)
                        filename=f'{env.arm.seed}_{gid}_{variant}.npz'
                        np.savez_compressed(root/filename,**trajectory)
                        manifest['episodes'].append({'file':filename,'body_seed':env.arm.seed,'family':env.arm.family,
                            'split':'train','dof':env.arm.dof,'goal_id':gid,'variant':variant,**result})
            print(role,'accepted',len(accepted),'/',count,'candidates',attempt+1,flush=True)
            if len(accepted)==count: break
        if len(accepted)!=count: raise RuntimeError(f'Feasibility failure for {role}; audit retained')
        manifest['goals'][role]=accepted
    for env in envs:
        if env.arm.split!='train': continue
        for index in range(8):
            filename=f'calibration_{env.arm.seed}_{index}.npy'
            np.save(root/filename,explore(env,seed+500000+index))
            manifest['calibrations'].append({'file':filename,'body_seed':env.arm.seed,'split':'train'})
    manifest['asset_hashes']={p.name:fingerprint(p) for p in root.iterdir() if p.is_file()}
    manifest['source_hashes']={p.name:fingerprint(p) for p in source.glob('*.py')}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (root/'manifest.sha256').write_text(fingerprint(root/'manifest.json'))
    print('dataset complete',len(manifest['episodes']),'episodes',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); p.add_argument('--seed',type=int,default=202610170)
    p.add_argument('--train-goals',type=int,default=40); p.add_argument('--eval-goals',type=int,default=8)
    build(**vars(p.parse_args()))

"""E62: exact native replay and paired fixed-reference cube-contact ablation."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .benchmark import fingerprint
from .real_arms import verify_assets
from .real_arm_reach import load_models
from .real_arm_pick_place import NativePickPlace
from .retention_fresh import GuardedPose
from .pick_place_control import calibrate
from .native_reliability import rollout
from .action_objective_audit import cloned_state,copy_controller,operator


def replay(env,controller,schedule):
    obs=env.observe();trace=[]
    for s in schedule:
        error=np.r_[np.asarray(s['reference'])-obs['position'],.15*(np.array([0,0,1])-obs['rotation'][:,2])]
        jac,bias=controller.predict()
        motion=jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(len(error)),.35*error)
        action=np.clip(motion+bias,-.04,.04)
        before=obs;obs,command,info=env.step(action,s['aperture']);controller.update(before,obs,command)
        trace.append(dict(phase=s['phase'],position=obs['position'].tolist(),rotation=obs['rotation'].tolist(),
            tracking_error=float(np.linalg.norm(np.asarray(s['reference'])-obs['position'])),
            direction_error=float(np.linalg.norm(np.array([0,0,1])-obs['rotation'][:,2])),bad=bool(info['bad'])))
        if env.bad:break
    return trace


def run(assets,parents,training,reference,candidates,output):
    torch.set_num_threads(4);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    asset=verify_assets(assets);ref=Path(reference);cr=Path(candidates)
    tasks=json.loads((ref/'cases.json').read_text());old=json.loads((ref/'records.json').read_text());new=json.loads((cr/'records.json').read_text())
    models={'analytical':None};hashes={}
    for group,folder in [('original',parents),('uncorrected',str(Path(training)/'prediction')),('corrected',str(Path(training)/'control'))]:
        mm,hh=load_models(folder);hashes.update(hh)
        for seed in range(3):models[f'{group}_{seed}']=mm[f'neural_{seed}']
    protocol=dict(experiment='E62',checkpoints=hashes,asset_commit=asset['commit'],cases_sha256=fingerprint(ref/'cases.json'),
        reference_sha256=fingerprint(ref/'records.json'),candidates_sha256=fingerprint(cr/'records.json'),
        design='All 12 observed E46 cases, analytical + E35 three + E61 uncorrected three + corrected three. Exact original native rollout first. Record ramped target each step, aperture and phase before action. Restore identical calibrated simulator/controller state. Replay same finite reference/aperture schedule with only cube collision contype/conaffinity zero. Table, arm, gripper mechanics unchanged. Stop at forbidden contact. No model fitting or privileged control.',
        limits='Paired prefixes, not object-free full-task success. Original failures truncate schedules and models produce different schedules; comparisons between models are not common-task success comparisons. Cube ablation removes grasp load/contact but table/self-contact remain. Recorded object-dependent goals are frozen, avoiding alternate goal feedback. These are observed native diagnostics, no policy promotion or final test.',test_evaluated=False)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));(root/'source').mkdir();(root/'trials').mkdir()
    for name in ['native_contact_isolation.py','native_reliability.py','action_objective_audit.py','real_arm_pick_place.py','pick_place_control.py','retention_fresh.py']:
        shutil.copy2(Path(__file__).with_name(name),root/'source'/name)
    rows=[]
    for method,model in models.items():
        group=method.rsplit('_',1)[0] if '_' in method else method;seed=int(method[-1]) if '_' in method else None
        for case,task in enumerate(tasks):
            expected=next(r for r in old if r['case']==case and r['method']==('analytical' if seed is None else f'old_{seed}')) if group in ['analytical','original'] else next(r for r in new if r['case']==case and r['seed']==seed and r['condition']==('prediction' if group=='uncorrected' else 'control'))
            env=NativePickPlace(assets,task['robot'],task['xy'],task['target'],.2,True);controller=GuardedPose(env.n,model)
            schedule=[];actual_motion=[]
            def before(phase,t,reference,aperture):
                schedule.append(dict(phase=phase,reference=reference.tolist(),aperture=float(aperture)))
            def after(record,obs,after,command,action,motion,bias,jac,error):
                actual_motion.append(dict(phase=record['phase'],position=after['position'].tolist(),rotation=after['rotation'].tolist(),
                    tracking_error=float(np.linalg.norm(np.asarray(schedule[-1]['reference'])-after['position'])),
                    direction_error=float(np.linalg.norm(np.array([0,0,1])-after['rotation'][:,2])),bad=bool(record['bad']),pads=record['pads']))
            with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged control forbidden')):
                probe=calibrate(env,task['home'],controller,41000+case);assert probe==expected['probe']
                with cloned_state(env):
                    result=rollout(env,copy_controller(controller,model),object_release=task['robot']=='panda',reference_observer=before,step_observer=after)
                for key,value in result.items():assert value==expected[key],(method,case,key)
                assert len(schedule)==len(actual_motion)==len(result['trace'])
                # Control replay verifies reference capture before any intervention.
                with cloned_state(env):
                    intact=replay(env,copy_controller(controller,model),schedule)
                assert len(intact)==len(actual_motion)
                for a,b in zip(intact,actual_motion):
                    np.testing.assert_allclose(a['position'],b['position'],atol=1e-10,rtol=0)
                    np.testing.assert_allclose(a['rotation'],b['rotation'],atol=1e-10,rtol=0)
                    assert a['bad']==b['bad']
                saved=(int(env.model.geom_contype[env.cube]),int(env.model.geom_conaffinity[env.cube]))
                env.model.geom_contype[env.cube]=0;env.model.geom_conaffinity[env.cube]=0
                try:
                    ghost=replay(env,copy_controller(controller,model),schedule)
                finally:env.model.geom_contype[env.cube],env.model.geom_conaffinity[env.cube]=saved
            first_pad=next((i for i,r in enumerate(actual_motion) if r['pads']>0),None)
            paired=min(len(ghost),len(actual_motion))
            row=dict(method=method,group=group,seed=seed,case=case,robot=task['robot'],original_success=result['success'],original_failure=result['failure'],
                original_collision=result['collision'],schedule_steps=len(schedule),ghost_steps=len(ghost),ghost_collision=any(r['bad'] for r in ghost),first_finger_contact=first_pad,
                original_tracking_rmse=float(np.sqrt(np.mean([r['tracking_error']**2 for r in actual_motion[:paired]]))),
                ghost_tracking_rmse=float(np.sqrt(np.mean([r['tracking_error']**2 for r in ghost[:paired]]))),
                original_direction_rmse=float(np.sqrt(np.mean([r['direction_error']**2 for r in actual_motion[:paired]]))),
                ghost_direction_rmse=float(np.sqrt(np.mean([r['direction_error']**2 for r in ghost[:paired]]))),
                ghost_final_within_tolerance=bool(ghost[-1]['tracking_error']<.006 and ghost[-1]['direction_error']<.08 and len(ghost)==len(schedule) and not any(r['bad'] for r in ghost)),
                schedule=schedule,original=actual_motion,ghost=ghost,exact_original=True,intact_reference_replay=True)
            (root/'trials'/f'{method}_{case}.json').write_text(json.dumps(row));rows.append(row)
            print(method,case,result['failure'],'ghost_contact',row['ghost_collision'],'tracks',row['ghost_final_within_tolerance'],flush=True)
    for path,sha in hashes.items():assert fingerprint(path)==sha
    (root/'records.json').write_text(json.dumps(rows));(root/'audit.json').write_text(json.dumps(dict(records=120,exact_original=120,intact_reference_replays=120,ghost_replays=120,records_sha256=fingerprint(root/'records.json'),unchanged_weights=True,test_evaluated=False),indent=2))
    (root/'status.json').write_text(json.dumps(dict(status='complete',test_evaluated=False)));report(root)


def report(root):
    root=Path(root);rows=json.loads((root/'records.json').read_text());assert len(rows)==120
    lines=['# E62: native cube-contact isolation','',json.loads((root/'protocol.json').read_text())['design'],'',json.loads((root/'protocol.json').read_text())['limits'],'',
        '| Group | Robot | Prefixes | Original task successes | Original contacts | Cube-disabled contacts | Cube-disabled endpoint tracks | Original XYZ tracking RMS | Cube-disabled XYZ tracking RMS |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in ['analytical','original','uncorrected','corrected']:
        for robot in ['piper','panda']:
            rr=[r for r in rows if r['group']==group and r['robot']==robot]
            lines.append(f'| {group} | {robot} | {len(rr)} | '+ ' | '.join(str(sum(r[k] for r in rr)) for k in ['original_success','original_collision','ghost_collision','ghost_final_within_tolerance'])+' | '+ ' | '.join(f"{np.sqrt(np.mean([r[k]**2 for r in rr])):.6f}" for k in ['original_tracking_rmse','ghost_tracking_rmse'])+' |')
    lines+=['','Endpoint tracks means only the final reference of the recorded prefix is within 6 mm and 0.08 tool-Z error, without forbidden contact. It is not completion of a full task or an eight-step hold. RMS uses each pair’s shared prefix, then equal-weight trial RMS aggregation; differing original failures produce different horizons.', '',
        'All original outcomes reproduce exactly; all intact fixed-reference replays match position/rotation to 1e-10. Cube-disabled replays retain table/self-collisions. No checkpoint changes or final-test evaluation.']
    (root/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['assets','parents','training','reference','candidates','output']:p.add_argument('--'+key,required=True)
    run(**vars(p.parse_args()))

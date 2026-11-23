"""Matched pre-action histories and exact one-step physical counterfactuals."""
from pathlib import Path
import shutil
import numpy as np
import mujoco
from .matched_unseen_prepare import ROOT,SOURCE,CASES
from .unseen_estimator import old,prior,geometry_ids,distances,ROT_SCALE,rotation_error
from .local_history import LocalPose
from .arm_pick_place import PickPlace
from .mixed_sim import Arm,AXES
from .bounded_history_correction import bounded_action
from .abstaining_task_value_gate import read,save,fingerprint

METHODS=('e120','analytical','e120_no_correction','oracle_local','e120_analytical_metric','analytical_e120_metric')
MODELS=('reference','correction','analytical','oracle')

def prepare():
    (ROOT/'analysis').mkdir(exist_ok=False)
    save(ROOT/'analysis_protocol.json',dict(candidates=METHODS,prediction_models=MODELS,pre_analysis_extension='Two crossed nominal-action/metric combinations added before any diagnostic outcomes to distinguish action-target error from projection-metric error. They are one-step offline probes, not new deployed policies.',oracle='Exact local kinematic Jacobian and its DLS action are privileged diagnostic references, not globally optimal actions or policy inputs.',clean_motion_phases=['approach','descend','lift','transport','lower','retreat'],exact_source_branch_tolerance=1e-12,action_reconstruction_tolerance=1e-12,model_prediction_reconstruction='Exact array equality for the source estimator',controller_changed=False))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);save(ROOT/'analysis_frozen.json',{str(ROOT/'analysis_protocol.json'):fingerprint(ROOT/'analysis_protocol.json'),str(Path(__file__)):fingerprint(__file__)})

def dls(j,error,bias):
    return np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)

class DiagnosticArm(PickPlace):
    def contacts(self):
        result=super().contacts()
        if getattr(self,'record_minimum',False):self.minimum=np.minimum(self.minimum,distances(self.model,self.data,self.ids,self.table))
        return result

def restore(env,state):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(env.model,env.data,state,spec);mujoco.mj_forward(env.model,env.data)
    # Forward kinematics is needed by observe(); restore all integration fields again.
    mujoco.mj_setState(env.model,env.data,state,spec)
    env.bad=False;env.grasped=False;env.lifted=False;env.steps=0;env.record_minimum=False

def analyze(source,limit=None):
    assert source in ('e120_ensemble','analytical')
    for m in ('e120_ensemble','analytical'):assert read(ROOT/f'replay_{m}/verification.json')['exact_replays']==6
    _,models,hashes=old.broad.setup();AXES.update(read(ROOT/'axis_catalog.json'));tasks=read(ROOT/'cases.json');controller=LocalPose(6,None)
    out=ROOT/'analysis'/('preflight_'+source if limit else source);out.mkdir(exist_ok=True);source_match=0;state_match=0;total=0
    for case in CASES:
        stem=f'{case}_{source}_frozen_bias';folder=ROOT/f'replay_{source}';trial=read(folder/'trials'/f'{stem}.json');logs=read(folder/'clearance'/f'{stem}.json');states=np.load(folder/'states'/f'{stem}.npz')['states'];contexts=np.load(folder/'histories'/f'{stem}.npz');task=tasks[case];env=DiagnosticArm(Arm(**task['arm']),task['xy'],task['target']);assert env.xml==(folder/'states'/f'{stem}.xml').read_text();env.ids,env.table=geometry_ids(env.model,6,True)
        rows=[];arrays={k:[] for k in ('jacobians','nominal_actions','final_actions','branch_states','observed_dx','observed_dq','position_predictions','branch_metrics')}
        selected=range(len(logs)) if limit is None else sorted(set([0,min(20,len(logs)-1),next((i for i,x in enumerate(trial['trace']) if x['bad']),len(logs)-1)]))
        for i in selected:
            l=logs[i];integration=np.array(l['integration_state']);restore(env,integration);before=env.observe();q=before['q'];bias=np.array(l['bias']);goal=np.array(l['goal']);error=np.r_[goal-before['position'],ROT_SCALE*rotation_error(np.eye(3),before['rotation'])];initial_object=env.data.xpos[env.obj].copy();pre_pads=len(env.contacts()[0]);pe=float(np.linalg.norm(error[:3]));axis_error=float(np.linalg.norm(error[3:])/ROT_SCALE)
            np.testing.assert_array_equal(env.data.qpos,l['full_qpos_before']);np.testing.assert_array_equal(env.data.qvel,l['full_qvel_before'])
            controller.gram=np.array(l['estimator_gram']);controller.cross=np.array(l['estimator_cross']);controller.bias=bias
            js={}
            for group in ('reference','correction'):
                values=[]
                for seed in range(3):controller.model=models[f'{group}_{seed}'];values.append(controller.predict()[0])
                js[group]=np.mean(values,axis=0)
            js['analytical']=np.linalg.solve(controller.gram,controller.cross).T;js['oracle']=env.jacobian()
            actual_j=js['reference'] if source=='e120_ensemble' else js['analytical'];np.testing.assert_array_equal(actual_j,l['estimator_jacobian']);source_match+=1
            actions={k:dls(v,error,bias) for k,v in js.items()};e120=bounded_action(actions['reference'],actions['correction']);nominals=[e120,actions['analytical'],actions['reference'],actions['oracle'],e120,actions['analytical']]
            if l['phase']=='probe':nominals=[np.array(l['nominal_action']) for _ in METHODS]
            source_index=0 if source=='e120_ensemble' else 1;np.testing.assert_allclose(nominals[source_index],l['nominal_action'],atol=1e-12,rtol=0)
            metric_js=[js['reference'],js['analytical'],js['reference'],js['oracle'],js['analytical'],js['reference']];final=[];solves=[]
            for nominal,j in zip(nominals,metric_js):
                target=np.array(l['optimization_target']) if l['route_active'] or l['handoff_control'] else nominal
                if l['eligible']:
                    jd=np.asarray(j,dtype=np.float64);u,solve,_=prior.project(target,bias,q,np.array(l['effective_distance']),np.array(l['gradient']),jd.T@jd+.015**2*np.eye(6))
                else:u=target.copy();solve=dict(attempted=False,fallback=False,solver_success=True)
                final.append(u);solves.append(solve)
            np.testing.assert_allclose(final[source_index],l['final_action'],atol=1e-12,rtol=0)
            # Measured next-transition target is the same for all model comparisons.
            post=states[i];env.data.qpos[:]=post[:env.model.nq];env.data.qvel[:]=post[env.model.nq:];mujoco.mj_forward(env.model,env.data);actual=env.observe();dq=actual['q']-q;dx=np.r_[actual['position']-before['position'],ROT_SCALE*(actual['rotation'][:,2]-before['rotation'][:,2])]
            predictions=np.array([js[k]@dq for k in MODELS]);pred_errors=np.linalg.norm(predictions[:,:3]-dx[:3],axis=1)*1000;axis_errors=np.linalg.norm(predictions[:,3:]-dx[3:],axis=1)/ROT_SCALE
            predictions_command=np.array([js[k]@(np.array(l['executed_command'])-bias) for k in MODELS]);command_errors=np.linalg.norm(predictions_command[:,:3]-dx[:3],axis=1)*1000
            branches=[];branch_states=[];branch_metrics=[];cache={}
            for name,u,solve in zip(METHODS,final,solves):
                key=(np.asarray(u).tobytes(),l['aperture'])
                if key not in cache:
                    restore(env,integration);env.minimum=distances(env.model,env.data,env.ids,env.table);env.record_minimum=True;after,command,info=env.step(u,l['aperture']);env.record_minimum=False;bad=sorted({tuple(sorted(p)) for p in info['bad']});state=np.r_[env.data.qpos.copy(),env.data.qvel.copy()];minimum=env.minimum.copy();postpe=float(np.linalg.norm(goal-after['position']));posta=float(np.linalg.norm(after['rotation'][:,2]-[0,0,1]));obj=env.data.xpos[env.obj].copy();cache[key]=(state,bad,command,postpe,posta,obj,minimum)
                state,bad,command,postpe,posta,obj,minimum=cache[key];branch_states.append(state.copy());branch_metrics.append([postpe,posta,float(bool(bad)),np.linalg.norm(obj[:2]-initial_object[:2]),np.min(minimum)])
                branches.append(dict(method=name,position_progress_mm=1000*(pe-postpe),axis_progress=axis_error-posta,collision=bool(bad),bad=[list(p) for p in bad],object_xy_motion_mm=1000*float(np.linalg.norm(obj[:2]-initial_object[:2])),minimum_clearance_mm=1000*float(np.min(minimum)),fallback=solve['fallback'],projection_action_change_mrad=1000*float(np.linalg.norm(u-(np.array(l['optimization_target']) if l['route_active'] or l['handoff_control'] else nominals[len(branches)]))),oracle_action_distance_mrad=1000*float(np.linalg.norm(u-final[3]))))
            np.testing.assert_allclose(branch_states[source_index],states[i],atol=1e-12,rtol=0);assert branches[source_index]['bad']==trial['trace'][i]['bad'];state_match+=1
            clean=bool(contexts['clean'][i]);eig=np.linalg.eigvalsh(controller.gram)
            rows.append(dict(case=case,source=source,group='E120_loss' if case in (0,8,37) else 'E120_win_control',step=i,phase=l['phase'],clean_before=clean,pre_pads=pre_pads,local=l['estimator_local'],position_error_mm=1000*pe,axis_error=axis_error,dq_norm=float(np.linalg.norm(dq)),gram_condition=float(np.linalg.cond(controller.gram)),gram_min_eigenvalue=float(eig[0]),eligible=l['eligible'],route_active=l['route_active'],handoff_control=l['handoff_control'],prediction_error_mm=dict(zip(MODELS,pred_errors.tolist())),axis_prediction_error=dict(zip(MODELS,axis_errors.tolist())),command_prediction_error_mm=dict(zip(MODELS,command_errors.tolist())),jacobian_position_error=dict((k,float(np.linalg.norm(js[k][:3]-js['oracle'][:3]))) for k in MODELS),bounded_correction_mrad=1000*float(np.linalg.norm(e120-actions['reference'])),nominal_E120_analytical_gap_mrad=1000*float(np.linalg.norm(nominals[0]-nominals[1])),final_E120_analytical_gap_mrad=1000*float(np.linalg.norm(final[0]-final[1])),branches=branches))
            for k,v in dict(jacobians=np.array([js[k] for k in MODELS]),nominal_actions=np.array(nominals),final_actions=np.array(final),branch_states=np.array(branch_states),observed_dx=dx,observed_dq=dq,position_predictions=predictions,branch_metrics=np.array(branch_metrics)).items():arrays[k].append(v)
            total+=1
        save(out/f'{case}.json',rows);np.savez_compressed(out/f'{case}.npz',steps=np.array(list(selected)),**{k:np.array(v) for k,v in arrays.items()});print(source,case,len(rows),'matched histories; source branches verified',flush=True)
    for p,h in {**hashes,**read(ROOT/'frozen.json'),**read(ROOT/'analysis_frozen.json')}.items():assert fingerprint(p)==h
    save(out/'verification.json',dict(histories=total,source_predictions_exact=source_match,source_physical_steps_verified=state_match,candidates_per_history=len(METHODS),controller_changed=False,preflight=limit is not None))

if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    else:analyze(sys.argv[1],3 if len(sys.argv)>2 and sys.argv[2]=='preflight' else None)

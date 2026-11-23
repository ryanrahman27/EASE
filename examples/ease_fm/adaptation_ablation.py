"""E114: observation-only history attribution on reused unseen development bodies."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .arm_pick_place import PickPlace,targets,ROT_SCALE
from .local_probe import probe_rollout,probe_offset
from .local_history import LocalPose
from .pick_place_control import calibrate
from .mixed_sim import Arm,AXES
from .motion_transformer import MotionTransformer
from .benchmark import fingerprint

ROOT=Path('outputs/ease_adaptation_ablation_v1')
TASKS=Path('outputs/ease_unseen_progress_v1')
MODES=('online','frozen_response','frozen_all')

def load():
    AXES.update(json.loads((TASKS/'axis_catalog.json').read_text()))
    tasks=json.loads((TASKS/'cases.json').read_text());models={'analytical':None};hashes={}
    for seed in range(3):
        p=Path(f'outputs/ease_task_retention_training_v1/retained/seed_{seed}/policy.pt')
        m=MotionTransformer().cuda();m.load_state_dict(torch.load(p,map_location='cuda',weights_only=True));models[f'learned_{seed}']=m.eval();hashes[str(p)]=fingerprint(p)
    return tasks,models,hashes

def evaluate(case,t,method,model,mode,capture=None,save_states=False):
    telemetry=[];states=[];updates=0;active=False
    class Controller(LocalPose):
        def update(self,before,after,command):
            nonlocal updates
            if not active:return super().update(before,after,command)
            j,_=self.predict();dq=after['q']-before['q'];dx=np.r_[after['position']-before['position'],ROT_SCALE*(after['rotation'][:,2]-before['rotation'][:,2])]
            telemetry.append(dict(phase=env.phase,position_prediction_error=float(np.linalg.norm(dx[:3]-(j@dq)[:3])),command_norm=float(np.linalg.norm(command))))
            if mode=='online':super().update(before,after,command);updates+=1
            elif mode=='frozen_response':self.bias=.8*self.bias+.2*(command-dq)
    class ObservedArm(PickPlace):
        def step(self,action,aperture):
            if getattr(self,'phase',None)=='probe':
                action=np.clip(self.anchor+probe_offset(self.probe_step,self.n)-self.observe()['q']+c.bias,-.04,.04);self.probe_step+=1
            r=super().step(action,aperture)
            if active:
                if save_states:states.append(np.r_[self.data.qpos.copy(),self.data.qvel.copy()])
                if capture:capture(self)
            return r
    env=ObservedArm(Arm(**t['arm']),t['xy'],t['target']);c=Controller(env.n,model)
    def state():return dict(gram=c.gram.tolist(),cross=c.cross.tolist(),bias=c.bias.tolist(),history_count=len(c.history))
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
        calibration=calibrate(env,t['home'],c,38000+case);initial=state();active=True
        def phases():
            for name,goal,aperture,budget in targets(env.xy,env.target):
                env.phase=name
                if name=='close':
                    if mode=='online':c.local=True;c.rebuild()
                    env.anchor=env.observe()['q'].copy();env.probe_step=0;env.phase='probe'
                    yield 'probe',env.observe()['position'].copy(),0.,32
                    assert env.probe_step==32;env.phase=name
                yield name,goal,aperture,budget
        r=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
    final=state()
    if mode!='online':
        for key in ('gram','cross','history_count'):assert initial[key]==final[key]
    if mode=='frozen_all':assert initial['bias']==final['bias']
    assert updates==(len(r['trace']) if mode=='online' else 0)
    result=dict(case=case,body=t['arm']['seed'],family=t['arm']['family'],dof=env.n,method=method,mode=mode,calibration=calibration,initial_estimator=initial,final_estimator=final,response_updates=updates,telemetry=telemetry,**r)
    if save_states:
        stem=f'{case}_{method}_{mode}';np.savez_compressed(ROOT/'states'/f'{stem}.npz',states=np.array(states));(ROOT/'states'/f'{stem}.xml').write_text(env.xml)
    return result

def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','source'):(ROOT/d).mkdir()
    tasks,models,hashes=load();assert len(tasks)==16
    trained=json.loads(Path('outputs/ease_task_retention_training_v1/protocol.json').read_text())['training_bodies']
    assert set(trained).isdisjoint(t['arm']['seed'] for t in tasks)
    inputs={str(TASKS/n):fingerprint(TASKS/n) for n in ('cases.json','axis_catalog.json')}
    spec=dict(experiment='E114',checkpoints=hashes,inputs=inputs,episodes=192,
        design='16 reused E112 unseen-body development tasks (8 bodies, 4 previously studied axis families; 2 placements/body), three learned seeds plus analytical estimator, each online/frozen_response/frozen_all. 192 full tasks. No outcome filtering or tuning. Reserved final test untouched.',
        controller='One frozen E99 retained checkpoint throughout each learned rollout; no parent/student phase switch. Analytical is the existing regularized measured-history least-squares estimator. Identical DLS action mapping, joint clipping, phase goals/budgets and physical scene. No geometry filter, route planner, physics selector, gravity oracle, model Jacobian or home-Jacobian prior. Scene supplies observed q/tool pose and fixed reset configuration.',
        history='Identical32 home-relative calibration transitions per case, tiny identity/zero statistics initialization. After calibration: online updates response and bias, switches to recent32 history before local probe; frozen_response preserves calibrated gram/cross/history but updates measured command bias; frozen_all preserves gram/cross/history and bias. Every condition executes32 local probe steps before close, no reset; probe commands use condition-specific measured bias. Frozen means no further adaptation after calibration, not empty history.',
        metrics='Full-task success, collision, grasp/lift, phase progress, steps and own-trajectory prediction residuals; six/seven-joint, seed and body breakdown. Paired task gains/losses. Analytical16 episodes per mode, not triplicated as independent seeds. Reused development tasks, no confirmatory heldout or flow-policy claim.',
        verification='Block mj_jacSite through calibration and execution. Direct PickPlace subclass has no selector or geometry querying API. Assert frozen state and exact paired calibration. Save complete state trajectories/models and checkpoint hashes; exact video replay.',
        videos='First success and first failure per method/mode where available, plus all modes for first learned-online gain and loss against each frozen control for seed0. Outcome-selected diagnostic examples; preserve all trial records and states.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('adaptation_ablation.py','arm_pick_place.py','pick_place_control.py','local_history.py','local_probe.py','retention_fresh.py','motion_transformer.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[]
    for case,t in enumerate(tasks):
        reference=None
        for method,model in models.items():
            for mode in MODES:
                r=evaluate(case,t,method,model,mode,save_states=True)
                if reference is None:reference=r['calibration']
                assert r['calibration']==reference
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('trace','calibration','initial_estimator','final_estimator','telemetry')})
                (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
        print('case',case,'complete',len(rows),'episodes',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'verification.json').write_text(json.dumps(dict(episodes=len(rows),matched_calibrations=True,frozen_state_assertions=True,weights_inputs_unchanged=True,privileged_jacobian_blocked=True,physics_selector=False,geometry_filter=False),indent=2))

if __name__=='__main__':run()

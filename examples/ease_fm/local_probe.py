"""E95: fixed local probe versus interaction-matched hold after planned descent."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .arm_pick_place import targets, ROT_SCALE, rotation_error
from .local_history import LocalPose, ROOT as REFERENCE, PRIOR, models
from .learned_path_transfer import route_check
from .novelaxis_path_execution import PathArm, PATHS
from .pick_place_control import calibrate
from .mixed_sim import AXES, Arm
from .history_excitation import excitation
from .benchmark import fingerprint

ROOT=Path('outputs/ease_local_probe_v1')


def probe_offset(step,n):
    offset=np.zeros(n)
    if step<4*n:offset[step//4]=.004 if step%4<2 else -.004
    return offset


class ProbeArm(PathArm):
    def step(self,action,aperture):
        if getattr(self,'phase',None) in ('probe','hold'):
            offset=probe_offset(self.probe_step,self.n) if self.phase=='probe' else np.zeros(self.n)
            action=np.clip(self.anchor+offset-self.observe()['q']+self.controller.bias,-.04,.04)
            self.probe_step+=1
        return super().step(action,aperture)

def probe_rollout(env,home,controller=None,render=None,reset=True,sequence=None):
    obs=env.reset(home) if reset else env.observe();trace=[];phases=[];renderer=None
    if render:
        from PIL import Image
        Path(render).mkdir(parents=True,exist_ok=True);renderer=mujoco.Renderer(env.model,height=480,width=640)
        camera=mujoco.MjvCamera();camera.lookat[:]=[.25,0,.35];camera.distance=1.5;camera.azimuth=130;camera.elevation=-25
    for phase,goal,aperture,budget in (targets(env.xy,env.target) if sequence is None else sequence):
        held=0;reached=False
        for t in range(budget):
            error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
            if controller is None:
                jac=env.jacobian()
                bias=env.privileged_bias() if hasattr(env,'privileged_bias') else env.data.qfrc_bias[:env.n]/(800*env.arm.strength)
            else:jac,bias=controller.predict()
            action=np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)
            before=obs;obs,command,info=env.step(action,aperture)
            info['bad']=[list(pair) for pair in sorted({tuple(sorted(p)) for p in info['bad']})]
            if controller is not None:controller.update(before,obs,command)
            pe=float(np.linalg.norm(goal-obs['position']));re=float(np.linalg.norm(rotation_error(np.eye(3),obs['rotation'])))
            trace.append(dict(phase=phase,position=obs['position'].tolist(),object=env.data.xpos[env.obj].tolist(),position_error=pe,rotation_error=re,bad=info['bad'],pads=info['pads']))
            held=held+1 if pe<.008 and re<.08 else 0
            if held>=5 and phase not in ('close','release','settle','probe','hold'):
                reached=True;break
        phases.append(dict(phase=phase,reached=reached or (pe<.008 and re<.08),steps=t+1))
        if renderer:
            renderer.update_scene(env.data,camera=camera);Image.fromarray(renderer.render()).save(Path(render)/f'{phase}.png')
    if renderer:renderer.close()
    obj=env.data.xpos[env.obj].copy();addr=env.model.jnt_dofadr[env.oj];velocity=env.data.qvel[addr:addr+6]
    success=bool(env.grasped and env.lifted and not env.bad and np.linalg.norm(obj[:2]-env.target)<.025 and abs(obj[2]-.215)<.005 and np.linalg.norm(velocity)<.02 and not env.contacts()[0])
    return dict(success=success,grasped=bool(env.grasped),lifted=bool(env.lifted),collision=bool(env.bad),steps=env.steps,final_object=obj.tolist(),phases=phases,trace=trace)


def evaluate(case,t,method,model,mode,record_step=None):
    old=json.loads((REFERENCE/'trials'/f'{case}_{method}_window32.json').read_text())
    p=json.loads((PATHS/'cases'/f'{case}.json').read_text());selected=next((r for r in p['attempts'] if r['attempt']==p['selected_attempt']),None)
    available=selected is not None and selected['contact_free'] and selected['margin_2mm']
    env=ProbeArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.route=None;env.path_log=[];env.waypoint=0;env.probe_step=0
    c=LocalPose(env.n,model);assert calibrate(env,t['home'],c,872000+t['candidate'])==json.loads((PRIOR/f'probe_{case}.json').read_text())
    env.controller=c;env.filter_enabled=True;env.filter_log=[];handoff={};measurements={}
    if record_step is not None:
        original=env.step
        def captured(action,aperture):
            result=original(action,aperture);record_step(env);return result
        env.step=captured
    def phases():
        for name,goal,aperture,budget in targets(env.xy,env.target):
            env.phase=name
            if name=='descend' and available:
                route,distances=route_check(env,np.array(selected['endpoint']))
                assert distances==old['route_check']['clearances']
                if min(distances)>=.002-1e-6:env.route=route;budget=300
            if name=='close':
                handoff.update(qpos=env.data.qpos.copy().tolist(),qvel=env.data.qvel.copy().tolist(),gram=c.gram.tolist(),cross=c.cross.tolist(),bias=c.bias.tolist())
                if env.route is not None:
                    c.local=True;c.rebuild();measurements['before']=excitation(c.gram)
                    if mode!='no_probe':
                        env.anchor=env.observe()['q'].copy();env.phase='probe' if mode=='probe32' else 'hold'
                        yield env.phase,env.observe()['position'].copy(),0.,32
                        assert env.probe_step==32
                    measurements['after']=excitation(c.gram);env.phase=name
            yield name,goal,aperture,budget
    with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
        result=probe_rollout(env,t['home'],c,reset=False,sequence=phases())
    prefix=lambda trace:[r for r in trace if r['phase'] in ('approach','descend')]
    assert prefix(result['trace'])==prefix(old['trace'])
    if mode=='no_probe' or env.route is None:assert all(result[k]==old[k] for k in result)
    return dict(case=case,method=method,mode=mode,dof=env.n,body=t['body'],route_used=env.route is not None,handoff=handoff,excitation=measurements,probe_steps=env.probe_step,**result)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    tasks=json.loads((PRIOR/'cases.json').read_text());AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()));loaded,hashes=models()
    paths=[REFERENCE/'protocol.json',REFERENCE/'records.json',PRIOR/'cases.json',PRIOR/'axis_catalog.json',*sorted((REFERENCE/'trials').glob('*_window32.json')),*sorted((PATHS/'cases').glob('*.json'))]
    inputs={str(p):fingerprint(p) for p in paths}
    spec=dict(experiment='E95',inputs=inputs,checkpoints=hashes,test_evaluated=False,
        design='All96 checkpoint/task pairs x no_probe/hold32/probe32 =288 tasks. Common E93 window32 hybrid prefix and planner. Insert exactly32 physical control steps after accepted-route descent, before closing fingers; aperture open. Anchor is actual post-descent q. Probe offsets each joint in index order +.004rad for2steps, -.004rad for2steps, then anchor for remaining8steps (6DOF) or4steps (7DOF). Hold uses anchor all32steps. Both command anchor+offset-q+measured bias, clipped+/-.04, through original geometry filter. All32 observed transitions update recent32 statistics and measured bias. No resets, analytic task Jacobians, gravity oracle or altered contact masks. No-route cases receive no extra steps and reproduce E93.',
        budgets='Hold and probe have identical32-step intervention budgets. No-probe quantifies added interaction cost. All original subsequent phase budgets unchanged; probe/hold cannot early-exit. Contacts and object motion during interventions count toward full-task success. Report first collision phase and grasp/lift/final outcomes.',
        gate='Probe must exceed hold task success, not increase hold collisions, retain every hold-successful six-joint case, and also retain every no-probe-successful six-joint case. Report every loss versus both controls. No amplitude/order/window tuning after outcomes.',
        limits='Fixed prospective intervention on reused development cases with privileged route planning; not independent transfer validation. Commanded offsets do not guarantee actual excitation. Probe effects include pose/object/bias changes, not just identification.',
        videos='Prespecified candidate_0 cases0 and4 in all three modes. Append first new probe regression against hold in both modes if it is not already represented, explicitly outcome-selected.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('local_probe.py','local_history.py','learned_path_transfer.py','novelaxis_path_execution.py','arm_pick_place.py','pick_place_control.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    rows=[];fallbacks=0
    for case,t in enumerate(tasks):
        for method,model in loaded.items():
            handoff=None
            for mode in ('no_probe','hold32','probe32'):
                r=evaluate(case,t,method,model,mode)
                if handoff is None:handoff=r['handoff']
                assert handoff==r['handoff']
                if mode!='no_probe' and not r['route_used']:fallbacks+=1
                (ROOT/'trials'/f'{case}_{method}_{mode}.json').write_text(json.dumps(r))
                rows.append({k:v for k,v in r.items() if k not in ('trace','handoff')})
        print('case',case,'complete',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2));(ROOT/'audit.json').write_text(json.dumps(dict(exact_replays=96,exact_fallbacks=fallbacks,identical_handoffs=True,hashes_unchanged=True,test_evaluated=False),indent=2));(ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()


"""E79: geometry attribution of successful-run warnings and missed contact."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
import torch
from .response_allphase import AllPhaseResponseArm
from .geometry_filter import allowed_pair,MARGIN
from .arm_pick_place import rollout
from .mixed_sim import AXES,Arm
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models
from .benchmark import fingerprint

ROOT=Path('outputs/ease_warning_geometry_v1')
SOURCE=Path('outputs/ease_response_allphase_v1')


class AttributionArm(AllPhaseResponseArm):
    def geometry_details(self):
        entries=[]
        for c in self.geometry_data.contact[:self.geometry_data.ncon]:
            ids=(int(c.geom1),int(c.geom2))
            if allowed_pair(set(ids),self.gids):continue
            entries.append(dict(distance=float(c.dist),pair=[mujoco.mj_id2name(self.geometry_model,mujoco.mjtObj.mjOBJ_GEOM,g) or str(g) for g in ids],
                bodies=[mujoco.mj_id2name(self.geometry_model,mujoco.mjtObj.mjOBJ_BODY,int(self.geometry_model.geom_bodyid[g])) for g in ids]))
        return dict(clearance=min([MARGIN]+[r['distance'] for r in entries]),contacts=entries)

    def clearance(self,q,fingers):
        value=super().clearance(q,fingers)
        if getattr(self,'collect',False):
            details=self.geometry_details();assert details['clearance']==value
            self.queries.append(dict(q=np.r_[q,fingers].tolist(),**details))
        return value

    def actual_geometry(self):
        # Full measured state, including current object pose; shadow model only.
        mujoco.mj_copyData(self.geometry_data,self.geometry_model,self.data)
        mujoco.mj_forward(self.geometry_model,self.geometry_data)
        return dict(q=self.data.qpos[:self.n+2].tolist(),**self.geometry_details())

    def step(self,action,aperture):
        index=self.steps-32
        selected=getattr(self,'audit_enabled',False) and index in self.selected_steps
        if not selected:return super().step(action,aperture)
        self.queries=[];self.collect=True;actual=[self.actual_geometry()]
        original=mujoco.mj_step
        def observed(model,data,*args):
            original(model,data,*args)
            actual.append(self.actual_geometry())
        with patch.object(mujoco,'mj_step',side_effect=observed):
            result=super().step(action,aperture)
        self.collect=False
        assert len(actual)==21
        paths={name:self.queries[-84+i::4] for i,name in enumerate(('command','response_linear','velocity_linear','velocity_curve'))}
        assert all(len(v)==21 for v in paths.values())
        record=self.prediction_log[-1]
        assert all(min(x['clearance'] for x in v)==record['clearances'][name] for name,v in paths.items())
        self.attribution.append(dict(step=index,phase=record['phase'],collision=record['collision'],paths=paths,actual=actual))
        return result


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={}
    def read(p):
        inputs[str(p)]=fingerprint(p);return json.loads(p.read_text())
    prior=read(SOURCE/'protocol.json');runs=read(SOURCE/'records.json');selected=[];tasks=[]
    for run in runs:
        if run['method']=='analytical':continue
        full=read(SOURCE/'trials'/f"{run['case']}_{run['method']}.json")
        steps=[r['step'] for r in full['records'] if r['evaluated'] and ((run['success'] and r['clearances']['response_linear']<.002-1e-6) or (r['collision'] and r['clearances']['response_linear']>=.002-1e-6))]
        if steps:selected.append(dict(**run,selected_steps=steps))
    for folder in ('ease_arm_pick_place_v2','ease_arm_pick_place_unseen_v1'):
        p=Path('outputs')/folder;AXES.update(read(p/'axis_catalog.json'));tasks.extend(dict(local_case=i,**t) for i,t in enumerate(read(p/'cases.json')))
    models={};hashes={}
    for group,folder in [('parent','outputs/ease_robust_training_v1'),('candidate','outputs/ease_estimator_distillation_v1/analytical')]:
        mm,hh=load_models(folder);hashes.update(hh);models.update({f'{group}_{s}':mm[f'neural_{s}'] for s in range(3)})
    assert hashes==prior['checkpoints']
    spec=dict(experiment='E79',selection=selected,checkpoints=hashes,
        design='All E78 successful learned trajectories containing response warnings plus every learned missed first contact. Selected after outcomes for causal-geometry diagnosis, not representative performance estimation.',
        measurement='Record forbidden geometry pairs at all 21 frozen prediction path samples; record 21 actual states including pre-step and 20 physics substeps using a shadow geometry model and full measured object state. No action, threshold, phase or predictor changes. Actual-path clearance is retrospective only.',
        limits='Known privileged geometry; discrete samples do not prove continuous clearance. No interventions or task recoveries; E78 gate remains failed.',test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('warning_geometry_audit.py','response_allphase.py','response_prediction.py','geometry_filter.py','arm_pick_place.py'):
        p=Path(__file__).with_name(name);assert not (SOURCE/'source'/name).exists() or fingerprint(p)==fingerprint(SOURCE/'source'/name)
        shutil.copy2(p,ROOT/'source'/name)
    results=[]
    for run in selected:
        case=run['case'];method=run['method'];t=tasks[case]
        old=read(Path('outputs/ease_geometry_filter_v2/trials')/f'{case}_{method}_geometry.json')
        previous=read(SOURCE/'trials'/f'{case}_{method}.json')
        env=AttributionArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry();env.initialize_response();c=GuardedPose(env.n,models[method])
        probe=calibrate(env,t['home'],c,38000+t['local_case'])
        assert probe==read(Path('outputs/ease_grasp_settling_v1')/f'probe_{case}.json')
        env.controller=c;env.filter_enabled=True;env.filter_log=[];env.audit_enabled=True;env.phases=[r['phase'] for r in old['trace']];env.selected_steps=set(run['selected_steps']);env.attribution=[]
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Task Jacobian forbidden')):
            replay=rollout(env,t['home'],c,reset=False)
        assert replay['trace']==old['trace'] and replay['success']==old['success'] and env.filter_log==old['filter_log']
        assert env.prediction_log==previous['records']
        assert len(env.attribution)==len(run['selected_steps'])
        result=dict(**run,attribution=env.attribution)
        (ROOT/'trials'/f'{case}_{method}.json').write_text(json.dumps(result));results.append(run)
        print('exact replay',case,method,'diagnosed steps',len(env.attribution),flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'records.json').write_text(json.dumps(results,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(inputs=inputs,hashes_unchanged=True,exact_replays=len(results),predictions_exact=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()

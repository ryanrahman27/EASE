"""E88: exact-Jacobian substitution, retaining measured bias and task logic."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import mujoco
import numpy as np
import torch
from .geometry_filter import GeometryArm
from .arm_pick_place import rollout
from .mixed_sim import Arm, AXES
from .pick_place_control import calibrate, PoseController
from .benchmark import fingerprint

ROOT=Path('outputs/ease_novelaxis_oracle_v1')
PRIOR=Path('outputs/ease_direction_novelaxis_v1')


class ExactPose(PoseController):
    """Privileged local kinematics only; bias remains measured and adaptive."""
    def __init__(self,env):
        super().__init__(env.n)
        self.env=env
        self.calls=0

    def predict(self):
        self.calls+=1
        return self.env.jacobian(),self.bias


def run():
    torch.set_num_threads(4)
    ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    inputs={str(PRIOR/name):fingerprint(PRIOR/name) for name in ('cases.json','axis_catalog.json','protocol.json','records.json')}
    tasks=json.loads((PRIOR/'cases.json').read_text());assert len(tasks)==16
    AXES.update(json.loads((PRIOR/'axis_catalog.json').read_text()))
    previous=json.loads((PRIOR/'protocol.json').read_text())
    spec=dict(experiment='E88',inputs=inputs,checkpoints=previous['checkpoints'],test_evaluated=False,
        design='All16 E87 cases, unchanged home/physics/calibration/phase budgets/filter/success rules. Two arms per case: measured analytical Jacobian baseline and exact instantaneous simulator Jacobian substitution. Same measured adaptive bias and online statistic updates in both. Original geometry filter only; no E82 direction correction. No additional IK screening, replacement, retraining or tuning.',
        interpretation='Privileged diagnostic, not a deployable policy, upper bound or learned result. Exact local kinematics need not solve global path constraints, dynamics, grasping or phase sequencing. Different closed-loop trajectories cause later histories and biases to differ. Only the Jacobian source is intervened upon.',
        verification='Every analytical result/trace must equal E87 analytical baseline; every probe equals saved E87 probe. Exact Jacobian calls equal task steps and zero during calibration. Input/checkpoint/source hashes checked. Report all paired gains and losses; no success gate or policy promotion.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for name in ('novelaxis_oracle.py','geometry_filter.py','arm_pick_place.py','pick_place_control.py'):
        p=Path(__file__).with_name(name);shutil.copy2(p,ROOT/'source'/name)
    rows=[];exact=0
    for case,t in enumerate(tasks):
        for mode in ('analytical','exact_jacobian'):
            env=GeometryArm(Arm(**t['arm']),t['xy'],t['target']);env.prepare_geometry()
            c=PoseController(env.n) if mode=='analytical' else ExactPose(env)
            probe=calibrate(env,t['home'],c,872000+t['candidate'])
            assert probe==json.loads((PRIOR/f'probe_{case}.json').read_text())
            if mode=='exact_jacobian':assert c.calls==0
            env.controller=c;env.filter_enabled=True;env.filter_log=[]
            if mode=='analytical':
                with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Analytical Jacobian access forbidden')):
                    result=rollout(env,t['home'],c,reset=False)
                old=json.loads((PRIOR/'trials'/f'{case}_analytical_baseline.json').read_text())
                assert all(result[k]==old[k] for k in result)
                assert sum(x['queries'] for x in env.filter_log)==old['filter_queries'];exact+=1
            else:
                result=rollout(env,t['home'],c,reset=False)
                assert c.calls==len(result['trace'])==env.steps-32
            row=dict(case=case,body=t['body'],family=t['arm']['family'],dof=env.n,mode=mode,
                filter_queries=sum(x['queries'] for x in env.filter_log),**result)
            (ROOT/'trials'/f'{case}_{mode}.json').write_text(json.dumps(dict(**row,filter_log=env.filter_log)))
            rows.append({k:v for k,v in row.items() if k!='trace'})
        print('case',case,[(r['mode'],r['success'],r['collision']) for r in rows[-2:]],flush=True)
    for p,h in {**inputs,**previous['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'audit.json').write_text(json.dumps(dict(exact_baselines=exact,probes_identical=True,hashes_unchanged=True,exact_call_counts_verified=True,test_evaluated=False),indent=2))
    (ROOT/'status.json').write_text(json.dumps(dict(status='complete')))


if __name__=='__main__':run()

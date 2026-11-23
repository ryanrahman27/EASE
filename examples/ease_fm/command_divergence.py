"""E100 read-only same-state command audit on E99 losses and controls."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import task_retention_training as prior
from .geometry_filter import GeometryArm
from .local_history import LocalPose
from .arm_pick_place import targets, ROT_SCALE, rotation_error
from .mixed_sim import AXES
from .benchmark import fingerprint

ROOT=Path('outputs/ease_command_divergence_v1')


def command(jac,bias,error):
    return np.clip(jac.T@np.linalg.solve(jac@jac.T+.015**2*np.eye(6),.35*error)+bias,-.04,.04)


def run():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=False);(ROOT/'trials').mkdir();(ROOT/'source').mkdir()
    rows=json.loads((prior.ROOT/'records.json').read_text())
    find=lambda c,s,m:next(r for r in rows if r['case']==c and r['method']==f'candidate_{s}' and r['mode']==m)
    losses=[(r['case'],int(r['method'][-1])) for r in rows if r['mode']=='retained' and not r['success'] and find(r['case'],int(r['method'][-1]),'parent')['success']]
    controls=[]
    for case,seed in losses:
        eligible=[s for s in range(3) if find(case,s,'parent')['success'] and find(case,s,'retained')['success']]
        if eligible:controls.append((case,min(eligible)))
    pairs=sorted(set(losses+controls));parents,hashes=prior.previous.models();students,hh=prior.load_students();hashes.update(hh)
    tasks=json.loads((prior.previous.PRIOR/'cases.json').read_text());AXES.update(json.loads((prior.previous.PRIOR/'axis_catalog.json').read_text()))
    inputs={str(p):fingerprint(p) for p in [prior.ROOT/'records.json',prior.previous.PRIOR/'cases.json',*Path(__file__).parent.glob('*.py')]}
    spec=dict(experiment='E100',losses=losses,controls=controls,pairs=pairs,checkpoints=hashes,inputs=inputs,test_evaluated=False,
        design='All three E99 retained regressions plus lowest-seed unchanged success on each same task, selected from outcomes for diagnosis. Replay parent and retained paths. At every post-probe step query both frozen networks on identical live Gram/cross/bias and physical state, then query both geometry-filter commands on shadow data. Execute original action unchanged. Compare full serialized trajectory with E99. No intervention or new success-rate estimate.',
        limits='Same-state command comparisons within each trajectory; parent and retained future trajectories differ. Geometry is privileged. Commands are not dynamic counterfactual outcomes. Cumulative norms measure discrepancy, not causal error. No target data training, seed selection, or test evaluation.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    original=GeometryArm.select_action;all_rows=[]
    for case,seed in pairs:
        method=f'candidate_{seed}'
        for mode in ('parent','retained'):
            measurements=[]
            def audit(env,action,aperture,bias):
                c=env.controller
                if getattr(c,'switched',False) and env.phase not in ('approach','descend','probe'):
                    obs=env.observe();goal=next(g for name,g,a,b in targets(env.xy,env.target) if name==env.phase)
                    error=np.r_[goal-obs['position'],ROT_SCALE*rotation_error(np.eye(3),obs['rotation'])]
                    predictions={};filtered={};entries={};saved=c.model
                    state=(env.data.qpos.copy(),env.data.qvel.copy(),env.data.ctrl.copy(),c.gram.copy(),c.cross.copy(),c.bias.copy())
                    try:
                        for name,model in [('parent',parents[method]),('retained',students[('retained',seed)])]:
                            c.model=model;jac,b=LocalPose.predict(c);predictions[name]=command(jac,b,error)
                            filtered[name],entries[name]=original(env,predictions[name],aperture,bias)
                    finally:c.model=saved
                    assert np.array_equal(action,predictions[mode])
                    for before,after in zip(state,(env.data.qpos,env.data.qvel,env.data.ctrl,c.gram,c.cross,c.bias)):assert np.array_equal(before,after)
                    measurements.append(dict(step=env.steps,phase=env.phase,already_bad=bool(env.bad),q=obs['q'].tolist(),error=error.tolist(),raw={k:v.tolist() for k,v in predictions.items()},filtered={k:v.tolist() for k,v in filtered.items()},filter=entries,
                        raw_l2=float(np.linalg.norm(predictions['parent']-predictions['retained'])),filtered_l2=float(np.linalg.norm(filtered['parent']-filtered['retained'])),scale_disagrees=entries['parent']['scale']!=entries['retained']['scale']))
                return original(env,action,aperture,bias)
            with patch.object(GeometryArm,'select_action',audit):result=prior.evaluate(case,tasks[case],method,parents[method],students[('retained',seed)],mode)
            ref=prior.ROOT/'trials'/f'{case}_{method}_{mode}.json';assert json.loads(json.dumps(result))==json.loads(ref.read_text())
            stem=f'{case}_{method}_{mode}';(ROOT/'trials'/f'{stem}.json').write_text(json.dumps(result));(ROOT/'trials'/f'{stem}_commands.json').write_text(json.dumps(measurements))
            for phase in dict.fromkeys(r['phase'] for r in measurements):
                rr=[r for r in measurements if r['phase']==phase];clean=[r for r in rr if not r['already_bad']]
                all_rows.append(dict(case=case,seed=seed,mode=mode,group='regression' if (case,seed) in losses else 'control',phase=phase,n=len(rr),before_contact=len(clean),raw_mean=float(np.mean([r['raw_l2'] for r in rr])),filtered_mean=float(np.mean([r['filtered_l2'] for r in rr])),filtered_sum=float(sum(r['filtered_l2'] for r in rr)),scale_disagreements=sum(r['scale_disagrees'] for r in rr),clean_filtered_mean=float(np.mean([r['filtered_l2'] for r in clean])) if clean else None))
            print(stem,'exact replay;',len(measurements),'audited steps',flush=True)
    for p,h in {**inputs,**hashes}.items():assert fingerprint(p)==h
    (ROOT/'summary.json').write_text(json.dumps(all_rows,indent=2));(ROOT/'verification.json').write_text(json.dumps(dict(exact_replays=2*len(pairs),state_preserved=True,executed_command_exact=True,hashes_unchanged=True,test_evaluated=False),indent=2))


if __name__=='__main__':run()

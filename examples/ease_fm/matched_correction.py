"""E123: phase-entry matched-state correction pulses, E120 unchanged."""
import json,shutil,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
import mujoco
from . import bounded_history_correction as previous
from .bounded_history_correction import fingerprint,ROT_SCALE
ROOT=Path('outputs/ease_matched_correction_v1')
OLD=Path('outputs/ease_bounded_history_v1')
PHASES=('approach','descend','close','lift','transport','lower','release','retreat','settle')
PULSE=8

def setup():
    tasks,models,hashes=previous.setup();previous.ROOT=ROOT;previous.core.ROOT=ROOT
    return tasks,models,hashes

def snapshot(env,c,phase_step):
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    state=np.empty(mujoco.mj_stateSize(env.model,spec));mujoco.mj_getState(env.model,env.data,state,spec)
    arrays=dict(integration=state,gram=c.gram.copy(),cross=c.cross.copy(),bias=c.bias.copy(),history_dq=np.array([x[0] for x in c.history]),history_dx=np.array([x[1] for x in c.history]),goal=np.array(env.goal))
    if hasattr(env,'anchor'):arrays['anchor']=env.anchor.copy()
    meta=dict(phase=env.phase,phase_step=phase_step,steps=env.steps,bad=bool(env.bad),grasped=bool(env.grasped),lifted=bool(env.lifted),local=bool(c.local),probe_step=getattr(env,'probe_step',None))
    h=hashlib.sha256(json.dumps(meta,sort_keys=True).encode())
    for name,value in sorted(arrays.items()):h.update(name.encode());h.update(value.tobytes())
    return h.hexdigest(),arrays,meta

def evaluate(case,task,seed,base,correction,off_phase=None,save=True):
    method=f'on_{seed}' if off_phase is None else f'off_{off_phase}_{seed}'
    stem=f'{case}_{method}_frozen_bias';signatures=[];metrics=[];counts={};off=False
    original_rollout=previous.core.probe_rollout;original_action=previous.bounded_action
    expected=None if off_phase is None else json.loads((ROOT/'branches'/f'{case}_on_{seed}.json').read_text())
    intervention=None;entries={}
    def bounded(a,b):return a.copy() if off else original_action(a,b)
    def observer(env,home,c,**kwargs):
        original_step=env.step
        def step(action,aperture):
            nonlocal off,intervention
            phase=env.phase;k=counts.get(phase,0);counts[phase]=k+1;i=len(signatures)
            signature,arrays,meta=snapshot(env,c,k);signatures.append(signature)
            if k==0:
                entries[phase]=dict(step=i,signature=signature,metadata=meta)
                if save:np.savez_compressed(ROOT/'snapshots'/f'{case}_{method}_{phase}.npz',**arrays)
            off=phase==off_phase and k<PULSE
            if off and intervention is None:intervention=i
            if expected is not None and (intervention is None or i<=intervention):assert signature==expected['signatures'][i],(case,seed,phase,i)
            before=env.observe();clean=not env.bad;goal=env.goal.copy();before_error=float(np.linalg.norm(goal-before['position']))
            after,command,info=original_step(action,aperture)
            metrics.append(dict(step=i,phase=phase,phase_step=k,off=off,clean_before=clean,position_error_before=before_error,position_error_after=float(np.linalg.norm(goal-after['position'])),rotation_error_after=float(np.linalg.norm(np.array([0.,0.,1.])-after['rotation'][:,2])),object_goal_xy_error=float(np.linalg.norm(env.data.xpos[env.obj,:2]-env.target)),object_height=float(env.data.xpos[env.obj,2]),collision_seen=bool(env.bad),grasped=bool(env.grasped),lifted=bool(env.lifted)))
            return after,command,info
        env.step=step
        return original_rollout(env,home,c,**kwargs)
    with patch.object(previous.core,'probe_rollout',observer),patch.object(previous,'bounded_action',bounded):
        result,logs=previous.evaluate(case,task,method,base,correction,'bounded',save=save)
    if off_phase is not None:assert intervention is not None
    branch=dict(case=case,seed=seed,off_phase=off_phase,intervention=intervention,entries=entries,signatures=signatures,metrics=metrics,prefix_identical=True,off_steps=sum(m['off'] for m in metrics))
    if save:(ROOT/'branches'/f'{case}_{method}.json').write_text(json.dumps(branch))
    return result,branch

def prepare():
    tasks,models,hashes=setup();ROOT.mkdir(exist_ok=False)
    for d in ('trials','states','actions','source','branches','snapshots'):(ROOT/d).mkdir()
    shutil.copy2(OLD/'cases.json',ROOT/'cases.json');pairs=[]
    for p in json.loads((OLD/'summary.json').read_text())['paired']:
        for c in p['changes']:pairs.append(dict(case=c['case'],seed=p['seed'],e120_gain=c['gain'],group=c['group']))
    assert len(pairs)==10;(ROOT/'pairs.json').write_text(json.dumps(pairs,indent=2))
    (ROOT/'protocol.json').write_text(json.dumps(dict(experiment='E123',checkpoints=hashes,inputs={str(OLD/n):fingerprint(OLD/n) for n in ('records.json','summary.json','cases.json')},phases=PHASES,pulse_steps=PULSE,
        intervention='For all six E120 gains and four regressions versus E118, exactly replay E120 on. At each of all nine task-phase entries independently replay the identical prefix then disable only the E120 residual for up to first8 actions within that phase. Resume unchanged E120 afterwards. Explicit probe is never disabled. Full MuJoCo integration state (including warmstart/control), controller Gram/cross/bias/recent history, local flag, task flags, phase/probe counters and goals hashed at every pre-action step. Assert full prefix and intervention snapshot equality. Deterministic prefix replay provides the identical-state branch, no qpos-only reset.',
        outcomes='Save all90 off continuations plus10 on references, every prefix/snapshot/action/full state trace. Report one-step and eighth-phase-step progress separately from final task success/collision. If phase ends before step8, fixed-horizon pair unavailable; record actual pulse duration. Mark branches already contaminated by earlier contact. These are selected development pairs, including known held placements; not an unbiased benchmark or an inference-time physics selector. No learned gate, retraining, checkpoint promotion or unseen bodies.'),indent=2))
    (ROOT/'records.json').write_text('[]')
    for name in ('matched_correction.py','bounded_history_correction.py','training_estimator_core.py','local_probe.py','local_history.py','arm_pick_place.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)

def run():
    tasks,models,hashes=setup();pairs=json.loads((ROOT/'pairs.json').read_text());rows=[]
    assert json.loads((ROOT/'records.json').read_text())==[]
    for pair in pairs:
        case,seed=pair['case'],pair['seed']
        for phase in (None,*PHASES):
            r,b=evaluate(case,tasks[case],seed,models[f'reference_{seed}'],models[f'correction_{seed}'],phase)
            if phase is None:
                old=json.loads((OLD/'trials'/f'{case}_candidate_{seed}_frozen_bias.json').read_text());old['method']=r['method'];assert r==old
                np.testing.assert_array_equal(np.load(ROOT/'states'/f"{case}_{r['method']}_frozen_bias.npz")['states'],np.load(OLD/'states'/f'{case}_candidate_{seed}_frozen_bias.npz')['states'])
            rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(seed=seed,off_phase=phase,e120_gain=pair['e120_gain']))
            (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('done',case,seed,phase,r['success'],flush=True)
    for p,h in hashes.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()

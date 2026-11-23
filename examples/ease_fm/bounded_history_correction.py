"""E120: paired first-divergence audit and bounded action residual, frozen weights."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from . import candidate_history_repair as prior
from .candidate_history_repair import core, MotionTransformer, fingerprint
from .arm_pick_place import ROT_SCALE, rotation_error

ROOT = Path('outputs/ease_bounded_history_v1')
OLD = prior.ROOT
ALPHA = .25
CAP = .001


def setup():
    torch.set_num_threads(4)
    tasks, _, models, hashes = prior.setup()
    for seed in range(3):
        p = OLD / f'seed_{seed}/policy.pt'
        m = MotionTransformer().cuda()
        m.load_state_dict(torch.load(p, map_location='cuda', weights_only=True))
        models[f'correction_{seed}'] = m.eval()
        hashes[str(p)] = fingerprint(p)
    for m in models.values():
        if m is not None:
            for p in m.parameters(): p.requires_grad_(False)
    core.ROOT = ROOT
    return tasks, models, hashes


def bounded_action(base, corrected, alpha=ALPHA, cap=CAP):
    return np.clip(base + np.clip(alpha * (corrected - base), -cap, cap), -.04, .04)


def evaluate(case, task, method, base, correction, kind, save=True):
    logs = []
    original_rollout = core.probe_rollout
    def rollout(env, home, controller, **kwargs):
        original_step = env.step
        def step(action, aperture):
            obs = env.observe()
            error = np.r_[env.goal-obs['position'], ROT_SCALE*rotation_error(np.eye(3), obs['rotation'])]
            old_model = controller.model
            try:
                controller.model = correction
                j, bias = controller.predict()
            finally:
                controller.model = old_model
            proposed = np.clip(j.T @ np.linalg.solve(j@j.T + .015**2*np.eye(6), .35*error) + bias, -.04, .04)
            if env.phase == 'probe':
                executed = action.copy()  # Original evaluator overrides this with the fixed probe.
            elif kind == 'baseline': executed = action.copy()
            elif kind == 'unbounded': executed = proposed.copy()
            else: executed = bounded_action(action, proposed)
            if env.phase != 'probe':
                assert np.max(np.abs(executed)) <= .04 + 1e-12
                if kind == 'bounded': assert np.max(np.abs(executed-action)) <= CAP + 1e-12
            after, command, info = original_step(executed, aperture)
            logs.append(dict(step=len(logs), phase=env.phase, base_action=action.tolist(), correction_action=proposed.tolist(), executed_command=command.tolist(), nominal_action=executed.tolist(), active=env.phase!='probe', same_history_delta_max=float(np.max(np.abs(proposed-action))), applied_delta_max=0. if env.phase=='probe' else float(np.max(np.abs(executed-action))), q_before=obs['q'].tolist(), position_before=obs['position'].tolist(), gram=controller.gram.tolist(), cross=controller.cross.tolist()))
            return after, command, info
        env.step = step
        return original_rollout(env, home, controller, **kwargs)
    with patch.object(core, 'probe_rollout', rollout):
        result = core.evaluate(case, task, method, base, 'frozen_bias', save_states=save)
    if save:
        stem = f'{case}_{method}_frozen_bias'
        (ROOT/'trials'/f'{stem}.json').write_text(json.dumps(result))
        (ROOT/'actions'/f'{stem}.json').write_text(json.dumps(logs))
    return result, logs


def prepare():
    tasks, models, hashes = setup()
    ROOT.mkdir(exist_ok=False)
    for name in ('trials','states','actions','source','diagnosis'): (ROOT/name).mkdir()
    shutil.copy2(OLD/'cases.json', ROOT/'cases.json')
    protocol = dict(experiment='E120', checkpoints=hashes, inputs={str(OLD/n):fingerprint(OLD/n) for n in ('records.json','summary.json','cases.json')}, alpha=ALPHA, cap_rad=CAP,
        design='Freeze E118 baseline and E119 correction weights. On each same measured history compute both clipped DLS actions. Apply baseline + clip(0.25*(correction-baseline), +/-0.001 rad/joint), then original +/-0.04 clip. No correction to calibration or fixed local probe; history and frozen calibrated bias rules unchanged. This bounds action changes, not physical state deviation or safety.',
        diagnosis='Replay all seven E119 lost-success pairs under baseline and unbounded correction. Verify states, task traces, calibration and final estimator exactly. Save both proposals on each actual history. Locate first numerical action divergence (>1e-10 rad), first material divergence (>1e-4 rad), and first forbidden collision. Held failures inspected diagnostically, never optimized.',
        evaluation='One prespecified bound, all24 training-body placements x3 seeds. No fitting, sweep, best seed/checkpoint selection, physics selector or privileged Jacobian. Compare to E118 and E119. Require higher success than E118, no lost E118 success, no more collision episodes. Held placements reported separately; no unseen bodies.', test_evaluated=False)
    (ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for row in json.loads((OLD/'records.json').read_text()):
        if not (row['method']=='analytical' or row['method'].startswith(('reference_','candidate_'))): continue
        original=row['method']; method=original.replace('candidate_','unbounded_'); row['method']=method
        a=f"{row['case']}_{original}_frozen_bias"; b=f"{row['case']}_{method}_frozen_bias"
        trial=json.loads((OLD/'trials'/f'{a}.json').read_text()); trial['method']=method
        (ROOT/'trials'/f'{b}.json').write_text(json.dumps(trial))
        for ext in ('npz','xml'): shutil.copy2(OLD/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(row)
    assert len(rows)==168
    (ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    for name in ('bounded_history_correction.py','training_estimator_core.py','local_probe.py','local_history.py','pick_place_control.py','retention_fresh.py'):
        shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)


def diagnose():
    tasks, models, hashes=setup(); summary=json.loads((OLD/'summary.json').read_text()); findings=[]
    for pair in summary['paired']:
        seed=pair['seed']
        for change in pair['changes']:
            if change['gain']: continue
            case=change['case']; logs={}
            for kind, family in (('baseline','reference'),('unbounded','unbounded')):
                method=f'{family}_{seed}'; stem=f'{case}_{method}_frozen_bias'
                old=json.loads((ROOT/'trials'/f'{stem}.json').read_text())
                old_states=np.load(ROOT/'states'/f'{stem}.npz')['states'].copy()
                result,log=evaluate(case,tasks[case],method,models[f'reference_{seed}'],models[f'correction_{seed}'],kind)
                for key in ('trace','calibration','initial_estimator','final_estimator','success','collision','phases'): assert result[key]==old[key],key
                np.testing.assert_array_equal(old_states,np.load(ROOT/'states'/f'{stem}.npz')['states'])
                # Preserve inherited telemetry: diagnostic core prediction errors
                # use the frozen base, while original E119 telemetry uses E119.
                (ROOT/'trials'/f'{stem}.json').write_text(json.dumps(old))
                logs[kind]=log
            numerical=next((r for r in logs['baseline'] if r['active'] and r['same_history_delta_max']>1e-10),None)
            material=next((r for r in logs['baseline'] if r['active'] and r['same_history_delta_max']>1e-4),None)
            trial=json.loads((ROOT/'trials'/f'{case}_unbounded_{seed}_frozen_bias.json').read_text())
            collision=next((dict(step=i,phase=r['phase'],contacts=r['bad']) for i,r in enumerate(trial['trace']) if r['bad']),None)
            compact=lambda r:None if r is None else {k:r[k] for k in ('step','phase','base_action','correction_action','same_history_delta_max')}
            finding=dict(case=case,seed=seed,group=change['group'],first_numerical=compact(numerical),first_material=compact(material),first_collision=collision,exact_replays=2)
            findings.append(finding);(ROOT/'diagnosis/first_divergence.json').write_text(json.dumps(findings,indent=2)); print('diagnosed',case,seed,flush=True)
    for p,h in hashes.items(): assert fingerprint(p)==h


def run():
    tasks,models,hashes=setup();rows=json.loads((ROOT/'records.json').read_text());assert len(rows)==168
    for case,t in enumerate(tasks):
        for seed in range(3):
            method=f'candidate_{seed}';r,logs=evaluate(case,t,method,models[f'reference_{seed}'],models[f'correction_{seed}'],'bounded')
            old=json.loads((ROOT/'trials'/f'{case}_reference_{seed}_frozen_bias.json').read_text())
            assert r['calibration']==old['calibration'] and r['initial_estimator']==old['initial_estimator']
            rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')})
        (ROOT/'records.json').write_text(json.dumps(rows,indent=2));print('evaluated',case,flush=True)
    for p,h in hashes.items(): assert fingerprint(p)==h


if __name__=='__main__':
    import sys
    {'prepare':prepare,'diagnose':diagnose,'evaluate':run}[sys.argv[1]]()

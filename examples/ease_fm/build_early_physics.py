"""Generate an isolated E109 evaluator without changing historical experiments."""
from pathlib import Path
base=Path(__file__).parent
s=(base/'single_phase.py').read_text().split('\ndef run():')[0]
s=s.replace('E104: single-phase ramps with phase-matched time controls.','E109 rollout: identical physical loop with approach-only prefix invariant.')
s=s.replace("if r['phase'] in ('approach','descend','probe','close')","if r['phase'] in ('approach',)")
(base/'early_physics_rollout.py').write_text(s)
s=(base/'physics_action.py').read_text()
s=s.replace('E108: fixed 40ms physics action selection on training bodies only.','E109: pre-first-collision physics selection across five motion phases.')
s=s.replace('from . import single_phase as prior','from . import early_physics_rollout as prior')
s=s.replace('outputs/ease_physics_action_v1','outputs/ease_early_physics_v1')
s=s.replace("MODES=('baseline','physics')","MODES=('baseline','physics')\nPHASES=('descend','lift','transport','lower','retreat')")
s=s.replace("getattr(env,'phase',None)!='retreat'","getattr(env,'phase',None) not in PHASES").replace("getattr(env,'phase',None)=='retreat'","getattr(env,'phase',None) in PHASES")
s=s.replace("and pred['pairs'] and not current:","and pred['pairs'] and not current and not env.bad:")
s=s.replace("return dict(pairs=sorted(hits),qpos=d.qpos.copy(),qvel=d.qvel.copy())","error=np.r_[env.goal-d.site_xpos[env.site],.15*(np.array([0.,0.,1.])-d.site_xmat[env.site].reshape(3,3)[:,2])]\n        return dict(pairs=sorted(hits),qpos=d.qpos.copy(),qvel=d.qvel.copy(),error_sq=float(error@error))")
s=s.replace("row=dict(step=env.steps,clean=not current", "obs=env.observe();error=np.r_[env.goal-obs['position'],.15*(np.array([0.,0.,1.])-obs['rotation'][:,2])]\n        row=dict(phase=env.phase,before_error_sq=float(error@error),original_after_error_sq=pred['error_sq'],selected_after_error_sq=chosen_pred['error_sq'],step=env.steps,clean=not current")
s=s.replace("if v['phase'] not in ('retreat','settle')","if v['phase']=='approach'")
s=s.replace("sum(x['clean'] and bool(x['original_pairs']) for x in a)","sum(x['clean'] and not x['prior_episode_collision'] and bool(x['original_pairs']) for x in a)")
s=s.replace('exact_pre_retreat_prefixes=72','exact_pre_descent_prefixes=72')
start=s.index("    spec=dict(experiment='E108'");end=s.index("    (ROOT/'protocol.json')",start)
s=s[:start]+'''    spec=dict(experiment='E109',inputs=inputs,checkpoints=hashes,
        design='All36 original training-body pairs x baseline/physics=72 full tasks. Same24 train/12 held-placement split. Same E108 five alternatives,40ms horizon, static eligibility, nearest original target ranking, unchanged budgets/weights/apertures; no route planner. Extend intervention to descend/lift/transport/lower/retreat, only before ANY earlier forbidden contact. Calibration and approach remain exactly unchanged. Probe, close, release and settle unchanged. Altered descent can change subsequent history; this is not a fixed post-probe prefix.',
        rule='Preserve original static-selected command if prediction contact-free, current contact exists, or episode previously collided. Otherwise test E108 half/quarter/bias/hold/brake candidates; require same static threshold and zero predicted forbidden contact, choose nearest original joint target with fixed tie order, preserve original if none. No new progress objective in selection: isolate extending phase coverage and disabling post-collision interventions.',
        progress='Log phase completion, phase steps, grasp/lift, final object errors; for changed actions record same-state original versus selected40ms squared XYZ plus .15-scaled tool-axis error. Reduced one-step error is a diagnostic, not task success. Report collision-free task failures and stalled phases separately; preserve all paired gains/losses.',
        gate='Strict training success increase, preserve every baseline success on both train and held placements, no collision increase in either split, no training seed decline. No automatic transfer evaluation.',
        verification='Every selected prediction matches executed qpos/qvel/contact union;36 full baselines reproduce E104;72 approach prefixes identical. Hash frozen inputs and checkpoints.',
        videos='Case0 seed0 both modes, all paired task gains/losses, earliest changed-action pair and missing outcome category; additionally first collision-free physics failure with its baseline if present. Full-result replay verification.',
        limits='Exact simulator-state/model access remains privileged. Same four training bodies; held placements are not unseen embodiments. No E105 outcomes or reserved final test. New actions may change estimator histories and later phases. No horizon/candidate tuning after outcomes.',test_evaluated=False)
'''+s[end:]
s=s.replace("shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);rows=[]", "shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);shutil.copy2(Path(prior.__file__),ROOT/'source'/Path(prior.__file__).name);rows=[]")
(base/'early_physics.py').write_text(s)
s=(base/'physics_action_videos.py').read_text().replace('from .physics_action import','from .early_physics import').replace('E108','E109').replace('Physics-based action selection on training bodies','Early-phase physics action selection on training bodies')
where=s.index('    for outcome in (True,False):')
s=s[:where]+'''    collision_free_failures=sorted([r for r in rows if r['mode']=='physics' and not r['success'] and not r['collision']],key=lambda r:(r['case'],r['seed']))
    if collision_free_failures:
        for m in ('baseline','physics'):
            key=(collision_free_failures[0]['case'],collision_free_failures[0]['seed'],None,m)
            if key not in selection:selection.append(key)
'''+s[where:]
s=s.replace('earliest changed-action pair, missing outcome category.','earliest changed-action pair, first collision-free physics failure, missing outcome category.')
(base/'early_physics_videos.py').write_text(s)

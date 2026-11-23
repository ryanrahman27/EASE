"""Adapt E175's independent physical audit to the two fixed estimators."""
from pathlib import Path

def run():
    base=Path(__file__).parent;s=(base/'early_object_report.py').read_text()
    s=s.replace('from .early_object import','from .frozen_estimator_comparison import').replace('from .link_clearance_report import outcome_counts','from .frozen_estimator_report import outcome_counts\nfrom .arm_pick_place import ROT_SCALE,rotation_error')
    s=s.replace('E174_guarded','E175_frozen')
    s=s.replace("stem=f'{case}_e120_ensemble_frozen_bias';t=", "stem=f'{case}_{condition}_frozen_bias';reference_stem=f'{case}_e120_ensemble_frozen_bias';t=")
    s=s.replace("REFERENCE/'trials'/f'{stem}.json'", "REFERENCE/'trials'/f'{reference_stem}.json'").replace("REFERENCE/'states'/f'{stem}.npz'", "REFERENCE/'states'/f'{reference_stem}.npz'")
    old="assert t['trace'][:i]==ref['trace'][:i];np.testing.assert_array_equal(states[:i],np.load(REFERENCE/'states'/f'{reference_stem}.npz')['states'][:i])"
    s=s.replace(old,"if condition=='e120_ensemble':\n                            "+old)
    old="offline=read((ROOT/'plans' if early else REFERENCE/'planning')/f'{case}.json');assert plan['available']==offline['available'];assert plan['route']==offline['route']"
    s=s.replace(old,"if condition=='e120_ensemble':\n                            offline=read(REFERENCE/'planning'/f'{case}.json');assert comparable(plan)==comparable(offline)")
    s=s.replace('def run():',"def comparable(value):\n    if isinstance(value,dict):return {k:comparable(v) for k,v in value.items() if k!='planning_seconds'}\n    if isinstance(value,list):return [comparable(v) for v in value]\n    return value\n\ndef run():",1)
    s=s.replace("scratch.site_xpos[site]-l['goal']", "scratch.site_xpos[site]-pick")
    s=s.replace("if trigger_reason!='early_object_tilt':", "if condition=='e120_ensemble':")
    s=s.replace("strict_screen=counts['success']>47 and counts['collisions']==0 and not paired['E175_frozen']['losses']", "matches_E175_successes=counts['success']==48 and counts['collisions']==0")
    s=s.replace('online_plans_equal_static_diagnosis=True','E120_plans_equal_reference=condition==\'e120_ensemble\'')
    s=s.replace('matched_preintervention_prefix=True',"matched_preintervention_prefix=condition=='e120_ensemble'")
    start=s.index('        for case in (0,26,43,62,65,67):');s=s[:start]+'''    for p,h in {**read(ROOT/'frozen.json'),**read(ROOT/'protocol.json')['checkpoints']}.items():assert fingerprint(p)==h
    for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name)),p.name
    assert checks['e120_ensemble']['exact_nonactivated_episodes']==48
    save(ROOT/'summary.json',summary);save(ROOT/'verification.json',dict(conditions=checks,new_episodes=96,preflight_exact_replays=5,full_E175_exact_replays=48,privileged=True,held_evaluated=False))
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print(summary['conditions'],flush=True)

if __name__=='__main__':run()
'''
    marker="                if l['phase']=='descend':\n                    pe="
    extra="""                if condition=='analytical':
                    analytical=np.linalg.solve(np.array(l['estimator_gram']),np.array(l['estimator_cross'])).T
                    np.testing.assert_allclose(analytical,l['estimator_jacobian'],atol=1e-14,rtol=0)
                    if l['phase']!='probe':
                        error=np.r_[np.array(l['goal'])-d.site_xpos[site],ROT_SCALE*rotation_error(np.eye(3),d.site_xmat[site].reshape(3,3))]
                        nominal=np.clip(analytical.T@np.linalg.solve(analytical@analytical.T+.015**2*np.eye(6),.35*error)+np.array(l['bias']),-.04,.04)
                        np.testing.assert_allclose(nominal,l['nominal_action'],atol=1e-12,rtol=0)
                if l['eligible']:np.testing.assert_array_equal(l['estimator_jacobian'],l['estimated_jacobian'])
"""
    assert marker in s;s=s.replace(marker,extra+marker)
    (base/'frozen_estimator_audit.py').write_text(s)

if __name__=='__main__':run()

"""Reuse independent E176 physical checks for new paired bodies."""
from pathlib import Path

def run():
    base=Path(__file__).parent;s=(base/'frozen_estimator_audit.py').read_text().replace('from .frozen_estimator_comparison import','from .unseen_estimator import').replace('from .frozen_estimator_report import outcome_counts','from .unseen_estimator_report import outcome_counts')
    start=s.index("    refs={'E120_release'");end=s.index('    for condition in CONDITIONS:',start)
    s=s[:start]+"    refs={};summary={'references':{},'conditions':{}};checks={}\n"+s[end:]
    s=s.replace("assert len(rows)==48 and read(root/'verification.json')['episodes']==48", "assert len(rows)==len(read(ROOT/'cases.json')) and read(root/'verification.json')['episodes']==len(rows)")
    s=s.replace("ref=read(REFERENCE/'trials'/f'{reference_stem}.json')", "other='analytical' if condition=='e120_ensemble' else 'e120_ensemble';ref=read(ROOT/other/'trials'/f'{case}_{other}_frozen_bias.json')")
    start=s.index("                        if condition=='e120_ensemble':");end=s.index("                        if plan['available']:",start);s=s[:start]+s[end:]
    start=s.index("            if condition=='e120_ensemble':");end=s.index('            desc=next',start);s=s[:start]+s[end:]
    s=s.replace(",matches_E175_successes=counts['success']==48 and counts['collisions']==0",'').replace("E120_plans_equal_reference=condition=='e120_ensemble'","static_plan_feasibility_verified=True").replace("matched_preintervention_prefix=condition=='e120_ensemble'","paired_initial_statistics_verified=True")
    s=s.replace("    assert checks['e120_ensemble']['exact_nonactivated_episodes']==48\n",'')
    s=s.replace('new_episodes=96,preflight_exact_replays=5,full_E175_exact_replays=48,privileged=True,held_evaluated=False',"new_episodes=2*len(read(ROOT/'cases.json')),preflight_exact_replays=10,privileged=True,prospective_unseen_bodies=True,reserved_legacy_tests_evaluated=False")
    s=s.replace("    save(ROOT/'summary.json',summary);", "    for method in CONDITIONS:assert read(ROOT/f'replay_{method}/verification.json')['exact_replays']==5\n    save(ROOT/'summary.json',summary);")
    (base/'unseen_estimator_audit.py').write_text(s)

if __name__=='__main__':run()

from pathlib import Path
base=Path(__file__).parent
s=(base/'early_physics_finalize.py').read_text().replace('from .early_physics import ROOT','from .progress_selector import ROOT')
s=s.replace('Revision 95','Revision 97').replace('Revision 94','Revision 96')
s=s.replace('E109','E111').replace('| E108 |','| E110 |').replace('4.99','4.101')
s=s.replace('ease_early_physics_v1','ease_progress_selector_v1')
s=s.replace("('early_physics_finalize.py','build_early_physics.py')","('progress_selector_finalize.py','build_progress_selector.py')")
start=s.index('    lead=f"');end=s.index('\n    p=base/',start)
s=s[:start]+'''    lead=f"**E111 completed:** Frozen broad progress-aware selector with nonnegative predicted geometric clearance, tested in72 paired training-body episodes: {scores}. Training/held-placement gate {'PASS' if summary['passes'] else 'FAIL'}. Same weights and task budgets; no route planner. Reports phase progress,2mm-margin fraction and search compute cost. All36 baseline replays,72 approach prefixes and {verify['exact_selected_predictions']} selected next states/contact unions verify exactly. Saves {len(manifest['videos'])} verified executed videos. Privileged exact model/state; no unseen-body or final-test evaluation. See Section4.101."
'''+s[end:]
s=s.replace('Earlier-phase physics selection and progress','Broad progress-aware physics selection').replace('tests earlier-phase physics selection','tests frozen broad progress-aware physics selection')
(base/'progress_selector_finalize.py').write_text(s)

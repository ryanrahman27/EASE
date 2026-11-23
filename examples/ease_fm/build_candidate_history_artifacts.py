from pathlib import Path
b=Path(__file__).parent
s=(b/'broad_history_videos.py').read_text().replace('from .broad_history_repair import ROOT','from .candidate_history_repair import ROOT').replace('==204','==276').replace('E118','E119').replace('Broader training histories / Phase retention','Candidate-history correction / Retention')
(b/'candidate_history_videos.py').write_text(s)
s=(b/'broad_history_phase_audit.py').read_text().replace('from .broad_history_repair import','from .candidate_history_repair import')
(b/'candidate_history_phase_audit.py').write_text(s)
s=(b/'broad_history_finalize.py').read_text().replace('from .broad_history_repair import ROOT,setup,core','from .candidate_history_repair import ROOT,setup,core').replace('inherited_episodes=84,new_reference_episodes=48','inherited_episodes=204,fitting_history_replays=48')
(b/'candidate_history_finalize.py').write_text(s)

"""Freeze E110 bank at40ms and E111 reporting/video entry points."""
from pathlib import Path
base=Path(__file__).parent
s=(base/'goal_action_audit.py').read_text();body=s[s.index('def audit_state('):s.index('    banks={}')] 
body=body.replace('def audit_state(','def search_actions(').replace('range(1,81)','range(1,21)').replace('if k in (20,80):','if k==20:')
body=body.replace('pairs=sorted(hits)','pairs=[list(p) for p in sorted(hits)]')
body=body.replace("vector=v.tolist(),object=data.xpos[env.obj].tolist())","vector=v.tolist(),object=data.xpos[env.obj].tolist(),qpos=data.qpos.tolist(),qvel=data.qvel.tolist())")
header='"""Frozen E110 action bank,40ms only; no filesystem effects."""\nimport numpy as np\nimport mujoco\nfrom .geometry_filter import allowed_pair\nfrom .physics_prediction_audit import forbidden\n\n'
(base/'progress_action_bank.py').write_text(header+body+"    return dict(candidates=candidates,unique_physics_branches=len(cache),before_error_sq=before)\n")
s=(base/'early_physics_videos.py').read_text().replace('from .early_physics import','from .progress_selector import').replace('E109','E111').replace('Early-phase physics action selection on training bodies','Progress-aware physics selector on training bodies')
(base/'progress_selector_videos.py').write_text(s)
s=(base/'early_physics_report.py').read_text().replace('from .early_physics import','from .progress_selector import').replace('E109','E111')
s=s.replace('safer actions before first collision across motion phases','progress-aware selection with geometric nonpenetration')
s=s.replace('Same E108 five alternatives,40ms prediction, static admissibility and nearest-original-target ranking.','Frozen E110 broad bank,40ms prediction, original static admissibility, zero physical contacts, nonnegative minimum geometric clearance and at least1% pose-error reduction (floor1e-8). Rank by smallest predicted pose error, then larger clearance, then name. Preserve original commands unless contact or penetration is predicted; preserve original if no candidate qualifies.')
start=s.index("    lines+=['','## Interpretation of this run'");end=s.index("    (ROOT/'report.md').write_text",start)
s=s[:start]+"    lines+=['','The broader search uses exact physics and finite-difference simulated response. This is a privileged model-based controller component, not a new learned policy. See protocol.json for frozen candidate counts, random seeds and thresholds.']\n"+s[end:]
(base/'progress_selector_report.py').write_text(s)

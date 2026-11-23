"""Create isolated E112 adapters without changing frozen E111 control logic."""
from pathlib import Path

base=Path(__file__).parent
s=(base/'early_physics_rollout.py').read_text()
start=s.index('    original=json.loads(')
end=s.index('    return dict(',start)
s=s[:start]+s[end:]
s=s.replace('"""E109 rollout: identical physical loop with approach-only prefix invariant."""','"""E112 identical physical loop; reference checks live in the experiment runner."""')
(base/'unseen_progress_rollout.py').write_text(s)
s=(base/'progress_selector.py').read_text()
s=s[:s.index('\ndef run():')]
s=s.replace('from . import early_physics_rollout as prior','from . import unseen_progress_rollout as prior')
s=s.replace("ROOT=Path('outputs/ease_progress_selector_v1')","ROOT=Path('outputs/ease_unseen_progress_v1')")
start=s.index('    old=json.loads(')
end=s.index("    r['physics_audit']",start)
s=s[:start]+s[end:]
s=s.replace('"""E111: frozen progress-aware broad action selector in full training episodes."""','"""E112 adapter: E111 selection and physical prediction logic unchanged."""')
(base/'unseen_progress_selector.py').write_text(s)

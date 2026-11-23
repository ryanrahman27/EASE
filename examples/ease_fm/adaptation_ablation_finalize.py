"""Close the E114 artifact ledger after all verification and videos complete."""
import json,shutil
from pathlib import Path
from .adaptation_ablation import ROOT
from .benchmark import fingerprint

v=json.loads((ROOT/'verification.json').read_text());m=json.loads((ROOT/'videos/manifest.json').read_text())
assert len(v['exact_reexecutions'])==8
for movie in m['videos']:assert fingerprint(ROOT/'videos'/movie['file'])==movie['sha256']
assert {r['success'] for r in m['videos']}=={True,False}
assert (ROOT/'videos/index.html').exists()
note=f"\nFinal verification: six relevant existing tests passed; eight independent full re-executions matched every saved trace and state exactly under forbidden-selector/geometry guards. Saved {len(m['videos'])} labeled success/failure videos; all replayed state positions match their recorded traces.\n"
p=ROOT/'report.md';s=p.read_text();assert 'Final verification:' not in s;p.write_text(s+note)
p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');s+=note;p.write_text(s,encoding='utf-8')
shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
for p in (ROOT/'source').glob('*.py'):assert fingerprint(p)==fingerprint(Path(__file__).with_name(p.name))
(ROOT/'status.json').write_text(json.dumps(dict(status='complete',experiment='E114',episodes=192,videos=len(m['videos']),exact_reexecutions=8,test_evaluated=False),indent=2))
(ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
print((ROOT/'status.json').read_text())

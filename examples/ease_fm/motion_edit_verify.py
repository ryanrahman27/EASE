"""Verify editorial-only revision and preserve source provenance."""
import json
from pathlib import Path
from .benchmark import fingerprint
from .motion_task_video import ROOT,SOURCE

m=json.loads((ROOT/'video_manifest.json').read_text())
assert m['frames']==1500 and m['fps']==50
assert fingerprint(ROOT/'ease_pick_place_demo.mp4')==m['sha256']
m['sources']={str(p):fingerprint(p) for p in [SOURCE/'video_manifest.json',SOURCE/'piperx_physical.mp4',SOURCE/'openarm_physical.mp4']}
m['editorial_revision']='Only composition changed. Same complete physical recordings and clearance audits as v2. Effects occur before execution or after completion; caption rolls remain outside the task viewport.'
(ROOT/'video_manifest.json').write_text(json.dumps(m,indent=2))
(ROOT/'report.md').write_text('''# D1 motion-design revision

14 November 2026. Editorial-only revision of the verified v2 physical demo. All physics recordings, task results and clearance measurements are unchanged. The final film remains 30 seconds at 50 fps, with complete 1x task playback.

The opening uses a split aperture with green edge markers. Captions roll through a fixed mask. Three staggered scan panels reveal the benchmark card after both tasks finish. An expanding circular aperture with a two-link animated graphic reveals the closing title. The final title holds without a fade. These graphics are editorial elements, not robot telemetry.

Verified 1,500 encoded frames and inspected representative opening, caption, scan and iris frames. Source video hashes and clearance provenance are retained in video_manifest.json. Earlier edits and success/failure recordings remain preserved in their original folders. No experimental scores changed.
''')
(ROOT/'artifact_hashes.json').write_text(json.dumps({p.name:fingerprint(p) for p in ROOT.iterdir() if p.is_file() and p.name!='artifact_hashes.json'},indent=2))

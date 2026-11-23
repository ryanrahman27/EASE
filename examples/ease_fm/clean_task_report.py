"""Save demo correction provenance without changing research benchmark scores."""
import json
from pathlib import Path
from .benchmark import fingerprint
from .clean_task_video import ROOT

def run():
    manifest=json.loads((ROOT/'video_manifest.json').read_text())
    assert fingerprint(ROOT/'ease_pick_place_demo.mp4')==manifest['sha256']
    report='''# D1 revision: smooth known-arm task demonstration

14 November 2026. The previous video had visible fingertip/table penetration because its task frame and simplified grasp pads sat proximal to the visible fingertips. This revision aligns both with the distal fingers and enables gripper mesh collision with the table and floor. The free cube still moves only through simulated contact. No object animation, welding, cropping to conceal contact, or trajectory interpolation is used.

Both frozen-neural tasks succeed and repeat with exactly matching full control traces. Original attempts and the original failure video remain in ease_known_arm_task_demo_v1. Both new attempts per robot are saved here. These development demonstrations are not additional independent benchmark episodes.

The physical simulation runs at 500 Hz, the controller updates at 25 Hz, and the renderer now uses actual snapshots at 50 Hz, replacing the previous 12.5 Hz rendered motion. The 30-second edit plays the complete tasks at 1x, uses short stable captions and smooth editorial dissolves, and preserves the black/white/green styling. History updates are measured online; network weights remain frozen. Demonstration servos, pads, gravity compensation and the model-derived home prior remain configured as documented in D1.

## Rendered geometry verification

Every 20 ms snapshot is audited by transforming the visible gripper mesh vertices into world coordinates and measuring their height over the tabletop footprint and floor. All audited gripper clearances are positive. This audit covers rendered gripper geometry at each video sample; it is not a general whole-robot continuous collision certificate.

'''
    report+='```json\n'+json.dumps(manifest['clearance'],indent=2)+'\n```\n'
    report+='\nThe results card refers separately to E111–E113, whose scores are unchanged. Source, states, traces, models, raw videos, clearance audits and a hashed manifest are retained alongside the final film.\n'
    (ROOT/'report.md').write_text(report)
    p=Path(__file__).with_name('TECHNICAL_REPORT.md');s=p.read_text(encoding='utf-8');marker='### D1 revision: fingertip clearance and smooth playback'
    if marker not in s:
        s=s.replace('Revision 100 · 14 November 2026','Revision 101 · 14 November 2026',1)
        s+='\n'+marker+'\n\nCorrected proximal grasp frames/pads that allowed the native visual fingertips to intersect the tabletop. Enabled gripper mesh/table and floor contacts. Both known-arm tasks still succeed and repeat exactly. All 840 rendered snapshots per arm have positive gripper/table clearance (PiperX 7.76 mm; OpenArm 8.02 mm). New 50 fps physics replay and gentle dissolves replace the 12.5 fps render and abrupt edits. Prior artifacts preserved. See `outputs/ease_known_arm_task_demo_v2/report.md`; no benchmark scores changed.\n'
        p.write_text(s,encoding='utf-8')
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))

if __name__=='__main__':run()

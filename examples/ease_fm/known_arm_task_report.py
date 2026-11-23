"""Document the showcase separately from controlled transfer benchmarks."""
import json,shutil
from pathlib import Path
from .known_arm_task_demo import ROOT
from .benchmark import fingerprint

def run():
    verification=json.loads((ROOT/'demo_verification.json').read_text());video=json.loads((ROOT/'video_manifest.json').read_text());assert fingerprint(ROOT/'ease_pick_place_demo.mp4')==video['sha256']
    attempts=[]
    for p in sorted(ROOT.glob('*_attempt*.json')):
        r=json.loads(p.read_text());attempts.append(dict(file=p.name,robot=r['robot'],method=r.get('method','analytical'),success=r['success'],grasped=r['grasped'],lifted=r['lifted']))
    report=['# D1: PiperX / OpenArm physical pick-and-place showcase','',
        '14 November 2026. Two recorded development tasks execute the same frozen E99 retained seed0 neural response estimator on six-joint PiperX and seven-joint OpenArm visual/kinematic models. Both grasp, lift, transfer10cm, release and settle a physical free cube. The30-second film replays these trajectories at1x simulated time; the object is never welded, attached or assigned a scripted motion.',
        '', '## What adapts','',
        'A48-step home probe and subsequent measured joint/tool transitions update the controller normal-equation history and command bias. The frozen transformer predicts a local6-by-n response matrix from these statistics. A damped least-squares task controller maps pose error through that estimate into joint targets. Network weights do not change. During close/release/settle, joint targets are held while observations continue updating history.',
        '', 'The film displays actual history length and a12-transition rolling XYZ motion-prediction residual. Mean updated-estimate error is compared with the initial estimate applied to the same observed transitions; this is a diagnostic prediction comparison, not a closed-loop no-history ablation.',
        '',json.dumps(verification,indent=2),'','## Task configuration and scope','',
        'Native source visual meshes and joint hierarchies from the previous demo are retained. This integration uses configured600kp arm position servos, gravity compensation,1500kp finger servos, simplified frictional contact pads, and disabled original mesh collision geometry. PiperX finger armature0.1 stabilizes the demonstration servo. The free cube is3cm wide and35g on a20cm-high table. Success requires bilateral grasp, object elevation above28cm, final XY error<2.5cm, resting-height tolerance8mm, low object velocity and no current finger contact. No native-arm collision-free benchmark claim is made.',
        '', 'Home pose uses exact-model IK; initial estimator statistics contain a0.8-scaled exact home Jacobian prior. No further model-Jacobian queries occur in the motion loop. The task sequence, control gains, servos and contacts differ from E111-E113. These two tuned development demonstrations are not an independent transfer evaluation, hardware result, history-only identification result, or evidence of arbitrary unknown-arm adaptation.',
        '', '## Development record','',
        'The initial PiperX analytical attempt failed to grasp because its visual-only finger inertia produced an unstable demo servo. Adding finger armature resolved this. OpenArm analytical initialization succeeded. Each subsequent frozen-neural run succeeded and was repeated; full numeric traces and qpos/qvel arrays match exactly. All failed and successful attempts remain saved. Do not pool these development attempts into a success-rate estimate.',
        '',json.dumps(attempts,indent=2),'','## Relation to research results','',
        'E111: physics-assisted selector improves familiar-body tasks22/36 to31/36, nine gains and no losses. E112: frozen selector on prospective body parameters28/48 to31/48, three gains all on one seven-joint body. E113: release-only extension on reused six-joint cases14/24 to16/24, two gains and no losses. These support incremental full-task control improvements. They do not yet establish broad learned adaptation to unseen morphology. Required attribution remains frozen/no-history/analytical comparisons on prospectively defined bodies under matched privileges.',
        '', 'The new known-arm film illustrates a functioning shared estimator/controller architecture with observed-history updates. Its on-screen results card retains the original procedural-benchmark scope. Full configuration details are in this report and the video manifest; brief labels in the film keep the visual presentation concise.',
        '', '## Artifacts','', 'ease_pick_place_demo.mp4:30-second short film. piperx_physical.mp4 and openarm_physical.mp4: physical trajectory replays. *_states.npz: complete executed state sequences. *_attempt*.json/xml: every result and model. demo_verification.json: repeatability and prediction diagnostics. video_manifest.json: source hashes and timing.']
    (ROOT/'report.md').write_text('\n'.join(report)+'\n')
    base=Path(__file__).parent
    shutil.copy2(__file__,ROOT/Path(__file__).name)
    p=base/'TECHNICAL_REPORT.md';text=p.read_text(encoding='utf-8');marker='## Appendix D1: known-arm task demonstration'
    assert marker not in text
    text=text.replace('Revision 99 · 14 November 2026','Revision 100 · 14 November 2026',1)
    text+='\n'+marker+'\n\nTwo configured PiperX/OpenArm physical pick-and-place showcases use the same frozen E99 neural estimator with online measured-history updates and a model-derived home prior. Both succeed and repeat exactly. Demonstration servos and simplified contact pads; not a new independent transfer benchmark. See `outputs/ease_known_arm_task_demo_v1/report.md` for all attempts, assumptions and prediction diagnostics; `ease_pick_place_demo.mp4` is the30-second showcase.\n'
    p.write_text(text,encoding='utf-8')
    (ROOT/'artifact_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):fingerprint(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'},indent=2))
    print('Saved showcase report, all attempt provenance, technical-report appendix and artifact hashes.')

if __name__=='__main__':run()

import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .retreat_contact_audit import ROOT

def run():
    rows=json.loads((ROOT/'summary.json').read_text());verification=json.loads((ROOT/'verification.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(11,7));lines=['# E106: opposite retreat contact outcomes','',
    'Six unchanged replays cover both E105 changed pairs and all three controls. Every complete serialized result matches E105. The diagnostic observes existing contact calls at2ms and existing geometry-filter decisions at40ms; it adds no physics steps or forward calls. All cases are outcome-selected diagnostics, not a new performance benchmark.','',
    '| Case / seed | Mode | Success | First retreat contact (s) | Min tool height (mm) | Filter fallbacks |','|---|---|---|---:|---:|---:|']
    for r in rows:
        contact=r['first_contact'];time='none' if contact is None else f"{contact['time']-r['retreat_start']['time']:.3f}"
        lines.append(f"| {r['case']} / {r['seed']} | {r['mode']} | {r['success']} | {time} | {1000*r['retreat_min_tool_z']:.2f} | {r['retreat_filter_fallbacks']} |")
        index=0 if r['case']==2 else 1;data=json.loads((ROOT/'trials'/f"{r['case']}_{r['seed']}_{r['mode']}.json").read_text());samples=[s for s in data['contacts'] if s['phase']=='retreat'];t=np.array([s['time'] for s in samples])-r['retreat_start']['time']
        axes[index,0].plot(t,[s['tool'][2]*1000 for s in samples],label=r['mode'])
        tilt=np.degrees(np.arccos(np.clip([s['tool_z'][2] for s in samples],-1,1)));axes[index,1].plot(t,tilt,label=r['mode'])
        if contact is not None:
            for ax in axes[index]:ax.axvline(contact['time']-r['retreat_start']['time'],color='red',alpha=.3,linestyle=':')
    for i,case in enumerate((2,5)):
        for j in range(2):axes[i,j].set_xlabel('Seconds since retreat');axes[i,j].set_title(f'Case {case}');axes[i,j].legend();axes[i,j].axvspan(0,.8,color='gray',alpha=.1)
        axes[i,0].set_ylabel('Tool height (mm)');axes[i,1].set_ylabel('Tool-axis tilt (degrees)')
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(ROOT/f'retreat_motion.{ext}')
    plt.close(fig)
    lines+=['','## Findings','',
    '- Case2/seed1: baseline and time-only control first contact the table1.082s after retreat begins. The right pad contacts; gentle retreat has no forbidden contact. At baseline first contact the preceding filter has already found negative current clearance, no feasible sampled candidate, and selects scale0 (measured bias). Baseline minimum tool height is230.47mm versus245.00mm for gentle.',
    '- Case5/seed2: gentle retreat first contacts the table3.150s into retreat, after the0.8s ramp has ended. The left pad contacts. The immediately preceding decision is marked feasible, with both current and selected clearance at the filter cap of2mm; scale0 selects the measured-bias command. Physical contact appears30ms later. This is direct evidence that this static check does not guarantee contact-free execution.',
    '- The successful case5 baseline reaches227.56mm tool height, below gentle at229.00mm. Tool height alone therefore cannot distinguish safe and unsafe retreat; pad pose and the executed joint trajectory matter. Both failed trajectories have at least one joint near the2.5rad limit at first contact. These observations do not isolate a causal role for joint limits, velocity, bias error, history estimation, or orientation.',
    '- The ramp changes the early trajectory and subsequent history. Because the new collision occurs long after the ramp ends, it should not be described as a contact during the directly attenuated commands. A full causal mechanism requires controlled state-matched tests.',
    '', '## Next experiment','',
    'Keep the original controller and failed candidate unchanged. On training-body states, compare static clearance predictions with short physical replay predictions, logging pad clearance, joint velocity and limit proximity. First establish whether a dynamics-aware predictor detects impending contact without rejecting successful motions. Any new control rule must be chosen on training data and pass retention there before a new prospective cohort. E105 is now diagnostic development data and must not be reused as independent confirmation.',
    '', '## Saved artifacts and limits','',
    f"{verification['contact_samples']} contact samples; six exact full-result replays; all checkpoint/input hashes unchanged. Six existing E105 success/failure videos are copied unchanged and retain original hashes. No new control result, training, or reserved final-test evaluation. Static geometry and all contact forces are privileged simulator diagnostics. Contact arrays are sampled at solver states; they do not certify continuous clearance. Figure: retreat_motion.png/pdf; gallery: videos/index.html."]
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n');shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)

if __name__=='__main__':run()

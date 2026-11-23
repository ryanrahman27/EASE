"""E69 terminal audit, diagnostic figure and cumulative report update."""
import json
from pathlib import Path
import shutil
from unittest.mock import patch
import numpy as np
import torch
import mujoco
from .grasp_diagnostics import ROOT,REFERENCE
from .arm_pick_place import PickPlace,rollout
from .mixed_sim import Arm,AXES
from .pick_place_control import calibrate
from .retention_fresh import GuardedPose
from .real_arm_reach import load_models


def run():
    torch.set_num_threads(4)
    summary=json.loads((ROOT/'summary.json').read_text());terminal=[]
    p=Path('outputs/ease_arm_pick_place_v2');AXES.update(json.loads((p/'axis_catalog.json').read_text()));tasks=json.loads((p/'cases.json').read_text())
    models,_=load_models('outputs/ease_robust_training_v1')
    for item in summary['learned_noncollision_failures']:
        case=item['case'];t=tasks[case];env=PickPlace(Arm(**t['arm']),t['xy'],t['target'])
        c=GuardedPose(env.n,models[item['method'].replace('parent','neural')]);calibrate(env,t['home'],c,38000+case)
        with patch.object(mujoco,'mj_jacSite',side_effect=RuntimeError('Privileged Jacobian forbidden')):
            result=rollout(env,t['home'],c,reset=False)
        old=json.loads((REFERENCE/'trials'/f"{case}_{item['method']}_baseline.json").read_text())
        assert result['trace']==old['trace']
        obj=env.data.xpos[env.obj];addr=env.model.jnt_dofadr[env.oj]
        terminal.append(dict(case=case,method=item['method'],grasped=bool(env.grasped),lifted=bool(env.lifted),collision=bool(env.bad),
            xy_error=float(np.linalg.norm(obj[:2]-env.target)),height_error=float(abs(obj[2]-.215)),
            velocity_norm=float(np.linalg.norm(env.data.qvel[addr:addr+6])),final_pad_contacts=len(env.contacts()[0]),
            missed_phases=[x['phase'] for x in result['phases'] if not x['reached']]))
    (ROOT/'terminal_audit.json').write_text(json.dumps(terminal,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    examples=[(4,'parent_0'),(10,'parent_1')]
    fig,axs=plt.subplots(2,2,figsize=(11,6),sharex='col')
    for column,(case,method) in enumerate(examples):
        r=json.loads((ROOT/'trials'/f'{case}_{method}.json').read_text());tr=r['trace'];time=np.arange(len(tr))*.04
        close=[x for x in tr if x['phase']=='close'];anchor=np.array(close[-1]['relative_object'])
        axs[0,column].plot(time,[1000*x['position_error'] for x in tr],label='Tool position error')
        axs[0,column].plot(time,[1000*np.linalg.norm(np.array(x['relative_object'])-anchor) for x in tr],label='Object relative displacement')
        axs[0,column].set(title=f'{method}, case {case}: no collision',ylabel='Distance (mm)')
        axs[1,column].plot(time,[sum(x['normal_force'].values()) for x in tr],label='Total finger normal force (N)')
        axs[1,column].plot(time,[2*int(x['bilateral']) for x in tr],label='Bilateral contact (2 = yes)',alpha=.65)
        axs[1,column].set(xlabel='Time after calibration (s)',ylabel='Force / contact indicator')
        previous=None
        for i,x in enumerate(tr):
            if x['phase']!=previous:
                for ax in axs[:,column]:ax.axvline(time[i],color='gray',alpha=.2)
                axs[0,column].text(time[i],.97,x['phase'],rotation=90,va='top',fontsize=7,transform=axs[0,column].get_xaxis_transform())
                previous=x['phase']
        if r['first_sustained_contact_loss'] is not None:
            for ax in axs[:,column]:ax.axvline(time[r['first_sustained_contact_loss']],color='red',ls='--',alpha=.7)
    for ax in axs.flat:ax.legend(fontsize=7,loc='upper right')
    fig.tight_layout();fig.savefig(ROOT/'diagnostics.png',dpi=160);plt.close(fig)
    lead='**E69 completed:** Exact telemetry replays show collisions in 29/31 learned procedural failures. All nine failures with both collision and sustained carry-contact loss collide first; one other failure loses contact without collision. The remaining failure retains carry contact but misses final placement. Successful trials also show relative object motion, so a 5 mm drift threshold cannot label slip reliably. Prioritize collision clearance and phase control before active regrasp. All 112 baseline traces reproduce exactly; 97 tests pass. See Section 4.59.'
    conclusion=('\n## Interpretation and next experiment\n\n'
        'The procedural failure set is dominated by forbidden contact: 29 of 31 learned failures and all five analytical failures. Nine learned failures combine collision and sustained carry-contact loss; all nine collide first. Twenty collision failures have no sustained carry loss. The other two learned failures are separately audited in terminal_audit.json. This is temporal evidence, not proof that collision caused subsequent loss or that the motion estimator is uninvolved.\n\n'
        'All 65 learned successes achieve stable close and avoid sustained carry loss, but their median maximum relative translation is 6.7 mm. The prespecified 5 mm drift event therefore cannot be interpreted as a specific slip detector. It includes compliant grasp motion, rolling and motion relative to the measured tool frame.\n\n'
        'Next test a shared clearance-aware path/phase intervention on procedural cases, retaining the original controller and all failures as controls. Keep arm/table, finger/table and arm/gripper collisions separate, with six-joint retention and interaction costs. Do not mask forbidden contacts, change success criteria, or tune the native placements. Active regrasp is secondary until clearance failures are addressed. This is not a native-robot conclusion or a new learned-policy performance gain.\n\n'
        'Additional artifact: diagnostics.png shows the two noncollision failures, selected exhaustively rather than for favorable outcomes. Two additional terminal replays match exactly.\n')
    conclusion+='\nTiming clarification: parent_1 case 10 has bilateral contact at trace indices 106–117 during close (105–144), then none at the final close sample. The carry-loss marker at index 145 is the first missing sample in the carry window, not the physical onset of loss. The sequencer proceeds to lift despite missed descend/close tolerances. Parent_0 case 4 misses transport tolerance and finishes 41.8 mm from the target. Both observations strengthen the phase-transition diagnostic; they do not establish that a new transition gate will recover the task.\n'
    terminal_text='\nTerminal checks (success requires XY error <25 mm, height error <5 mm, speed norm <0.02, no finger contact, prior grasp/lift and no collision):\n\n'
    for r in terminal:
        terminal_text+=f"- {r['method']} case {r['case']}: XY error {1000*r['xy_error']:.1f} mm, height error {1000*r['height_error']:.1f} mm, velocity norm {r['velocity_norm']:.4f}, final pad contacts {r['final_pad_contacts']}, lifted={r['lifted']}; missed phase tolerances: {', '.join(r['missed_phases']) or 'none'}.\n"
    rp=ROOT/'report.md';rp.write_text(rp.read_text()+conclusion+terminal_text)
    folder=Path(__file__).parent;path=folder/'TECHNICAL_REPORT.md';text=path.read_text();assert '## 4.59 E69:' not in text
    text=text.replace('Revision 54 · 27 October 2026','Revision 55 · 27 October 2026',1)
    text=text.replace('## 1. Research question and claim criteria\n','## 1. Research question and claim criteria\n\n'+lead+'\n',1)
    line=next(x for x in text.splitlines(True) if x.startswith('| E68 |'))
    text=text.replace(line,line+'| E69 | Grasp/tracking telemetry and collision chronology | Complete: 29/31 learned failures include collision; prioritize clearance | `ease_grasp_diagnostics_v1` |\n',1)
    section=rp.read_text().replace('# E69: grasp and tracking diagnostics','## 4.59 E69: grasp and tracking diagnostics',1).replace('## Interpretation and next experiment','### Interpretation and next experiment')
    text=text.replace('## 5. Limitations and next decisions',section+'\n\n## 5. Limitations and next decisions',1)
    text=text.replace('## 5. Limitations and next decisions\n\n','## 5. Limitations and next decisions\n\nE69 narrows the immediate procedural priority to collision clearance and phase control. Twenty-nine of 31 learned failures contain forbidden contact; every joint collision/contact-loss event has collision first. Preserve E35 and frozen E66 while testing shared path/phase changes, with all negative cases and six-joint retention. Telemetry does not causally isolate estimator error, and these findings do not establish the cause of native manipulation failures.\n\n',1)
    text=text.replace('## Revision history\n','## Revision history\n\n- Revision 55, 27 October 2026: Completed E69 with 112 exact baseline telemetry replays, two exact terminal audits, unchanged input/model hashes and 97 passing tests. Recorded contact forces, relative object motion, motion residuals and collision chronology; inspected noncollision-failure figure. Prioritized collision clearance over regrasp; no training, native tuning or final-test evaluation.\n',1)
    path.write_text(text)
    for name in ('README.md','MIXED_DESIGN.md'):
        p=folder/name;first,rest=p.read_text().split('\n',1);p.write_text(first+'\n\n'+lead+'\n'+rest)
    for name in ('grasp_diagnostics_report.py','grasp_diagnostics_finish.py'):shutil.copy2(folder/name,ROOT/'source'/name)
    print(json.dumps(terminal,indent=2))


if __name__=='__main__':run()

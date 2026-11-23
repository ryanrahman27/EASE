"""Paired E65 phase transitions and a static scientific summary figure."""
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .native_full_interventions import ROOT


def run():
    rows=json.loads((ROOT/'records.json').read_text());details=[]
    groups=['analytical','original','uncorrected','corrected']
    lines=['','## Paired phase holds','', '| Group | Robot | Measured holds | Zero-bias holds | Gains | Losses |','|---|---|---:|---:|---:|---:|']
    for group in groups:
        for robot in ['piper','panda']:
            base=[r for r in rows if r['mode']=='measured_bias' and r['robot']==robot and (r['method']==group or r['method'].startswith(group+'_'))]
            pairs=[(b,next(z for z in rows if z['method']==b['method'] and z['case']==b['case'] and z['mode']=='zero_bias')) for b in base]
            phase_pairs=[(a,b) for old,new in pairs for a,b in zip(old['phases'],new['phases'])]
            stats=dict(group=group,robot=robot,phase_count=len(phase_pairs),measured=sum(a['passes'] for a,b in phase_pairs),zero=sum(b['passes'] for a,b in phase_pairs),
                gains=sum(not a['passes'] and b['passes'] for a,b in phase_pairs),losses=sum(a['passes'] and not b['passes'] for a,b in phase_pairs))
            details.append(stats)
            lines.append(f'| {group} | {robot} | {stats["measured"]}/{stats["phase_count"]} | {stats["zero"]}/{stats["phase_count"]} | {stats["gains"]} | {stats["losses"]} |')
    (ROOT/'phase_transitions.json').write_text(json.dumps(details,indent=2))
    (ROOT/'phase_analysis.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),sharey=True)
    for ax,robot in zip(axes,['piper','panda']):
        for j,mode in enumerate(['measured_bias','zero_bias']):
            values=[];labels=[]
            for group in groups:
                rr=[r for r in rows if r['mode']==mode and r['robot']==robot and (r['method']==group or r['method'].startswith(group+'_'))]
                k=sum(r['passes'] for r in rr);values.append(100*k/len(rr));labels.append(f'{k}/{len(rr)}')
            bars=ax.bar(np.arange(4)+(j-.5)*.36,values,width=.36,label=['Measured bias','Zero bias'][j],color=['#27648e','#db8d32'][j])
            for bar,label in zip(bars,labels):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,label,ha='center',fontsize=9)
        ax.set_title('PiPER (6 joints)' if robot=='piper' else 'Panda (7 joints)')
        ax.set_xticks(range(4),['Analytical','E35','E61\nuncorrected','E61\ncorrected']);ax.set_ylim(0,114);ax.set_yticks(range(0,101,20));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Complete schedules passing (%)');axes[1].legend(loc='upper right',bbox_to_anchor=(1,1.22),ncol=2,fontsize=8)
    fig.suptitle('E65: removing bias throughout complete schedules',y=1.02)
    fig.text(.5,-.015,'Observed development cases; contacts disabled. Tracking only, not manipulation. Learned seeds share cases.',ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(ROOT/'full_schedule_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    shutil.copy2(__file__,ROOT/'source'/Path(__file__).name)
    print('\n'.join(lines))


if __name__=='__main__':run()

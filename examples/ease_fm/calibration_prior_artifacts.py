"""Archive analytical prior control without crediting it as a learned gain."""
import shutil
from pathlib import Path
from .calibration_prior_control import ROOT
from .history_estimator_control import ROOT as OLD
from .abstaining_task_value_gate import read,save

def run():
    rows=read(ROOT/'records.json');s=read(ROOT/'summary.json');refs=[r for r in read(OLD/'records_references.json') if r['method']=='uniform32'];pairs=[dict(case=r['case'],dof=r['dof'],before=next(v['success'] for v in refs if v['case']==r['case']),after=r['success']) for r in rows];g=sum(r['after'] and not r['before'] for r in pairs);l=sum(r['before'] and not r['after'] for r in pairs);save(ROOT/'paired.json',dict(gains=g,losses=l,changes=[r for r in pairs if r['before']!=r['after']]))
    lines=['# E145: measured calibration prior','','A single fixed analytical control: regularize the local32-transition Jacobian toward the initial measured calibration instead of zero. Same.98 recency,1e-6 ridge, calibration, frozen bias and task controller as E144 uniform32. No learned weights, privileged Jacobian or physics selector.','',f"Success: **{s['all']['success']}/96**, with {s['all']['collisions']} collision episodes. Six joints: {s['6']['success']}/48; seven joints: {s['7']['success']}/48. Uniform32 reference: {sum(r['success'] for r in refs)}/96. Paired rescues/losses: **{g}/{l}**.",'','Although matched next-motion prediction improved, this analytical prior did not establish a general task-success improvement. Preserving stale calibration directions can trade off local adaptation. All96 fitting outcomes remain saved; fresh and old held placements remain closed. This is known-body development, not unseen-embodiment evidence. E120 unchanged.','']
    (ROOT/'report.md').write_text('\n'.join(lines));p=Path(__file__).with_name('TECHNICAL_REPORT.md');text=p.read_text(encoding='utf-8');assert '## E145:' not in text;p.write_text(text+'\n## E145: measured calibration prior control\n\n'+lines[4]+f' All outcomes retained. Report: `{ROOT}/report.md`.\n',encoding='utf-8')
    selected={}
    for dof in (6,7):
        for success in (True,False):
            rr=[r for r in rows if r['dof']==dof and r['success']==success]
            if rr:r=rr[0];selected[r['case'],'calibration_prior']=r
    for gain in (True,False):
        rr=[r for r in pairs if r['after']==gain and r['before']!=r['after']]
        if rr:
            case=rr[0]['case'];selected[case,'calibration_prior']=next(r for r in rows if r['case']==case);ref=next(r for r in refs if r['case']==case);selected[case,'uniform32']=ref;stem=f'{case}_uniform32_frozen_bias'
            for folder,exts in (('trials',('json',)),('states',('npz','xml'))):
                for ext in exts:shutil.copy2(OLD/folder/f'{stem}.{ext}',ROOT/folder/f'{stem}.{ext}')
    save(ROOT/'video_records.json',list(selected.values()))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4));ax.bar(['Uniform32','Calibration prior'],[sum(r['success'] for r in refs),s['all']['success']]);ax.set_ylim(0,96);ax.set_ylabel('Completed fitting tasks /96');ax.set_title('Analytical controls; no learned gain');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    st=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','calibration_prior_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E145').replace('Task-value ranking diagnosis','Calibration prior control').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Outcome-selected fitting successes and failures, including prior-versus-uniform rescues and regressions. Analytical controls; no fresh test.');Path(__file__).with_name('calibration_prior_videos.py').write_text(st);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('Prepared',len(selected),'E145 videos',flush=True)

if __name__=='__main__':run()

"""Preserve rejected E158 and distinguish it from the E159 prospective candidate."""
import shutil
from pathlib import Path
from .conservative_lift_control import ROOT
from .abstaining_task_value_gate import read,save

def run():
    d=read(ROOT/'diagnostic_summary.json');rows=read(ROOT/'records.json');assert len(rows)==12 and not d['ready_for_full'];lines=['# E158: conservative weights with lift-only posture correction','','One fixed.25-power attenuation of E153 history weights, with the unchanged E156 lift-only projection. No retraining or parameter sweep. Four outcome-selected fitting cases13,1,6,10 are diagnostics, not a success-rate benchmark. Four additional conservative standard rollouts supply exact matched pre-lift references.','', '| Case | Uniform lift correction | Conservative learned lift correction | E120 ensemble lift correction |','|---|---:|---:|---:|']
    for case in (13,1,6,10):lines.append('| '+str(case)+' | '+' | '.join('success' if next(r['success'] for r in rows if r['case']==case and r['method']==m) else 'failure' for m in ('uniform32','learned_ensemble','e120_ensemble'))+' |')
    lines+=['','The conservative model retains case6 but still fails case13. The diagnostic gate fails; no full fitting or fresh evaluation of this variant follows. All pre-lift states match their corresponding standard controller exactly. All16 trajectories, including failures, remain saved.','', 'E159 instead selects the unattenuated standard fixed-kernel model that reached77/96 in E154. E159 records an explicit amendment before fresh outcomes to test net improvement; it does not claim these failed retention gates passed. E120 remains unchanged.',''];(ROOT/'report.md').write_text('\n'.join(lines));report=Path(__file__).with_name('TECHNICAL_REPORT.md');text=report.read_text(encoding='utf-8');assert '## E158:' not in text;report.write_text(text+'\n## E158: conservative lift diagnostic\n\nFails legacy case13 despite retaining case6. Twelve outcome-selected diagnostic episodes plus four matched standard controls; exact pre-lift equality. No full/fresh test of this variant. Report: `outputs/ease_conservative_lift_control_v1/report.md`.\n',encoding='utf-8')
    video=list(rows)
    for r in read(ROOT/'standard/records.json'):
        case=r['case'];oldstem=f'{case}_learned_ensemble_frozen_bias';alias='conservative_standard';stem=f'{case}_{alias}_frozen_bias';t=read(ROOT/'standard/trials'/f'{oldstem}.json');t['method']=alias;save(ROOT/'trials'/f'{stem}.json',t)
        for ext in ('npz','xml'):shutil.copy2(ROOT/'standard/states'/f'{oldstem}.{ext}',ROOT/'states'/f'{stem}.{ext}')
        video.append(dict(case=case,method=alias,mode='frozen_bias',success=t['success'],collision=t['collision']))
    save(ROOT/'video_records.json',video)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=['uniform32','learned_ensemble','e120_ensemble'];fig,ax=plt.subplots(figsize=(7,3));ax.bar(methods,[sum(r['success'] for r in rows if r['method']==m) for m in methods]);ax.set_ylim(0,4);ax.set_title('E158 | four outcome-selected diagnostic cases');fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=150);plt.close(fig)
    s=Path(__file__).with_name('task_value_ranking_videos.py').read_text().replace('task_value_ranking_diagnosis import ROOT','conservative_lift_control import ROOT').replace("ROOT/'records.json'","ROOT/'video_records.json'").replace('E128','E158').replace('Task-value ranking diagnosis','Conservative lift diagnostic').replace('Outcome-selected excluded-fitting-family pairs: E120 on versus short correction-off then E120. Diagnosis only; no new policy.','Four outcome-selected fitting diagnostics and matched standard controls. Variant rejected; no fresh test.');Path(__file__).with_name('conservative_lift_control_videos.py').write_text(s);shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('E158 archived;16 success/failure videos prepared',flush=True)

if __name__=='__main__':run()

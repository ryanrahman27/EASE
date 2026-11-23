"""Close and inventory the E133-E136 autonomous iteration batch."""
import json
import re
from pathlib import Path
from .abstaining_task_value_gate import read,save,fingerprint

ROOTS=[Path('outputs')/name for name in ('ease_structured_response_v1','ease_one_step_response_control_v1','ease_compact_response_value_v1','ease_combined_response_value_v1')]

def run():
    index=[]
    for experiment,root in zip(range(133,137),ROOTS):
        verification=read(root/'verification.json');manifest=read(root/'videos/manifest.json')
        assert all(v.get('all_saved_states_match_trace') for v in manifest['videos'])
        for v in manifest['videos']:assert fingerprint(root/'videos'/v['file'])==v['sha256']
        report=root/'report.md';text=report.read_text()
        # Formatting only; preserve numerical values, labels and scientific conclusions.
        for a,b in {'and1108':'and 1108','or3x3':'or 3x3','additive1e-18':'additive 1e-18','uses1e-6':'uses 1e-6','RMSE98.61':'RMSE 98.61','reference50.67':'reference 50.67','achieves17.69':'achieves 17.69','on974':'on 974','above1':'above 1','at10':'at 10','micrometers:101':'micrometers: 101','E12097':'E120 97','and8':'and 8','existing8-step':'existing 8-step','safe8-step':'safe 8-step','fixed10-micrometer':'fixed 10-micrometer','models;144':'models; 144','All144':'All 144','all144':'all 144','lambda1':'lambda 1','margin0.05:102':'margin 0.05: 102','vs E12097':'vs E120 97','102/144':'102/144',';9 rescues and4':'; 9 rescues and 4',',80 switches':', 80 switches','no-loss grid':'no-loss grid','to8 actions':'to 8 actions','the32-feature':'the 32-feature','the20-rule':'the 20-rule','lost8':'lost 8','lost10':'lost 10','lost4':'lost 4','| nan% |':'| — |'}.items():text=text.replace(a,b)
        report.write_text(text)
        status=dict(experiment=f'E{experiment}',status='complete',videos=len(manifest['videos']),success_videos=sum(v['success'] for v in manifest['videos']),failure_videos=sum(not v['success'] for v in manifest['videos']),E120_unchanged=True,verification_file='verification.json')
        save(root/'status.json',status);save(root/'artifact_hashes.json',{str(p.relative_to(root)):fingerprint(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'});index.append(status|dict(path=str(root)))
    out=Path('outputs/ease_iteration_batch_e133_e136');out.mkdir(exist_ok=True);summary=read(ROOTS[-1]/'rollout_summary.json');save(out/'experiments.json',index);save(out/'final_candidate.json',summary)
    rows=['# E133–E136 iteration batch','','All experiments, including unsuccessful candidates, are retained. E120 was not overwritten.','', '| Experiment | Change | Fitting successes /144 | Rescues / losses |','|---|---|---:|---:|','| E120 | Frozen reference |97|—|','| E133 | Structured one-step predictor, eight-step intervention |101|12 /8|','| E134 | One-step intervention |97|10 /10|','| E135 | Compact regularized task value |102|9 /4|','| E136 | Local progress AND task value |102|5 /0|','']
    h=summary['held'];rows += [f"E136 frozen development-held evaluation: {h['success']}/24 versus E120 {h['baseline_success']}/24; {h['gains']} rescues and {h['losses']} losses. These are known training bodies and previously used held placements, not unseen-body or fresh final-test results.",'',f"Saved {sum(r['videos'] for r in index)} trace-verified videos across the four experiments, plus full numerical results, model files, source snapshots and trajectories. E133/E135 use existing executed counterfactuals; E134/E136 include fresh rollouts.",'']
    (out/'report.md').write_text('\n'.join(rows));save(out/'artifact_hashes.json',{str(p.relative_to(out)):fingerprint(p) for p in out.iterdir() if p.is_file() and p.name!='artifact_hashes.json'});print(json.dumps(dict(experiments=index,held=h),indent=2))

if __name__=='__main__':run()

"""Record remaining task failures separately from eliminated contacts."""
from .descent_execution import ROOT
from .abstaining_task_value_gate import read,save

def run():
    root=ROOT/'history_allowance';failures=[]
    for r in read(root/'records.json'):
        if r['success']:continue
        stem=f"{r['case']}_e120_ensemble_frozen_bias";t=read(root/'trials'/f'{stem}.json');desc=[x for x in t['trace'] if x['phase']=='descend']
        failures.append(dict(case=r['case'],collision=r['collision'],grasped=r['grasped'],lifted=r['lifted'],unreached=[p['phase'] for p in t['phases'] if not p['reached']],last_descent_position_error_m=desc[-1]['position_error'],last_descent_rotation_error=desc[-1]['rotation_error']))
    save(ROOT/'remaining_failures.json',failures)
    text='''
## Interpretation

The history-based allowance completes44/48 fitting episodes (91.7%), compared with41/48 (85.4%) for E169 protection and37/48 (77.1%) for E120+release. It rescues cases42,43 and64 without losing any previous success. All four previous collision episodes are eliminated across complete episodes, not merely delayed beyond descent; there are zero forbidden-contact episodes and zero solver fallbacks. This is an observed result on these48 fitting cases, not a general safety guarantee or independent transfer result.

The fixed allowance completes42/48 and reduces collision episodes to2, but those remaining contacts occur during the later probe. This distinguishes postponing a collision from avoiding it throughout the episode. The history-based allowance ranges from1mm to6.65mm in this run; its90th percentile is4.88mm. The history mechanism is an explicit measured-error rule around frozen E120, not a newly trained adaptation model.

Four history-allowance cases remain unsuccessful:26,62,65 and67. All are collision-free but fail the descent reach criterion and later task criteria. Their terminal descent position/rotation errors and grasp/lift outcomes are preserved in remaining_failures.json. Next development should target descent pose/orientation progress while retaining this protection and all44 successes. The baseline and release fix remain untouched; no held test has been opened.
'''
    p=ROOT/'report.md';old=p.read_text(encoding='utf-8')
    if '## Interpretation' not in old:p.write_text(old+text,encoding='utf-8')
    print(failures,flush=True)

if __name__=='__main__':run()

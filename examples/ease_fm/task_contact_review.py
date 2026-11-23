"""Save the completed contact-control research checkpoint and index."""
from pathlib import Path
import json
from .task_contact_control import ROOT as RATE
from .release_clearance_control import ROOT as RELEASE
from .release_clearance_transfer import ROOT as TRANSFER
from .abstaining_task_value_gate import read,save

def run():
    roots=(RATE,RELEASE,TRANSFER)
    for root in roots:assert read(root/'status.json')['status']=='complete'
    rate=read(RATE/'summary.json');release=read(RELEASE/'summary.json');transfer=read(TRANSFER/'summary.json');review=Path('outputs/ease_iteration_review_v1');p=review/'report.md';text=p.read_text(encoding='utf-8')
    text=text.replace('The strongest development task result was E154:', 'Before the task-controller studies below, the strongest development task result was E154:')
    paragraph='''## Fitting task-control follow-up (E160–E162)

The global measured-motion rate limiter was rejected: it reduced task success. The fixed release margin changes only lower/release height from245 to265mm, permitting a short drop. It uses no new estimator training or contact/geometry/physics input. All original success and contact criteria are retained. Every state and trace before lowering matches the corresponding original reference exactly. No held placements were executed.

'''
    paragraph+='| Estimator | Original success | Elevated release | Rescues / regressions | Collision episodes |\n|---|---:|---:|---:|---:|\n'
    for name,v in transfer['methods'].items():
        a,b=v['baseline'],v['candidate'];paired=v['paired'];paragraph+=f"| {name} | {a['all']['success']}/96 | {b['all']['success']}/96 | {paired['gains']} / {paired['losses']} | {a['all']['collisions']} → {b['all']['collisions']} |\n"
    paragraph+='\nThese are fitting controller improvements, not new unseen-embodiment evidence. E120 remains unchanged as the baseline. First-contact reductions must not be confused with all-contact elimination: earlier failed trajectories can still collide later. The next unresolved control problem is early approach/descent/lift contact, especially arm-link/table contact on six-joint bodies. The elevated release is not gentle placement.\n\n'
    at=text.index('| Study |');text=text[:at]+paragraph+text[at:]
    additions=[('E160',RATE,'Fixed motion-rate limiter: rejected; all192 intervention episodes and16 exact baseline replays saved.'),('E161',RELEASE,f"E120 elevated release:72 to{release['candidate']['all']['success']}/96;{release['gains']} rescues/{release['losses']} regressions."),('E162',TRANSFER,'Matched elevated-release comparison with frozen analytical and learned estimators; all pre-lower states exact.')]
    at=text.index('\n\nInterpretation:');table='\n'.join(f'| {e} | {finding} | [report](../{root.name}/report.md) | [videos](../{root.name}/videos/index.html) |' for e,root,finding in additions);text=text[:at]+'\n'+table+text[at:];p.write_text(text,encoding='utf-8')
    studies=read(review/'studies.json');existing={r['experiment'] for r in studies};assert not any(e in existing for e,_,_ in additions)
    studies += [dict(experiment=e,folder=str(root),report_exists=True,videos=read(root/'status.json')['videos']) for e,root,_ in additions];save(review/'studies.json',studies)
    summary=dict(studies=['E160','E161','E162'],new_intervention_episodes=480,exact_baseline_replays=16,success_failure_videos=sum(read(root/'status.json')['videos'] for root in roots),E120_unchanged=True,held_evaluated=False,release_comparison=transfer['methods'],rate_limit_rejected=True);save(review/'task_contact_summary.json',summary)
    state=review/'WORKING_STATE.md';state.write_text('# Latest checkpoint: E160–E162 complete\n\n'+paragraph+'All480 new intervention episodes,16 baseline replays and success/failure videos are complete and audited. No jobs remain active. E159 remains a previously used test and was not touched. See task_contact_summary.json for concise final outcomes and the three linked study reports for all failures.\n\n---\n\n'+state.read_text(encoding='utf-8'),encoding='utf-8');print(summary,flush=True)

if __name__=='__main__':run()

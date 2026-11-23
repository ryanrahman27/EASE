import json
from .single_phase import ROOT

def run():
    out=ROOT/'transfer';rows=json.loads((out/'records.json').read_text());phases=json.loads((out/'protocol.json').read_text())['phases'];table=[];changes=[];gates={}
    assert len(rows)==48*(1+2*len(phases))
    for phase in phases:
        selected=[r for r in rows if r['mode']=='baseline' or r['affected_phase']==phase]
        for group in ('all','6','7','seed0','seed1','seed2','route','no_route'):
            for mode in ('baseline','budget','gentle'):
                rr=[r for r in selected if r['mode']==mode and (group=='all' or group==str(r['dof']) or group==f"seed{r['seed']}" or (group=='route' and r['route_used']) or (group=='no_route' and not r['route_used']))]
                table.append(dict(phase=phase,group=group,mode=mode,n=len(rr),success=sum(r['success'] for r in rr),collisions=sum(r['collision'] for r in rr),steps=sum(r['steps'] for r in rr)))
        gates[phase]={}
        for control in ('baseline','budget'):
            new=[r for r in selected if r['mode']=='gentle'];old=[r for r in selected if r['mode']==control]
            for r in new:
                ref=next(x for x in old if x['case']==r['case'] and x['seed']==r['seed'])
                if r['success']!=ref['success']:changes.append(dict(phase=phase,control=control,case=r['case'],seed=r['seed'],dof=r['dof'],change='gain' if r['success'] else 'loss',collision=r['collision']))
            gates[phase][control]=dict(success_improves=sum(r['success'] for r in new)>sum(r['success'] for r in old),collisions_nonincreasing=sum(r['collision'] for r in new)<=sum(r['collision'] for r in old),retains_all_successes=not any(r['phase']==phase and r['control']==control and r['change']=='loss' for r in changes),per_seed_nondecreasing=all(sum(r['success'] for r in new if r['seed']==s)>=sum(r['success'] for r in old if r['seed']==s) for s in range(3)))
        gates[phase]['passes']=all(all(g.values()) for g in gates[phase].values())
    summary=dict(table=table,changes=changes,gates=gates);(out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['## Conditional development transfer','',f"Training-gate passing phases only: {phases}. All16 novel-axis development tasks x3 seeds are evaluated in baseline and paired budget/gentle modes. There are{len(rows)} new rollouts. Fixed E99 retained checkpoints, privileged route planning and identical original-parent/probe prefix; no-route cases remain exactly unchanged. The baseline reproduces all48 E99 retained results. This is reused development data, not the reserved final test.",'','| Phase | Group | Mode | Success | Collisions | Steps |','|---|---|---|---:|---:|---:|']
    for r in table:lines.append(f"| {r['phase']} | {r['group']} | {r['mode']} | {r['success']}/{r['n']} | {r['collisions']} | {r['steps']} |")
    lines+=['','### Transfer paired changes','','| Phase | Reference | Case | Seed | DOF | Change | Collision |','|---|---|---:|---:|---:|---|---|']
    for r in changes:lines.append(f"| {r['phase']} | {r['control']} | {r['case']} | {r['seed']} | {r['dof']} | {r['change']} | {r['collision']} |")
    lines+=['','### Transfer gates','',str(gates),'','Only full-suite success improvement over both references with no lost success, increased collisions or seed decline can pass. No seed or DOF switching. Transfer video gallery: transfer/videos/index.html.']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(phases),figsize=(8*len(phases),4),squeeze=False)
    for ax,phase in zip(axes[0],phases):
        for j,mode in enumerate(('baseline','budget','gentle')):
            values=[next(r['success'] for r in table if r['phase']==phase and r['group']==g and r['mode']==mode) for g in ('all','6','7')]
            bars=ax.bar([i+(j-1)*.24 for i in range(3)],values,width=.24,label=mode)
            ax.bar_label(bars,padding=3)
        ax.set_xticks(range(3),['All (48)','Six-joint (24)','Seven-joint (24)'])
        ax.set_ylabel('Successful tasks');ax.set_ylim(0,48);ax.set_title('Novel-axis development: '+phase);ax.legend()
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/f'transfer_comparison.{ext}')
    plt.close(fig)
    lines+=['','Figure: transfer_comparison.png / .pdf. Counts are descriptive; placements and model seeds share embodiments.']
    (out/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':run()

"""Lightweight progress ledger for the authorized E138 collection."""
from pathlib import Path
from .expanded_structured_pairs import ROOT,COLLECT
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    counts={str(s):len(read(COLLECT/f'records_seed_{s}.json')) if (COLLECT/f'records_seed_{s}.json').exists() else 0 for s in range(3)}
    summary=dict(status='collection_complete' if sum(counts.values())==1440 else 'collecting',episodes=sum(counts.values()),target_episodes=1440,by_seed=counts,fresh_test_executed=False,E120_unchanged=True)
    save(ROOT/'progress.json',summary)
    report=f'''# E138: expanded fitting counterfactuals

Status: {summary['status']}. Saved {summary['episodes']} of 1,440 planned new fitting episodes. This progress file is not a completed result report.

Add 48 independent randomized fitting placements on the four known training bodies. Every placement receives all three E120 estimator seeds, an E120-on rollout, and nine independently paired phase-entry off pulses. Save every outcome, exact state, action, history snapshot and paired prefix. Do not filter placements by success or reachability.

The separate 48-placement test generated in E137 remains unexecuted and reserved. E120 is unchanged. The fixed comparisons are scalar response plus ridge task value, family-deletion consensus, and a shallow extra-trees task-value model. All use measured structured features, fixed margins and the original family grouping. No model is selected from fresh-test outcomes.

Preceding experiments: E136 improved fitting success from 97/144 to 102/144 without losses, but its frozen old-held result was 15/24 with one rescue and one loss. E137 confidence erased the fitting gains. Neither was promoted. E138 tests whether independent fitting coverage improves this generalization failure.
'''
    if not (ROOT/'summary.json').exists():(ROOT/'report.md').write_text(report)
    print(summary,flush=True)

def close_e137():
    root=Path('outputs/ease_family_consensus_gate_v1');m=read(root/'videos/manifest.json')
    for p,h in read(root/'frozen.json').items():assert fingerprint(p)==h
    for v in m['videos']:assert fingerprint(root/'videos'/v['file'])==v['sha256']
    save(root/'status.json',dict(status='complete',videos=len(m['videos']),success_videos=sum(v['success'] for v in m['videos']),failure_videos=sum(not v['success'] for v in m['videos']),promoted=False,fresh_test_executed=False,E120_unchanged=True))
    save(root/'artifact_hashes.json',{str(p.relative_to(root)):fingerprint(p) for p in root.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})

if __name__=='__main__':
    import sys
    close_e137() if 'close_e137' in sys.argv else run()

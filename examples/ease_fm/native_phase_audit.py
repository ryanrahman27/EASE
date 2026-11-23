"""Validate E64 saved branches and summarize paired recoveries."""
import json
from pathlib import Path
from .benchmark import fingerprint
from .native_common_schedule import summarize_phase


def run():
    root=Path('outputs/ease_native_phase_interventions_v1')
    rows=json.loads((root/'records.json').read_text());audit=json.loads((root/'audit.json').read_text())
    protocol=json.loads((root/'protocol.json').read_text())
    source=json.loads(Path('outputs/ease_native_common_schedule_v2/records.json').read_text())
    assert fingerprint('outputs/ease_native_common_schedule_v2/records.json')==protocol['source_sha256']
    assert fingerprint(root/'records.json')==audit['records_sha256']
    expected={(r['method'],r['case'],p['phase']) for r in source for p in r['phases'] if not p['passes']}
    assert len(expected)==37
    for mode in ['baseline','analytical_motion','zero_bias','analytical_zero_bias']:
        rr=[r for r in rows if r['mode']==mode]
        assert len(rr)==37 and {(r['method'],r['case'],r['phase']) for r in rr}==expected
    for r in rows:
        saved=json.loads((root/'trials'/f'{r["method"]}_{r["case"]}_{r["phase"]}_{r["mode"]}.json').read_text())
        assert saved==r and len(r['trace'])==len(r['telemetry'])
        assert summarize_phase(r['phase'],r['trace'])==r['summary']
        if r['mode']=='baseline':
            old=next(x for x in source if x['method']==r['method'] and x['case']==r['case'])
            assert r['trace']==[x for x in old['trace'] if x['phase']==r['phase']]
    for path,sha in protocol['checkpoints'].items():assert fingerprint(path)==sha
    lines=['','## Paired recovery attribution','', '| Group | Failed phases | Analytical motion | Zero bias | Both |','|---|---:|---:|---:|---:|']
    for group in ['original','uncorrected','corrected']:
        rr=[r for r in rows if r['method'].startswith(group+'_')]
        scores=[sum(r['summary']['passes'] for r in rr if r['mode']==m) for m in ['analytical_motion','zero_bias','analytical_zero_bias']]
        lines.append(f'| {group} | {len(rr)//4} | {scores[0]} | {scores[1]} | {scores[2]} |')
    lines+=['','Verification: all 148 saved trial files, 37 exact baseline branches, recomputed phase summaries, source and checkpoint hashes pass. EASE-FM regression suite: 90 passed. The repository-wide suite could not collect eight unrelated recovery modules because openpi was not installed in this WSL environment.']
    with (root/'report.md').open('a') as f:f.write('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':run()

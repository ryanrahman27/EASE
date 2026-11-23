"""Reuse E126 aggregate calculation and audit, adding E127 support verification."""
from pathlib import Path

def run():
    root = Path(__file__).parent
    source = (root/'abstaining_gate_artifacts.py').read_text()
    target = root/'history_similarity_artifacts.py'
    s = target.read_text()
    assert '\ndef report(' not in s and '\ndef verify(' not in s
    report = source[source.index('\ndef report('):source.index('    selected, grid =')]
    report += '    write_report(summary)\n'
    verify = source[source.index('\ndef verify('):source.index('\ndef finish(')]
    verify = verify.replace("scores = outcome_grid(oof['probabilities'], meta, tasks, read(ROOT / 'protocol.json')['grid'])", 'scores = support_audit()')
    old = "selected = max([s for s in scores if s['losses'] == 0], key=lambda s: (s['gains'], -s['switches'], s['rule']['min_advantage'], -s['rule']['max_spread']))"
    assert old in verify
    verify = verify.replace(old, 'selected = select(scores)')
    verify = verify.replace("assert bool(accepts(prob, rule)) == t['selected']", "support = hs.SUPPORT.query(np.asarray(t['features'], np.float32), rule['quantile'])\n            assert support == t['support']\n            assert (bool(accepts(prob, rule)) and support['supported']) == t['selected']")
    verify = verify.replace('ancestral_families_disjoint=True,', 'history_support_reconstructed=True, unchanged_E126_models=True, ancestral_families_disjoint=True,')
    target.write_text(s+report+verify+"\nif __name__ == '__main__':\n    import sys\n    {'build': build, 'report': report, 'verify': verify, 'finish': finish}[sys.argv[1]]()\n")

if __name__ == '__main__':
    run()

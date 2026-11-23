"""Append audited E126 rollout machinery with E127 history support checks."""
from pathlib import Path

def run():
    root = Path(__file__).parent
    source = (root/'abstaining_task_value_gate.py').read_text()
    target = root/'history_similarity_gate.py'
    s = target.read_text()
    assert '\ndef evaluate(' not in s
    runtime = source[source.index('\ndef evaluate('):source.index("\nif __name__ == '__main__':")]
    runtime = runtime.replace('selected = bool(accepts(prob, rule))', "support = SUPPORT.query(x, rule['quantile'])\n                selected = bool(accepts(prob, rule)) and support['supported']")
    runtime = runtime.replace('probabilities=prob.tolist(), selected=selected', 'probabilities=prob.tolist(), support=support, selected=selected')
    target.write_text(s+runtime+"\nif __name__ == '__main__':\n    import sys\n    {'prepare': prepare, 'train': train, 'evaluate': run}[sys.argv[1]]()\n")

if __name__ == '__main__':
    run()

"""Verify the prospective data adapter preserves frozen control code."""
from pathlib import Path
import ast,shutil
from .unseen_estimator import ROOT,REFERENCE
from .abstaining_task_value_gate import save,fingerprint

def run():
    base=Path(__file__).parent;a=ast.parse((base/'frozen_estimator_comparison.py').read_text());b=ast.parse((base/'unseen_estimator.py').read_text());names=('geometry_ids','distances','distance_gradient','ProtectedArm','targets','rollout','route_rollout');checked=[]
    for name in names:
        aa=[n for n in ast.walk(a) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name];bb=[n for n in ast.walk(b) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name];assert len(aa)==len(bb)==1,name;assert ast.dump(aa[0])==ast.dump(bb[0]),name;checked.append(name)
    source=(base/'history_estimator_control.py').read_text();fn=source[source.index('def evaluate('):source.index('\ndef run(worker):')];line="    source=DATA/'collection' if case<56 else COLLECT;baseline=read(source/'trials'/f'{case}_on_0_frozen_bias.json');assert r['calibration']==baseline['calibration'] and r['initial_estimator']==baseline['initial_estimator']\n";assert line in fn
    actual=(base/'unseen_estimator_core.py').read_text();assert actual[actual.index('def evaluate('):]==fn.replace(line,'')
    save(ROOT/'controller_equivalence.json',dict(exact_AST_matches=checked,estimator_evaluator_exact_except_fitting_archive_assertion=True,controller_sha256=fingerprint(base/'frozen_estimator_comparison.py'),adapter_sha256=fingerprint(base/'unseen_estimator.py'),reference=str(REFERENCE),no_control_rule_or_budget_changes=True));shutil.copy2(__file__,ROOT/'source'/Path(__file__).name);print('Frozen controller and estimator equivalence verified',flush=True)

if __name__=='__main__':run()

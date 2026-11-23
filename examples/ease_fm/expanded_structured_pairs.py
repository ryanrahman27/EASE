"""E138: expand fitting counterfactuals; reserve E137 fresh test unchanged."""
import json
import shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from .abstaining_task_value_gate import read,save,fingerprint,family,DATA,OLD
from . import broad_task_value_gate as broad
from . import matched_correction as paired

ROOT=Path('outputs/ease_expanded_structured_pairs_v1')
COLLECT=ROOT/'collection'
FRESH=Path('outputs/ease_family_consensus_gate_v1/fresh_cases.json')

def setup():
    _,models,hashes=broad.setup();paired.ROOT=COLLECT;broad.previous.ROOT=COLLECT;broad.previous.core.ROOT=COLLECT
    return read(ROOT/'cases.json'),models,hashes

def prepare():
    ROOT.mkdir(exist_ok=False);COLLECT.mkdir()
    for folder in ('trials','states','actions','branches','snapshots','source'):(COLLECT/folder).mkdir()
    for folder in ('models','source'):(ROOT/folder).mkdir()
    tasks=read(DATA/'cases.json');roots=sorted({family(tasks,i) for i,t in enumerate(tasks) if not t['validation']});assert len(roots)==8
    rng=np.random.default_rng(138000);new=[]
    for source in roots:
        for variant in range(6):
            t=json.loads(json.dumps(tasks[source]));xy=rng.uniform([.275,-.1],[.425,.1]);delta=xy-np.asarray(t['xy']);t['xy']=xy.tolist();t['target']=(np.asarray(t['target'])+delta+rng.uniform(-.02,.02,2)).tolist();t.update(validation=False,group='expanded_e138',source_case=source,variant=variant);new.append(t)
    tasks+=new;assert len(tasks)==104
    fresh=read(FRESH);sign=lambda t:(t['arm']['seed'],*t['xy'],*t['target']);assert not {sign(t) for t in tasks}&{sign(t) for t in fresh}
    save(ROOT/'cases.json',tasks);shutil.copy2(FRESH,ROOT/'fresh_cases.json')
    _,models,hashes=setup()
    inputs=[DATA/'cases.json',DATA/'gate_rows.json',DATA/'gate_dataset.npz',DATA/'collection/records.json',FRESH,OLD/'records.json',Path('outputs/ease_compact_response_value_v1/dataset.npz'),Path('outputs/ease_structured_response_v1/dataset.npz')]
    save(ROOT/'protocol.json',dict(experiment='E138',checkpoints=hashes,inputs={str(p):fingerprint(p) for p in inputs},
        design='Add48 fitting placements (six per original ancestral source), seed138000, identical distribution to E125 expansion but independent draws. No reachability/success filtering. Known four training bodies and original homes. All3 motion-estimator seeds. Each on baseline plus nine independently paired8-step off pulses:1440 new full episodes. Keep all outcomes and all exact states/actions; exclude only starts already contaminated by forbidden collision from fitting.',
        training='Join original1108 clean starts with all new clean starts. Same four ancestral-family folds, descendants remain grouped. Compare fixed E136 scalar+ridge1 conjunction; E137 leave-one-training-family-out consensus; and a prespecified moderately nonlinear task-value model (ExtraTrees128 trees, depth5, minimum12 leaf samples, random_state138000), always paired with fitted scalar motion response. Targets remain one-step paired Cartesian response and final-success advantage.32 measured structured features only. Fixed value margin.05 and progress10um; no threshold search. Select positive gains with zero E120 losses on all288 fitting episodes, maximize gains, fewer switches, favor ridge then consensus then trees. Original144 subset retention reported separately.',
        fresh_test='E137 seed137000 fresh48 placements were saved before its outcome analysis and remain unexecuted. Preserve exactly. If a nontrivial no-loss candidate qualifies, freeze all-fitting model(s) and rule, then evaluate all144 fresh placement/estimator episodes paired against E120. No fresh-test based tuning, no unseen-body claim.',
        cache='Within a case/seed, memoize frozen model Jacobian predictions by exact model identity,n,Gram bytes,cross bytes; return live bias separately. Cache cleared for each case. Same optimization used by original E125 collection. No approximate caching.',
        E120_unchanged=True,held_evaluated=False,fresh_evaluated=False))
    for name in ('expanded_structured_pairs.py','matched_correction.py','bounded_history_correction.py','training_estimator_core.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    save(ROOT/'frozen_collection.json',{str(p):fingerprint(p) for p in [ROOT/'protocol.json',ROOT/'cases.json',ROOT/'fresh_cases.json',*(ROOT/'source').glob('*.py')]})
    print('Prepared1440 fitting rollouts; reserved48 fresh placements untouched',flush=True)

def collect(seed):
    tasks,models,hashes=setup();path=COLLECT/f'records_seed_{seed}.json';rows=read(path) if path.exists() else [];done={(r['case'],r['off_phase']) for r in rows};original=broad.previous.core.LocalPose.predict;cache={}
    def cached(c):
        if c.model is None:return original(c)
        key=id(c.model),c.n,c.gram.tobytes(),c.cross.tobytes()
        if key not in cache:cache[key]=original(c)[0].copy()
        return cache[key].copy(),c.bias
    with patch.object(broad.previous.core.LocalPose,'predict',cached):
        for case in range(56,104):
            cache.clear()
            for phase in (None,*paired.PHASES):
                if (case,phase) in done:continue
                r,b=paired.evaluate(case,tasks[case],seed,models[f'reference_{seed}'],models[f'correction_{seed}'],phase)
                rows.append({k:v for k,v in r.items() if k not in ('trace','telemetry','calibration','initial_estimator','final_estimator')}|dict(seed=seed,off_phase=phase));tmp=path.with_suffix('.tmp');save(tmp,rows);tmp.replace(path)
            print('E138 collected',seed,case,len(rows),'of480 episodes',flush=True)
    assert len(rows)==480
    for p,h in {**hashes,**read(ROOT/'frozen_collection.json')}.items():assert fingerprint(p)==h

if __name__=='__main__':
    import sys
    if sys.argv[1]=='collect':collect(int(sys.argv[2]))
    else:prepare()

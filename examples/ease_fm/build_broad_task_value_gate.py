"""Generate the E125 data-expansion experiment from the audited E124 pipeline."""
from pathlib import Path

def build():
    p=Path(__file__).with_name('task_value_gate.py');s=p.read_text()
    s=s.replace('E124: fitting-only expanded paired outcomes and one-pulse learned task value.','E125: broader fitting placements with unchanged learned task-value gate.')
    s=s.replace("ROOT=Path('outputs/ease_task_value_gate_v1')","ROOT=Path('outputs/ease_broad_task_value_gate_v1')")
    s=s.replace("REUSE=Path('outputs/ease_matched_correction_v1')","REUSE=Path('outputs/ease_task_value_gate_v1')")
    s=s.replace('    return tasks,models,hashes','    if (ROOT/\'cases.json\').exists():tasks=json.loads((ROOT/\'cases.json\').read_text())\n    return tasks,models,hashes',1)
    start=s.index('def prepare():');end=s.index('def collect(seed):')
    prepare='''def prepare():
    tasks,models,hashes=setup();ROOT.mkdir(exist_ok=False);COLLECT.mkdir()
    for root in (ROOT,COLLECT):
        for name in ('trials','states','actions','source','branches','snapshots'):(root/name).mkdir()
    rng=np.random.default_rng(125000);sources=[i for i,t in enumerate(tasks) if not t['validation']];assert len(sources)==16
    expanded=[]
    for case in sources:
        for variant in range(2):
            t=json.loads(json.dumps(tasks[case]));xy=rng.uniform([.275,-.10],[.425,.10]);delta=xy-np.array(t['xy']);target_delta=delta+rng.uniform([-.020,-.020],[.020,.020])
            t['xy']=xy.tolist();t['target']=(np.array(t['target'])+target_delta).tolist();t.update(validation=False,group='expanded_train',source_case=case,variant=variant,object_offset=delta.tolist(),target_offset=target_delta.tolist());expanded.append(t)
    tasks+=expanded;assert len(tasks)==56 and sum(not t['validation'] for t in tasks)==48
    keys=[(t['arm']['seed'],*np.round(t['xy'],12),*np.round(t['target'],12)) for t in tasks];assert len(set(keys))==len(keys)
    assert tasks[:24]==json.loads((OLD/'cases.json').read_text());assert all(t['arm']['split']=='train' for t in tasks)
    (ROOT/'cases.json').write_text(json.dumps(tasks,indent=2));pairs=[dict(case=i,seed=seed,e120_gain=False,group=t['group']) for i,t in enumerate(tasks) if not t['validation'] for seed in range(3)]
    assert len(pairs)==144
    reuse_rows=json.loads((REUSE/'collection/records.json').read_text());assert len(reuse_rows)==480 and all(not tasks[r['case']]['validation'] for r in reuse_rows)
    # Reused scientific data are copied, not linked to mutable source files.
    for folder in ('trials','states','actions','branches','snapshots'):
        shutil.copytree(REUSE/'collection'/folder,COLLECT/folder,dirs_exist_ok=True)
    (COLLECT/'reused_records.json').write_text(json.dumps(reuse_rows,indent=2));(COLLECT/'pairs.json').write_text(json.dumps(pairs,indent=2))
    rows=[]
    for r in json.loads((REUSE/'records.json').read_text()):
        source=r['method'];method=source.replace('candidate_','previousgate_');r['method']=method;a=f"{r['case']}_{source}_frozen_bias";b=f"{r['case']}_{method}_frozen_bias"
        trial=json.loads((REUSE/'trials'/f'{a}.json').read_text());trial['method']=method;(ROOT/'trials'/f'{b}.json').write_text(json.dumps(trial))
        for ext in ('npz','xml'):shutil.copy2(REUSE/'states'/f'{a}.{ext}',ROOT/'states'/f'{b}.{ext}')
        rows.append(r)
    assert len(rows)==240;(ROOT/'records.json').write_text(json.dumps(rows,indent=2))
    spec=dict(experiment='E125',checkpoints=hashes,inputs={str(OLD/n):fingerprint(OLD/n) for n in ('records.json','summary.json','cases.json')},reused_inputs={str(REUSE/n):fingerprint(REUSE/n) for n in ('collection/records.json','gate.pt','summary.json','records.json')},placement_seed=125000,
        placements='Two randomized variants per16 original fitting placements, all four existing training bodies. Object XY uniform in[0.275,0.425] x[-0.10,0.10] meters. Destination preserves source task displacement plus independent +/-20mm jitter each axis. Same home, calibration seeds38000+new case. Workspace coverage chosen after inspecting placement geometry, before any new outcome collection. No held labels/outcomes, IK or success filtering. Exact held placements excluded; spatial region no longer intentionally excluded. All32 new placements and outcomes retained. Original24 evaluation cases unchanged.',
        collection='48 fitting placements x3 frozen estimator seeds x10 trajectories =1440 collection episodes. Reuse480 E124 fitting-only episodes, collect960 new episodes. Nine independent phase-entry off-pulses plus E120-on. Exact prefix/snapshot checks for1296 pairs; exclude only prior-collision starts from fitting. No held outcomes in training.',
        model='Same139-64-32-2 tanh gate, seed124000,1500 full-batch AdamW updates, learning rate0.001 weight decay0.001, BCE paired success labels. Refit from scratch on all clean fitting data; normalization fitting-only. No architecture, threshold or training-step change and no held-driven selection.',
        inference='Same one-pulse rule: predicted off-success minus on-success >0.10 at a phase entry, up to8 off actions within that phase, then resume E120 for remainder. At most one intervention. No physics selector, IDs or future outcomes as gate features.',
        evaluation='Retest original24 placements x3 estimator seeds =72 episodes. Compare E120 baseline and prior E124 gate, separate original fitting versus unchanged held placements, audit all previous gains. New placements are training data, not a new held benchmark. E120 remains immutable; no unseen bodies.')
    (ROOT/'protocol.json').write_text(json.dumps(spec,indent=2))
    for n in ('broad_task_value_gate.py','build_broad_task_value_gate.py','matched_correction.py','bounded_history_correction.py','training_estimator_core.py','local_probe.py','local_history.py'):shutil.copy2(Path(__file__).with_name(n),ROOT/'source'/n)
    print('Prepared48 fitting placements; reused480 episodes;960 new episodes planned',flush=True)

'''
    s=s[:start]+prepare+s[end:]
    s=s.replace('            if phase is None:\n                old=', '            if phase is None and case<24:\n                old=')
    s=s.replace('assert len(rows)==160','assert len(rows)==480').replace('assert len(rows)==480\n    (COLLECT/\'records.json\')','assert len(rows)==1440\n    (COLLECT/\'records.json\')')
    s=s.replace('independent_pairs=48','independent_pairs=144')
    s=s.replace("assert len(rows)==168\n    for case,t in enumerate(tasks):","assert len(rows)==240\n    for case,t in enumerate(tasks[:24]):")
    Path(__file__).with_name('broad_task_value_gate.py').write_text(s)

if __name__=='__main__':build()

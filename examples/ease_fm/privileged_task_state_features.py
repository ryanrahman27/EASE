"""E131 helper source embedded by its builder. Explicit simulator-only inputs."""

def extract(meta,tasks):
    import mujoco
    cache={};extra=[];provenance={}
    base=np.load(MEASURED/'features.npz')['x']
    for i,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];key=(case,seed)
        assert not tasks[case]['validation']
        if key not in cache:
            xml=COLLECT/'states'/f'{case}_on_{seed}_frozen_bias.xml';m=mujoco.MjModel.from_xml_path(str(xml.resolve()));d=mujoco.MjData(m)
            site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center');obj=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'object')
            gids={name:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name) for name in ('cube','table','gripper_left_pad','gripper_right_pad')}
            assert min([site,obj,*gids.values()])>=0
            branch_path=COLLECT/'branches'/f'{case}_on_{seed}.json';branch=read(branch_path)
            cache[key]=(m,d,site,obj,gids,branch);provenance[str(xml)]=fingerprint(xml);provenance[str(branch_path)]=fingerprint(branch_path)
        m,d,site,obj,gids,branch=cache[key]
        path=COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz';snap=np.load(path);provenance[str(path)]=fingerprint(path)
        mujoco.mj_setState(m,d,snap['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        tool=d.site_xpos[site];position=d.xpos[obj];rotation=d.xmat[obj].reshape(3,3)
        np.testing.assert_allclose(base[i,139:142],tool,atol=1e-7,rtol=1e-6)
        np.testing.assert_allclose(base[i,30:33],snap['goal']-tool,atol=1e-7,rtol=1e-5)
        contacts=np.zeros(3)
        for c in d.contact[:d.ncon]:
            pair={int(c.geom1),int(c.geom2)}
            if gids['cube'] not in pair:continue
            for j,name in enumerate(('gripper_left_pad','gripper_right_pad','table')):
                if gids[name] in pair:contacts[j]=1
        entry=branch['entries'][phase]
        assert entry['signature']==r['signature']
        # Cumulative grasp flag is captured before the candidate action, not from final outcomes.
        grasped=float(entry['metadata']['grasped']);current_bilateral=float(contacts[0] and contacts[1])
        target=np.r_[tasks[case]['target'],.215]
        value=np.r_[position-tool,rotation.ravel(),target-position,contacts,grasped,current_bilateral].astype(np.float32)
        assert value.shape==(20,) and np.isfinite(value).all()
        extra.append(value)
    return np.stack(extra),provenance


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','paired'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json','fold_audit.json'):shutil.copy2(PRIOR/name,ROOT/name)
    inputs=[DATA/n for n in ('gate_dataset.npz','gate_rows.json','cases.json','verification.json')]
    inputs += [COLLECT/'records.json',PRIOR/'protocol.json',PRIOR/'oof_predictions.npz',PRIOR/'validation_grid.json',PRIOR/'frozen_selection.json',OLD/'records.json',OLD/'summary.json']
    inputs += [MEASURED/n for n in ('features.npz','richer_oof.npz','primary_outcomes.json','validation_grids.json','verification.json')]
    inputs += [p for base in (PRIOR,MEASURED) for p in (base/'models').rglob('*') if p.is_file()]
    save(ROOT/'protocol.json',dict(experiment='E131',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},
        label='PRIVILEGED SIMULATOR-STATE INFORMATION UPPER BOUND. Not a measured-sensor or deployable policy result; extra information does not guarantee improved performance.',
        design='Same two-head BCE,4 ancestral-family fitting folds,3 seeds126100..126102,1500 full-batch AdamW steps, lr.001 weight_decay.001, widths64/32. Input expands measured162 to182. No reweighting/resampling/early stopping or held-driven selection.',
        initialization='Same-seed original139-column weights, biases, hidden and2-output weights copied exactly; all43 extra measured/privileged columns start at zero, matching E130 initial function. Input capacity grows with features.',
        features='Measured162 unchanged. Append20 privileged inputs: object-to-tool position3, world object rotation matrix9, destination-minus-object position3 (task landing height0.215m), instantaneous object contacts with left pad/right pad/table3, pre-action cumulative grasp flag1, instantaneous bilateral finger contact1.',
        timing='All added values use E120-on pre-action integration/history snapshot and same-time branch metadata. Grasp flag is from entry metadata, never final trial success/grasp/lift. No future state, final label, collision/bad flag, physical parameters, or true Jacobian in features.',
        normalization='Measured162 normalization retained exactly per fold from E130. Added20 mean/std fitted only on training families, scale floor1e-6; clipping[-10,10].',
        primary='Same min-advantage0.1/max-spread2 first-clean-phase intervention rule,144 fitting-placement/estimator episodes. Compare original139, measured162 and privileged182. Secondary same E126 zero-loss grid, tuning-validation only.',
        outcomes='Reconstruct success from existing exact paired full-task continuations: one up-to8-step correction-off pulse then E120. Post-collision starts omitted because success irreversibly false. No fresh online-policy rollout or post-collision trajectory equivalence claim.',
        baseline='E120 unchanged. No original held-placement benchmark, unseen bodies, sensing deployment, or promotion from this comparison.',grid=read(PRIOR/'protocol.json')['grid']))
    tasks,meta=read(ROOT/'cases.json'),read(DATA/'gate_rows.json');new,provenance=extract(meta,tasks)
    base=np.load(MEASURED/'features.npz');np.savez_compressed(ROOT/'features.npz',x=np.c_[base['x'],new],y=base['y'])
    save(ROOT/'feature_provenance.json',provenance)
    for name in ('privileged_task_state.py','privileged_task_state_features.py','build_privileged_task_state.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print('Prepared182 inputs:162 measured plus20 PRIVILEGED',flush=True)


def data():
    f=np.load(ROOT/'features.npz');o=np.load(PRIOR/'oof_predictions.npz')
    return f['x'],f['y'],o['row_fold'],read(DATA/'gate_rows.json'),read(ROOT/'cases.json')


def verify_features():
    x,y,folds,meta,tasks=data();new,provenance=extract(meta,tasks)
    np.testing.assert_array_equal(new,x[:,162:]);np.testing.assert_array_equal(x[:,:162],np.load(MEASURED/'features.npz')['x'])
    assert provenance==read(ROOT/'feature_provenance.json')
    for fold in range(4):
        n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz');old=np.load(MEASURED/'models'/f'fold_{fold}'/'normalization.npz')
        np.testing.assert_array_equal(n['mean'][:162],old['mean']);np.testing.assert_array_equal(n['scale'][:162],old['scale'])


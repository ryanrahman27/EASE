"""E130 helper source, embedded into the runnable experiment by its builder."""

def extract(meta,tasks):
    import mujoco
    from .arm_pick_place import rotation_error
    from .broad_task_value_gate import features
    cache={};extra=[];provenance={}
    base=np.load(DATA/'gate_dataset.npz')['x']
    for i,r in enumerate(meta):
        case,seed,phase=r['case'],r['seed'],r['phase'];key=(case,seed)
        assert not tasks[case]['validation']
        if key not in cache:
            path=COLLECT/'states'/f'{case}_on_{seed}_frozen_bias.xml'
            m=mujoco.MjModel.from_xml_path(str(path.resolve()));d=mujoco.MjData(m)
            site=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_SITE,'grasp_center')
            joints=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,f'gripper_{side}_slide') for side in ('left','right')]
            assert all(j>=0 for j in joints)
            actuators=[int(np.flatnonzero(m.actuator_trnid[:,0]==j)[0]) for j in joints]
            cache[key]=(m,d,site,joints,actuators);provenance[str(path)]=fingerprint(path)
        m,d,site,joints,actuators=cache[key]
        path=COLLECT/'snapshots'/f'{case}_on_{seed}_{phase}.npz';snap=np.load(path);provenance[str(path)]=fingerprint(path)
        mujoco.mj_setState(m,d,snap['integration'],mujoco.mjtState.mjSTATE_INTEGRATION);mujoco.mj_forward(m,d)
        n=int(base[i,9:16].sum());position=d.site_xpos[site].copy();rot=d.site_xmat[site].reshape(3,3)
        np.testing.assert_allclose(base[i,16:16+n],d.qpos[:n],rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(base[i,30:33],snap['goal']-position,rtol=1e-5,atol=1e-7)
        # Full base feature equality ensures the reconstruction is at the decision time.
        rebuilt=features(n,snap['gram'],snap['cross'],snap['bias'],d.qpos[:n],position,snap['goal'],phase,bool(base[i,138]),base[i,124:124+n],base[i,131:131+n])
        np.testing.assert_array_equal(rebuilt,base[i])
        dq=np.zeros(7);dx=np.zeros(6)
        if len(snap['history_dq']):dq[:n]=snap['history_dq'][-1]
        if len(snap['history_dx']):dx[:]=snap['history_dx'][-1]
        encoder=np.array([d.qpos[m.jnt_qposadr[j]] for j in joints]);command=d.ctrl[actuators].copy()
        value=np.r_[position,rotation_error(np.eye(3),rot),dq,dx,encoder,command].astype(np.float32)
        assert value.shape==(23,) and np.isfinite(value).all()
        extra.append(value)
    return np.stack(extra),provenance


def prepare():
    ROOT.mkdir(exist_ok=False)
    for folder in ('models','source','trials','states','paired'):(ROOT/folder).mkdir()
    for name in ('cases.json','folds.json','fold_audit.json'):shutil.copy2(PRIOR/name,ROOT/name)
    inputs=[DATA/n for n in ('gate_dataset.npz','gate_rows.json','cases.json','verification.json')]
    inputs += [COLLECT/'records.json',PRIOR/'protocol.json',PRIOR/'oof_predictions.npz',PRIOR/'validation_grid.json',PRIOR/'frozen_selection.json',OLD/'records.json',OLD/'summary.json']
    inputs += [p for p in (PRIOR/'models').rglob('*') if p.is_file()]
    save(ROOT/'protocol.json',dict(experiment='E130',date='2026-11-15',inputs={str(p):fingerprint(p) for p in inputs},
        design='Same E126 two-success-head BCE objective,4 ancestral fitting-family folds,3 seeds126100..126102,1500 full-batch AdamW updates, lr.001 weight_decay.001, no weighting/resampling/model selection. Same64/32 tanh hidden layers and2 output logits. Input grows139 to162.',
        initialization='Copy original139-column first-layer weights, all original biases, hidden and output weights from the same-seed baseline initialization. Added23 input columns start at zero. Thus existing pathways and initial behavior match; only new input weights/capacity are added.',
        features='Append tool position3, tool orientation error3, latest measured joint increment padded7, latest measured Cartesian/rotation increment6, two gripper encoder positions, two preceding gripper actuator commands. Reconstruct from the pre-action E120-on integration/history snapshot. No post-action or future state.',
        observability='Ideal noise-free simulated observations. Tool pose and joint history already available to controller; gripper positions model encoder readings, prior commands are known controller signals. No object pose, contact/grasp/lift/bad flags, object velocity, privileged Jacobian, or physical parameters. Gripper sensing is not a tested hardware sensor pipeline.',
        normalization='Original139 columns retain exact original fold-local normalization; added23 mean/std from that fold training families only, scale floor1e-6; all features clipped to[-10,10].',
        primary='Same predeclared minimum-advantage0.1/max-spread2 first-clean-phase rule on144 fitting-placement/estimator episodes. Secondary unchanged E126 grid and zero-loss selection. No original held-placement or unseen-body evaluation.',
        outcomes='Reconstruct success using existing exact paired continuations: first accepted clean phase gets up-to8 off-pulse steps, then E120 resumes. Post-collision starts omitted because success cannot recover. No fresh online-policy rollout or post-collision trajectory equivalence claim.',
        baseline='E120 unchanged; no promotion on fitting-only results. Same success/failure-video provenance workflow.',grid=read(PRIOR/'protocol.json')['grid']))
    tasks,meta=read(ROOT/'cases.json'),read(DATA/'gate_rows.json')
    new,provenance=extract(meta,tasks)
    base=np.load(DATA/'gate_dataset.npz')
    np.savez_compressed(ROOT/'features.npz',x=np.c_[base['x'],new],y=base['y'])
    save(ROOT/'feature_provenance.json',provenance)
    for name in ('richer_task_state.py','richer_task_state_features.py','build_richer_task_state.py'):shutil.copy2(Path(__file__).with_name(name),ROOT/'source'/name)
    print('Prepared162 inputs from1108 pre-action snapshots',flush=True)


def data():
    f=np.load(ROOT/'features.npz');o=np.load(PRIOR/'oof_predictions.npz')
    return f['x'],f['y'],o['row_fold'],read(DATA/'gate_rows.json'),read(ROOT/'cases.json')


def verify_features():
    x,y,folds,meta,tasks=data();new,provenance=extract(meta,tasks)
    np.testing.assert_array_equal(new,x[:,139:]);np.testing.assert_array_equal(x[:,:139],np.load(DATA/'gate_dataset.npz')['x'])
    assert provenance==read(ROOT/'feature_provenance.json')
    for fold in range(4):
        n=np.load(ROOT/'models'/f'fold_{fold}'/'normalization.npz');old=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz')
        np.testing.assert_array_equal(n['mean'][:139],old['mean']);np.testing.assert_array_equal(n['scale'][:139],old['scale'])


"""Preserve and audit the E166 checker-only dtype correction; no policy change."""
from pathlib import Path
import shutil
import numpy as np
from .descent_reference_control import ROOT
from .abstaining_task_value_gate import read,save,fingerprint

def prepare():
    old=ROOT/'source/descent_reference_control.py';new=Path(__file__).with_name('descent_reference_control.py');a=old.read_text();b=new.read_text();assert a.replace("j=np.array(l['j']);solve=", "j=np.array(l['j'],dtype=np.float32);solve=")==b
    root=ROOT/'audit_initial';root.mkdir(exist_ok=False);shutil.copy2(old,root/'original_source.py');shutil.copy2(ROOT/'frozen.json',root/'original_frozen.json');stem='12_e120_ensemble_frozen_bias'
    for folder,ext in (('trials','json'),('states','npz'),('control','json')):shutil.copy2(ROOT/folder/f'{stem}.{ext}',root/f'{folder}.{ext}')
    # Prove the corrected checker matches the already-executed commands before resuming.
    for l in read(root/'control.json'):
        if not l['active']:continue
        j=np.array(l['j'],dtype=np.float32);solve=lambda e:np.clip(j.T@np.linalg.solve(j@j.T+.015**2*np.eye(6),.35*np.array(e))+l['bias'],-.04,.04)
        np.testing.assert_array_equal(solve(l['error']),l['original_base']);np.testing.assert_array_equal(solve(l['capped_error']),l['new_base'])
    save(ROOT/'audit_precision_correction.json',dict(reason='Audit reconstructed logged float32 E120 Jacobian as float64, changing the J@J.T rounding. Preserve float32 in checker; strict equality retained. Controller computation, parameters, checkpoint, protocol and outcomes unchanged.',old_source_sha256=fingerprint(old),new_source_sha256=fingerprint(new),saved_initial_case=12,existing_commands_pass_corrected_exact_audit=True))
    shutil.copy2(new,old);frozen=read(ROOT/'frozen.json');frozen[str(old)]=fingerprint(old);save(ROOT/'frozen.json',frozen)

def finish():
    stem='12_e120_ensemble_frozen_bias';assert read(ROOT/'audit_initial/trials.json')==read(ROOT/'trials'/f'{stem}.json');np.testing.assert_array_equal(np.load(ROOT/'audit_initial/states.npz')['states'],np.load(ROOT/'states'/f'{stem}.npz')['states']);v=read(ROOT/'audit_precision_correction.json');v['rerun_identical_full_trial_and_states']=True;save(ROOT/'audit_precision_correction.json',v);print('E166 checker-only correction verified; original episode reproduced exactly',flush=True)

if __name__=='__main__':
    import sys
    {'prepare':prepare,'finish':finish}[sys.argv[1]]()

"""Build E131 from the audited measured-state fitting-family workflow."""
from pathlib import Path

def run():
    root=Path(__file__).parent
    s=(root/'richer_task_state.py').read_text().replace('E130','E131').replace('richer_task_state','privileged_task_state').replace('RichGate','PrivilegedGate').replace('richer','privileged')
    s=s.replace("PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')", "PRIOR=Path('outputs/ease_abstaining_task_value_gate_v1')\nMEASURED=Path('outputs/ease_richer_task_state_v1')")
    s=s.replace('nn.Linear(162,64)','nn.Linear(182,64)')
    a=s.index('def extract(');b=s.index('\ndef train():')
    helper=(root/'privileged_task_state_features.py').read_text()
    s=s[:a]+helper[helper.index('def extract('):]+s[b:]
    s=s.replace("n=np.load(PRIOR/'models'/f'fold_{fold}'/'normalization.npz');mean=np.r_[n['mean'],x[train_ids,139:].mean(0)];scale=np.r_[n['scale'],np.maximum(x[train_ids,139:].std(0),1e-6)]", "n=np.load(MEASURED/'models'/f'fold_{fold}'/'normalization.npz');mean=np.r_[n['mean'],x[train_ids,162:].mean(0)];scale=np.r_[n['scale'],np.maximum(x[train_ids,162:].std(0),1e-6)]")
    s=s.replace("mean[:139],n['mean']", "mean[:162],n['mean']").replace("scale[:139],n['scale']", "scale[:162],n['scale']")
    s=s.replace("n['mean'][139:],x[ids,139:].mean(0)","n['mean'][162:],x[ids,162:].mean(0)").replace("n['scale'][139:],np.maximum(x[ids,139:].std(0),1e-6)","n['scale'][162:],np.maximum(x[ids,162:].std(0),1e-6)")
    s=s.replace("for name,adv in (('two_head',two),('privileged',privileged)):", "for name,adv in (('two_head',two),('measured',np.load(MEASURED/'richer_oof.npz')['advantages']),('privileged',privileged)):")
    # Include changes relative to the measured-state control in the video selection.
    needle="    for name,s in selected.items():"
    extra="    for key,a in maps['measured'].items():\n        b=maps['privileged'][key]\n        if a['success']!=b['success']:\n            for r in (a,b):add(r['case'],r['seed'],r['phase'],'measured versus privileged')\n            add(*key,None,'E120 reference')\n"
    assert needle in s;s=s.replace(needle,extra+needle)
    s=s.replace("'trained privileged fold'","'trained privileged fold'").replace('privilegedory','directory')
    s=s.replace('Richer measured task-state comparison','PRIVILEGED simulator task-state comparison').replace('Measured-history control | No physics selector or geometry filter','PRIVILEGED state gate study | Counterfactual replay')
    s=s.replace("assert primary['two_head']==old", "assert primary['two_head']==old\n    assert primary['measured']==read(MEASURED/'primary_outcomes.json')['richer']\n    assert grids['measured']==read(MEASURED/'validation_grids.json')['richer']")
    needle="    Path(__file__).with_name('privileged_task_state_videos.py').write_text(source)"
    s=s.replace(needle,"    source=source.replace('Measured-history control | No physics selector or geometry filter','PRIVILEGED simulator-state study | Counterfactual replay')\n"+needle)
    s=s.replace('measured_features_reconstructed=True','privileged_features_reconstructed=True,measured_features_unchanged=True')
    (root/'privileged_task_state.py').write_text(s)

if __name__=='__main__':run()

"""Finalize E138-E140 without rewriting immutable collection files."""
import shutil
from pathlib import Path
from .abstaining_task_value_gate import read,save,fingerprint

def run():
    studies=[('E138','ease_expanded_structured_pairs_v1',None),('E139','ease_rich_structured_value_v1','rich_and_classifier_audit.py'),('E140','ease_paired_effect_classifier_v1','rich_and_classifier_audit.py')]
    inventory=[]
    for label,name,audit in studies:
        root=Path('outputs')/name;verification=read(root/'verification.json');manifest=read(root/'videos/manifest.json');hashes=read(root/'artifact_hashes.json')
        for v in manifest['videos']:assert fingerprint(root/'videos'/v['file'])==v['sha256']
        changed=[]
        if audit:
            dst=root/'source'/audit;shutil.copy2(Path(__file__).with_name(audit),dst);changed.append(dst)
        status=dict(experiment=label,status='complete',selected='abstain',promoted=False,fresh_test_executed=False,E120_unchanged=True,videos=len(manifest['videos']),success_videos=sum(v['success'] for v in manifest['videos']),failure_videos=sum(not v['success'] for v in manifest['videos']))
        save(root/'status.json',status);changed.append(root/'status.json')
        for p in changed:hashes[str(p.relative_to(root))]=fingerprint(p)
        save(root/'artifact_hashes.json',hashes);inventory.append(status)
    save(Path('outputs/ease_iteration_batch_e133_e136')/'later_gate_studies.json',inventory);print(inventory,flush=True)

if __name__=='__main__':run()

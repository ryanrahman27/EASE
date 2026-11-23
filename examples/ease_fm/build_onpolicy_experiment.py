"""Construct E101 from the audited E99 collection/training harness."""
from pathlib import Path

p=Path(__file__).with_name('task_retention_training.py')
s=p.read_text()
s=s.replace('E99: measured training-task contexts, matched data-only/action-retention fits.', 'E101: fixed student-collected histories versus matched parent replay.')
s=s.replace("from . import probe_distillation as prior", "from . import probe_distillation as prior\nfrom . import task_retention_training as e99")
s=s.replace("ROOT=Path('outputs/ease_task_retention_training_v1')", "ROOT=Path('outputs/ease_onpolicy_history_v1')")
s=s.replace("CONDITIONS=('data_only','retained')", "CONDITIONS=('replay','onpolicy')")
start=s.index("    spec=dict(experiment='E99'")
end=s.index("    (ROOT/'protocol.json')",start)
s=s[:start]+'''    drivers,driver_hashes=e99.load_students();hashes.update(driver_hashes)
    inputs=[prior.ROOT/'contexts.npz',prior.ROOT/'provenance.json',e99.ROOT/'task_contexts.npz',e99.ROOT/'provenance.json',TASKS/'cases.json',TASKS/'axis_catalog.json']
    spec=dict(experiment='E101',inputs={str(p):fingerprint(p) for p in inputs},parents=hashes,training_bodies=sorted(training_bodies),test_evaluated=False,
        data='36 full trajectories on the same12 original training placements, original E66 parent prefix and32-step probe. Fixed E99 retained student drives post-probe continuation. All failures/contacts retained. Train only post-probe contexts; third placement per body held out across every driver. This is one collection round from the starting student, not iterative on-policy aggregation. No evaluation-arm data.',
        training='Both conditions start E99 retained checkpoint for each seed. Fixed2000 updates, AdamW lr2e-5, clip1, batch128:64 synthetic+32 local+16 parent-task+16 extra-task. Replay extra-task comes from E99 parent contexts; onpolicy extra-task comes from newly collected student contexts. Same synthetic/local/parent rows and identical extra-index uniform draws per pair. Analytical ridge fit and synthetic E66 parent anchor (six2/seven.5) plus .01 action retention against E66 on all32 task rows in BOTH conditions. Final checkpoints only. Extra source distribution is the only training intervention.',
        evaluation='All16 development tasks x3 seeds x original parent/replay/onpolicy=144 full tasks, unchanged original-parent/planner/probe prefix. Also compare saved E99 retained starting checkpoints. Strict gate: improve pooled success, nonincreasing collisions, retain every original parent AND starting-student success, no seed success decline against either reference. Report all outcomes, no seed/DOF selection.',
        videos='Cases2 seed0,10 seed2,15 seed2 allthree modes; append first regression against either reference for each condition with corresponding reference video. Include success and failure. Full replay/hash/frame verification.',
        limits='Training tasks cover four original training bodies, local data eight. Privileged geometry/route planning at evaluation. Reused development tasks, no final test, no broad policy promotion.')
''' + s[end:]
s=s.replace("return super().predict()", """saved=self.model
                    if self.local and env.phase!='probe':self.model=drivers[('retained',seed)]
                    try:return super().predict()
                    finally:self.model=saved""",1)
# Collection hashes are verified immediately, not only after fitting.
s=s.replace("    arrays={key:", "    for p,h in {**spec['inputs'],**hashes}.items():assert fingerprint(p)==h\n    arrays={key:",1)
s=s.replace("[('local',prior.ROOT/'contexts.npz'),('task',ROOT/'task_contexts.npz')]", "[('local',prior.ROOT/'contexts.npz'),('task',e99.ROOT/'task_contexts.npz'),('onpolicy',ROOT/'task_contexts.npz')]")
s=s.replace("m.load_state_dict(teacher.state_dict());models[condition]=m", "m.load_state_dict(torch.load(e99.ROOT/'retained'/f'seed_{seed}/policy.pt',map_location='cuda',weights_only=True));models[condition]=m")
s=s.replace("if name=='task':", "if name in ('task','onpolicy'):")
s=s.replace("out['task_action_drift']=", "out[name+'_action_drift']=")
s=s.replace("initial=metrics(teacher);gen=", "initial=metrics(models['replay']);gen=")
start=s.index("            ids={name:")
end=s.index("                assert torch.isfinite(loss)",start)
s=s[:start]+'''            ids={name:v['train_idx'][torch.randint(len(v['train_idx']),(32 if name=='local' else 16,),device='cuda',generator=gen)] for name,v in values.items() if name!='onpolicy'}
            uniform=torch.rand(16,device='cuda',generator=gen);sg,sc,sm,st,_=batch(64,'cuda',gen)
            for x in (sg,sc,sm,st,ids['local'],ids['task'],uniform):digest.update(x.detach().cpu().numpy().tobytes())
            for condition,m in models.items():
                extra=values['task' if condition=='replay' else 'onpolicy'];ei=extra['train_idx'][(uniform*len(extra['train_idx'])).long()]
                chunks=[(values['local'],ids['local']),(values['task'],ids['task']),(extra,ei)]
                g=torch.cat([sg,*[v['g'][ix] for v,ix in chunks]]);c=torch.cat([sc,*[v['c'][ix] for v,ix in chunks]]);mask=torch.cat([sm,*[v['mask'][ix] for v,ix in chunks]]);wanted=torch.cat([st,*[v['target'][ix] for v,ix in chunks]])
                error=torch.cat([v['error'][ix] for v,ix in chunks[1:]]);bias=torch.cat([v['bias'][ix] for v,ix in chunks[1:]]);tm=mask[96:]
                with torch.no_grad():old=predict(teacher,g,c,mask);old_action=action_tensor(old[96:],error,bias,tm)
                pred=predict(m,g,c,mask);fit=((pred-wanted).square()*mask[:,:,None]).sum()/(6*mask.sum());weight=torch.where(sm.sum(-1)==6,2.,.5)[:,None,None];anchor=((pred[:64]-old[:64]).square()*sm[:,:,None]*weight).sum()/(6*sm.sum())
                retain=((action_tensor(pred[96:],error,bias,tm)-old_action).square()*tm).sum()/(tm.sum()*.04**2);loss=fit+anchor+.01*retain
''' + s[end:]
s=s.replace("'data_only'", "'replay'")
s=s.replace("parent_sha256=fingerprint(prior.PARENTS", "starting_sha256=fingerprint(e99.ROOT/'retained'/f'seed_{seed}/policy.pt'),parent_sha256=fingerprint(prior.PARENTS")
Path(__file__).with_name('onpolicy_history.py').write_text(s)

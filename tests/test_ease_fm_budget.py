import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from examples.ease_fm.full_control import fit
from examples.ease_fm.budget_experiment import gate


class BudgetTests(unittest.TestCase):
    def test_checkpoint_serialization_preserves_training_trajectory(self):
        torch.set_num_threads(2)
        rng=torch.Generator().manual_seed(7)

        class Dataset:
            tensors=(torch.randn(4,6,2,generator=rng),torch.randn(4,6,generator=rng),
                     torch.ones(4,6,dtype=torch.bool),torch.randn(4,16,6,generator=rng),torch.ones(4,16,6))
            contexts=[np.random.default_rng(7).normal(size=(4,6,11)).astype(np.float32)]

            def histories(self,indices,rng):
                h=rng.normal(size=(len(indices),4,6,11)).astype(np.float32)
                return torch.from_numpy(h),torch.full((len(indices),),4,dtype=torch.long)

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            plain=fit(Dataset(),root/'plain','direct',3,'cpu','fixture')

            def checkpoint(step,model,opt,losses,states):
                torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':states},root/f'{step}.pt')

            saved=fit(Dataset(),root/'saved','direct',3,'cpu','fixture',checkpoint_steps=[1,3],on_checkpoint=checkpoint)
            for name,value in plain.state_dict().items():
                self.assertTrue(torch.equal(value,saved.state_dict()[name]),name)
            state=torch.load(root/'3.pt',weights_only=False)
            self.assertTrue(all(v['step']==3 for v in state['optimizer']['state'].values()))
            self.assertEqual(set(state['rng']),{'batch','noise','context','torch','cuda'})

    def test_gate_rejects_weak_body_despite_high_overall_success(self):
        records=[{'body_seed':1,'success':True} for _ in range(95)]
        records += [{'body_seed':2,'success':False} for _ in range(5)]
        self.assertFalse(gate(records)['passed'])


if __name__=='__main__': unittest.main()

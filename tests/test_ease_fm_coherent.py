import unittest
from types import SimpleNamespace
import numpy as np
import torch
from examples.ease_fm.coherent_pose_data import pack
from examples.ease_fm.coherent_pose_training import context_batch,mixed_batch


class CoherentContextTest(unittest.TestCase):
    def test_pack_preserves_state_and_excludes_padding(self):
        c=SimpleNamespace(n=6,gram=np.eye(6)*2,cross=np.ones((6,6))*3,bias=np.arange(6)*.001)
        obs={'q':np.arange(6)*.1};truth=np.arange(36).reshape(6,6)/10
        row=pack(c,obs,np.arange(6)*.002,truth)
        np.testing.assert_allclose(row['j'][:6],truth.T)
        self.assertFalse(row['mask'][6]);self.assertTrue(np.all(row['j'][6]==0))
        self.assertTrue(np.all(row['c'][6]==0));self.assertEqual(row['q'][6],0)
        c.gram[:]=9;self.assertEqual(row['g'][0,0],2)

    def test_mixture_uses_only_selected_partition(self):
        values={'g':torch.eye(7).repeat(4,1,1),'c':torch.zeros(4,7,6),'mask':torch.ones(4,7,dtype=torch.bool),
            'j':torch.ones(4,7,6),'error':torch.zeros(4,6),'bias':torch.zeros(4,7),'q':torch.zeros(4,7)}
        values['j'][2:]=99
        def make():
            return mixed_batch(values,torch.tensor([0,1]),torch.Generator().manual_seed(1),torch.Generator().manual_seed(2),torch.Generator().manual_seed(3))
        (g,c,m,t,j),ctx=make();second,sc=make()
        self.assertEqual(len(g),128);self.assertTrue(torch.all(j[64:]==1))
        self.assertTrue(torch.equal(t[64:],j[64:]))
        for x,y in zip((g,c,m,t,j),second):self.assertTrue(torch.equal(x,y))
        for x,y in zip(ctx,sc):self.assertTrue(torch.equal(x,y))


if __name__=='__main__':unittest.main()

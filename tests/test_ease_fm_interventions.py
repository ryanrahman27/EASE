import unittest
import numpy as np
from examples.ease_fm.history_audit import intervene


class InterventionTests(unittest.TestCase):
    def test_feature_boundaries_and_input_preservation(self):
        x=np.random.default_rng(9).normal(size=(2,32,7,11)).astype(np.float32)
        mask=np.array([[True]*6+[False],[True]*7]); x*=mask[:,None,:,None]
        saved=x.copy()
        shuffled=intervene(x,'shuffle_actions',mask)
        np.testing.assert_array_equal(shuffled[:,:,:,0:2],x[:,:,:,0:2])
        np.testing.assert_array_equal(shuffled[:,:,:,3:],x[:,:,:,3:])
        np.testing.assert_array_equal(np.sort(shuffled[:,:,:,2],axis=1),np.sort(x[:,:,:,2],axis=1))
        static=intervene(x,'repeat_initial_static',mask)
        for col in [0,5,6,7]:
            np.testing.assert_array_equal(static[:,:,:,col],np.repeat(x[:,:1,:,col],32,axis=1))
        self.assertFalse(static[:,:,:,[1,2,3,4,8,9,10]].any())
        np.testing.assert_array_equal(x,saved)
        self.assertFalse(static[0,:,6].any())


if __name__=='__main__': unittest.main()

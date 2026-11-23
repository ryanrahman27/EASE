import unittest
from dataclasses import replace
import numpy as np
from examples.ease_fm.diverse_data import TRAIN,NOVEL,TEST
from examples.ease_fm.mixed_sim import AXES, MixedReach, sample_arm
from examples.ease_fm.diverse_oracle import oracle_tokens


class DiverseAxesTests(unittest.TestCase):
    def test_oracle_tokens_exclude_identity_and_pad_absent_joint(self):
        arm=sample_arm(19,'six_c','train')
        reference=oracle_tokens(arm,padded=True)
        np.testing.assert_array_equal(reference,oracle_tokens(replace(arm,seed=999,split='validation'),padded=True))
        self.assertEqual(reference.shape,(32,7,11))
        self.assertFalse(reference[:,6].any())
        np.testing.assert_allclose(reference[0,:6,3],arm.lengths)
        np.testing.assert_array_equal(reference[0],reference[-1])

    def test_distinct_axis_families_and_shared_reset(self):
        groups=[set(tuple(AXES[k]) for k in names) for names in [TRAIN,NOVEL,TEST]]
        for i in range(3):
            for j in range(i): self.assertTrue(groups[i].isdisjoint(groups[j]))
        self.assertEqual(len(set(tuple(AXES[k]) for k in TRAIN+NOVEL+TEST)),12)
        for count in [6,7]:
            home=tuple([0,.5,-.9,0,.6,0] if count==6 else [0,.5,0,-.9,0,.6,0])
            states=[]
            for family in TRAIN+NOVEL+TEST:
                if len(AXES[family])!=count: continue
                arm=replace(sample_arm(19,family,'train'),home=home)
                env=MixedReach(arm,synchronize=True)
                states.append(env.reset(123,[0,0,.9])['joints'])
                self.assertEqual(env.model.nu,count)
            for state in states[1:]: np.testing.assert_array_equal(state,states[0])


if __name__=='__main__': unittest.main()

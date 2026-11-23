import unittest
import torch
from examples.ease_fm.pose_policy import initial_pose, PosePolicy


class PoseTests(unittest.TestCase):
    def test_ignores_dynamics_and_subsequent_poses(self):
        torch.set_num_threads(2)
        a=torch.randn(2,32,7,11); b=torch.randn_like(a)
        b[:,0,:,0]=a[:,0,:,0]; b[:,0,:,5:8]=a[:,0,:,5:8]
        torch.testing.assert_close(initial_pose(a),initial_pose(b),rtol=0,atol=0)
        policy=PosePolicy(history=True).eval(); mask=torch.ones(2,7,dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(policy.encode(a,torch.tensor([32,0]),mask),policy.encode(b,torch.tensor([32,0]),mask),rtol=0,atol=0)


if __name__=='__main__': unittest.main()

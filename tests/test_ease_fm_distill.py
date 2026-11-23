import unittest
import numpy as np
import torch
from examples.ease_fm.joint_policy import JointFlowPolicy
from examples.ease_fm.history_distill import make_student, frozen_equal, encoder_key


class DistillTests(unittest.TestCase):
    def test_student_seed_changes_encoder_but_not_controller(self):
        teacher=JointFlowPolicy(history=True).eval()
        contexts=[np.random.default_rng(1).normal(size=(32,6,11)).astype(np.float32)]
        a=make_student(teacher,contexts,'cpu',seed=1)
        b=make_student(teacher,contexts,'cpu',seed=1)
        c=make_student(teacher,contexts,'cpu',seed=2)
        self.assertTrue(all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items()))
        self.assertFalse(torch.equal(a.history_input.weight,c.history_input.weight))
        self.assertTrue(frozen_equal(a,teacher) and frozen_equal(c,teacher))

    def test_only_history_changes_during_distillation(self):
        torch.set_num_threads(2)
        teacher=JointFlowPolicy(history=True).eval().requires_grad_(False)
        contexts=[np.random.default_rng(1).normal(size=(32,6,11)).astype(np.float32)]
        student=make_student(teacher,contexts,'cpu')
        self.assertTrue(frozen_equal(student,teacher))
        self.assertTrue(all(p.requires_grad==encoder_key(n) for n,p in student.named_parameters()))
        before=student.history_output.weight.detach().clone()
        opt=torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],lr=.001)
        h=torch.tensor(np.stack(contexts)); mask=torch.ones(1,6,dtype=torch.bool)
        loss=student.encode(h,torch.tensor([32]),mask).square().mean()
        loss.backward(); opt.step()
        self.assertFalse(torch.equal(before,student.history_output.weight))
        self.assertTrue(frozen_equal(student,teacher))
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))


if __name__=='__main__': unittest.main()

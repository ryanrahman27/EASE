import unittest
import torch
from examples.ease_fm.joint_policy import JointFlowPolicy


class ReactiveBoundaryTests(unittest.TestCase):
    def test_empty_context_is_independent_of_history_with_trainable_constant(self):
        torch.manual_seed(6)
        model = JointFlowPolicy(history=True)
        mask = torch.tensor([[True]*6 + [False], [True]*7])
        lengths = torch.zeros(2, dtype=torch.long)
        left = model.encode(torch.randn(2, 32, 7, 11), lengths, mask)
        right = model.encode(torch.randn(2, 32, 7, 11)*10, lengths, mask)
        torch.testing.assert_close(left, right, rtol=0, atol=0)
        left.sum().backward()
        self.assertGreater(model.empty.grad.abs().sum().item(), 0)
        for name, parameter in model.named_parameters():
            if name.startswith(('history_input', 'history_spatial')) and parameter.grad is not None:
                self.assertEqual(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__': unittest.main()

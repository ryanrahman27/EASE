import unittest
import torch
from examples.ease_fm.joint_policy import JointFlowPolicy


class ResponseTests(unittest.TestCase):
    def test_padding_and_history_gradient(self):
        torch.set_num_threads(2)
        torch.manual_seed(4)
        model = JointFlowPolicy(history=True, auxiliary=True)
        history = torch.randn(2, 8, 7, 11)
        mask = torch.tensor([[True]*6+[False]]*2)
        lengths = torch.tensor([5, 8])
        q = torch.randn(2, 7, 2)
        action = torch.randn(2, 7)
        z = model.encode(history, lengths, mask)
        prediction = model.response(q, action, mask, z)
        changed_q, changed_action = q.clone(), action.clone()
        changed_q[:, -1] = 999
        changed_action[:, -1] = 999
        torch.testing.assert_close(prediction, model.response(changed_q, changed_action, mask, z))
        short_z = model.encode(history[:, :, :6], lengths, mask[:, :6])
        torch.testing.assert_close(prediction, model.response(q[:, :6], action[:, :6], mask[:, :6], short_z), atol=2e-6, rtol=2e-5)
        prediction.square().mean().backward()
        self.assertGreater(model.history_input.weight.grad.abs().sum().item(), 0)

    def test_capacity_matched_initialization(self):
        torch.manual_seed(0)
        a = JointFlowPolicy(history=True, auxiliary=True)
        torch.manual_seed(0)
        b = JointFlowPolicy(history=True, auxiliary=True)
        for key, value in a.state_dict().items():
            torch.testing.assert_close(value, b.state_dict()[key])


if __name__ == '__main__':
    unittest.main()

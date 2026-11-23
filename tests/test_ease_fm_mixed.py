import unittest
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from examples.ease_fm.joint_policy import JointFlowPolicy
from examples.ease_fm.benchmark import fingerprint
from examples.ease_fm.mixed_data import MixedData
from examples.ease_fm.history_control import estimate_jacobian
from examples.ease_fm.mixed_sim import MixedReach, sample_arm, token


class MixedArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_real_six_and_seven_joint_models(self):
        for family, count in [("six", 6), ("seven", 7), ("six_novel", 6)]:
            env = MixedReach(sample_arm(42, family, "train"))
            before = env.reset(10, [.1, .05, .95])
            self.assertEqual(env.model.nv, count)
            self.assertEqual(env.model.nu, count)
            after, _, info = env.step(env.expert())
            self.assertEqual(before["joints"].shape, (count, 2))
            self.assertEqual(token(before, info["executed_action"], after).shape, (count, 11))
            with self.assertRaises(ValueError):
                env.step(np.zeros(count+1))

    def test_padding_cannot_change_valid_actions(self):
        torch.manual_seed(3)
        model = JointFlowPolicy().eval()
        action, joints, task = torch.randn(1, 16, 7), torch.randn(1, 7, 2), torch.randn(1, 6)
        mask = torch.tensor([[True]*6+[False]])
        first = model(action, torch.tensor([.5]), joints, task, mask)
        action[:, :, -1], joints[:, -1] = 10000, -10000
        changed = model(action, torch.tensor([.5]), joints, task, mask)
        torch.testing.assert_close(first, changed)
        trimmed = model(action[:, :, :6], torch.tensor([.5]), joints[:, :6], task, mask[:, :6])
        torch.testing.assert_close(first[:, :, :6], trimmed, atol=1e-5, rtol=1e-5)
        self.assertEqual(first[:, :, -1].count_nonzero(), 0)

    def test_history_causal_and_padded_joint_invariance(self):
        model = JointFlowPolicy(history=True).eval()
        history = torch.randn(2, 32, 7, 11)
        mask = torch.tensor([[True]*6+[False], [True]*7])
        lengths = torch.tensor([0, 5])
        reference = model.encode(history, lengths, mask)
        history[0], history[1, 5:] = 1000, -1000
        torch.testing.assert_close(reference, model.encode(history, lengths, mask))
        history[0] = torch.randn_like(history[0])
        lengths[0] = 32
        reference = model.encode(history, lengths, mask)
        history[0, :, -1] = 10000
        torch.testing.assert_close(reference, model.encode(history, lengths, mask))

    def test_dynamic_joint_count_and_history_gradients(self):
        model = JointFlowPolicy(history=True)
        for count in [6, 7, 12]:
            mask = torch.ones(1, count, dtype=torch.bool)
            history = torch.randn(1, 8, count, 11)
            latent = model.encode(history, torch.tensor([8]), mask)
            output = model(torch.randn(1, 16, count), torch.tensor([.4]),
                           torch.randn(1, count, 2), torch.randn(1, 6), mask, latent)
            output.square().mean().backward()
            self.assertTrue(model.history_input.weight.grad.isfinite().all())
            self.assertGreater(model.history_input.weight.grad.abs().sum(), 0)
            self.assertEqual(output.shape, (1, 16, count))
            model.zero_grad()

    def test_mixed_loader_masks_and_training_only_contexts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episodes, calibrations, arms = [], [], []
            for body, count in [(1, 6), (2, 7), (999, 6)]:
                split = "validation" if body == 999 else "train"
                arms.append({"seed": body, "lengths": [1]*count, "split": split})
                for eid in range(2):
                    filename = f"{body}_{eid}.npz"
                    transition = np.full((3, count, 11), body, np.float32)
                    transition[:, :, 8:11] = body + .25
                    np.savez(root/filename, joints=np.full((3, count, 2), body, np.float32),
                        task=np.zeros((3, 6), np.float32), action=np.ones((3, count), np.float32)*.01,
                        history=transition)
                    episodes.append({"file": filename, "body_seed": body, "split": split,
                                     "success": True, "collision": False})
                if split == "train":
                    filename = f"context_{body}.npy"
                    np.save(root/filename, np.full((32, count, 11), body, np.float32))
                    calibrations.append({"file": filename, "body_seed": body, "split": "train"})
            manifest = {"schema": "ease_mixed_v1", "arms": arms, "episodes": episodes,
                        "calibrations": calibrations,
                        "asset_hashes": {p.name: fingerprint(p) for p in root.iterdir()}}
            (root/"manifest.json").write_text(json.dumps(manifest))
            (root/"manifest.sha256").write_text(fingerprint(root/"manifest.json"))
            dataset = MixedData(root)
            joints, _, joint_mask, _, valid = dataset.tensors
            self.assertEqual(len(joints), 12)
            self.assertEqual(joints.max(), 2)
            self.assertEqual(dataset.responses.shape, (12, 3))
            torch.testing.assert_close(dataset.responses, joints[:, 0, 0, None].expand(-1, 3)+.25)
            self.assertEqual(dataset.responses.max(), 2.25)  # validation labels excluded
            self.assertEqual(joint_mask[:6, -1].count_nonzero(), 0)
            self.assertEqual(valid[:6, :, -1].count_nonzero(), 0)
            self.assertEqual(valid[2, 1:].count_nonzero(), 0)
            for eid, pools in dataset.pools.items():
                for context in np.concatenate(pools):
                    self.assertEqual(dataset.context_body[context], dataset.episode_body[eid])
                    self.assertNotEqual(dataset.context_episode[context], eid)
            indices = np.tile(np.arange(12), 5)
            history, lengths = dataset.histories(indices, np.random.default_rng(9))
            for row, idx in enumerate(indices):
                if lengths[row]:
                    self.assertEqual(history[row, 0, 0, 0], joints[idx, 0, 0])

    def test_history_only_identification_recovers_known_linear_mapping(self):
        rng = np.random.default_rng(5)
        for count in [6, 7]:
            true_jacobian = rng.normal(size=(3, count))
            changes = rng.normal(scale=.02, size=(32, count))
            displacement = changes @ true_jacobian.T
            history = np.zeros((32, count, 11))
            history[:, :, 3] = changes
            history[:, :, 8:11] = displacement[:, None, :]
            estimated = estimate_jacobian(history, ridge=1e-10)
            np.testing.assert_allclose(estimated, true_jacobian, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

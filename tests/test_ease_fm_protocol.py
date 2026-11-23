"""Scientific-contract checks for the v2 experiment."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import mujoco
import torch

from examples.ease_fm.baseline import FlowPolicy
from examples.ease_fm.benchmark import SCHEMA, fingerprint, load_manifest, transition_token
from examples.ease_fm.experiment import TrainingData
from examples.ease_fm.history import HistoryBuffer, HistoryFlowPolicy
from examples.ease_fm.report import paired_interval
from examples.ease_fm.recovery_data import perturb_state
from examples.ease_fm.sim import ReachEnv, sample_body


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_goal_excluded_and_angle_delta_wrapped(self):
        before = np.zeros(20)
        after = np.zeros(20)
        before[0], after[0] = np.pi-.01, -np.pi+.01
        token = transition_token(before, np.zeros(7), after)
        self.assertAlmostEqual(float(token[24]), .02, places=5)
        before[-3:] = 100
        after[-3:] = -100
        np.testing.assert_array_equal(token, transition_token(before, np.zeros(7), after))

    def test_causal_padding_and_empty_context(self):
        model = HistoryFlowPolicy().eval()
        tokens = torch.randn(2, 32, 41)
        lengths = torch.tensor([0, 5])
        reference = model.encode(tokens, lengths)
        changed = tokens.clone()
        changed[0] = 1000
        changed[1, 5:] = -1000
        torch.testing.assert_close(reference, model.encode(changed, lengths))
        changed[1, 0] += 3
        self.assertFalse(torch.allclose(reference[1], model.encode(changed, lengths)[1]))
        reference.square().mean().backward()
        self.assertGreater(model.history_input.weight.grad.abs().sum(), 0)
        self.assertTrue(torch.isfinite(model.empty_history.grad).all())

    def test_buffer_is_rolling_and_records_executed_action(self):
        buffer = HistoryBuffer()
        for index in range(40):
            before = np.full(20, index, dtype=float)
            buffer.append(before, np.ones(7)*.02, before+.1)
        tokens, length = buffer.array()
        self.assertEqual(length, 32)
        self.assertEqual(tokens[0, 0], 8)
        np.testing.assert_array_equal(tokens[:, 17:24], .5)

    def test_shared_initialization_and_paired_noise(self):
        torch.manual_seed(42)
        baseline = FlowPolicy().eval()
        torch.manual_seed(42)
        history = HistoryFlowPolicy().eval()
        for key, value in baseline.state_dict().items():
            torch.testing.assert_close(value, history.state_dict()[key])
        obs, noise = torch.zeros(2, 20), torch.randn(2, 16, 7)
        first = baseline.sample(obs, noise=noise)
        torch.randn(1000)
        torch.testing.assert_close(first, baseline.sample(obs, noise=noise))

    def test_transient_collision_and_sustained_success(self):
        env = ReachEnv(sample_body(42))
        env.goal = np.zeros(3)
        env.data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(7), ctrl=np.zeros(7),
                                   site_xpos=np.zeros((1, 3)), ncon=0)
        with patch("examples.ease_fm.sim.mujoco.mj_step"):
            for _ in range(4):
                self.assertFalse(env.step(np.zeros(7))[2]["success"])
            self.assertTrue(env.step(np.zeros(7))[2]["success"])
        calls = 0

        def transient(*args):
            nonlocal calls
            calls += 1
            env.data.ncon = int(calls == 1)

        with patch("examples.ease_fm.sim.mujoco.mj_step", side_effect=transient):
            info = env.step(np.zeros(7))[2]
        self.assertTrue(info["collision"])
        self.assertFalse(info["success"])
        self.assertEqual(env.data.ncon, 0)
        with patch("examples.ease_fm.sim.mujoco.mj_step"):
            for _ in range(6):
                self.assertFalse(env.step(np.zeros(7))[2]["success"])

    def test_cross_episode_training_and_manifest_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries, calibrations = [], []
            for body in [1, 2]:
                for episode in range(2):
                    name = f"body{body}_episode{episode}.npz"
                    np.savez(root / name, observation=np.full((4, 20), body, dtype=np.float32),
                             action=np.zeros((4, 7), dtype=np.float32),
                             next_observation=np.full((4, 20), body+.1, dtype=np.float32))
                    entries.append({"file": name, "body_seed": body, "split": "train",
                                    "success": True, "collision": False})
                name = f"calibration{body}.npy"
                np.save(root / name, np.full((32, 41), body, dtype=np.float32))
                calibrations.append({"file": name, "body_seed": body, "split": "train"})
            manifest = {"schema": SCHEMA, "episodes": entries, "calibrations": calibrations,
                        "asset_hashes": {p.name: fingerprint(p) for p in root.iterdir()}}
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "manifest.sha256").write_text(fingerprint(root / "manifest.json"))
            dataset = TrainingData(root)
            indices = np.tile(np.arange(len(dataset.observation)), 10)
            tokens, lengths, chosen = dataset.sample_history(indices, np.random.default_rng(0))
            self.assertEqual(tokens.shape, (160, 32, 41))
            self.assertTrue((lengths == 0).any())
            for idx, context in zip(indices, chosen):
                episode = dataset.row_episode[idx]
                self.assertEqual(dataset.episode_body[episode], dataset.context_body[context])
                self.assertNotEqual(episode, dataset.context_episode[context])
            (root / "manifest.json").write_text("{}")
            with self.assertRaises((ValueError, KeyError)):
                load_manifest(root)

    def test_paired_bootstrap_preserves_equal_conditions(self):
        values = np.random.default_rng(4).integers(2, size=(3, 3, 12))
        self.assertEqual(paired_interval(values, values, samples=50),
                         {"difference": 0., "low": 0., "high": 0.})
        self.assertEqual(paired_interval(np.ones((2, 2, 2)), np.zeros((2, 2, 2)), samples=50),
                         {"difference": 1., "low": 1., "high": 1.})

    def test_recovery_perturbs_all_joints_and_recomputes_observation(self):
        env = ReachEnv(sample_body(42))
        original = env.reset(12)
        perturbed, collision = perturb_state(env, original, 100)
        self.assertFalse(collision)
        self.assertTrue(np.all(np.abs(perturbed[:7]-original[:7]) > 1e-5))
        self.assertLessEqual(np.max(np.abs(perturbed[:7]-original[:7])), .35)
        np.testing.assert_array_equal(perturbed[-3:], original[-3:])
        np.testing.assert_allclose(perturbed[14:17], env.data.site_xpos[0], atol=1e-7)
        repeated, _ = perturb_state(env, original, 100)
        np.testing.assert_array_equal(perturbed, repeated)

    def test_fixture_last_joint_is_position_null(self):
        env = ReachEnv(sample_body(42))
        env.reset(12)
        endpoint = env.data.site_xpos[0].copy()
        env.data.qpos[6] += .4
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(endpoint, env.data.site_xpos[0], atol=1e-12)
        self.assertAlmostEqual(float(env.expert()[6]), 0., places=8)


if __name__ == "__main__":
    unittest.main()

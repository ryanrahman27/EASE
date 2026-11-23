import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from examples.ease_fm.baseline import FlowPolicy, training_arrays
from examples.ease_fm.sim import ACTION_LIMIT, ReachEnv, sample_body


def test_seeded_dynamics_and_action_contract():
    a, b = [ReachEnv(sample_body(42)) for _ in range(2)]
    np.testing.assert_array_equal(a.reset(12), b.reset(12))
    for _ in range(10):
        x, _, info = a.step(np.ones(7))
        y, _, _ = b.step(np.ones(7))
        np.testing.assert_array_equal(x, y)
        assert np.max(np.abs(info["executed_action"])) <= ACTION_LIMIT + 1e-7
    with unittest.TestCase().assertRaises(ValueError):
        a.step(np.full(7, np.nan))


def test_expert_reaches_across_bodies():
    for seed in range(5):
        env = ReachEnv(sample_body(seed))
        env.reset(500)
        done = False
        while not done:
            _, done, info = env.step(env.expert())
        assert info["success"], (seed, info)


def test_training_split_and_episode_boundaries(tmp_path):
    entries = []
    for split, value in [("train", 1.), ("interpolation", 999.)]:
        name = f"{split}.npz"
        np.savez(tmp_path / name, observation=np.full((2, 20), value, dtype=np.float32),
                 action=np.full((2, 7), .01, dtype=np.float32))
        entries.append({"split": split, "file": name, "success": True, "collision": False})
    (tmp_path / "manifest.json").write_text(json.dumps({"episodes": entries}))
    obs, actions, mask = training_arrays(tmp_path)
    assert len(obs) == 2 and obs.max() == 1
    assert mask[0].sum() == 14 and mask[1].sum() == 7
    assert actions[1, 1:].count_nonzero() == 0


def test_flow_backward_and_bounded_sampling():
    torch.set_num_threads(2)
    model = FlowPolicy()
    obs = torch.zeros(2, 20)
    prediction = model(torch.randn(2, 16, 7), torch.rand(2), obs)
    prediction.square().mean().backward()
    assert model.output.weight.grad.isfinite().all()
    action = model.eval().sample(obs, steps=2)
    assert action.shape == (2, 16, 7)
    assert action.abs().max() <= ACTION_LIMIT


class EaseFMTests(unittest.TestCase):
    def test_dynamics(self):
        test_seeded_dynamics_and_action_contract()

    def test_expert(self):
        test_expert_reaches_across_bodies()

    def test_data(self):
        with tempfile.TemporaryDirectory() as directory:
            test_training_split_and_episode_boundaries(Path(directory))

    def test_flow(self):
        test_flow_backward_and_bounded_sampling()


if __name__ == "__main__":
    unittest.main()

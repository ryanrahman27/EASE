"""Procedural seven-DoF reaching development fixture (not a Franka model)."""

from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ACTION_LIMIT = 0.04
OBS_DIM = 20  # q, dq, measured end-effector position, goal
CONTROL_HZ = 25
SUCCESS_HOLD_STEPS = 5


@dataclass(frozen=True)
class Body:
    seed: int
    split: str
    lengths: tuple
    masses: tuple
    strength: float
    damping: float

    def manifest(self):
        return asdict(self)


def sample_body(seed, split="train"):
    if split not in {"train", "validation", "interpolation", "extrapolation"}:
        raise ValueError(split)
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0.8, 1.2, 7)
    if split == "extrapolation":
        scale = rng.uniform(1.25, 1.35, 7)
    return Body(seed, split, tuple(np.array([.16, .18, .16, .14, .12, .10, .08]) * scale),
                tuple(rng.uniform(.7, 1.3, 7)), float(rng.uniform(.5, 1.5)), float(rng.uniform(1, 3)))


def model_xml(body):
    root = ET.Element("mujoco", model="ease_development_arm")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="true")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="2 2 .1", rgba=".3 .3 .3 1")
    parent = world
    axes = ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"]
    actuators = ET.SubElement(root, "actuator")
    for i, (length, mass, axis) in enumerate(zip(body.lengths, body.masses, axes)):
        parent = ET.SubElement(parent, "body", name=f"link{i}",
                               pos="0 0 .15" if i == 0 else f"0 0 {body.lengths[i-1]}")
        ET.SubElement(parent, "joint", name=f"j{i}", axis=axis, range="-2.5 2.5",
                      damping=str(body.damping), armature=".05")
        ET.SubElement(parent, "geom", type="capsule", fromto=f"0 0 0 0 0 {length}",
                      size=".025", mass=str(mass))
        ET.SubElement(actuators, "position", joint=f"j{i}", kp=str(800 * body.strength),
                      kv="50", ctrlrange="-2.5 2.5", forcerange=f"{-100*body.strength} {100*body.strength}")
    ET.SubElement(parent, "site", name="ee", pos=f"0 0 {body.lengths[-1]}", size=".015")
    return ET.tostring(root, encoding="unicode")


class ReachEnv:
    def __init__(self, body, horizon=150):
        self.body = body
        self.xml = model_xml(body)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.horizon = horizon
        self.goal = np.zeros(3)
        self.steps = 0
        self.success_streak = 0
        self.episode_collision = False

    def observation(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.data.site_xpos[0], self.goal]).astype(np.float32)

    def reset(self, seed, goal=None):
        rng = np.random.default_rng(seed)
        home = np.array([0., .5, 0., -.9, 0., .6, 0.])
        # Kinematic reachability only; collector separately reports expert success.
        for _ in range(100):
            mujoco.mj_resetData(self.model, self.data)
            self.data.qpos[:] = home + rng.uniform(-.4, .4, 7)
            mujoco.mj_forward(self.model, self.data)
            if self.data.ncon == 0:
                self.goal = self.data.site_xpos[0].copy()
                break
        else:
            raise RuntimeError("No collision-free target configuration")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = home + rng.uniform(-.05, .05, 7)
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        if self.data.ncon:
            raise RuntimeError("Initial configuration collides")
        self.steps = 0
        self.success_streak = 0
        self.episode_collision = False
        if goal is not None:
            goal = np.asarray(goal, dtype=float)
            if goal.shape != (3,) or not np.isfinite(goal).all():
                raise ValueError("Expected a finite XYZ goal")
            self.goal = goal.copy()
        return self.observation()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("Expected seven finite joint increments")
        target = np.clip(self.data.qpos + np.clip(action, -ACTION_LIMIT, ACTION_LIMIT), -2.5, 2.5)
        executed = target - self.data.qpos
        self.data.ctrl[:] = target
        # Detect transient contact at every physics step, not only at 25 Hz.
        collision = False
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
            collision |= bool(self.data.ncon)
        self.steps += 1
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite simulation state")
        error = float(np.linalg.norm(self.data.site_xpos[0] - self.goal))
        self.episode_collision |= collision
        self.success_streak = self.success_streak + 1 if error < .03 and not collision else 0
        info = {"error": error,
                "success": self.success_streak >= SUCCESS_HOLD_STEPS and not self.episode_collision,
                "collision": collision,
                "executed_action": executed.astype(np.float32)}
        return self.observation(), info["success"] or self.steps >= self.horizon, info

    def expert(self):
        # Privileged Jacobian is confined to demonstration generation.
        jac = np.zeros((3, 7))
        mujoco.mj_jacSite(self.model, self.data, jac, None, 0)
        error = self.goal - self.data.site_xpos[0]
        delta = jac.T @ np.linalg.solve(jac @ jac.T + .015**2 * np.eye(3), .5 * error)
        # Gravity feedforward compensates the position servo's static error.
        delta += self.data.qfrc_bias / (800 * self.body.strength)
        return np.clip(delta, -ACTION_LIMIT, ACTION_LIMIT)

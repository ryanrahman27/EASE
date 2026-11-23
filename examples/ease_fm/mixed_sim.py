"""Six/seven-DoF serial-arm fixtures with different joint-axis sequences.

These are procedural kinematic families, not models of particular commercial arms.
Only joint states, joint validity and measured end-effector position reach students.
"""
from dataclasses import asdict, dataclass
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .sim import ACTION_LIMIT, SUCCESS_HOLD_STEPS

AXES = {
    "six": ["0 0 1", "0 1 0", "0 1 0", "1 0 0", "0 1 0", "1 0 0"],
    "seven": ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "1 0 0", "0 1 0", "0 0 1"],
    "six_novel": ["0 0 1", "0 1 0", "1 0 0", "0 1 0", "0 1 0", "1 0 0"],
}


@dataclass(frozen=True)
class Arm:
    seed: int
    split: str
    family: str
    lengths: tuple
    masses: tuple
    strength: float
    damping: float

    @property
    def dof(self):
        return len(self.lengths)

    def manifest(self):
        return asdict(self)


def sample_arm(seed, family, split):
    rng = np.random.default_rng(seed)
    nominal = np.array([.2, .22, .2, .16, .14, .12] if family.startswith("six")
                       else [.16, .18, .16, .14, .12, .10, .08])
    return Arm(seed, split, family, tuple(nominal*rng.uniform(.8, 1.2, len(nominal))),
               tuple(rng.uniform(.7, 1.3, len(nominal))), float(rng.uniform(.5, 1.5)),
               float(rng.uniform(1, 3)))


def arm_xml(arm):
    root = ET.Element("mujoco", model=f"ease_{arm.family}")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="true")
    ET.SubElement(root, "option", timestep=".002", integrator="implicitfast")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "geom", type="plane", size="2 2 .1")
    actuators = ET.SubElement(root, "actuator")
    parent = world
    for i, (length, mass, axis) in enumerate(zip(arm.lengths, arm.masses, AXES[arm.family])):
        parent = ET.SubElement(parent, "body", name=f"link{i}",
                               pos="0 0 .15" if i == 0 else f"0 0 {arm.lengths[i-1]}")
        ET.SubElement(parent, "joint", name=f"j{i}", axis=axis, range="-2.5 2.5",
                      damping=str(arm.damping), armature=".05")
        ET.SubElement(parent, "geom", type="capsule", fromto=f"0 0 0 0 0 {length}", size=".025", mass=str(mass))
        ET.SubElement(actuators, "position", joint=f"j{i}", kp=str(800*arm.strength), kv="50",
                      ctrlrange="-2.5 2.5", forcerange=f"{-100*arm.strength} {100*arm.strength}")
    ET.SubElement(parent, "site", name="ee", pos=f"0 0 {arm.lengths[-1]}", size=".015")
    return ET.tostring(root, encoding="unicode")


class MixedReach:
    def __init__(self, arm, horizon=150):
        self.arm, self.horizon = arm, horizon
        self.xml = arm_xml(arm)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.goal = np.zeros(3)
        self.steps, self.streak, self.collision = 0, 0, False

    def observe(self):
        return {"joints": np.stack([self.data.qpos, self.data.qvel], axis=-1).astype(np.float32),
                "task": np.concatenate([self.data.site_xpos[0], self.goal]).astype(np.float32)}

    def reset(self, seed, goal, spread=.05):
        rng = np.random.default_rng(seed)
        home = np.array([0, .5, -.9, 0, .6, 0] if self.arm.family == "six"
                        else [0, .5, 0, -.9, .6, 0] if self.arm.family == "six_novel"
                        else [0, .5, 0, -.9, 0, .6, 0], dtype=float)
        mujoco.mj_resetData(self.model, self.data)
        self.goal = np.asarray(goal, dtype=float).copy()
        self.data.qpos[:] = home + rng.uniform(-spread, spread, self.arm.dof)
        self.data.ctrl[:] = self.data.qpos
        mujoco.mj_forward(self.model, self.data)
        self.steps, self.streak, self.collision = 0, 0, bool(self.data.ncon)
        if self.collision:
            raise ValueError("Colliding initial state")
        return self.observe()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (self.arm.dof,) or not np.isfinite(action).all():
            raise ValueError("Action must match the active joints")
        target = np.clip(self.data.qpos + np.clip(action, -ACTION_LIMIT, ACTION_LIMIT), -2.5, 2.5)
        executed = (target - self.data.qpos).astype(np.float32)
        self.data.ctrl[:] = target
        contact = False
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)
            contact |= bool(self.data.ncon)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("Nonfinite mixed-arm state")
        self.steps += 1
        self.collision |= contact
        error = float(np.linalg.norm(self.data.site_xpos[0]-self.goal))
        self.streak = self.streak+1 if error < .03 and not contact else 0
        success = self.streak >= SUCCESS_HOLD_STEPS and not self.collision
        return self.observe(), success or self.steps >= self.horizon, {
            "success": success, "error": error, "collision": self.collision, "executed_action": executed}

    def expert(self):
        jac = np.zeros((3, self.arm.dof))
        mujoco.mj_jacSite(self.model, self.data, jac, None, 0)
        delta = jac.T @ np.linalg.solve(jac @ jac.T+.015**2*np.eye(3), .5*(self.goal-self.data.site_xpos[0]))
        delta += self.data.qfrc_bias / (800*self.arm.strength)
        return np.clip(delta, -ACTION_LIMIT, ACTION_LIMIT)


def token(before, action, after):
    delta = after["joints"]-before["joints"]
    delta[:, 0] = (delta[:, 0]+np.pi) % (2*np.pi)-np.pi
    global_state = np.concatenate([before["task"][:3], after["task"][:3]-before["task"][:3]])
    return np.concatenate([before["joints"], np.asarray(action)[:, None]/ACTION_LIMIT, delta,
                           np.broadcast_to(global_state, (len(action), 6))], axis=-1).astype(np.float32)


def explore(env, seed, length=32):
    obs = env.reset(seed, [0, 0, .9])
    rng = np.random.default_rng(seed)
    result = []
    for step in range(length):
        if step % 4 == 0:
            action = rng.uniform(-.018, .018, env.arm.dof)
        after, _, info = env.step(action)
        if info["collision"]:
            raise RuntimeError("Exploration contact")
        result.append(token(obs, info["executed_action"], after))
        obs = after
    return np.asarray(result)

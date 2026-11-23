"""E182 retrospective matched-history analysis of E181 discordant outcomes."""
from pathlib import Path
import sys
import numpy as np
import mujoco

from .fresh_recovery import ROOT as SOURCE, old, prior, read, save, fingerprint
from .matched_unseen_analyze import dls, restore
from .arm_pick_place import PickPlace, rotation_error, ROT_SCALE
from .mixed_sim import Arm, AXES
from .local_history import LocalPose
from .pick_place_control import calibrate
from .bounded_history_correction import bounded_action

ROOT = Path("outputs/ease_fresh_update_diagnosis_v1")
METHODS = ("e120_ensemble", "e120_frozen", "analytical")
CASES = (25, 6, 38, 43, 9, 35)


def networks(controller, models):
    result = []
    for role in ("reference", "correction"):
        predictions = []
        for seed in range(3):
            controller.model = models[f"{role}_{seed}"]
            predictions.append(controller.predict()[0])
        result.append(np.mean(predictions, axis=0))
    controller.model = None
    return result


def analyze(source):
    assert source in METHODS
    root = ROOT / source
    root.mkdir(parents=True,exist_ok=True)
    tasks = read(SOURCE/"cases.json")
    AXES.update(read(SOURCE/"axis_catalog.json"))
    _,models,hashes=old.broad.setup()
    for p,h in {**read(SOURCE/"frozen.json"),**hashes}.items():
        assert fingerprint(p)==h
    for case in CASES:
        task = tasks[case]
        stem = f"{case}_{source}_frozen_bias"
        trial = read(SOURCE/source/"trials"/f"{stem}.json")
        logs = read(SOURCE/source/"clearance"/f"{stem}.json")
        states = np.load(SOURCE/source/"states"/f"{case}_{source}_{trial['mode']}.npz")["states"]
        env = PickPlace(Arm(**task["arm"]),task["xy"],task["target"])
        clone = PickPlace(Arm(**task["arm"]),task["xy"],task["target"])
        c = LocalPose(6,None)
        assert calibrate(env, task["home"], c, 38000+case)==trial["calibration"]
        frozen = networks(c,models)
        bias = c.bias.copy()
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        snapshot=np.empty(mujoco.mj_stateSize(env.model,spec))
        rows=[]
        for i,l in enumerate(logs):
            phase = l["phase"]
            if phase=="probe" and not c.local:
                c.local=True
                c.rebuild()
            if source!="e120_frozen":
                np.testing.assert_array_equal(c.gram,l["estimator_gram"])
                np.testing.assert_array_equal(c.cross,l["estimator_cross"])
            online=networks(c,models)
            analytical=np.linalg.solve(c.gram,c.cross).T
            before=env.observe()
            goal=np.asarray(l["goal"])
            error=np.r_[goal-before["position"], ROT_SCALE*rotation_error(np.eye(3),before["rotation"])]
            candidates=[bounded_action(dls(j,error,bias),dls(k,error,bias)) for j,k in (online,frozen)]
            candidates.append(dls(analytical,error,bias))
            jacobians=[online[0],frozen[0],analytical]
            idx=METHODS.index(source)
            np.testing.assert_array_equal(jacobians[idx],l["estimator_jacobian"])
            final=[]
            for nominal,j in zip(candidates,jacobians):
                if phase=="probe":
                    nominal=np.array(l["nominal_action"])
                target=np.array(l["optimization_target"]) if l["route_active"] or l["handoff_control"] else nominal
                if l["eligible"]:
                    jd=j.astype(float)
                    target=prior.project(target,bias,before["q"],np.array(l["effective_distance"]),np.array(l["gradient"]),jd.T@jd+.015**2*np.eye(6))[0]
                final.append(target)
            np.testing.assert_allclose(final[idx],l["final_action"],atol=1e-12,rtol=0)
            mujoco.mj_getState(env.model,env.data,snapshot,spec)
            after,command,actual=env.step(np.array(l["final_action"]),l["aperture"])
            np.testing.assert_array_equal(np.r_[env.data.qpos,env.data.qvel],states[i])
            dq=after["q"]-before["q"]
            dx=after["position"]-before["position"]
            branches=[]
            for m,(j,action) in enumerate(zip(jacobians,final)):
                restore(clone,snapshot)
                nxt,_,info=clone.step(action,l["aperture"])
                if m==idx:
                    np.testing.assert_allclose(np.r_[clone.data.qpos,clone.data.qvel],states[i],atol=1e-12,rtol=0)
                    assert sorted(set(map(tuple,map(sorted,info["bad"]))))==sorted(map(tuple,trial["trace"][i]["bad"]))
                branches.append(dict(prediction_error_mm=1000*float(np.linalg.norm((j@dq)[:3]-dx)),
                                     progress_mm=1000*float(np.linalg.norm(error[:3])-np.linalg.norm(goal-nxt["position"])),
                                     collision=bool(info["bad"]),action=action.tolist()))
            rows.append(dict(step=i,phase=phase,close_ready=l["close_ready"],aligning=l["aligning"],
                             aborted=l["aborted"],routed=l["route_active"],error_mm=float(np.linalg.norm(error[:3])*1000),
                             branches=branches))
            c.update(before,after,command)
            c.bias=bias.copy()
        save(root/f"{case}.json",rows)
        print(source,case,len(rows),flush=True)


if __name__=="__main__":
    analyze(sys.argv[1])

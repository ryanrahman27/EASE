"""Case 25 cube displacement for each one-step action on the failing descent."""
import numpy as np
import mujoco
from .fresh_update_diagnosis import (ROOT,SOURCE,METHODS,Arm,PickPlace,LocalPose,AXES,
                                     calibrate,read,save,restore)


def run():
    case=25
    task=read(SOURCE/"cases.json")[case]
    AXES.update(read(SOURCE/"axis_catalog.json"))
    env,clone=[PickPlace(Arm(**task["arm"]),task["xy"],task["target"]) for _ in range(2)]
    calibrate(env,task["home"],LocalPose(6,None),38000+case)
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    snapshot=np.empty(mujoco.mj_stateSize(env.model,spec))
    src=METHODS[0]
    logs=read(SOURCE/src/"clearance"/f"25_{src}_frozen_bias.json")
    results=[]
    for l,r in zip(logs,read(ROOT/src/"25.json")):
        if l["phase"]=="probe":break
        initial=env.data.xpos[env.obj,:2].copy()
        if l["phase"]=="descend":
            mujoco.mj_getState(env.model,env.data,snapshot,spec)
            changes=[]
            for b in r["branches"]:
                restore(clone,snapshot)
                clone.step(b["action"],l["aperture"])
                changes.append(1000*float(np.linalg.norm(clone.data.xpos[clone.obj,:2]-initial)))
            results.append(dict(step=r["step"],pre_drift_mm=1000*float(np.linalg.norm(initial-task["xy"])),
                                cube_motion_mm=dict(zip(METHODS,changes))))
        env.step(l["final_action"],l["aperture"])
    save(ROOT/"case25_contact.json",results)
    print([r for r in results if r["cube_motion_mm"][src]>.5])


if __name__=="__main__":run()

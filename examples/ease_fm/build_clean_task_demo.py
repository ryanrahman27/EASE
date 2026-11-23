"""Isolated revised demo: aligned grasp frames and mesh/table collisions."""
from pathlib import Path
b=Path(__file__).parent
s=(b/'known_arm_task_demo.py').read_text()
s=s.replace("ease_known_arm_task_demo_v1","ease_known_arm_task_demo_v2")
s=s.replace("pos='0 0 .08' if name=='piperx' else '0 0 .065'","pos='0 0 .15' if name=='piperx' else '0 0 .09'")
s=s.replace("('0 -.055 -.004' if i==0 else '0 .055 -.004') if name=='piperx' else '0 .002 .05'","('0 .015 -.004' if i==0 else '0 -.015 -.004') if name=='piperx' else '0 .002 .075'")
s=s.replace("    world=root.find('worldbody');xy=", "    for geom in mount.findall('.//geom'):\n        if geom.get('type')=='mesh' and geom.get('group')!='3':geom.set('contype','2');geom.set('conaffinity','4')\n    world=root.find('worldbody');xy=")
s=s.replace("rgba='.10 .12 .10 1',contype='1',conaffinity='1'","rgba='.10 .12 .10 1',contype='4',conaffinity='3'")
s=s.replace("rgba='.025 .03 .025 1',contype='0',conaffinity='0'","rgba='.025 .03 .025 1',contype='4',conaffinity='3'")
s=s.replace("Native visual/kinematic models; demonstration-specific gravcomp position servos and simplified contact pads; physical free object, no welding or scripted object pose.","Native visual/kinematic models; aligned fingertip grasp frames, demonstration servos/pads. Gripper visual meshes collide with table/floor; cube is free and moved by pad contact. No welding or object animation.")
s=s.replace('records=[];states=[];', 'records=[];states=[];render_states=[];')
s=s.replace('for _ in range(20):mujoco.mj_step(m,d)', 'for substep in range(20):\n                mujoco.mj_step(m,d)\n                if substep in (9,19):render_states.append(np.r_[d.qpos.copy(),d.qvel.copy()])')
s=s.replace('states=np.array(states),nq=', 'states=np.array(states),render_states=np.array(render_states),nq=')
(b/'known_arm_task_clean.py').write_text(s)

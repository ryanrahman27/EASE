"""World-frame rotation increments for measured rigid-body control."""
import mujoco
import numpy as np


def rotation_vector(matrix):
    q=np.empty(4);mujoco.mju_mat2Quat(q,np.asarray(matrix,dtype=float).reshape(9))
    if q[0]<0:q=-q
    length=np.linalg.norm(q[1:])
    return 2*q[1:] if length<1e-10 else (2*np.arctan2(length,q[0])/length)*q[1:]


def upright_target(rotation,align):
    x=rotation[:,0].copy();x[2]=0
    if align:
        axes=np.array([[1.,0,0],[0,1.,0],[-1.,0,0],[0,-1.,0]])
        x=axes[np.argmax(axes@x)]
    else:x=x/np.linalg.norm(x)
    z=np.array([0.,0,1.]);return np.column_stack([x,np.cross(z,x),z])

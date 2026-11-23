"""Capacity-matched context encoder restricted to the first observed pose."""
import torch
from .joint_policy import JointFlowPolicy


def initial_pose(history):
    result=torch.zeros_like(history)
    result[:,:,:,0]=history[:,:1,:,0]
    result[:,:,:,5:8]=history[:,:1,:,5:8]
    return result


class PosePolicy(JointFlowPolicy):
    def encode(self,history,lengths,joint_mask):
        return super().encode(initial_pose(history),lengths,joint_mask)

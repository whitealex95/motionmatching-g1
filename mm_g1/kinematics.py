"""Planar root-motion stitching: place library frames into a continuous world path.

A segment is played under one planar "alignment" (a yaw rotation about world Z +
xy translation). Playing contiguous library frames under a fixed alignment lets the
clip's own root motion carry the character; at a transition we recompute the alignment
so the new frame coincides with the current world pose -> C0-continuous root path.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def rotz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def transform_qpos(qpos, dyaw, pivot, offset):
    """Place library qpos rows (..,36) into the WORLD via a planar (SE2) alignment:
    root_xy <- Rz(dyaw)*(root_xy - pivot) + offset, and the heading is pre-rotated by dyaw
    about world-Z. Root z and the 29 joint angles are unchanged. pivot/offset are world xy."""
    q = np.atleast_2d(qpos).astype(np.float64).copy()
    q[:, 0:2] = (rotz(dyaw) @ (q[:, 0:2] - pivot).T).T + offset   # root xy: planar move in world
    xyzw = q[:, [4, 5, 6, 3]]                                     # qpos quat wxyz -> scipy xyzw
    rot = R.from_euler("z", dyaw) * R.from_quat(xyzw)             # rotate heading by dyaw (world Z)
    q[:, 3:7] = rot.as_quat()[:, [3, 0, 1, 2]]                    # scipy xyzw -> qpos wxyz
    return q


def alignment_to(lib_xy, lib_yaw, world_xy, world_yaw):
    """Planar alignment (dyaw, pivot, offset) for transform_qpos that lands a library frame
    exactly at a target world pose: dyaw = world_yaw - lib_yaw (heading delta), pivot = the
    library frame's xy (rotate about it), offset = target world xy. xy in metres, yaw in rad."""
    return world_yaw - lib_yaw, np.asarray(lib_xy, float), np.asarray(world_xy, float)


def blend_qpos(frozen, live, w):
    """Ease joints + root orientation from a frozen pose toward the live pose (w:0->1).

    Root position is taken from `live` so the world path stays continuous; only the
    body pose (joint angles + root tilt/heading) is smoothed across a transition.
    """
    out = live.copy()
    out[7:36] = (1 - w) * frozen[7:36] + w * live[7:36]
    from scipy.spatial.transform import Slerp
    key = R.concatenate([R.from_quat(frozen[[4, 5, 6, 3]]), R.from_quat(live[[4, 5, 6, 3]])])
    out[3:7] = Slerp([0, 1], key)([w])[0].as_quat()[[3, 0, 1, 2]]
    return out

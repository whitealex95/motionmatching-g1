"""GenoView-style 27-D motion-matching feature database, built from our G1 library.

Math + layout mirror ../GenoViewPython-MotionMatching/genoview_g1.py's build_database:
  Xpos (6)  local foot positions (L,R) relative to the root
  Xvel (9)  local velocities of both feet + the pelvis (= root)
  XtrajPos (6) future root xy at +10/+20/+30 frames, in the heading frame
  XtrajDir (6) future heading xy at the same horizons
Everything is expressed in each frame's yaw-only (heading) frame, then normalized by a
per-block scale (one shared std per block) so the Euclidean search weights blocks sensibly.
"""
import numpy as np

from . import config as C
from . import quat

FPS = C.FPS
HORIZONS = np.array(C.HORIZONS)
FORWARD = np.array([1.0, 0.0, 0.0])     # G1 pelvis forward axis
UP = np.array([0.0, 0.0, 1.0])


def yaw_quat(theta):
    """Quaternion (wxyz) for a rotation of theta about world +Z."""
    return quat.from_angle_axis(np.asarray(theta), UP)


def heading_dir(rootquat):
    """World-space forward direction (xy, z=0, normalized) of a root quaternion (wxyz)."""
    fwd = quat.mul_vec(rootquat, FORWARD) * np.array([1.0, 1.0, 0.0])
    return fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)


def build_db(lib):
    """Assemble the GenoView feature DB (a dict) from the precomputed G1 library."""
    qpos = lib["qpos"].astype(np.float64)
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(qpos))
    spans = list(zip(starts, stops))

    rootPos = qpos[:, 0:3].copy()
    rootQuat = qpos[:, 3:7].copy()
    dof = qpos[:, 7:].copy()
    theta = lib["yaw"].astype(np.float64)               # heading angle (T,)
    headDir = heading_dir(rootQuat)                      # (T,3)
    footL = lib["feet_world"][:, 0].astype(np.float64)   # world foot positions
    footR = lib["feet_world"][:, 1].astype(np.float64)
    pelvis = rootPos                                     # floating base == pelvis

    def clipwise_vel(arr):                               # finite-diff, never crossing seams
        v = np.zeros_like(arr)
        for rs, re in spans:
            v[rs:re] = np.gradient(arr[rs:re], axis=0) * FPS
        return v

    footLvel, footRvel = clipwise_vel(footL), clipwise_vel(footR)
    pelvisVel, rootVel = clipwise_vel(pelvis), clipwise_vel(rootPos)
    dofVel = clipwise_vel(dof)
    yawRate = np.zeros(len(qpos))
    for rs, re in spans:
        yawRate[rs:re] = np.gradient(np.unwrap(theta[rs:re])) * FPS

    T = len(qpos)
    qh = yaw_quat(theta)                                 # per-frame heading quats
    to_local = lambda v: quat.inv_mul_vec(qh, v)         # world vec -> heading frame

    Xpos = np.concatenate([to_local(footL - rootPos), to_local(footR - rootPos)], -1)  # (T,6)
    Xvel = np.concatenate([to_local(footLvel), to_local(footRvel), to_local(pelvisVel)], -1)  # (T,9)
    XtrajPos = np.zeros((T, 6))
    XtrajDir = np.zeros((T, 6))
    for rs, re in spans:
        idx = np.arange(rs, re)
        for k, h in enumerate(HORIZONS):
            ft = np.clip(idx + h, rs, re - 1)
            XtrajPos[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(
                qh[rs:re], rootPos[ft] - rootPos[rs:re])[:, 0:2]
            XtrajDir[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(qh[rs:re], headDir[ft])[:, 0:2]

    X = np.concatenate([Xpos, Xvel, XtrajPos, XtrajDir], -1)        # (T,27)
    Xoffset = X.mean(0)
    Xscale = np.concatenate([                                       # one shared std per block
        np.repeat(Xpos.std(0).mean(), Xpos.shape[1]),
        np.repeat(Xvel.std(0).mean(), Xvel.shape[1]),
        np.repeat(XtrajPos.std(0).mean(), XtrajPos.shape[1]),
        np.repeat(XtrajDir.std(0).mean(), XtrajDir.shape[1])])
    Xscale = np.where(Xscale < 1e-5, 1.0, Xscale)
    Xn = (X - Xoffset) / Xscale

    return dict(
        starts=starts, stops=stops, spans=spans,
        rootPos=rootPos, rootQuat=rootQuat, dof=dof, dofVel=dofVel,
        theta=theta, rootVel=rootVel, yawRate=yawRate,
        X=Xn.astype(np.float32), Xoffset=Xoffset, Xscale=Xscale)

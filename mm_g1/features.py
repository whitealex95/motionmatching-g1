"""GenoView-style feature database with a smoothed simulation root, built from our library.

Math + layout mirror ../GenoViewPython-MotionMatching/genoview_g1.py's build_database:
a per-clip Savitzky-Golay-smoothed "simulation root" (ground position + facing) carries the
character, with the pelvis stored as a local offset of it; the 27-D search feature is then
expressed in that smoothed root frame:
  Xpos (6)  local foot positions (L,R) relative to the sim root
  Xvel (9)  local velocities of both feet + the pelvis
  XtrajPos (6) future sim-root xy at +10/+20/+30 frames, in the sim-heading frame
  XtrajDir (6) future sim heading xy at the same horizons
Normalized by a per-block scale (one shared std per block) so the search weights blocks sensibly.
"""
import numpy as np
from scipy.signal import savgol_filter

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


def smooth_root(pelvisPos_world, headDir):
    """Build the smoothed simulation root for one clip range.
    Returns (simPos (N,3) ground, simTheta (N,), headDirSmooth (N,3))."""
    n = len(pelvisPos_world)
    pw = min(C.ROOT_POS_SMOOTH, n if n % 2 == 1 else n - 1)
    dw = min(C.ROOT_DIR_SMOOTH, n if n % 2 == 1 else n - 1)
    simXY = pelvisPos_world[:, :2]
    if pw >= 5:
        simXY = savgol_filter(simXY, pw, 3, axis=0, mode='interp')
    simPos = np.concatenate([simXY, np.zeros((n, 1))], axis=1)
    d = headDir[:, :2].copy()
    if dw >= 5:
        d = savgol_filter(d, dw, 3, axis=0, mode='interp')
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    headDirSmooth = np.concatenate([d, np.zeros((n, 1))], axis=1)
    simTheta = np.arctan2(d[:, 1], d[:, 0])
    return simPos, simTheta, headDirSmooth


def central_diff(x, fps):
    """Central-difference velocity along axis 0 with linear endpoint extrapolation."""
    v = np.empty_like(x)
    if len(x) < 4:
        v[:] = (np.gradient(x, axis=0) * fps) if len(x) > 1 else 0.0
        return v
    v[1:-1] = 0.5 * (x[2:] - x[1:-1]) * fps + 0.5 * (x[1:-1] - x[:-2]) * fps
    v[0] = v[1] - (v[3] - v[2])
    v[-1] = v[-2] + (v[-2] - v[-3])
    return v


def central_diff_ang(rot, fps):
    """Angular velocity (scaled-angle-axis) from a quaternion series via central differences."""
    n = len(rot)
    ang = np.zeros((n, 3))
    if n < 4:
        if n >= 2:
            ang[1:] = quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[1:], rot[:-1]))) * fps
            ang[0] = ang[1]
        return ang
    ang[1:-1] = (0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[2:], rot[1:-1]))) * fps +
                 0.5 * quat.to_scaled_angle_axis(quat.abs(quat.mul_inv(rot[1:-1], rot[:-2]))) * fps)
    ang[0] = ang[1] - (ang[3] - ang[2])
    ang[-1] = ang[-2] + (ang[-2] - ang[-3])
    return ang


def build_db(lib):
    """Assemble the GenoView (smoothed-sim-root) feature DB (a dict) from the G1 library."""
    qpos = lib["qpos"].astype(np.float64)
    fic = lib["frame_in_clip"]
    starts = np.where(fic == 0)[0]
    stops = np.append(starts[1:], len(qpos))
    spans = list(zip(starts, stops))

    rootQuat = qpos[:, 3:7].copy()
    dof = qpos[:, 7:].copy()
    footL = lib["feet_world"][:, 0].astype(np.float64)   # world foot positions
    footR = lib["feet_world"][:, 1].astype(np.float64)
    pelvis = qpos[:, 0:3]                                 # floating base == pelvis
    headDirRaw = heading_dir(rootQuat)                   # (T,3)

    # ---- Smoothed simulation root + pelvis-local offset, per range ----
    T = len(qpos)
    simPos = np.zeros((T, 3))
    simTheta = np.zeros(T)
    headDir = np.zeros((T, 3))                            # smoothed heading
    pelvLocalPos = np.zeros((T, 3))
    pelvLocalRot = np.zeros((T, 4))
    for rs, re in spans:
        sp, st, hd = smooth_root(pelvis[rs:re], headDirRaw[rs:re])
        simPos[rs:re], simTheta[rs:re], headDir[rs:re] = sp, st, hd
        qh = yaw_quat(st)
        pelvLocalPos[rs:re] = quat.inv_mul_vec(qh, pelvis[rs:re] - sp)
        pelvLocalRot[rs:re] = quat.mul(quat.inv(qh), rootQuat[rs:re])

    # ---- Per-range central-difference velocities (never cross range seams) ----
    def clipwise_vel(arr):
        v = np.zeros_like(arr)
        for rs, re in spans:
            v[rs:re] = central_diff(arr[rs:re], FPS)
        return v

    footLvel, footRvel = clipwise_vel(footL), clipwise_vel(footR)
    pelvisVel, simVel = clipwise_vel(pelvis), clipwise_vel(simPos)
    dofVel, pelvLocalVel = clipwise_vel(dof), clipwise_vel(pelvLocalPos)
    yawRate = np.zeros(T)
    pelvLocalAng = np.zeros((T, 3))
    for rs, re in spans:
        yawRate[rs:re] = central_diff(np.unwrap(simTheta[rs:re])[:, None], FPS)[:, 0]
        pelvLocalAng[rs:re] = central_diff_ang(pelvLocalRot[rs:re], FPS)

    # ---- Features in the smoothed sim-root frame ----
    qh_all = yaw_quat(simTheta)
    to_local = lambda v: quat.inv_mul_vec(qh_all, v)

    Xpos = np.concatenate([to_local(footL - simPos), to_local(footR - simPos)], -1)        # (T,6)
    Xvel = np.concatenate([to_local(footLvel), to_local(footRvel), to_local(pelvisVel)], -1)  # (T,9)
    XtrajPos = np.zeros((T, 6))
    XtrajDir = np.zeros((T, 6))
    for rs, re in spans:
        idx = np.arange(rs, re)
        for k, h in enumerate(HORIZONS):
            ft = np.clip(idx + h, rs, re - 1)
            XtrajPos[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(
                qh_all[rs:re], simPos[ft] - simPos[rs:re])[:, 0:2]
            XtrajDir[rs:re, 2 * k:2 * k + 2] = quat.inv_mul_vec(qh_all[rs:re], headDir[ft])[:, 0:2]

    X = np.concatenate([Xpos, Xvel, XtrajPos, XtrajDir], -1)        # (T,27)
    # Normalize over LOCOMOTION frames only (skill==0), so the search space statistics match
    # genoview's loco-only database -- jump frames (a triggered-only extra) don't skew them.
    m = (lib["skill"] == 0) if "skill" in lib else np.ones(T, bool)
    Xoffset = X[m].mean(0)
    Xscale = np.concatenate([                                       # one shared std per block
        np.repeat(Xpos[m].std(0).mean(), Xpos.shape[1]),
        np.repeat(Xvel[m].std(0).mean(), Xvel.shape[1]),
        np.repeat(XtrajPos[m].std(0).mean(), XtrajPos.shape[1]),
        np.repeat(XtrajDir[m].std(0).mean(), XtrajDir.shape[1])])
    Xscale = np.where(Xscale < 1e-5, 1.0, Xscale)
    Xn = (X - Xoffset) / Xscale

    return dict(
        starts=starts, stops=stops, spans=spans,
        dof=dof, dofVel=dofVel,
        simPos=simPos, simTheta=simTheta, simVel=simVel, yawRate=yawRate,
        pelvLocalPos=pelvLocalPos, pelvLocalVel=pelvLocalVel,
        pelvLocalRot=pelvLocalRot, pelvLocalAng=pelvLocalAng,
        X=Xn.astype(np.float32), Xoffset=Xoffset, Xscale=Xscale)

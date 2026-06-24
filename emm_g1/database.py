"""Build / load the EMM feature database from *our* dataset.

Reuses the very same clips ``mm_g1`` uses -- GMR-retargeted LAFAN1 locomotion
(``data/gmr_lafan1_g1``: walk / run / push-and-stumble, GenoView-trimmed) plus the
walk->jump->walk CSVs (``data/g1_jump``) -- each added normal + L/R mirrored. On
top of the standard motion-matching features it precomputes the two extra feature
blocks the paper needs:

  * ``ellipse`` (T, 3, 3): per future-tap footprint ellipse in the frame's
    heading-local frame, ``[axis_x*ext_p, axis_y*ext_p, ext_secondary]``.
  * ``height``  (T, 3, 2): per future-tap (min, max) body height **above the
    world floor (z = 0)**. (Our pelvis root is not ground-projected, so unlike the
    reference we measure height against the floor, which is what an absolute-height
    obstacle band needs for the jump gating to work.)

Unlike ``mm_g1`` -- which hides the jump clips from the search and enters them only
via the ``J`` trigger -- here EVERY frame (jumps included) is searchable; the
env-aware search (see :mod:`emm_g1.search`) decides when a jump wins, purely from
the obstacle penalization. Cached to ``data/emm_lib.npz``.
"""

import os
import numpy as np
from scipy.signal import savgol_filter

from . import quat
from . import features as feat
from . import g1_model as g1
from mm_g1 import config as C
from mm_g1 import data as mmdata
from mm_g1.g1_model import G1Model

CACHE_PATH = os.path.join(C.ROOT, "data", "emm_lib.npz")
DB_VERSION = 1   # bump to invalidate a stale data/emm_lib.npz


def _load_clips():
    """Load our dataset as a list of ``(name, qpos(T,36) wxyz, is_jump)`` clips,
    each followed by its L/R mirror -- identical selection + trims to
    ``mm_g1.data.build_library`` (the same dataset)."""
    model = G1Model()
    loco = [c for c in C.CLIPS
            if os.path.exists(os.path.join(C.DATA_DIR, c + ".pkl"))]
    jumps = [c for c in C.JUMP_CLIPS
             if os.path.exists(os.path.join(C.JUMP_DATA_DIR, c + ".csv"))]
    if not loco:
        raise FileNotFoundError(f"No locomotion clips found in {C.DATA_DIR}")

    base = [(c, False) for c in loco] + [(c, True) for c in jumps]
    out = []
    for name, is_jump in base:
        q = mmdata._load_jump_csv(name) if is_jump else mmdata._load_clip(name)
        out.append((name, q.astype(np.float32), is_jump))
        if C.MIRROR:
            out.append((name + "_mirror", model.mirror_qpos(q).astype(np.float32), is_jump))
    return out


def _clipwise(arr, starts, stops, fn):
    out = np.zeros_like(arr)
    for rs, re in zip(starts, stops):
        out[rs:re] = fn(arr[rs:re])
    return out


def build_database():
    """Assemble the EMM feature DB (a dict) from our dataset via MuJoCo FK."""
    import mujoco
    clips = _load_clips()

    qpos_list, starts, stops, off = [], [], [], 0
    clip_names, clip_is_jump, clip_id = [], [], []
    for cid, (name, q, is_jump) in enumerate(clips):
        if len(q) < int(g1.HORIZONS[-1]) + 2:
            continue
        qpos_list.append(q)
        starts.append(off); stops.append(off + len(q)); off += len(q)
        clip_names.append(name); clip_is_jump.append(is_jump)
        clip_id.append(np.full(len(q), len(clip_names) - 1, np.int32))
    qpos = np.concatenate(qpos_list, 0).astype(np.float64)
    starts = np.array(starts); stops = np.array(stops)
    clip_id = np.concatenate(clip_id)
    T = len(qpos)

    model = g1.load_model()
    data = mujoco.MjData(model)
    foot_ids = [model.body(b).id for b in g1.FOOT_BODIES]
    pelvis_id = model.body(g1.PELVIS_BODY).id
    extent_ids = g1.extent_body_ids(model)
    all_ids = foot_ids + [pelvis_id] + extent_ids
    world = feat.fk_body_positions(model, data, qpos, all_ids)        # (T, N, 3)
    footL = world[:, 0]; footR = world[:, 1]; pelvis = world[:, 2]
    extent = world[:, 3:]                                            # (T, B, 3)

    rootPos = qpos[:, 0:3].copy()
    rootQuat = qpos[:, 3:7].copy()
    dof = qpos[:, 7:].copy()

    fps = g1.FPS
    vel = lambda a: _clipwise(a, starts, stops, lambda s: np.gradient(s, axis=0) * fps)
    footLvel, footRvel, pelvisVel = vel(footL), vel(footR), vel(pelvis)
    rootVel = vel(rootPos)

    # Heading = smoothed PELVIS facing (body orientation), GenoView / mm_g1 style.
    pelvisFace = g1.heading_dir(rootQuat)[:, 0:2]
    headDir2 = np.zeros((T, 2), np.float32)
    for rs, re in zip(starts, stops):
        n = re - rs
        d = pelvisFace[rs:re].astype(np.float64)
        win = min(31, n if n % 2 == 1 else n - 1)
        if win >= 5:
            d = savgol_filter(d, win, 3, axis=0, mode='interp')
        headDir2[rs:re] = d
    headDir2 /= (np.linalg.norm(headDir2, axis=1, keepdims=True) + 1e-9)
    theta = np.arctan2(headDir2[:, 1], headDir2[:, 0]).astype(np.float32)
    headDir = np.concatenate([headDir2, np.zeros((T, 1), np.float32)], axis=1)
    qh = g1.yaw_quat(theta)
    dofVel = vel(dof)
    yawRate = np.zeros(T, np.float32)
    for rs, re in zip(starts, stops):
        yawRate[rs:re] = np.gradient(np.unwrap(theta[rs:re])) * fps

    def local(v):
        return quat.inv_mul_vec(qh, v)

    # ---- Pose block (15), Unity-interleaved order ----
    pose = np.concatenate([
        local(footL - rootPos), local(footLvel),
        local(footR - rootPos), local(footRvel),
        local(pelvisVel)], axis=-1)

    # ---- Trajectory + ellipse + height per tap ----
    trajPos = np.zeros((T, 6), np.float32)
    trajDir = np.zeros((T, 6), np.float32)
    ellipse = np.zeros((T, 3, 3), np.float32)
    height = np.zeros((T, 3, 2), np.float32)

    # footprint primary axis = pelvis ground displacement (fallback: facing).
    pelvisXY = pelvis[:, 0:2]
    displ = np.zeros((T, 2), np.float32)
    for rs, re in zip(starts, stops):
        d = np.zeros((re - rs, 2), np.float32)
        d[:-1] = pelvisXY[rs + 1:re] - pelvisXY[rs:re - 1]
        d[-1] = d[-2] if re - rs >= 2 else 0.0
        displ[rs:re] = d
    motionDirW = displ.copy()
    nrm = np.linalg.norm(motionDirW, axis=-1)
    motionDirW[nrm < 1e-2] = headDir[nrm < 1e-2, 0:2]
    motionDirW /= (np.linalg.norm(motionDirW, axis=-1, keepdims=True) + 1e-9)

    for rs, re in zip(starts, stops):
        idx = np.arange(rs, re)
        for k, h in enumerate(g1.HORIZONS):
            ft = np.clip(idx + h, rs, re - 1)
            futPos = quat.inv_mul_vec(qh[idx], rootPos[ft] - rootPos[idx])
            futDir = quat.inv_mul_vec(qh[idx], headDir[ft])
            trajPos[idx, 2 * k:2 * k + 2] = futPos[:, 0:2]
            trajDir[idx, 2 * k:2 * k + 2] = futDir[:, 0:2]

            axisW = motionDirW[ft]                                    # (n,2) world
            ext = feat.footprint_ellipse(extent[ft][..., 0:2], pelvisXY[ft], axisW)
            axis3 = np.concatenate([axisW, np.zeros((len(idx), 1), np.float32)], -1)
            axisLocal = quat.inv_mul_vec(qh[idx], axis3)[:, 0:2]
            axisLocal /= (np.linalg.norm(axisLocal, axis=-1, keepdims=True) + 1e-9)
            ellipse[idx, k, 0:2] = axisLocal * ext[:, 0:1]
            ellipse[idx, k, 2] = ext[:, 1]
            # Height range of the FUTURE pose, ABSOLUTE above the world floor
            # (z = 0). During a jump's flight the lowest body part lifts above a
            # low bar, so the height gate stops penalizing that pose -> it wins.
            height[idx, k] = feat.height_range(extent[ft][..., 2], floor_z=0.0)

    # ---- Assemble + normalize the static block (per-feature shared scale) ----
    X = np.concatenate([trajPos, trajDir, pose], axis=-1)            # (T,27)
    blocks = [(0, 6), (6, 12), (12, 15), (15, 18), (18, 21), (21, 24), (24, 27)]
    Xmean = X.mean(0)
    Xstd = np.ones(X.shape[1], np.float32)
    for a, b in blocks:
        s = X[:, a:b].std(0).mean()
        Xstd[a:b] = s if s > 1e-5 else 1.0
    Xn = (X - Xmean) / Xstd

    return dict(
        qpos=qpos.astype(np.float32), starts=starts, stops=stops,
        clip_id=clip_id, lengths=np.array(stops) - np.array(starts),
        rootPos=rootPos.astype(np.float32), rootQuat=rootQuat.astype(np.float32),
        dof=dof.astype(np.float32), dofVel=dofVel.astype(np.float32),
        theta=theta.astype(np.float32), rootVel=rootVel.astype(np.float32),
        yawRate=yawRate.astype(np.float32),
        Xn=Xn.astype(np.float32), Xmean=Xmean.astype(np.float32), Xstd=Xstd.astype(np.float32),
        ellipse=ellipse, height=height,
        height_mode=np.array(True),
        clip_names=np.array(clip_names, dtype=object),
        clip_is_jump=np.array(clip_is_jump, bool),
        db_version=np.array(DB_VERSION))


def load_or_build(rebuild=False, path=CACHE_PATH):
    if os.path.exists(path) and not rebuild:
        d = dict(np.load(path, allow_pickle=True))
        if int(d.get("db_version", np.array(-1))) == DB_VERSION:
            return d
        print(f"[emm] cache stale (v{int(d.get('db_version', -1))} != v{DB_VERSION}); rebuilding...")
    print("[emm] building EMM feature database from the dataset (MuJoCo FK over all bodies)...")
    db = build_database()
    np.savez(path, **db)
    print(f"[emm] saved {len(db['Xn'])} frames, {len(db['clip_names'])} clips -> {path}")
    return db


if __name__ == "__main__":
    load_or_build(rebuild=True)

"""Motion-matching feature vectors, all expressed in each frame's root-local space.

Per frame:  trajectory = future root (dx,dy, facing_x,facing_y) at TRAJ_HORIZONS,
            pose       = local foot positions (2x3), foot velocities (2x3), root vel (3).
Query vectors are built to match this layout (commands.py / controller.py).
"""
import numpy as np
from . import config as C

POSE_DIM = 2 * 3 + 2 * 3 + 3          # 15
TRAJ_DIM = 4 * len(C.TRAJ_HORIZONS)   # 12
FEAT_DIM = TRAJ_DIM + POSE_DIM


def _local(vec_xy, yaw):
    """Rotate world xy vectors into the frame's local space (yaw about Z)."""
    c, s = np.cos(-yaw), np.sin(-yaw)
    x, y = vec_xy[..., 0], vec_xy[..., 1]
    return np.stack([c * x - s * y, s * x + c * y], -1)


def compute_features(lib):
    """Return (N, FEAT_DIM) raw features computed per-clip (no cross-clip sampling)."""
    qpos, yaw, feet = lib["qpos"], lib["yaw"], lib["feet_world"]
    xy, z = qpos[:, 0:2], qpos[:, 2]
    N = len(qpos)
    feat = np.zeros((N, FEAT_DIM), np.float32)

    for cid in np.unique(lib["clip_id"]):
        idx = np.where(lib["clip_id"] == cid)[0]
        a, b = idx[0], idx[-1]                       # clip span [a, b]
        for i in idx:
            yi = yaw[i]
            col = 0
            # --- trajectory: future root offset + facing, local ---
            for h in C.TRAJ_HORIZONS:
                j = min(i + h, b)
                feat[i, col:col + 2] = _local(xy[j] - xy[i], yi); col += 2
                face = np.array([np.cos(yaw[j]), np.sin(yaw[j])])
                feat[i, col:col + 2] = _local(face, yi); col += 2
            # --- pose: feet pos/vel + root vel, local ---
            inxt = min(i + 1, b)
            root = qpos[i, 0:3]
            for k in range(feet.shape[1]):
                fp = feet[i, k] - root
                feat[i, col:col + 2] = _local(fp[:2], yi); feat[i, col + 2] = fp[2]; col += 3
            for k in range(feet.shape[1]):
                fv = (feet[inxt, k] - feet[i, k]) / C.DT
                feat[i, col:col + 2] = _local(fv[:2], yi); feat[i, col + 2] = fv[2]; col += 3
            rv = (xy[inxt] - xy[i]) / C.DT
            feat[i, col:col + 2] = _local(rv, yi)
            feat[i, col + 2] = (z[inxt] - z[i]) / C.DT
    return feat


def standardize(feat, traj_w=1.0, pose_w=1.0):
    """Z-score + per-group weighting. Returns (feat_std, mean, std, weight)."""
    mean, std = feat.mean(0), feat.std(0) + 1e-6
    w = np.concatenate([np.full(TRAJ_DIM, traj_w), np.full(POSE_DIM, pose_w)]).astype(np.float32)
    return ((feat - mean) / std) * w, mean, std, w

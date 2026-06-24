"""Unitree G1 model constants, FK helpers and heading math for EMM.

Mirrors the role of the reference EMM ``g1_model`` but binds to *this* repo's
MuJoCo model + dataset (``mm_g1``). The qpos layout (36) is
``[root_pos(3), root_quat_wxyz(4), dof(29)]`` -- identical to ``mm_g1`` and the
dataset -- so the EMM database and the ``mm_g1`` library share one skeleton.

The root body (the floating base, ``pelvis``) is NOT ground-projected in our
data, so body *height* features are measured against the world floor (z = 0),
see :mod:`emm_g1.database`.
"""

import os
import numpy as np

from . import quat
from mm_g1 import config as _mmC   # reuse the existing model/data paths (read-only)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

# Reuse the existing self-contained G1 scene + dataset (mm_g1 is untouched).
MODEL_PATH = _mmC.SCENE_XML
DATA_DIR = _mmC.DATA_DIR             # GMR-retargeted LAFAN1 .pkl locomotion clips
JUMP_DATA_DIR = _mmC.JUMP_DATA_DIR   # walk->jump->walk .csv clips

NDOF = 29
FPS = _mmC.FPS
DT = 1.0 / FPS

# G1 pelvis local axes: +x forward, +z up (matches mm_g1.features.FORWARD/UP).
FORWARD = np.array([1.0, 0.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])

# Trajectory taps: future frames sampled for trajectory / ellipse / height
# features (~0.33 / 0.67 / 1.0 s ahead at 30 fps). Same horizons as mm_g1.
HORIZONS = np.array(_mmC.HORIZONS)

# Bodies used for pose features (feet), and the body that defines the footprint
# ellipse centre / character origin (pelvis == floating base).
FOOT_BODIES = list(_mmC.FOOT_BODIES)
PELVIS_BODY = 'pelvis'


def extent_body_ids(model):
    """All skeleton body ids used for the footprint-ellipse extent AND height
    range: every body except the world/base (id 0), matching the paper's use of
    all skeleton joints."""
    return list(range(1, model.nbody))


def load_model(path=MODEL_PATH):
    import mujoco
    return mujoco.MjModel.from_xml_path(path)


# ---------------------------------------------------------------------------
# Heading helpers (yaw-only, ground plane) -- identical math to mm_g1.features.
# ---------------------------------------------------------------------------

def yaw_quat(theta):
    """Quaternion (wxyz) for a rotation of ``theta`` about world +Z."""
    return quat.from_angle_axis(np.asarray(theta, dtype=float), UP)


def heading_dir(rootquat):
    """World-space forward direction (xy, z=0, normalized) of a root quaternion."""
    fwd = quat.mul_vec(rootquat, FORWARD)
    fwd = fwd * np.array([1.0, 1.0, 0.0])
    return fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)


def heading_angle(rootquat):
    d = heading_dir(rootquat)
    return np.arctan2(d[..., 1], d[..., 0])

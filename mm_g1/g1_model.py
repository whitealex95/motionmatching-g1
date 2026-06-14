"""MuJoCo G1 wrapper: qpos conversion + forward kinematics for features."""
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation as R

from . import config as C


def csv_to_qpos(rows):
    """Dataset rows [...,quat_xyzw,...] -> MuJoCo qpos [...,quat_wxyz,...]."""
    rows = np.atleast_2d(rows).astype(np.float64)
    q = rows.copy()
    q[:, 3:7] = rows[:, [6, 3, 4, 5]]  # xyzw -> wxyz
    return q


def quat_wxyz_yaw(quat_wxyz):
    """Heading (yaw about world Z) of a wxyz quaternion array (...,4)."""
    q = np.atleast_2d(quat_wxyz)
    xyzw = q[:, [1, 2, 3, 0]]
    return R.from_quat(xyzw).as_euler("xyz")[:, 2]


class G1Model:
    """Loads the menagerie G1 and exposes batched FK for feature extraction."""

    def __init__(self, xml=C.SCENE_XML):
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self.foot_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in C.FOOT_BODIES
        ]

    def fk_feet(self, qpos_seq):
        """World foot positions for each frame -> (T, n_feet, 3)."""
        qpos_seq = np.atleast_2d(qpos_seq)
        out = np.empty((len(qpos_seq), len(self.foot_ids), 3))
        for t, q in enumerate(qpos_seq):
            self.data.qpos[:] = q
            mujoco.mj_kinematics(self.model, self.data)
            for k, bid in enumerate(self.foot_ids):
                out[t, k] = self.data.xpos[bid]
        return out

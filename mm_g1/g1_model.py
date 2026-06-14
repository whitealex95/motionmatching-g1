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
        self._build_mirror_map()

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

    # --- sagittal mirror (GenoView-style left/right reflection) --------------
    def _build_mirror_map(self):
        """For each of the 29 hinge dofs, the qpos index it maps to under a sagittal
        (y->-y) reflection and its sign. Rule: swap left<->right joints; a rotation about
        the pitch axis (y) keeps sign, roll (x) / yaw (z) negate. Built from the model so
        it stays correct if the joint order ever changes."""
        m = self.model
        self._mir_src, self._mir_dst, self._mir_sign = [], [], []
        for j in range(m.njnt):
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            twin = (name.replace("left", "right") if "left" in name
                    else name.replace("right", "left") if "right" in name else name)
            tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, twin)
            sign = 1.0 if abs(m.jnt_axis[j][1]) > 0.5 else -1.0   # +1 pitch(y), -1 roll(x)/yaw(z)
            self._mir_src.append(m.jnt_qposadr[j])
            self._mir_dst.append(m.jnt_qposadr[tid])
            self._mir_sign.append(sign)
        self._mir_src = np.array(self._mir_src)
        self._mir_dst = np.array(self._mir_dst)
        self._mir_sign = np.array(self._mir_sign)

    def mirror_qpos(self, qpos_seq):
        """Left/right-mirror a (T,36) qpos sequence (wxyz). Root: negate y position and
        reflect the quaternion (w,x,y,z)->(w,-x,y,-z); joints: swap L/R with per-axis sign."""
        q = np.atleast_2d(qpos_seq).astype(np.float64)
        out = q.copy()
        out[:, 1] = -q[:, 1]                                   # root y -> -y
        out[:, 4] = -q[:, 4]; out[:, 6] = -q[:, 6]            # root quat (w,-x,y,-z)
        out[:, self._mir_dst] = self._mir_sign * q[:, self._mir_src]   # swap + sign joints
        return out

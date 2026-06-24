"""Feature extraction for EMM: pose, trajectory, footprint-ellipse and height.

Ports the paper's custom feature extractors to the G1:

  * ``EllipseFeatureExtractor`` -> :func:`footprint_ellipse` — an oriented
    ground-plane ellipse bounding the body, primary axis along motion.
  * ``HeightFeatureExtractor``  -> :func:`height_range` — min/max body height.

All world positions come from MuJoCo forward kinematics over the G1.
"""

import numpy as np

from . import quat
from . import g1_model as g1


def fk_body_positions(model, data, qpos_all, body_ids):
    """Run FK over every frame; return ``(T, B, 3)`` world positions for body_ids."""
    import mujoco
    T = len(qpos_all)
    out = np.zeros((T, len(body_ids), 3), np.float32)
    for t in range(T):
        data.qpos[:] = qpos_all[t]
        mujoco.mj_kinematics(model, data)
        for j, bid in enumerate(body_ids):
            out[t, j] = data.xpos[bid]
    return out


def fk_body_positions_one(model, data, qpos, body_ids):
    """FK for a single qpos; return ``(B, 3)``."""
    import mujoco
    data.qpos[:] = qpos
    mujoco.mj_kinematics(model, data)
    return np.array([data.xpos[bid] for bid in body_ids], np.float32)


def footprint_ellipse(body_xy, root_xy, primary_axis):
    """Oriented bounding ellipse of a body footprint about ``root_xy``.

    ``body_xy`` (..., B, 2), ``root_xy`` (..., 2), ``primary_axis`` (..., 2) unit.
    Returns ``(primary_extent, secondary_extent)`` (..., 2): the max absolute
    projection of the body points on the primary axis and its perpendicular.
    """
    perp = np.stack([-primary_axis[..., 1], primary_axis[..., 0]], axis=-1)
    rel = body_xy - root_xy[..., None, :]
    prim = np.abs(np.einsum('...bd,...d->...b', rel, primary_axis))
    sec = np.abs(np.einsum('...bd,...d->...b', rel, perp))
    return np.stack([prim.max(axis=-1), sec.max(axis=-1)], axis=-1)


def height_range(body_z, floor_z=0.0):
    """Min/max body height above the floor. ``body_z`` (..., B) -> (..., 2)."""
    return np.stack([body_z.min(axis=-1) - floor_z,
                     body_z.max(axis=-1) - floor_z], axis=-1)


def current_footprint_ellipse(body_xy, root_xy, facing_xy):
    """Runtime helper: world-frame ellipse of an agent's current pose.

    Returns ``(primary_axis(2), primary_extent, secondary_extent)`` where the
    primary axis is the (unit) facing direction. Used to publish an agent as an
    elliptical obstacle to its neighbours.
    """
    n = np.linalg.norm(facing_xy) + 1e-9
    axis = facing_xy / n
    ext = footprint_ellipse(body_xy[None], root_xy[None], axis[None])[0]
    return axis, float(ext[0]), float(ext[1])

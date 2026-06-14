"""Live keyboard input -> desired future trajectory used to query the motion database.

Each frame the viewer reports a desired (speed, heading); `predict_trajectory` slews the
heading toward that target at a fixed turn rate and integrates at the commanded speed to
produce a short predicted path, then expresses it in the character's local frame to match
the trajectory feature layout in features.py. Feeding the database a trajectory it must
"follow" is what makes motion matching responsive to the keys in real time.
"""
import numpy as np
from . import config as C
from .features import _local

MAX_H = max(C.TRAJ_HORIZONS)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def predict_trajectory(world_xy, world_yaw, speed, heading, turn_rate=C.TURN_RATE):
    """Predicted future path -> trajectory feature block (4*len(horizons),).

    world_xy / world_yaw : the character's current world pose.
    speed / heading      : the desired locomotion commanded by the keyboard this frame.
    """
    pos, head = np.asarray(world_xy, float).copy(), world_yaw
    traj = {0: (pos.copy(), head)}
    for f in range(1, MAX_H + 1):
        head = head + np.clip(_wrap(heading - head), -turn_rate * C.DT, turn_rate * C.DT)
        pos = pos + speed * C.DT * np.array([np.cos(head), np.sin(head)])
        traj[f] = (pos.copy(), head)
    block = []
    for h in C.TRAJ_HORIZONS:
        p, hd = traj[h]
        block += list(_local(p - world_xy, world_yaw))
        block += list(_local(np.array([np.cos(hd), np.sin(hd)]), world_yaw))
    return np.array(block, np.float32)

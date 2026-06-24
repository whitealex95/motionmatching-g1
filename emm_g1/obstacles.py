"""Obstacle model and the per-tap nearby-obstacle query.

Obstacles are circles ``(center, radius, height_range)`` or ellipses
``(center, axis, extents, height_range)`` on the ground plane. Dynamic obstacles
track a smoothed velocity and are queried at their *predicted future* position
for each trajectory tap (mirroring the paper's ``DynamicObstacle`` which spawns
future copies at ``v * horizon``).

Neighbour agents in a crowd are published Unity-style as BOTH a per-tap ELLIPSE
(at the neighbour's own predicted trajectory tap positions, oriented by its
footprint ellipse) AND a CIRCLE at the neighbour's CURRENT position.

The query (:meth:`Environment.get_nearby`) returns, per tap, the obstacles in the
searching character's heading-local frame, after per-tap culling against the
searching agent's predicted position at that tap (Unity ``GetNearbyObstacles``).
"""

import numpy as np

from . import quat
from . import g1_model as g1

MAXIMUM_ELLIPSE_LENGTH = 0.9       # Unity MaximumEllipseLength


class Obstacle:
    """A static or dynamic ground obstacle (circle or ellipse)."""

    def __init__(self, center, radius=None, axis=None, ext=None,
                 height=(-10.0, 10.0), dynamic=False, vel_smooth=0.9):
        self.center = np.asarray(center, float)[:2].copy()
        self.is_ellipse = radius is None
        self.radius = float(radius) if radius is not None else 0.0
        self.axis = np.asarray(axis, float) if axis is not None else np.array([1.0, 0.0])
        self.axis = self.axis / (np.linalg.norm(self.axis) + 1e-9)
        self.ext = np.asarray(ext, float) if ext is not None else np.array([0.3, 0.3])
        self.height = np.asarray(height, float)
        self.dynamic = dynamic
        self.vel_smooth = vel_smooth
        self.vel = np.zeros(2)              # per-frame displacement (m/frame)
        self._last = self.center.copy()

    def set_center(self, center):
        self.center = np.asarray(center, float)[:2].copy()

    def update_velocity(self):
        """Call once per frame after moving the obstacle (dynamic obstacles)."""
        if self.dynamic:
            self.vel = self.vel * self.vel_smooth + (self.center - self._last) * (1.0 - self.vel_smooth)
        self._last = self.center.copy()

    def future_center(self, horizon_frames):
        return self.center + self.vel * horizon_frames


class AgentEllipse:
    """A neighbour agent published as per-tap oriented ellipses (3 taps).

    ``taps`` (3, 2) world centres at the neighbour's predicted trajectory taps,
    ``axes`` (3, 2) unit primary axes, ``exts`` (3, 2) (primary, secondary).
    """

    def __init__(self, taps, axes, exts, height=(-10.0, 10.0)):
        self.taps = np.asarray(taps, float).reshape(3, 2)
        self.axes = np.asarray(axes, float).reshape(3, 2)
        self.exts = np.asarray(exts, float).reshape(3, 2)
        self.height = np.asarray(height, float)


class AgentCircle:
    """A neighbour agent published as a single circle at its current position."""

    def __init__(self, center, radius, height=(-10.0, 10.0)):
        self.center = np.asarray(center, float)[:2].copy()
        self.radius = float(radius)
        self.height = np.asarray(height, float)


class Environment:
    """A collection of obstacles plus dynamically-published agent ellipses."""

    def __init__(self, threshold=0.4, nearby_radius=3.0):
        self.threshold = threshold
        self.nearby_radius = nearby_radius
        self.obstacles = []           # persistent Obstacle objects
        self._transient = []          # per-frame AgentEllipse / AgentCircle

    def add(self, obs):
        self.obstacles.append(obs)
        return obs

    def clear_transient(self):
        self._transient = []

    def add_agent(self, taps, axes, exts, center, radius,
                  height=(-10.0, 10.0)):
        """Publish a neighbour agent Unity-style: a per-tap ellipse (at its own
        predicted taps) AND a circle at its current position."""
        self._transient.append(AgentEllipse(taps, axes, exts, height))
        self._transient.append(AgentCircle(center, radius, height))

    def add_agent_ellipse(self, taps, axes, exts, height=(-10.0, 10.0)):
        self._transient.append(AgentEllipse(taps, axes, exts, height))

    def add_agent_circle(self, center, radius, height=(-10.0, 10.0)):
        self._transient.append(AgentCircle(center, radius, height))

    def update_dynamics(self):
        for o in self.obstacles:
            o.update_velocity()

    def get_nearby(self, root_xy, root_yaw, height_mode=False, tap_world=None):
        """Return ``(circ, ell)`` — lists (len 3, per tap) of local-frame obstacles.

        ``circ[k]``: (Nk, 5) = [cx, cy, radius, hmin, hmax].
        ``ell[k]``:  (Mk, 8) = [cx, cy, axx, axy, extp, exts, hmin, hmax].
        Positions/axes are in the character heading-local frame.

        Per-tap culling (Unity ``GetNearbyObstacles``): an obstacle is kept for
        tap ``k`` only if its tap-position is within
        ``MaximumEllipseLength + ObstacleDistanceThreshold`` of the SEARCHING
        agent's predicted position at tap ``k`` (``tap_world[k]``), evaluated
        separately for ellipses vs circles. If ``tap_world`` is None the searching
        agent's current position is used for every tap.
        """
        root_xy = np.asarray(root_xy, float)[:2]
        qh = g1.yaw_quat(root_yaw)
        cull = MAXIMUM_ELLIPSE_LENGTH + self.threshold
        if tap_world is None:
            tap_world = np.repeat(root_xy[None], 3, axis=0)
        else:
            tap_world = np.asarray(tap_world, float).reshape(3, 2)

        def to_local_pt(p):
            v = np.array([p[0] - root_xy[0], p[1] - root_xy[1], 0.0])
            return quat.inv_mul_vec(qh, v)[0:2]

        def to_local_dir(d):
            v = np.array([d[0], d[1], 0.0])
            return quat.inv_mul_vec(qh, v)[0:2]

        circ = [[] for _ in range(3)]
        ell = [[] for _ in range(3)]

        # --- persistent obstacles: dynamic future-extrapolation per tap ---
        for o in self.obstacles:
            for k, h in enumerate(g1.HORIZONS):
                c = o.future_center(h)
                if np.linalg.norm(c - tap_world[k]) > cull:
                    continue
                cl = to_local_pt(c)
                if o.is_ellipse:
                    al = to_local_dir(o.axis)
                    al = al / (np.linalg.norm(al) + 1e-9)
                    ell[k].append([cl[0], cl[1], al[0], al[1], o.ext[0], o.ext[1],
                                   o.height[0], o.height[1]])
                else:
                    circ[k].append([cl[0], cl[1], o.radius, o.height[0], o.height[1]])

        # --- transient neighbour agents (Unity per-tap ellipse + circle) ---
        for o in self._transient:
            if isinstance(o, AgentEllipse):
                for k in range(3):
                    c = o.taps[k]
                    if np.linalg.norm(c - tap_world[k]) > cull:
                        continue
                    cl = to_local_pt(c)
                    al = to_local_dir(o.axes[k]); al = al / (np.linalg.norm(al) + 1e-9)
                    ell[k].append([cl[0], cl[1], al[0], al[1], o.exts[k, 0], o.exts[k, 1],
                                   o.height[0], o.height[1]])
            else:  # AgentCircle published at its current position to every tap
                for k in range(3):
                    if np.linalg.norm(o.center - tap_world[k]) > cull:
                        continue
                    cl = to_local_pt(o.center)
                    circ[k].append([cl[0], cl[1], o.radius, o.height[0], o.height[1]])

        circ = [np.array(c, float).reshape(-1, 5) for c in circ]
        ell = [np.array(e, float).reshape(-1, 8) for e in ell]
        return circ, ell

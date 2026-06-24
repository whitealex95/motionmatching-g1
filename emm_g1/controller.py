"""Environment-aware motion-matching controller (Ponton et al. 2025), G1 port.

One controller drives one kinematic G1. Each step it predicts a desired
trajectory from the input (spring smoothing), builds the query feature vector,
runs the environment-aware search (:mod:`emm_g1.search`) against the database,
inertializes into the matched frame, and integrates the root from the matched
clip's root velocity. The obstacle **penalization** with **height gating** is what
makes a low bar trigger a jump with no manual command. Direction-weight
**evasion** and environment **anticipation** reproduce the paper's adaptive
behaviour.

``step(left_stick, right_stick)`` returns a world-space ``qpos (36,)`` (wxyz
quat), the same layout ``mm_g1`` produces, so the viewer / web port are uniform.
"""

import numpy as np

from . import quat
from . import g1_model as g1
from . import springs as sp
from . import search as S
from . import features as feat   # noqa: F401  (kept for parity / external use)

# Static feature layout (27): [trajPos 0:6, trajDir 6:12, pose 12:27].
_TRAJPOS = slice(0, 6)
_TRAJDIR = slice(6, 12)
_POSE = slice(12, 27)


def default_weights():
    w = np.ones(27, np.float32)
    w[_TRAJPOS] = 1.0
    w[_TRAJDIR] = 1.5      # facing -- modulated down by evasion near obstacles
    w[12:15] = 1.0         # left foot position
    w[15:18] = 0.25        # left foot velocity
    w[18:21] = 1.0         # right foot position
    w[21:24] = 0.25        # right foot velocity
    w[24:27] = 0.25        # hips velocity
    return w


class EMMController:
    def __init__(self, db, env=None, weights=None, max_speed=1.7,
                 search_time=0.12, inert_halflife=0.10, vel_halflife=0.25,
                 rot_halflife=0.25, penalty_weight=18.0, evasion=None,
                 anticipation=2.0, start_frame=None):
        self.db = S.prepare(db)
        self.env = env
        self.height_mode = bool(np.asarray(db['height_mode']))
        self.w0 = (weights if weights is not None else default_weights()).astype(np.float32)
        self.w = self.w0.copy()
        self.start_dir_w = float(self.w0[_TRAJDIR][0])
        self.MAX_SPEED = max_speed
        self.SEARCH_TIME = search_time
        self.INERT = inert_halflife
        self.VELH = vel_halflife
        self.ROTH = rot_halflife
        self.penalty_weight = penalty_weight
        # Evasion default: 0.54 for the height (jump/crouch) scenario, else 0.01
        # (reference Search .asset values).
        self.evasion = (0.54 if self.height_mode else 0.01) if evasion is None else evasion
        self.anticipation = anticipation

        self.starts, self.stops = db['starts'], db['stops']
        self.Xn, self.Xmean, self.Xstd = db['Xn'], db['Xmean'], db['Xstd']
        self.dof, self.dofVel = db['dof'], db['dofVel']
        self.rootPosDB, self.rootQuatDB = db['rootPos'], db['rootQuat']
        self.thetaDB, self.rootVelDB, self.yawRateDB = db['theta'], db['rootVel'], db['yawRate']
        self.Ttimes = g1.HORIZONS / g1.FPS
        self._start_frame = start_frame
        self.spawn_xy = None        # if set, reset() places the root here
        self.spawn_yaw = 0.0
        self.reset(start_frame)

    def set_spawn(self, x, y, yaw=0.0):
        """Pin the reset (Space) pose to a world xy + heading (e.g. the lane start)."""
        self.spawn_xy = (float(x), float(y))
        self.spawn_yaw = float(yaw)
        self.reset()

    # -- state --
    def reset(self, start_frame=None):
        if start_frame is None:
            start_frame = self._start_frame
        f0 = int(self.starts[0] + 30) if start_frame is None else int(start_frame)
        self.animClip = int(np.searchsorted(self.stops, f0, side='right'))
        self.animFrame = f0
        self.rootPos = self.rootPosDB[f0].astype(float).copy()
        self.rootYaw = float(self.thetaDB[f0])
        if getattr(self, "spawn_xy", None) is not None:
            self.rootPos[0:2] = self.spawn_xy
            self.rootYaw = self.spawn_yaw
        self.rootRot = g1.yaw_quat(self.rootYaw)
        self.rootVel = np.zeros(3); self.rootAcc = np.zeros(3); self.rootAng = np.zeros(3)
        self.desiredDir = quat.mul_vec(self.rootRot, g1.FORWARD)
        self.offDof = np.zeros(g1.NDOF); self.offDofVel = np.zeros(g1.NDOF)
        self.offH = 0.0; self.offHVel = 0.0
        self.searchTimer = 0.0
        self.target_speed = 0.0
        self.w = self.w0.copy()
        self.Tpos = np.tile(self.rootPos, (len(g1.HORIZONS), 1))
        self.Tdir = np.tile(self.desiredDir, (len(g1.HORIZONS), 1))

    @property
    def cur(self):
        return self.animFrame

    @property
    def jumping(self):
        return bool(self.db['clip_is_jump'][self.animClip]) and self.rootPosDB[self.animFrame, 2] > 0.0

    def clip_name(self, f=None):
        return str(self.db['clip_names'][self._clip_of(self.animFrame if f is None else f)])

    def _clip_of(self, frame):
        return int(np.searchsorted(self.stops, frame, side='right'))

    def _query(self, Tpos, Tdir, qh_ctrl):
        pose = self.Xn[self.animFrame, _POSE] * self.Xstd[_POSE] + self.Xmean[_POSE]
        trajPos = quat.inv_mul_vec(qh_ctrl, Tpos - self.rootPos)[:, 0:2].ravel()
        trajDir = quat.inv_mul_vec(qh_ctrl, Tdir)[:, 0:2].ravel()
        q = np.concatenate([trajPos, trajDir, pose])
        return ((q - self.Xmean) / self.Xstd).astype(np.float32)

    # -- one real-time frame --
    def step(self, left_stick, right_stick):
        db, starts, stops = self.db, self.starts, self.stops
        desiredVel = self.MAX_SPEED * np.asarray(left_stick, float)
        self.target_speed = float(np.linalg.norm(desiredVel))
        if np.linalg.norm(right_stick) > 0.01:
            self.desiredDir = np.asarray(right_stick, float) / np.linalg.norm(right_stick)
        elif np.linalg.norm(left_stick) > 0.01:
            self.desiredDir = np.asarray(left_stick, float) / np.linalg.norm(left_stick)
        desiredRot = g1.yaw_quat(np.arctan2(self.desiredDir[1], self.desiredDir[0]))

        # Desired trajectory via springs.
        dt_col = self.Ttimes[:, None]
        Tpos, _, _ = sp.trajectory_spring_position(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, self.VELH, dt_col)
        Trot, _ = sp.trajectory_spring_rotation(
            self.rootRot, self.rootAng, desiredRot, self.ROTH, dt_col)
        Tdir = quat.mul_vec(Trot, g1.FORWARD)
        self.Tpos, self.Tdir = Tpos, Tdir

        # Nearby obstacles, per trajectory tap.
        circ = [np.zeros((0, 5))] * 3
        ell = [np.zeros((0, 8))] * 3
        if self.env is not None:
            circ, ell = self.env.get_nearby(self.rootPos[0:2], self.rootYaw,
                                            self.height_mode, tap_world=Tpos[:, 0:2])

        # Search (rate limited): env-aware nearest pose.
        if self.searchTimer <= 0.0:
            qh_ctrl = g1.yaw_quat(self.rootYaw)
            Xq = self._query(Tpos, Tdir, qh_ctrl)
            pw = self.penalty_weight * self.anticipation * max(0.5, self.target_speed)
            best = S.search_env(db, Xq, self.w, pw, circ, ell,
                                self.env.threshold if self.env else 0.6,
                                self.height_mode)
            if best != self.animFrame:
                self.offDof = (self.offDof + self.dof[self.animFrame]) - self.dof[best]
                self.offDofVel = (self.offDofVel + self.dofVel[self.animFrame]) - self.dofVel[best]
                self.offH = (self.offH + self.rootPosDB[self.animFrame, 2]) - self.rootPosDB[best, 2]
                self.animFrame = best
                self.animClip = self._clip_of(best)
            self.searchTimer = self.SEARCH_TIME

        # Evasion: shrink the facing weight when an obstacle is close.
        self._update_evasion(circ, ell)

        # Advance playhead.
        self.animFrame = int(np.clip(self.animFrame + 1,
                                     starts[self.animClip], stops[self.animClip] - 1))
        self.searchTimer -= g1.DT
        if self.animFrame >= stops[self.animClip] - 2:
            self.searchTimer = 0.0

        # Integrate controller root from the matched clip's root velocity.
        _, _, self.rootAcc = sp.trajectory_spring_position(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, self.ROTH, g1.DT)
        qh_clip = g1.yaw_quat(self.thetaDB[self.animFrame])
        clipVelLocal = quat.inv_mul_vec(qh_clip, self.rootVelDB[self.animFrame])
        self.rootVel = quat.mul_vec(self.rootRot, clipVelLocal)
        self.rootAng = np.array([0.0, 0.0, self.yawRateDB[self.animFrame]])
        self.rootPos = self.rootPos + self.rootVel * g1.DT
        self.rootYaw = self.rootYaw + self.yawRateDB[self.animFrame] * g1.DT
        self.rootRot = g1.yaw_quat(self.rootYaw)

        # Inertialized pose.
        self.offDof, self.offDofVel = sp.decay_spring_damper(
            self.offDof, self.offDofVel, self.INERT, g1.DT)
        self.offH, self.offHVel = sp.decay_spring_damper(
            self.offH, self.offHVel, self.INERT, g1.DT)
        f = self.animFrame
        dofOut = self.dof[f] + self.offDof
        tilt = quat.mul(quat.inv(qh_clip), self.rootQuatDB[f])
        rootQuatOut = quat.mul(self.rootRot, tilt)

        qpos = np.empty(36)
        qpos[0:2] = self.rootPos[0:2]
        qpos[2] = self.rootPosDB[f, 2] + self.offH
        qpos[3:7] = rootQuatOut
        qpos[7:] = dofOut
        return qpos

    def _update_evasion(self, circ, ell):
        """Lerp the facing-weight down near the closest obstacle (reference
        ``OnSearchCompleted``)."""
        from . import ellipse_geom as eg
        thr = self.env.threshold if self.env else 0.4
        closest = np.inf
        f = self.animFrame
        for k in range(3):
            center = self.db['posTaps'][f, k]
            ax = self.db['ellAxis'][f, k]
            ext = np.array([self.db['ellExtP'][f, k], self.db['ellExtS'][f, k]])
            if len(circ[k]):
                d, _ = eg.point_to_ellipse(center, ax, ext, circ[k][:, 0:2])
                d = np.maximum(d - circ[k][:, 2], eg.MAX_INSIDE)
                dmin = float(d.min())
                if dmin < thr:
                    closest = min(closest, dmin)
            for j in range(len(ell[k])):
                d, _, _ = eg.ellipse_to_ellipse(center, ax, ext, ell[k][j, 0:2],
                                                ell[k][j, 2:4], ell[k][j, 4:6])
                if d < thr:
                    closest = min(closest, d)
        target = self.start_dir_w
        if np.isfinite(closest) and closest < thr:
            factor = np.log10(max(closest / thr, 1e-3)) + 1.0
            target = self.start_dir_w * max(self.evasion, factor)
            rate = np.clip(g1.DT * 10.0, 0.0, 1.0)
        else:
            rate = np.clip(g1.DT * 100.0, 0.0, 1.0)
        cur = self.w[_TRAJDIR][0]
        self.w[_TRAJDIR] = cur + (target - cur) * rate

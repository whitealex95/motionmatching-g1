"""Real-time motion matching, GenoView (Holden "Simple Motion Matching") heuristics.

A faithful port of ../GenoViewPython-MotionMatching/genoview_g1.py's MotionMatcher driving
our G1 library one frame at a time. A per-clip Savitzky-Golay-smoothed "simulation root"
(ground position + facing) is the thing the matcher tracks and integrates; the pelvis is
stored as a local offset of it. Every SEARCH_TIME we nearest-neighbour search per-clip
KD-trees (biased toward staying put), inertialize joints + pelvis-local pos/rot toward the
winner, integrate the matched clip's smooth root velocity through the world, then place the
pelvis back on that root. All math + hyperparameters are identical to genoview_g1.py; the
extras here are (a) the L/R-mirrored database and (b) the J-triggered jump skill (jump frames
are kept out of the search; a jump is entered only via its run-up).
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from . import quat
from .features import build_db, yaw_quat, FORWARD, HORIZONS, FPS
from .jumps import jump_entries
from .springs import (DecaySpringDamperPosition, DecaySpringDamperRotation,
                      TrajectorySpringPosition, TrajectorySpringRotation)

DT = C.DT
NDOF = 29


class MotionMatcher:
    def __init__(self, lib, start_frame=None):
        self.lib = lib
        db = self.db = build_db(lib)
        self.starts, self.stops = db["starts"], db["stops"]
        self.X = db["X"]
        self.dof, self.dofVel = db["dof"], db["dofVel"]
        self.simPosDB, self.simThetaDB = db["simPos"], db["simTheta"]
        self.simVelDB, self.yawRateDB = db["simVel"], db["yawRate"]
        self.plpDB, self.plvDB = db["pelvLocalPos"], db["pelvLocalVel"]
        self.prDB, self.paDB = db["pelvLocalRot"], db["pelvLocalAng"]
        self.clip_id = lib["clip_id"]
        self.skill = lib["skill"] if "skill" in lib else np.zeros(len(self.X), np.int32)
        self.Ttimes = HORIZONS / FPS

        # Per-clip KD-trees over LOCOMOTION clips only (skill==0 everywhere), each trimming
        # the last HORIZONS[-1] frames so a full future trajectory always exists. Jump clips
        # are excluded so locomotion never matches into a jump -- it only happens on command.
        self.search = []        # (clip_index, tree, range_start)
        searchable = []
        for ci, (rs, re) in enumerate(zip(self.starts, self.stops)):
            if self.skill[rs:re].any() or re - rs <= HORIZONS[-1]:
                continue
            self.search.append((ci, cKDTree(self.X[rs:re - HORIZONS[-1]]), rs))
            searchable.append(re - rs - HORIZONS[-1])
        self.valid = np.empty(int(np.sum(searchable)), int)   # count of searchable frames

        # Pre-take-off run-up frames available to the J trigger (continuing jumps only).
        self.jump_enter, self.jump_land_of = jump_entries(lib)
        self.reset(start_frame)

    # --- state ---------------------------------------------------------------
    def reset(self, start_frame=None):
        if start_frame is None:
            start_frame = min(self.stops[0] - 1, self.starts[0] + 30)
        self.animRange = int(np.searchsorted(self.starts, start_frame, "right") - 1)
        self.animFrame = int(start_frame)
        # Controller root = the smoothed simulation root (ground position + yaw).
        self.rootPos = self.simPosDB[self.animFrame].copy()
        self.rootVel = np.zeros(3); self.rootAcc = np.zeros(3); self.rootAng = np.zeros(3)
        self.rootYaw = float(self.simThetaDB[self.animFrame])
        self.rootRot = yaw_quat(self.rootYaw)
        self.desiredDir = quat.mul_vec(self.rootRot, FORWARD)
        # Inertialization offsets: joints, pelvis-local position, pelvis-local rotation.
        self.offDof = np.zeros(NDOF); self.offDofVel = np.zeros(NDOF)
        self.offPP = np.zeros(3); self.offPPVel = np.zeros(3)
        self.offPR = np.array([1.0, 0.0, 0.0, 0.0]); self.offPAng = np.zeros(3)
        self.searchTimer = 0.0
        self.jump_pending = False
        self.jump_locked = 0
        self.Tpos = np.tile(self.rootPos, (len(HORIZONS), 1))   # command-trajectory viz
        self.Tdir = np.tile(self.desiredDir, (len(HORIZONS), 1))

    # --- jump skill ----------------------------------------------------------
    def trigger_jump(self):
        """Request a jump (J key). Honoured on the next step if not already jumping."""
        if self.jump_locked == 0:
            self.jump_pending = True

    @property
    def jumping(self):
        return self.jump_locked > 0

    @property
    def cur(self):
        return self.animFrame

    def _best_jump_entry(self):
        """Pre-take-off `ready` run-up frame whose features best match the current frame,
        so the jump is entered from the run-up (not mid-air) with a smooth take-off."""
        if len(self.jump_enter) == 0:
            return None
        d = np.linalg.norm(self.X[self.jump_enter] - self.X[self.animFrame], axis=1)
        return int(self.jump_enter[d.argmin()])

    def _inertialize_into(self, b, rng):
        """Capture the pose discontinuity from the current frame `a` to frame `b` -- joints,
        pelvis-local position, and pelvis-local rotation -- as decaying inertialization
        offsets, then switch the playhead there (no pop)."""
        a = self.animFrame
        self.offDof = (self.offDof + self.dof[a]) - self.dof[b]
        self.offDofVel = (self.offDofVel + self.dofVel[a]) - self.dofVel[b]
        self.offPP = (self.offPP + self.plpDB[a]) - self.plpDB[b]
        self.offPPVel = (self.offPPVel + self.plvDB[a]) - self.plvDB[b]
        self.offPR = quat.abs(quat.mul_inv(quat.mul(self.offPR, self.prDB[a]), self.prDB[b]))
        self.offPAng = (self.offPAng + self.paDB[a]) - self.paDB[b]
        self.animRange, self.animFrame = rng, b

    # --- one real-time frame -------------------------------------------------
    def step(self, speed, heading):
        """Advance one frame. speed [m/s] and heading [rad] are the desired locomotion this
        frame; returns the world-space qpos (36,) to display."""
        starts, stops, X = self.starts, self.stops, self.X

        # ---- Desired velocity / facing (springs need a target each frame) ----
        desiredVel = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
        if speed > 0.01:
            self.desiredDir = np.array([np.cos(heading), np.sin(heading), 0.0])
        desiredRot = yaw_quat(np.arctan2(self.desiredDir[1], self.desiredDir[0]))

        # ---- Predict desired trajectory (critically-damped springs) ----
        dt_col = self.Ttimes[:, None]
        self.Tpos, _, _ = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, C.VEL_HALFLIFE, dt_col)
        Trot, _ = TrajectorySpringRotation(
            self.rootRot, self.rootAng, desiredRot, C.ROT_HALFLIFE, dt_col)
        self.Tdir = quat.mul_vec(Trot, FORWARD)

        # ---- Jump trigger: inertialize into the best `ready` run-up, then lock ----
        if self.jump_pending and self.jump_locked == 0:
            self.jump_pending = False
            entry = self._best_jump_entry()
            if entry is not None:
                rng = int(np.searchsorted(starts, entry, "right") - 1)
                self._inertialize_into(entry, rng)
                land = self.jump_land_of[entry]
                after_end = min(land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER, stops[rng] - 1)
                self.jump_locked = max(1, after_end - entry)
                self.searchTimer = C.SEARCH_TIME

        # ---- Search (skipped while riding a jump) ----
        if self.jump_locked == 0 and self.searchTimer <= 0.0:
            qh_ctrl = yaw_quat(self.rootYaw)
            Xq = self._runtime_features(qh_ctrl)
            bestRange, bestFrame = self.animRange, self.animFrame
            if bestFrame < stops[bestRange] - HORIZONS[-1]:
                best = float(np.linalg.norm(Xq - X[bestFrame]) - C.CURRENT_BIAS)  # stay-in-clip bias
            else:
                best = np.inf
            for ci, tree, rs in self.search:
                dist, k = tree.query(Xq, eps=C.APPROX_BIAS, distance_upper_bound=best)
                if dist < best:
                    best, bestRange, bestFrame = dist, ci, int(rs + k)
            if bestRange != self.animRange or bestFrame != self.animFrame:
                self._inertialize_into(bestFrame, bestRange)   # seamless inertialized cut
            self.searchTimer = C.SEARCH_TIME

        # ---- Advance the playhead (30 fps data, 30 fps render) ----
        self.animFrame = int(np.clip(self.animFrame + 1,
                                     starts[self.animRange], stops[self.animRange] - 1))
        self.searchTimer -= DT
        if self.jump_locked > 0:
            self.jump_locked -= 1
            if self.jump_locked == 0:
                self.searchTimer = 0.0                         # search out of the jump at once
        elif self.animFrame >= stops[self.animRange] - 2:
            self.searchTimer = 0.0
        f = self.animFrame

        # ---- Integrate controller root from the matched clip's smooth root velocity ----
        _, _, self.rootAcc = TrajectorySpringPosition(
            self.rootPos, self.rootVel, self.rootAcc, desiredVel, C.ROT_HALFLIFE, DT)
        qh_clip = yaw_quat(self.simThetaDB[f])
        clipVelLocal = quat.inv_mul_vec(qh_clip, self.simVelDB[f])
        self.rootVel = quat.mul_vec(self.rootRot, clipVelLocal)
        self.rootAng = np.array([0.0, 0.0, self.yawRateDB[f]])
        self.rootPos = self.rootPos + self.rootVel * DT
        self.rootYaw = self.rootYaw + self.yawRateDB[f] * DT
        self.rootRot = yaw_quat(self.rootYaw)

        # ---- Inertialize joints + pelvis-local offset, then reconstruct the pose ----
        self.offDof, self.offDofVel = DecaySpringDamperPosition(
            self.offDof, self.offDofVel, C.INERT_HALFLIFE, DT)
        self.offPP, self.offPPVel = DecaySpringDamperPosition(
            self.offPP, self.offPPVel, C.INERT_HALFLIFE, DT)
        self.offPR, self.offPAng = DecaySpringDamperRotation(
            self.offPR, self.offPAng, C.INERT_HALFLIFE, DT)

        dofOut = self.dof[f] + self.offDof
        pelvLocalPos = self.plpDB[f] + self.offPP
        pelvLocalRot = quat.mul(self.offPR, self.prDB[f])
        # Place the pelvis on the controller's smooth root (which carries the heading).
        pelvWorldPos = self.rootPos + quat.mul_vec(self.rootRot, pelvLocalPos)
        pelvWorldRot = quat.mul(self.rootRot, pelvLocalRot)

        qpos = np.empty(36)
        qpos[0:3] = pelvWorldPos
        qpos[3:7] = pelvWorldRot
        qpos[7:] = dofOut
        return qpos

    def _runtime_features(self, qh_ctrl):
        """Query = current frame's pose blocks (from X) + the desired trajectory, normalized
        the same way as the database (genoview runtime_features)."""
        Xoffset, Xscale = self.db["Xoffset"], self.db["Xscale"]
        pose = self.X[self.animFrame, 0:15] * Xscale[0:15] + Xoffset[0:15]   # de-normalized pose
        trajPos = quat.inv_mul_vec(qh_ctrl, self.Tpos - self.rootPos)[:, 0:2].ravel()
        trajDir = quat.inv_mul_vec(qh_ctrl, self.Tdir)[:, 0:2].ravel()
        q = np.concatenate([pose, trajPos, trajDir])
        return (q - Xoffset) / Xscale

"""Real-time motion matching: an incremental, keyboard-driven nearest-neighbour search.

`MotionMatcher.generate` in the original offline project rolls a fixed command schedule
out for N seconds. Here we expose the same per-frame logic as `step(speed, heading)`, so
a live input loop (the viewer) can drive it one frame at a time:

  query = [ trajectory-predicted-from-the-keys | pose-of-the-current-frame ]

We search for the nearest database frame and, with hysteresis, either continue the
current clip or jump to that frame. Root motion is stitched continuously and pose pops
are cross-faded, exactly as in the offline matcher.
"""
import numpy as np
from scipy.spatial import cKDTree

from . import config as C
from . import features as F
from .commands import predict_trajectory
from .jumps import jump_entries
from .kinematics import transform_qpos, alignment_to, blend_qpos


class MotionMatcher:
    def __init__(self, lib, traj_w=1.0, pose_w=1.0, min_z=0.6, jump_margin=0.35,
                 start_frame=0):
        self.lib = lib
        self.raw = F.compute_features(lib)
        self.feat, self.mean, self.std, self.w = F.standardize(self.raw, traj_w, pose_w)
        self.xy, self.yaw, self.qpos = lib["qpos"][:, 0:2], lib["yaw"], lib["qpos"]
        self.fic, self.lengths = lib["frame_in_clip"], lib["lengths"]
        self.clip_id = lib["clip_id"]
        self.jump_margin = jump_margin
        # skill: 0 = locomotion, 1 = jump (ready..after). Jump frames are kept out of the
        # search pool so locomotion never matches into a jump; a jump only happens when the
        # user triggers it, entering via its `ready` run-up (see step / trigger_jump).
        self.skill = lib["skill"] if "skill" in lib else np.zeros(len(self.qpos), np.int32)
        # Search only over upright LOCOMOTION frames (never land in a degenerate/jump pose)
        # that are also at least SEARCH_TAIL frames from their clip's end -- GenoView's
        # cKDTree(X[rs:re-60]): the tail still plays, but the match can't land there.
        tail_ok = (self.lengths[self.clip_id] - 1 - self.fic) >= C.SEARCH_TAIL
        self.valid = np.where((lib["qpos"][:, 2] >= min_z) & tail_ok & (self.skill == 0))[0]
        self.tree = cKDTree(self.feat[self.valid])
        # Pre-take-off run-up frames available to the J trigger (continuing jumps only).
        self.jump_enter, self.jump_land_of = jump_entries(lib)
        self.reset(start_frame)

    # --- state ---------------------------------------------------------------
    def reset(self, start_frame=0):
        """Start the character at the world origin, facing +x, on `start_frame`."""
        self.cur = int(start_frame)
        self.dyaw = -self.yaw[self.cur]
        self.pivot = self.xy[self.cur].copy()
        self.offset = np.zeros(2)
        self.frozen = None
        self.blend_left = 0
        self.step_count = 0
        self.jump_pending = False      # set by trigger_jump(); consumed on the next step
        self.jump_locked = 0           # >0 while riding a jump clip (no search)

    def _is_clip_end(self, i):
        return self.fic[i] >= self.lengths[self.clip_id[i]] - 1

    # --- jump skill ----------------------------------------------------------
    def trigger_jump(self):
        """Request a jump (J key). Honoured on the next step if not already jumping."""
        if self.jump_locked == 0:
            self.jump_pending = True

    @property
    def jumping(self):
        return self.jump_locked > 0

    def _best_jump_entry(self, cur):
        """Pre-take-off `ready` run-up frame whose features best match the current frame,
        so the jump is entered from the run-up (not mid-air) with a smooth take-off."""
        if len(self.jump_enter) == 0:
            return None
        d = np.linalg.norm(self.feat[self.jump_enter] - self.feat[cur], axis=1)
        f = int(self.jump_enter[d.argmin()])
        return f, self.jump_land_of[f]

    def _qstd(self, traj_block, cur):
        """Standardized query: command trajectory + current frame's pose features."""
        query = np.concatenate([traj_block, self.raw[cur, F.TRAJ_DIM:]])
        return ((query - self.mean) / self.std) * self.w

    # --- one real-time frame -------------------------------------------------
    def step(self, speed, heading):
        """Advance one frame under the live command (speed [m/s], heading [rad]).

        Returns the world-space qpos (36,) to display this frame. Hysteresis: at a search
        we only jump to the nearest neighbour if it is clearly better (by jump_margin) than
        simply continuing the current clip, which keeps the character on long continuous
        fragments -> less jitter / foot-skating.
        """
        cur = self.cur
        world = transform_qpos(self.qpos[cur], self.dyaw, self.pivot, self.offset)[0]
        cwx, cwy = world[0:2].copy(), self.yaw[cur] + self.dyaw   # current world xy + heading

        if self.blend_left > 0:                                   # cross-fade a recent pop
            world = blend_qpos(self.frozen, world, 1 - self.blend_left / C.BLEND_FRAMES)
            self.blend_left -= 1

        # --- jump trigger: snap into the best `ready` run-up and lock the clip in ---
        if self.jump_pending and self.jump_locked == 0:
            self.jump_pending = False
            je = self._best_jump_entry(cur)
            if je is not None:
                entry, land = je
                self.dyaw, self.pivot, self.offset = alignment_to(
                    self.xy[entry], self.yaw[entry], cwx, cwy)
                self.frozen, self.blend_left = world.copy(), C.BLEND_FRAMES
                after_end = land + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER   # ready..after
                self.cur = entry
                self.jump_locked = max(1, after_end - entry)
                self.step_count += 1
                return world

        # --- ride an in-progress jump through landing (take-off/flight/touchdown) ---
        if self.jump_locked > 0:
            if not self._is_clip_end(cur):
                self.cur = cur + 1
            self.jump_locked -= 1
            self.step_count += 1
            return world

        step = self.step_count
        if step > 0 and (step % C.MM_SEARCH_INTERVAL == 0 or self._is_clip_end(cur)):
            block = predict_trajectory(cwx, cwy, speed, heading)
            qstd = self._qstd(block, cur)
            dist_best, vi = self.tree.query(qstd)
            best = int(self.valid[vi])
            end = self._is_clip_end(cur)
            # continuing costs the query's distance to the next frame's features
            cont = 1e9 if end else float(np.linalg.norm(qstd - self.feat[cur + 1]))
            if end or dist_best < cont * (1 - self.jump_margin):
                jump = not (self.clip_id[best] == self.clip_id[cur] and 0 <= best - cur <= 2)
                self.dyaw, self.pivot, self.offset = alignment_to(
                    self.xy[best], self.yaw[best], cwx, cwy)
                if jump:
                    self.frozen, self.blend_left = world.copy(), C.BLEND_FRAMES
                self.cur = best
            elif not end:
                self.cur = cur + 1
        elif not self._is_clip_end(cur):
            self.cur = cur + 1

        self.step_count += 1
        return world

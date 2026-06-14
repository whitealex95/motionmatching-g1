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
        # Search only over upright frames (never land in a degenerate pose) that are also at
        # least SEARCH_TAIL frames from their clip's end -- GenoView's cKDTree(X[rs:re-60]):
        # the tail still plays, but the match can't land there and run off the clip.
        tail_ok = (self.lengths[self.clip_id] - 1 - self.fic) >= C.SEARCH_TAIL
        self.valid = np.where((lib["qpos"][:, 2] >= min_z) & tail_ok)[0]
        self.tree = cKDTree(self.feat[self.valid])
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

    def _is_clip_end(self, i):
        return self.fic[i] >= self.lengths[self.clip_id[i]] - 1

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

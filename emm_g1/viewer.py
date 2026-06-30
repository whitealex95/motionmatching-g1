"""Interactive GLFW + MuJoCo viewer for the EMM (auto-jump) demo.

Same window / follow-camera / held-key handling as ``mm_g1.viewer`` but it drives
the environment-aware :class:`emm_g1.controller.EMMController` and draws the hurdle
walls. There is NO jump key -- the G1 hops over a wall on its own when the
env-aware search decides a jump clip clears the obstacle's height band.

Controls
  W / A / S / D ........ move (relative to the camera)
  Arrow keys ........... face direction, independent of travel
  Shift (hold) ......... walk instead of run
  Space ................ reset to the start pose
  T .................... toggle the command-trajectory gizmo
  Left-drag / right-drag / scroll ... orbit / pan / zoom
  Esc .................. quit
"""
import math
import numpy as np
import glfw
import mujoco

from . import g1_model as g1
from . import config as EC

_TRAJ_RGBA = np.array([0.9, 0.1, 0.1, 1.0], np.float32)
_WALL_RGBA = np.array([0.86, 0.32, 0.18, 0.92], np.float32)
_TRAJ_Z = 0.05
_SPHERE_R = 0.05
_STICK_LEN = 0.25
_STICK_W = 0.012

_MOVE_KEYS = {glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D}
_FACE_KEYS = {glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT}

WALK_SCALE = EC.WALK_SCALE   # Shift -> walk pace (same as mm_g1 / index.html)


def _axis_mat(ax, ay):
    a = np.array([ax, ay, 0.0]); a /= (np.linalg.norm(a) + 1e-9)
    b = np.array([-a[1], a[0], 0.0])
    return np.stack([a, b, [0.0, 0.0, 1.0]], axis=1)


class EMMViewer:
    def __init__(self, model, data, ctrl, walls=None, width=1280, height=720,
                 title="EMM G1 - WASD to move; the G1 jumps obstacles on its own"):
        self.model, self.data, self.ctrl = model, data, ctrl
        self.walls = walls if walls is not None else EC.obstacle_dicts()

        if not glfw.init():
            raise RuntimeError("glfw.init() failed -- this viewer needs a display.")
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create a GLFW window (no display available?).")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        # Behind the spawn, looking down the +x hurdle lane (the hurdles are ahead in
        # view), so WASD is intuitive: W walks forward into the screen toward the
        # hurdles, A/D strafe across the lane. (The old default, azimuth 135, ran the
        # lane diagonally so W drifted off it.) Drag to orbit.
        self.cam.azimuth = 0.0; self.cam.elevation = -12.0
        self.cam.distance = 4.5; self.cam.lookat[:] = [0.0, 0.0, 0.8]

        self.held = set(); self.shift = False
        self.show_traj = True; self._speed = 0.0
        self._mouse_last = None; self._button = {"left": False, "right": False}
        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # --- input callbacks ---
    def _on_key(self, window, key, scancode, action, mods):
        self.shift = bool(mods & glfw.MOD_SHIFT)
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.ctrl.reset()
            elif key == glfw.KEY_T:
                self.show_traj = not self.show_traj
            elif key in _MOVE_KEYS or key in _FACE_KEYS:
                self.held.add(key)
        elif action == glfw.RELEASE:
            self.held.discard(key)

    def _on_mouse_button(self, window, button, action, mods):
        press = action == glfw.PRESS
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._button["left"] = press
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._button["right"] = press
        self._mouse_last = glfw.get_cursor_pos(window) if press else None

    def _on_cursor(self, window, xpos, ypos):
        if self._mouse_last is None:
            return
        dx = xpos - self._mouse_last[0]; dy = ypos - self._mouse_last[1]
        self._mouse_last = (xpos, ypos)
        w, h = glfw.get_window_size(window)
        if self._button["left"]:
            act = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._button["right"]:
            act = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.model, act, dx / h, dy / h, self.scene, self.cam)

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoffset, self.scene, self.cam)

    # --- per-frame command -> (left_stick, right_stick) ---
    def _command(self):
        # "into the screen" heading: at the default azimuth 0 this is world +x, so W
        # walks the character forward down the +x lane toward the hurdles it can see
        # ahead (and the arrow keys face that way too).
        fwd = math.radians(self.cam.azimuth)
        right = fwd - math.pi / 2.0
        fdir = np.array([math.cos(fwd), math.sin(fwd), 0.0])
        rdir = np.array([math.cos(right), math.sin(right), 0.0])
        move = np.zeros(3)
        if glfw.KEY_W in self.held: move += fdir
        if glfw.KEY_S in self.held: move -= fdir
        if glfw.KEY_D in self.held: move += rdir
        if glfw.KEY_A in self.held: move -= rdir
        face = np.zeros(3)
        if glfw.KEY_UP in self.held:    face += fdir
        if glfw.KEY_DOWN in self.held:  face -= fdir
        if glfw.KEY_RIGHT in self.held: face += rdir
        if glfw.KEY_LEFT in self.held:  face -= rdir
        m = np.linalg.norm(move)
        # left_stick: unit direction * stick magnitude in [0,1] (controller scales
        # by MAX_SPEED). Shift -> walk pace.
        left = move / m * (WALK_SCALE if self.shift else 1.0) if m > 1e-6 else np.zeros(3)
        f = np.linalg.norm(face)
        right_stick = face / f if f > 1e-6 else np.zeros(3)
        return left, right_stick

    # --- main loop ---
    def run(self):
        last = glfw.get_time(); acc = 0.0
        while not glfw.window_should_close(self.window):
            now = glfw.get_time(); acc += now - last; last = now
            while acc >= g1.DT:
                left, right = self._command()
                self._speed = float(np.linalg.norm(left)) * self.ctrl.MAX_SPEED
                world = self.ctrl.step(left, right)
                self.data.qpos[:] = world
                mujoco.mj_forward(self.model, self.data)
                acc -= g1.DT
            self.cam.lookat[0] = float(self.data.qpos[0])
            self.cam.lookat[1] = float(self.data.qpos[1])

            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            self._draw_walls()
            if self.show_traj:
                self._draw_command()
            mujoco.mjr_render(viewport, self.scene, self.ctx)
            self._overlay(viewport)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()

    # --- decorations ---
    def _next_geom(self):
        if self.scene.ngeom >= self.scene.maxgeom:
            return None
        g = self.scene.geoms[self.scene.ngeom]; self.scene.ngeom += 1
        return g

    def _draw_walls(self):
        for o in self.walls:
            g = self._next_geom()
            if g is None:
                return
            zc = 0.5 * (o['hmin'] + o['hmax']); hh = max(0.05, 0.5 * (o['hmax'] - o['hmin']))
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_BOX,
                np.array([o['half_len'], o['half_thick'], hh], float),
                np.array([o['cx'], o['cy'], zc], float),
                _axis_mat(o['ax'], o['ay']).reshape(9), _WALL_RGBA)

    def _draw_command(self):
        for (px, py, _), (dx, dy, _) in zip(self.ctrl.Tpos, self.ctrl.Tdir):
            base = np.array([px, py, _TRAJ_Z])
            self._add_sphere(base, _SPHERE_R)
            self._add_stick(base, base + _STICK_LEN * np.array([dx, dy, 0.0]))

    def _add_sphere(self, pos, radius):
        g = self._next_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([radius, 0.0, 0.0]), np.asarray(pos, float),
                            np.eye(3).flatten(), _TRAJ_RGBA)

    def _add_stick(self, p0, p1):
        g = self._next_geom()
        if g is None:
            return
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                            np.zeros(3), np.zeros(3), np.eye(3).flatten(), _TRAJ_RGBA)
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, _STICK_W,
                             np.asarray(p0, float), np.asarray(p1, float))

    def _overlay(self, viewport):
        s = self._speed
        gait = "JUMP" if self.ctrl.jumping else \
               ("RUN" if s > self.ctrl.MAX_SPEED * 0.7 else ("WALK" if s > 1e-3 else "IDLE"))
        title = f"{gait}   {s:.1f} m/s   (env-aware: jumps obstacles automatically)"
        body = (f"clip: {self.ctrl.clip_name()}\n"
                f"frame: {self.ctrl.cur}\n"
                f"command gizmo: {'on' if self.show_traj else 'off'} (T)\n"
                "WALK (Shift) into a wall and the G1 hops it -- no jump key\n"
                "WASD move | arrows face | Shift walk | Space reset\n"
                "drag orbit | right-drag pan | scroll zoom | Esc quit")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport, title, body, self.ctx)

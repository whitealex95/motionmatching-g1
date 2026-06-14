"""Interactive GLFW + MuJoCo viewer: drive the G1 with the keyboard in real time.

We open our own GLFW window (rather than mujoco.viewer) so we get true held-key state
from the GLFW key callback -- press/repeat/release -- which is what "hold W to keep
walking" needs. Each rendered frame we read the held keys, turn them into a desired
(speed, heading) relative to the camera, advance the motion matcher one frame, write the
resulting qpos into MjData and draw. A follow-camera keeps the character centred.

Controls
  W / A / S / D ........ move (forward / left / back / right), relative to the camera
  Shift (hold) ......... run instead of walk
  Space ................ reset to the start pose at the origin
  Left-drag ............ orbit camera     Right-drag ... pan     Scroll ... zoom
  Esc .................. quit
"""
import math
import numpy as np
import glfw
import mujoco

from . import config as C


# Movement keys -> bit in a held-key set.
_MOVE_KEYS = {glfw.KEY_W, glfw.KEY_A, glfw.KEY_S, glfw.KEY_D}


class InteractiveViewer:
    def __init__(self, model, data, matcher, width=1280, height=720,
                 title="Motion Matching G1 - WASD to move, Shift to run"):
        self.model = model
        self.data = data
        self.matcher = matcher

        if not glfw.init():
            raise RuntimeError(
                "glfw.init() failed -- this viewer needs a display. On a headless host, "
                "run on a machine with a GPU/X display (MUJOCO_GL=glfw)."
            )
        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create a GLFW window (no display available?).")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)   # vsync

        # MuJoCo visualization objects.
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        # Follow-camera: orbit around the character at a comfortable height.
        self.cam.azimuth = 135.0
        self.cam.elevation = -20.0
        self.cam.distance = 4.0
        self.cam.lookat[:] = [0.0, 0.0, 0.8]

        # Input state.
        self.held = set()
        self.shift = False
        self.last_heading = 0.0
        self._mouse_last = None
        self._button = {"left": False, "right": False}

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor)
        glfw.set_scroll_callback(self.window, self._on_scroll)

    # --- input callbacks -----------------------------------------------------
    def _on_key(self, window, key, scancode, action, mods):
        self.shift = bool(mods & glfw.MOD_SHIFT)
        if action == glfw.PRESS:
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE:
                self.matcher.reset()
                self.last_heading = 0.0
            elif key in _MOVE_KEYS:
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
        dx = xpos - self._mouse_last[0]
        dy = ypos - self._mouse_last[1]
        self._mouse_last = (xpos, ypos)
        w, h = glfw.get_window_size(window)
        if self._button["left"]:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
        elif self._button["right"]:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return
        mujoco.mjv_moveCamera(self.model, action, dx / h, dy / h, self.scene, self.cam)

    def _on_scroll(self, window, xoffset, yoffset):
        mujoco.mjv_moveCamera(self.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                              0.0, -0.05 * yoffset, self.scene, self.cam)

    # --- per-frame command from the keys -------------------------------------
    def _command(self):
        """Map held WASD (camera-relative) to a desired (speed, heading)."""
        fwd = math.radians(self.cam.azimuth + 180.0)   # ground heading "into the screen"
        right = fwd - math.pi / 2.0
        vx = vy = 0.0
        if glfw.KEY_W in self.held:
            vx += math.cos(fwd); vy += math.sin(fwd)
        if glfw.KEY_S in self.held:
            vx -= math.cos(fwd); vy -= math.sin(fwd)
        if glfw.KEY_D in self.held:
            vx += math.cos(right); vy += math.sin(right)
        if glfw.KEY_A in self.held:
            vx -= math.cos(right); vy -= math.sin(right)
        if abs(vx) < 1e-6 and abs(vy) < 1e-6:
            return 0.0, self.last_heading       # idle: keep facing, command zero speed
        heading = math.atan2(vy, vx)
        self.last_heading = heading
        return (C.RUN_SPEED if self.shift else C.WALK_SPEED), heading

    # --- main loop -----------------------------------------------------------
    def run(self):
        last = glfw.get_time()
        acc = 0.0
        speed = 0.0
        while not glfw.window_should_close(self.window):
            now = glfw.get_time()
            acc += now - last
            last = now

            # Advance the matcher at a fixed 30 Hz, independent of render rate.
            while acc >= C.DT:
                speed, heading = self._command()
                world = self.matcher.step(speed, heading)
                self.data.qpos[:] = world
                mujoco.mj_forward(self.model, self.data)
                acc -= C.DT

            # Follow-camera tracks the character root.
            self.cam.lookat[0] = float(self.data.qpos[0])
            self.cam.lookat[1] = float(self.data.qpos[1])

            w, h = glfw.get_framebuffer_size(self.window)
            viewport = mujoco.MjrRect(0, 0, w, h)
            mujoco.mjv_updateScene(self.model, self.data, self.opt, None, self.cam,
                                   mujoco.mjtCatBit.mjCAT_ALL, self.scene)
            mujoco.mjr_render(viewport, self.scene, self.ctx)
            self._overlay(viewport, speed)

            glfw.swap_buffers(self.window)
            glfw.poll_events()
        glfw.terminate()

    def _overlay(self, viewport, speed):
        gait = "RUN" if (speed > (C.WALK_SPEED + C.RUN_SPEED) / 2) else \
               ("WALK" if speed > 1e-3 else "IDLE")
        clip = self.matcher.lib["clip_names"][self.matcher.clip_id[self.matcher.cur]]
        title = f"{gait}   {speed:.1f} m/s"
        body = (f"clip: {clip}\n"
                "WASD move | Shift run | Space reset\n"
                "drag orbit | right-drag pan | scroll zoom | Esc quit")
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,
                           mujoco.mjtGridPos.mjGRID_TOPLEFT, viewport,
                           title, body, self.ctx)

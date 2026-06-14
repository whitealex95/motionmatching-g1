"""Shared constants: paths, skeleton layout, and motion-matching settings.

Everything is resolved relative to this file so the project is fully relocatable --
clone the folder anywhere and the model, data and cache paths still line up.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "g1")
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")   # built on first run, then cached

FPS = 30
DT = 1.0 / FPS

# qpos layout (36-D), shared by the dataset and MuJoCo (same joint order):
#   [0:3]  root position (x, y, z) in world metres
#   [3:7]  root orientation quaternion -- DATASET stores xyzw, MuJoCo qpos stores wxyz
#          (csv_to_qpos / transform_qpos do the reorder; see g1_model.py)
#   [7:36] 29 joint angles (radians)
JOINTS = slice(7, 36)

# Foot bodies used for motion-matching pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# Motion-matching feature config.
TRAJ_HORIZONS = [10, 20, 30]   # future sample frames (~0.33 / 0.67 / 1.0 s ahead)
MM_SEARCH_INTERVAL = 15        # frames between nearest-neighbour searches (~0.5 s; fewer pops)
BLEND_FRAMES = 12              # cross-fade length at a transition (~0.4 s)

# Every LAFAN1 clip begins and ends in a T-pose (arms out) that blends into the motion
# over ~1.5 s. We DROP the first/last TRIM frames of every clip so the T-pose never
# appears in the library or the generated motion.
TRIM = 45

# The locomotion library: one walk + one run clip (subject5, G1-retargeted LAFAN1).
# A speed command then steers motion matching smoothly between standing-walk and run.
CLIPS = ["walk1_subject5", "run1_subject5"]

# Keyboard-control speeds (m/s) used to build the command trajectory each frame.
WALK_SPEED = 1.2               # base speed when a direction key is held
RUN_SPEED = 2.6                # speed while Shift is also held (selects the run clip)
TURN_RATE = 3.0               # rad/s the predicted heading slews toward the input direction

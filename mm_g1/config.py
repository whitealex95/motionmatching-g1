"""Shared constants: paths, skeleton layout, and motion-matching settings.

Everything is resolved relative to this file so the project is fully relocatable --
clone the folder anywhere and the model, data and cache paths still line up.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "gmr_lafan1_g1")   # GMR-retargeted LAFAN1 .pkl clips
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

# Every LAFAN1 clip begins and ends in a T-pose (arms out) that blends into the motion.
# GenoView crops this with per-clip, hand-picked start:stop frame indices rather than a
# single fixed amount; we mirror that intent here. GenoView's BVH database is 60 fps and
# its starts are walk=160 / run=172 frames (~2.67 / 2.87 s); our CSVs are 30 fps, so we
# halve those to 80 / 86 frames. Tails keep the original ~1.5 s (45-frame) T-pose crop.
# CLIP_TRIM maps clip name -> (head_frames, tail_frames) dropped from the database.
DEFAULT_TRIM = (45, 45)
CLIP_TRIM = {
    "walk1_subject5": (80, 45),            # GenoView head start 160 @60fps -> 80 @30fps (~2.67 s)
    "run1_subject5":  (86, 45),            # GenoView head start 172 @60fps -> 86 @30fps (~2.87 s)
    "pushAndStumble1_subject5": (45, 45),  # full clip minus T-pose ends; deep stumbles are
                                           # dropped from SEARCH anyway by the min_z filter.
}

# GenoView also drops the LAST second of each clip from the SEARCH only (cKDTree(X[rs:re-60])
# at 60 fps): the tail still plays out, but the match never lands there, so it can't run off
# the end of a clip. We replicate that as a search-only exclusion of SEARCH_TAIL frames.
SEARCH_TAIL = 30   # frames (1.0 s @30fps) excluded from the KD-tree at each clip's end

# The locomotion library: GMR-retargeted LAFAN1 clips (subject5) -- walk, run, and
# push-and-stumble. A speed command then steers motion matching smoothly between
# standing-walk, run, and the stumble-recovery variety.
CLIPS = ["walk1_subject5", "run1_subject5", "pushAndStumble1_subject5"]

# Keyboard-control speeds (m/s) used to build the command trajectory each frame.
WALK_SPEED = 1.2               # base speed when a direction key is held
RUN_SPEED = 2.6                # speed while Shift is also held (selects the run clip)
TURN_RATE = 3.0               # rad/s the predicted heading slews toward the input direction

"""Shared constants: paths, skeleton layout, and motion-matching settings.

Everything is resolved relative to this file so the project is fully relocatable --
clone the folder anywhere and the model, data and cache paths still line up.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "gmr_lafan1_g1")   # GMR-retargeted LAFAN1 .pkl clips
JUMP_DATA_DIR = os.path.join(ROOT, "data", "g1_jump")     # CAMDM walk->jump->walk .csv clips
SCENE_XML = os.path.join(ROOT, "assets", "unitree_g1", "scene.xml")
LIB_PATH = os.path.join(ROOT, "data", "motion_lib.npz")   # built on first run, then cached

FPS = 30
DT = 1.0 / FPS

# qpos layout (36-D), shared by the dataset and MuJoCo (same joint order):
#   [0:3]  root position (x, y, z) in world metres
#   [3:7]  root orientation quaternion -- DATASET stores xyzw, MuJoCo qpos stores wxyz
#          (csv_to_qpos / mirror_qpos handle the reorder; see g1_model.py)
#   [7:36] 29 joint angles (radians)
JOINTS = slice(7, 36)

# Foot bodies used for motion-matching pose features (names from menagerie g1.xml).
FOOT_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]

# --- Motion-matching: GenoView (Holden "Simple Motion Matching") heuristics ----------
# All math + hyperparameters mirror ../GenoViewPython-MotionMatching/genoview_g1.py.
HORIZONS = [10, 20, 30]        # future trajectory taps (frames) ~0.33 / 0.67 / 1.0 s @30fps
TRAJ_HORIZONS = HORIZONS       # (alias kept for any external reference)
SEARCH_TIME = 0.15             # seconds between database searches
INERT_HALFLIFE = 0.075         # inertialization (pose-transition) blend half-life
VEL_HALFLIFE = 0.2             # desired-trajectory position spring half-life
ROT_HALFLIFE = 0.2             # desired-trajectory rotation spring half-life
CURRENT_BIAS = 0.01            # stay-in-clip bias seeded onto the current frame's distance
APPROX_BIAS = 0.01             # cKDTree eps: slightly approximate (faster) nearest-neighbour

# Savitzky-Golay windows for the smoothed "simulation root" (genoview's 31/61 @60fps,
# time-matched to our 30 fps). Smooths per-step bob/sway out of the root the matcher tracks.
ROOT_POS_SMOOTH = 15
ROOT_DIR_SMOOTH = 31

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

# GenoView trims the last HORIZONS[-1] frames of each clip from the SEARCH only
# (cKDTree(X[rs:re-30])): the tail still plays out, but a match never lands there, so a
# full future trajectory always exists and the playhead can't run off the clip end.
SEARCH_TAIL = HORIZONS[-1]   # frames excluded from each clip's KD-tree (1.0 s @30fps)

# The locomotion library: GMR-retargeted LAFAN1 clips (subject5) -- walk, run, and
# push-and-stumble. Each clip is added twice (normal + L/R MIRRORED, GenoView-style) for
# symmetric left/right coverage. A desired-velocity command then steers motion matching
# smoothly between standing-walk, run, and the stumble-recovery variety.
CLIPS = ["walk1_subject5", "run1_subject5", "pushAndStumble1_subject5"]
MIRROR = True                  # append a left/right-mirrored copy of every clip

# Desired locomotion speed (m/s) fed to the trajectory springs. Full stick = MAX_SPEED
# (run pace; the G1 run clip peaks ~4.3 m/s); holding Shift scales it to a walk -- exactly
# GenoView's 5.0 m/s & 0.4 scale.
MAX_SPEED = 5.0
WALK_SCALE = 0.4

# --- Jump skill (triggered with J) ----------------------------------------------
# CAMDM walk->jump->walk clips (in JUMP_DATA_DIR, 30 fps CSVs, same 36-D layout). They
# are appended to the library and phase-labeled; a jump is ENTERED only from its run-up
# (the `ready` phase) and ridden through landing, never matched into during locomotion.
JUMP_CLIPS = ["walk_jump_walk", "walk_jump_walk2", "walk_jump_stop"]
JUMP_FOOT_THR = 0.13     # m: both feet above this == airborne (flight detection)

# Five phases carved around each detected flight (frame counts @30fps). A jump is entered
# only in `ready` (the run-up before push-off) and exited only after `after` (recovery).
JUMP_PHASES = ["walk", "ready", "takeoff", "flight", "touchdown", "after"]
PHASE_READY = 12         # run-up before push-off -- the only place a jump can be entered
PHASE_TAKEOFF = 10       # ground push-off / loading just before lift-off
PHASE_TOUCHDOWN = 6      # landing impact, just after the feet hit
PHASE_AFTER = 18         # landing absorption / recovery walk -- the only place to exit

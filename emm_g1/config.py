"""EMM tuning + the interactive obstacle layout (shared by MuJoCo and the web).

Jump is the *only* environment target: every obstacle here is a LOW bar whose
height band sits below the G1's standing body but is cleared by a jump's flight
phase, so the env-aware search hops over it automatically (no trigger).
"""

import numpy as np

from . import obstacles as OB
from mm_g1 import config as _mmC

# --- Controller defaults (reference height/jump scenario) ----------------------
# Walk/run speed is sourced from the mm_g1 (index.html) demo so the two demos move
# at the SAME pace: full stick = MAX_SPEED (run), Shift = MAX_SPEED*WALK_SCALE (walk).
# NOTE: the dataset's jump clips are walk->jump->walk (no sprint-jump), so the
# auto-jump fires at WALK pace (hold Shift into a wall); at full run the matcher
# tracks the run clip and does not hop.
MAX_SPEED = _mmC.MAX_SPEED      # 5.0 m/s -- identical to mm_g1 / index.html
WALK_SCALE = _mmC.WALK_SCALE    # 0.4    -- Shift -> walk pace (identical to mm_g1)
SEARCH_TIME = 0.12       # seconds between env-aware searches
INERT_HALFLIFE = 0.10
VEL_HALFLIFE = 0.25
ROT_HALFLIFE = 0.25
PENALTY_WEIGHT = 60.0    # base obstacle-penalization weight (tuned for clean hop)
EVASION = 0.54           # facing-weight floor near obstacles (height scenario)
ANTICIPATION = 2.0       # scales penalty influence with desired speed

# --- Obstacle field (the "adequate obstacles" added to the interactive scene) --
# Each hurdle is a LOW, THIN, WIDE WALL: an ellipse footprint long along the wall
# (lateral) and thin across it, with an absolute height band (hmin, hmax) above
# the floor. A standing G1 (~0..1.3 m tall) overlaps the band, so walking into one
# is penalized -- and because the wall is too wide to comfortably step around, the
# cheapest option the search finds is a JUMP clip, whose airborne phase lifts the
# whole body above HMAX and clears the height gate. (An isolated small circle is
# just side-stepped, so we use walls.) Thin walls matter: a too-thick band
# over-penalizes and the character stalls in the run-up instead of committing.
OBSTACLE_THRESHOLD = 0.4         # log-barrier distance threshold t (metres)
NEARBY_RADIUS = 4.0
PENALTY_W = PENALTY_WEIGHT       # alias

WALL_HEIGHT = (0.0, 0.30)        # low bar: clearly below standing body height
WALL_HALF_LEN = 2.5              # half-length ALONG the wall (lateral span 5 m)
WALL_HALF_THICK = 0.28           # half-thickness ACROSS the wall (thin!)

# A hurdle LANE down +x: hold W from spawn and the G1 clears each in turn. A wall
# is (cx, cy, axis_x, axis_y) -- ``axis`` is the wall's long (lateral) direction.
HURDLE_WALLS = [
    (3.0, 0.0, 0.0, 1.0),
    (6.5, 0.0, 0.0, 1.0),
    (10.0, 0.0, 0.0, 1.0),
]


def build_environment():
    """A fresh :class:`emm_g1.obstacles.Environment` with the hurdle lane."""
    env = OB.Environment(threshold=OBSTACLE_THRESHOLD, nearby_radius=NEARBY_RADIUS)
    for cx, cy, ax, ay in HURDLE_WALLS:
        env.add(OB.Obstacle(center=[cx, cy], axis=[ax, ay],
                            ext=[WALL_HALF_LEN, WALL_HALF_THICK], height=WALL_HEIGHT))
    return env


def obstacle_dicts():
    """Plain-data obstacle list for the web export / renderer (matches
    build_environment). ``axis`` is the wall's long axis; ``ext`` = (half_len,
    half_thick); a drawable box is len x thick x (hmax-hmin)."""
    return [dict(cx=float(cx), cy=float(cy), ax=float(ax), ay=float(ay),
                 half_len=WALL_HALF_LEN, half_thick=WALL_HALF_THICK,
                 hmin=WALL_HEIGHT[0], hmax=WALL_HEIGHT[1])
            for cx, cy, ax, ay in HURDLE_WALLS]

# Interactive Motion Matching for the Unitree G1

Steer a **Unitree G1** humanoid around a MuJoCo scene in real time with the keyboard.
Hold **WASD** and a [motion-matching](https://www.gdcvault.com/play/1023280/Motion-Matching-and-The-Road)
search stitches retargeted LAFAN1 **walk** and **run** clips into one continuous,
responsive gait — no neural network, no training, just nearest-neighbour search over a
pose/trajectory feature database.

This repo is **fully self-contained**: the G1 model, the motion data, and the code all
live in this one folder. Clone it, run `setup.sh`, and go.

```
W / A / S / D    move (relative to the camera)
Shift (hold)     run instead of walk
Space            reset to the start pose
left-drag        orbit camera
right-drag       pan camera
scroll           zoom
Esc              quit
```

## Quick start

```bash
git clone <this-repo> motionmatchin-g1
cd motionmatchin-g1
./setup.sh                      # makes .venv, installs deps, builds the cache
source .venv/bin/activate
python run.py                   # opens the window — WASD to move
```

The first launch builds a feature cache (`data/motion_lib.npz`, ~1 s); later launches
start instantly. The viewer needs a display — run it on a desktop or an X-forwarded
session with `MUJOCO_GL=glfw` (the default).

Already have a MuJoCo Python environment? Skip `setup.sh`:

```bash
pip install -r requirements.txt
python run.py
```

## How it works

Each frame the loop does four things:

1. **Read the keys → a desired trajectory.** Held WASD become a desired
   `(speed, heading)` relative to the camera. `predict_trajectory` slews the heading
   toward the input at a fixed turn rate and integrates forward to a short predicted
   path (`mm_g1/commands.py`).
2. **Build a query and search.** The query is `[ predicted trajectory | the current
   frame's pose features ]`, standardized the same way as the database. A KD-tree
   returns the nearest library frame (`mm_g1/controller.py`).
3. **Continue or jump, with hysteresis.** We only switch to the nearest neighbour when
   it is clearly better (by `jump_margin`) than continuing the current clip — this keeps
   the character on long continuous fragments, so the motion stays smooth.
4. **Stitch and blend.** The chosen frame is placed into the world by a planar (SE2)
   alignment so the root path is C0-continuous; pose pops at a switch are cross-faded
   over ~0.4 s (`mm_g1/kinematics.py`).

The feature vector (27-D, `mm_g1/features.py`) is computed entirely in each frame's
root-local frame so matching is heading-invariant:

| block      | dims | contents                                                |
|------------|------|---------------------------------------------------------|
| trajectory | 12   | future root offset + facing at +10/+20/+30 frames       |
| pose       | 15   | local foot positions (2×3), foot velocities (2×3), root velocity (3) |

Searching every `MM_SEARCH_INTERVAL` frames (default 15, ~0.5 s) rather than every frame
reduces pops; commanding a higher speed (Shift) pulls the match into the **run** clip,
a lower speed back into the **walk** clip.

## Layout

```
motionmatchin-g1/
├── run.py                       # entry point: python run.py
├── setup.sh                     # venv + install + build cache (self-contained)
├── requirements.txt
├── mm_g1/
│   ├── config.py                # paths, FPS, joint layout, feature + speed settings
│   ├── g1_model.py              # CSV→qpos conversion, quaternion yaw, FK for the feet
│   ├── data.py                  # build / load + cache the walk+run motion library
│   ├── features.py              # 27-D motion-matching feature vectors
│   ├── commands.py              # keyboard input → predicted query trajectory
│   ├── kinematics.py            # planar root stitching + pose cross-fade
│   ├── controller.py            # real-time nearest-neighbour matcher: step(speed, heading)
│   └── viewer.py                # GLFW + MuJoCo window, held-key input, follow-camera
├── assets/unitree_g1/           # MuJoCo G1 model (g1.xml, scene.xml, STL meshes)
└── data/g1/                     # G1-retargeted LAFAN1 clips (walk1_subject5, run1_subject5)
```

## Tuning

Edit `mm_g1/config.py`:

- `WALK_SPEED` / `RUN_SPEED` — commanded speeds (m/s) for walk and Shift-run.
- `TURN_RATE` — how fast the predicted heading chases the input direction.
- `MM_SEARCH_INTERVAL` — frames between searches (lower = more reactive, more pops).
- `CLIPS` — which clips form the library (drop extra CSVs into `data/g1/` and list them
  here; delete `data/motion_lib.npz` to rebuild).
- `--jump-margin` on `run.py` — hysteresis strength (higher = stickier clips, smoother).

## Credits

- **G1 model** — [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
  `unitree_g1` (license in `assets/unitree_g1/LICENSE`).
- **Motion data** — [LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)
  (Ubisoft La Forge), retargeted to the G1.
- **Approach** — adapted from a MuJoCo G1 motion-matching / motion-graph project and the
  real-time, keyboard-driven control of GenoView's motion-matching demo.

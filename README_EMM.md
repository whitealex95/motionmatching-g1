# Environment-aware Motion Matching (EMM) — auto-jump variant

An **additive, backward-compatible** variant of this repo's motion matcher that
implements [Ponton et al. 2025, *Environment-aware Motion
Matching*](https://doi.org/10.1145/3763334) (ACM TOG 44(6), Art. 232) on the
Unitree G1. Instead of pressing **J** to jump, the G1 **hops over low obstacles
on its own** when one is in its path.

Nothing in `mm_g1/`, `run.py`, `docs/index.html` or `docs/js/{mm,main,fk,quat}.js`
is modified — the EMM code lives entirely in new files and reuses the **same
dataset** (`data/gmr_lafan1_g1` locomotion + `data/g1_jump` jumps).

```
python run_emm.py          # MuJoCo window: WASD to walk the hurdle lane; the G1 jumps each wall
docs/emm.html              # the same demo in the browser (no jump key)
```

## How the auto-jump works (no trigger, no tagging)

Standard motion matching searches for the pose whose trajectory + pose features
best match your input. EMM adds two per-pose feature blocks and one extra search
term:

1. **Footprint ellipse** (`ellipse`, per future trajectory tap) — the body's
   ground-plane footprint.
2. **Body height range** (`height`, per tap) — the min/max body height **above
   the floor**. During a jump's airborne phase the *lowest* body part lifts above
   a low wall.
3. **Obstacle penalization** — each candidate pose is scored as `static distance +
   Σ log-barrier penalty` over nearby obstacles. In **height mode** a candidate is
   only penalized by an obstacle whose height band its body actually overlaps. A
   walking pose (body from the floor up) collides with a low wall and is
   penalized; a jump clip's flight pose clears the wall's height band and is *not*
   penalized — so the env-aware search selects the jump automatically.

The obstacles are **low, thin, wide walls** (a hurdle lane): a small isolated
circle is just side-stepped in open ground, so a wall — too wide to comfortably
go around — makes the jump the cheapest option the search can find. This is a
faithful port of the reference EMM "height" scenario, tuned for our dataset.

> Note vs. the paper: the reference projects its root onto the ground, so it
> measures body height relative to that root. Our pelvis root is **not**
> ground-projected, so `emm_g1/database.py` measures height against the world
> floor (`z = 0`), which is what an absolute-height obstacle band needs.

## Layout (all new files)

```
emm_g1/                         # the EMM Python package (imports mm_g1 read-only)
├── config.py                   # EMM tuning + the hurdle-wall layout
├── g1_model.py                 # model/dataset bindings, heading math
├── database.py                 # build data/emm_lib.npz from the SAME dataset (+ ellipse/height)
├── ellipse_geom.py             # ellipse distance + log-barrier penalty kernel
├── obstacles.py                # Obstacle / Environment / per-tap nearby query
├── search.py                   # env-aware branch-and-bound search with height gating
├── controller.py               # EMMController.step(left, right) -> qpos(36)
├── viewer.py                   # GLFW MuJoCo viewer (draws the walls; no jump key)
├── springs.py / quat.py / features.py   # spring + math helpers (reference ports)
run_emm.py                      # entry point: python run_emm.py
tools/export_emm_web_data.py    # export docs/data/emm.{json,bin} for the web demo
docs/emm.html                   # browser demo entry
docs/js/emm/                    # JS env-aware matcher (1:1 port of emm_g1/*)
└── ellipse_geom.js obstacles.js search.js springs.js controller.js main.js
```

The first launch of `run_emm.py` (or the export script) builds the EMM feature
cache `data/emm_lib.npz` (~30 k frames, MuJoCo FK over all bodies). The web demo
reuses the shared `docs/data/model.json` + `mesh.*`; rebuild the EMM data with
`python tools/export_emm_web_data.py`.

## Controls

```
W / A / S / D    move (relative to the camera)
Arrow keys       face direction, independent of travel
Shift (hold)     walk instead of run
Space            reset to the start of the lane
T                toggle the command-trajectory gizmo
Esc              quit (MuJoCo)
```

There is **no jump key** — walk into a wall and the G1 jumps it.

## Tuning (`emm_g1/config.py`)

- `PENALTY_WEIGHT` — obstacle-penalization strength (≈60; higher = more eager to jump).
- `EVASION` / `ANTICIPATION` — facing relaxation near obstacles / penalty scaling with speed.
- `WALL_HEIGHT` / `WALL_HALF_LEN` / `WALL_HALF_THICK` — hurdle geometry (low, wide, thin).
- `HURDLE_WALLS` — the obstacle layout shared by MuJoCo and the web export.
```

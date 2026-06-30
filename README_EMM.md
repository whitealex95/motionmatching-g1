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
docs/emm/                  # the same demo in the browser (served at /emm)
```

## How the auto-jump works (separate skill buckets, obstacle-triggered)

Locomotion and the jump are kept in **separate skill buckets**, exactly like
`mm_g1`'s J-key jump — *not* merged into one nearest-neighbour search. Merging
them is why a jump was sometimes missed: from a walking query the jump clip's
pose is far in feature space (large static distance), so even with the obstacle
penalty the walk pose usually won the single search and the hop never fired.

Instead:

1. **Locomotion bucket** — the motion-matching search runs over locomotion frames
   **only** (`cand_mask`); it still carries the env-aware footprint-ellipse +
   body-height features and the obstacle **penalization / evasion** so the walk
   steers around things, but it can never match into a jump clip.
2. **Jump bucket** (`emm_g1/jumps.py`, `emm_g1/controller.py`) — a distinct skill
   entered through its **run-up**: when a wall whose height band the standing body
   overlaps crosses `JUMP_TRIGGER_DIST` metres ahead, the controller seeks the
   run-up ('ready') frame that best continues the current pose, then **locks** the
   search and rides the clip through take-off → flight → landing (mm_g1's exact
   mechanism). The obstacle distance simply replaces the J key, and the take-off
   is auto-timed off that distance so the flight's whole-body-clear window lands
   over the wall. The jump-clip phases (take-off / apex / landing) are detected
   from each clip's pelvis-height hop, since the EMM feature DB carries no phase
   tags.

The obstacles are **low, thin, wide walls** (a hurdle lane): too wide to step
around, so the jump is the natural way past. Verified on the 3-wall lane — the G1
clears all three (every body part stays above the 0.30 m band over each wall); see
`media/emm_3hurdles_*`.

> **Why this design?** Earlier the jump was merged into the single nearest-neighbour
> search and was often out-competed by the walk, so the G1 skipped hurdles. The
> diagnosis, the per-skill-bucket fix, and a note on wall **thickness** (the real
> clearance limit, not height) are written up in
> [`docs/EMM_AUTOJUMP_FIX.md`](docs/EMM_AUTOJUMP_FIX.md).

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
├── jumps.py                    # jump-bucket index: run-up entries + take-off/apex/landing
├── search.py                   # env-aware branch-and-bound search (loco-only via cand_mask)
├── controller.py               # EMMController.step(left, right) -> qpos(36); jump trigger
├── viewer.py                   # GLFW MuJoCo viewer (draws the walls; no jump key)
├── springs.py / quat.py / features.py   # spring + math helpers (reference ports)
run_emm.py                      # entry point: python run_emm.py
tools/export_emm_web_data.py    # export docs/data/emm.{json,bin} for the web demo
docs/emm/index.html             # browser demo entry (served at /emm)
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

There is **no jump key** — **walk** (hold Shift) into a wall and the G1 jumps it.

Walk/run speed matches the original `index.html` demo (`MAX_SPEED` and `WALK_SCALE`
are imported from `mm_g1/config.py`): full stick = run (5 m/s), Shift = walk
(2 m/s). The auto-jump fires at **walk** pace — the dataset's jump clips are
`walk→jump→walk` (no sprint-jump), so at full run the matcher tracks the run clip
and does not hop. Hold Shift to clear a hurdle.

## Tuning (`emm_g1/config.py`)

- `MAX_SPEED` / `WALK_SCALE` — imported from `mm_g1/config.py` so both demos move
  at the same walk/run pace; the jump fires at walk pace (see above).
- `JUMP_TRIGGER_DIST` — forward distance to a wall at which the jump fires (≈0.85 m;
  auto-times the take-off so the flight clears the wall). The jump is now a separate
  skill bucket, not a product of the penalty weight.
- `PENALTY_WEIGHT` — obstacle-penalization strength for the **locomotion** search
  (steering/evasion near obstacles); it no longer selects the jump.
- `EVASION` / `ANTICIPATION` — facing relaxation near obstacles / penalty scaling with speed.
- `WALL_HEIGHT` / `WALL_HALF_LEN` / `WALL_HALF_THICK` — hurdle geometry (low, wide, thin).
- `HURDLE_WALLS` — the obstacle layout shared by MuJoCo and the web export.
```

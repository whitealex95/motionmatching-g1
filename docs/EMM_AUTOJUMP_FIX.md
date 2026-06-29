# EMM auto-jump: why the G1 sometimes didn't jump, and the fix

> Companion to [`README_EMM.md`](../README_EMM.md). Records the debugging of the
> "G1 skips a hurdle" bug and the per-skill-bucket fix. All numbers below were
> measured on the 3-wall lane (`HURDLE_WALLS` at x = 3.0 / 6.5 / 10.0, walls
> `0.30 m` tall) with the G1 walking straight down it.

## Symptom

Walking the hurdle lane, the G1 cleared **only some** walls — typically the 3rd,
sometimes none of the first two — even though an obstacle was clearly in its path.

## How EMM picks a pose (recap)

Each frame the controller predicts a desired trajectory (3 future *taps* at
0.33 / 0.67 / 1.0 s), builds a query feature `Xq = [trajPos, trajDir, pose]`, and
searches the motion DB for the nearest pose. The env-aware variant adds, per tap,
a footprint **ellipse** and a body **height range**, and an obstacle
**penalization** (log-barrier, height-gated) so walking into a wall costs extra.

## Diagnosis (what the data actually showed)

A headless harness ran the lane and, at every search where an obstacle was near,
decomposed the candidate field into a **locomotion** bucket and a **jump** bucket
and printed each bucket's best `total = static + penalty`:

```
 rootX | LOCO total(stat+pen)  | JUMP total(stat+pen)  | winner
  2.24 |  0.07 ( 0.05+ 0.02)   |  2.81 ( 2.75+ 0.06)   |  loco   <- wall 1, no jump
  8.52 |  1.87 ( 1.86+ 0.00)   |  0.23 ( 0.23+ 0.00)   |  JUMP   <- wall 3, jumps
```

Two facts fell out:

1. **The search has no K-nearest cutoff.** Both `search.py` and the JS port scan
   *every* valid frame (exact branch-and-bound); the jump frame is always
   evaluated. So the original hypothesis ("the jump never enters the candidate
   set") was **not** the cause.
2. **The jump lost on the metric.** From a walking query the jump clip's pose is
   far in feature space (`static ≈ 1.7–2.8`), while the wall penalty on the walk
   pose peaked at only `~0.3` (a thin wall + the search's freedom to pick a
   low-penalty walking pose means there's always a near-zero-penalty walk). So
   `loco_total ≈ 0.07 < jump_total ≈ 2.8` — the walk always won. The 3rd wall only
   fired because a stray earlier jump had already put the character in a jump-like
   pose (self-reinforcing).

**Root cause:** locomotion and the jump were **merged into one nearest-neighbour
search**, and the two skills' static-distance scales are not comparable. A rare
skill (jump) can never win a single combined search against a well-matched walk,
no matter how the obstacle penalty is tuned.

## The fix: separate skill buckets (obstacle-triggered jump)

`mm_g1` already keeps the jump in its **own bucket** — jump frames are excluded
from the search and the skill is entered via the **J key**, seeking the best
run-up frame and then *locking* the search while the clip rides take-off → flight
→ landing. EMM had collapsed that separation; restoring it is the fix. The only
change vs `mm_g1` is the trigger: **an obstacle replaces the J key.**

- **Locomotion bucket** — the search runs over locomotion frames only
  (`search_env(..., cand_mask=loco_mask)`); it still carries the env features +
  penalty for steering/evasion, but can never match into a jump clip.
- **Jump bucket** (`emm_g1/jumps.py` + controller) — when a wall whose height band
  the standing body overlaps crosses `JUMP_TRIGGER_DIST` ahead, the controller
  seeks the run-up frame best matching the current pose, **locks** the search, and
  rides the clip through landing + recovery (`jump_locked`). Jump-clip phases
  (take-off / apex / landing) are detected from each clip's pelvis-height hop,
  since the EMM feature DB carries no phase tags.

## Sub-bugs found while verifying

The bucket split made the jump *fire*, but getting it to actually *clear* the wall
surfaced three more issues:

1. **Riding the walk-in, not the hop.** Selecting a jump *frame* by feature match
   lands in the clip's long *walk-up* portion; the playhead got pulled back to
   locomotion before the hop. → Seek the **run-up entry** and **lock** for the
   jump's duration (mm_g1's mechanism), so the airborne arc always plays out.
2. **A bogus clearance metric.** `geom_xpos[:,2].min()` was always ≈0 — it
   included the **floor plane** (geom 0). → Exclude world-body geoms; measure the
   robot only.
3. **Take-off timing vs gait phase.** Walls are 3.5 m apart, not an integer number
   of gait cycles, so the character met each wall at a different foot phase →
   inconsistent clearance. → Fire on a **fixed-distance crossing** (`_prev_fd >
   trigger_dist >= fd`) so every wall is launched from the same spot, and tune that
   one distance.

## Verification

FK collision test — *no robot geom may enter the wall box* (`|x−xw| ≤ half_thick`,
`z < 0.30`). With the shipped config (wall `0.20 m` thick, `JUMP_TRIGGER_DIST = 0.80`):

```
take-offs at rootX = [2.27, 5.73, 9.42]   (walls at 3.0 / 6.5 / 10.0)
lowest body-z over each wall = 0.48 / 0.48 / 0.40 m   (band top = 0.30)  -> 3/3 CLEARED
```

Side-profile capture: [`media/emm_3hurdles_side.mp4`](../media/emm_3hurdles_side.mp4),
montage [`media/emm_3hurdles_montage.png`](../media/emm_3hurdles_montage.png).

## It's the wall **thickness**, not its height (now fixed)

The jump clips vault the lowest body part to **0.58 m** (`walk_jump_walk`) /
**0.72 m** (`walk_jump_walk2`) — far above the `0.30 m` wall, so height is not the
problem. But a clip clears `0.30 m` only over a **narrow forward window**:

| clip            | vault peak | forward window clearing 0.30 m |
|-----------------|-----------:|-------------------------------:|
| walk_jump_walk  |   0.58 m   |  0.30 m                         |
| walk_jump_walk2 |   0.72 m   |  0.56 m                         |

The original wall footprint was **0.56 m** front-to-back (`WALL_HALF_THICK = 0.28`)
— as thick as that window, so its front/back edges sat in the ascending/descending
part of the arc where the feet are still low, giving the middle wall ~0 m margin.
(For a full 0.56 m-thick wall the clip can clear only ~0.03–0.11 m tall — a thick
wall is fundamentally a platform-vault, not this quick hop. You can see the raw
clip graze our old thick wall vs. cleanly clear a thin one in
[`media/rawclip_thickness_compare.png`](../media/rawclip_thickness_compare.png).)

**A thinner wall restores a comfortable margin** (height fixed at 0.30 m, best
`trigger_dist` per thickness):

| wall thickness | min clearance margin over the 3 walls |
|---------------:|--------------------------------------:|
| 0.56 m (old)   | 0.10 / **0.00** / 0.18 m              |
| 0.36 m         | 0.03 / 0.23 / 0.18 m                  |
| **0.20 m (now)** | **0.18 / 0.18 / 0.10 m** (→ 0.48 / 0.48 / 0.40 at D=0.80) |
| 0.16 m         | 0.34 / 0.26 / 0.18 m                  |

The old design made the wall thick so the *penalty* couldn't be side-stepped; with
the jump now a distance-triggered bucket, the wall no longer needs to be thick, so
`WALL_HALF_THICK` is now **0.10** (a 0.20 m hurdle bar) and the G1 clears all three
with ~0.4 m of margin.

## Tuning knobs (`emm_g1/config.py`)

- `JUMP_TRIGGER_DIST` (0.85 m) — forward distance to a wall at which the jump fires.
- `WALL_HEIGHT` / `WALL_HALF_THICK` — obstacle geometry; thinner = more clearance margin.
- `emm_g1/jumps.py`: `READY_LEN` (run-up entry window), `POST_LAND` (recovery ride),
  `SKIP_SUBSTR` (jump clips to exclude — `walk_jump_stop` is dropped as it halts).

## Files changed

`emm_g1/jumps.py` (new), `emm_g1/controller.py` (jump trigger + lock, loco-only
search), `emm_g1/search.py` (`cand_mask`), `emm_g1/config.py` (`JUMP_TRIGGER_DIST`),
`run_emm.py`, `README_EMM.md`.

> Not updated: the browser port (`docs/js/emm`) still uses the old merged search and
> has the same missed-jump bug; it needs the same bucket split to match.

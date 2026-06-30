#!/usr/bin/env python3
"""Interactive, keyboard-controlled ENVIRONMENT-AWARE motion matching for the G1.

    python run_emm.py

An additive, backward-compatible variant of ``run.py`` implementing Ponton et al.
2025 ("Environment-aware Motion Matching"). The scene contains low hurdle walls;
the G1 **jumps over them on its own** -- there is no jump key. Each candidate pose
carries a footprint ellipse + body-height range at its future trajectory taps, and
the search adds a log-barrier obstacle penalization that gates out poses colliding
with a wall's height band; a jump clip's airborne phase clears a low wall, so the
env-aware search selects it automatically.

Uses the SAME dataset as ``run.py`` (``data/gmr_lafan1_g1`` + ``data/g1_jump``),
built into its own cache ``data/emm_lib.npz`` on first launch.

Controls
  W / A / S / D ........ move, relative to the camera
  Arrow keys ........... face direction, independent of travel
  Shift (hold) ......... walk instead of run
  Space ................ reset to the start pose
  T .................... toggle the command-trajectory gizmo
  Left/right-drag / scroll ... orbit / pan / zoom
  Esc .................. quit
"""
import argparse
import mujoco

from emm_g1 import database as DB
from emm_g1 import config as EC
from emm_g1 import g1_model as g1
from emm_g1.controller import EMMController
from emm_g1.viewer import EMMViewer


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true",
                    help="build/refresh the EMM database cache and exit (no window)")
    ap.add_argument("--rebuild-db", action="store_true",
                    help="force a rebuild of data/emm_lib.npz")
    args = ap.parse_args()

    print("Loading EMM database (first run builds data/emm_lib.npz from the dataset)...")
    db = DB.load_or_build(rebuild=args.rebuild_db)
    print(f"  {len(db['Xn'])} frames, {len(db['clip_names'])} clips, "
          f"height_mode={bool(db['height_mode'])}")
    if args.build_only:
        print("Build complete. Run `python run_emm.py` to control the G1.")
        return

    env = EC.build_environment()
    ctrl = EMMController(db, env=env, max_speed=EC.MAX_SPEED, search_time=EC.SEARCH_TIME,
                        inert_halflife=EC.INERT_HALFLIFE, vel_halflife=EC.VEL_HALFLIFE,
                        rot_halflife=EC.ROT_HALFLIFE, penalty_weight=EC.PENALTY_WEIGHT,
                        evasion=EC.EVASION, anticipation=EC.ANTICIPATION,
                        trigger_dist=EC.JUMP_TRIGGER_DIST)
    # Spawn a couple of metres before the first hurdle, facing down the lane
    # (also where Space returns the G1).
    ctrl.set_spawn(-2.5, 0.0, 0.0)

    model = g1.load_model()
    data = mujoco.MjData(model)
    print("Opening viewer -- WASD to move; the G1 jumps the walls by itself. Esc to quit.")
    EMMViewer(model, data, ctrl).run()


if __name__ == "__main__":
    main()

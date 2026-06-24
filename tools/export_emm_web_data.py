#!/usr/bin/env python3
"""Export the EMM (auto-jump) database for the browser demo (docs/emm.html).

Run from the repo root with the mujoco env, e.g.:
    ~/miniconda3/envs/deploy_mujoco/bin/python tools/export_emm_web_data.py

Writes:
  docs/data/emm.json -- header: per-array {dtype, shape, byte offset} into emm.bin,
                        plus EMM hyperparameters, the obstacle (hurdle) layout and
                        clip metadata.
  docs/data/emm.bin  -- all runtime arrays the JS env-aware matcher needs
                        (features + precomputed ellipse/height search helpers).

It also (re)writes the SHARED docs/data/model.json + mesh.{json,bin} via the
existing exporter if missing, so the EMM page is self-contained. The JS modules
under docs/js/emm/ are a 1:1 port of emm_g1/{ellipse_geom,obstacles,search,
controller}.py.
"""
import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from emm_g1 import database as DB
from emm_g1 import config as EC
from emm_g1 import search as S
import tools.export_web_data as web   # reuse the shared model/mesh exporter

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "docs", "data")


def ensure_model_and_mesh():
    """Reuse the existing exporter to (re)write the shared skeleton + meshes."""
    m, bodies = web.export_model()
    if not os.path.exists(os.path.join(OUT, "model.json")):
        web.verify_fk(m, bodies)
        json.dump(dict(bodies=bodies, nbody=len(bodies)),
                  open(os.path.join(OUT, "model.json"), "w"))
        print(f"  model.json: {len(bodies)} bodies")
    if not os.path.exists(os.path.join(OUT, "mesh.bin")):
        web.export_meshes(m)
    else:
        print("  model.json + mesh.* already present (shared with the GenoView demo)")


def export_emm(db):
    S.prepare(db)   # adds valid / posTaps / ellAxis / ellExtP / ellExtS
    starts, stops = db['starts'], db['stops']

    arrays = {
        "Xn": db['Xn'].astype(np.float32),
        "Xmean": db['Xmean'].astype(np.float32),
        "Xstd": db['Xstd'].astype(np.float32),
        "dof": db['dof'].astype(np.float32),
        "dofVel": db['dofVel'].astype(np.float32),
        "rootPos": db['rootPos'].astype(np.float32),
        "rootQuat": db['rootQuat'].astype(np.float32),
        "theta": db['theta'].astype(np.float32),
        "rootVel": db['rootVel'].astype(np.float32),
        "yawRate": db['yawRate'].astype(np.float32),
        # precomputed search helpers (so the browser does no FK)
        "height": db['height'].astype(np.float32),        # (T,3,2)
        "posTaps": db['posTaps'].astype(np.float32),       # (T,3,2)
        "ellAxis": db['ellAxis'].astype(np.float32),       # (T,3,2)
        "ellExtP": db['ellExtP'].astype(np.float32),       # (T,3)
        "ellExtS": db['ellExtS'].astype(np.float32),       # (T,3)
        "valid": db['valid'].astype(np.int32),             # (T,)
        "starts": np.asarray(starts, np.int32),
        "stops": np.asarray(stops, np.int32),
        "clip_id": db['clip_id'].astype(np.int32),
        "clip_is_jump": db['clip_is_jump'].astype(np.int32),
        # default search weights (27) -- JS modulates trajDir via evasion
        "weights": _default_weights().astype(np.float32),
    }

    blob = bytearray()
    header = {}
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        header[name] = dict(dtype=a.dtype.name, shape=list(a.shape), offset=len(blob))
        blob += a.tobytes()

    meta = dict(
        fps=int(__import__('emm_g1.g1_model', fromlist=['FPS']).FPS),
        ndof=29, horizons=[int(h) for h in __import__('emm_g1.g1_model', fromlist=['HORIZONS']).HORIZONS],
        max_speed=EC.MAX_SPEED, walk_scale=EC.WALK_SCALE,
        search_time=EC.SEARCH_TIME, inert_halflife=EC.INERT_HALFLIFE,
        vel_halflife=EC.VEL_HALFLIFE, rot_halflife=EC.ROT_HALFLIFE,
        penalty_weight=EC.PENALTY_WEIGHT, evasion=EC.EVASION, anticipation=EC.ANTICIPATION,
        threshold=EC.OBSTACLE_THRESHOLD, nearby_radius=EC.NEARBY_RADIUS,
        height_mode=True, max_ellipse_length=0.9,
        crowd_weights=[1.0, 0.4, 0.1],
        obstacles=EC.obstacle_dicts(),
        clip_names=[str(n) for n in db['clip_names']],
        arrays=header, n_frames=int(len(db['Xn'])))
    return meta, bytes(blob)


def _default_weights():
    from emm_g1.controller import default_weights
    return default_weights()


def main():
    os.makedirs(OUT, exist_ok=True)
    print("Exporting EMM web demo data -> docs/data/")
    ensure_model_and_mesh()
    db = DB.load_or_build()
    meta, blob = export_emm(db)
    json.dump(meta, open(os.path.join(OUT, "emm.json"), "w"))
    with open(os.path.join(OUT, "emm.bin"), "wb") as f:
        f.write(blob)
    print(f"  emm.json + emm.bin: {meta['n_frames']} frames, "
          f"{len(meta['clip_names'])} clips, {len(meta['obstacles'])} hurdles, "
          f"{len(blob) / 1e6:.1f} MB")
    print("Done.")


if __name__ == "__main__":
    main()

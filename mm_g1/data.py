"""Build / load the G1 locomotion library from the GMR-retargeted LAFAN1 clips.

The library concatenates the walk, run and push-and-stumble clips into one continuous
array and precomputes the per-frame heading and FK foot positions that the feature
extractor needs. It is cached to data/motion_lib.npz so subsequent launches start
instantly. See data/gmr_lafan1_g1/README.md for the source pickle format.
"""
import os
import pickle
import numpy as np

from . import config as C
from .g1_model import G1Model, csv_to_qpos, quat_wxyz_yaw


def _gmr_rows(name, data_dir):
    """GMR pickle {root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,29)} -> (T,36) rows in
    the project's [xyz, quat_xyzw, 29 joints] layout (same as the old CSVs)."""
    with open(os.path.join(data_dir, name + ".pkl"), "rb") as f:
        d = pickle.load(f)
    return np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1)


def _load_clip(name, data_dir=C.DATA_DIR, trim=None):
    """Load a clip and drop its T-pose lead-in/out. `trim` is a (head, tail) frame pair;
    when None we use the GenoView-style per-clip CLIP_TRIM (falling back to DEFAULT_TRIM)."""
    rows = _gmr_rows(name, data_dir)
    head, tail = trim if trim is not None else C.CLIP_TRIM.get(name, C.DEFAULT_TRIM)
    rows = rows[head:len(rows) - tail]       # drop hand-picked T-pose blend frames
    return csv_to_qpos(rows)  # (T, 36) wxyz; csv_to_qpos reorders quat xyzw -> wxyz


def build_library(clips=None, out=C.LIB_PATH):
    """Concatenate clips into one array; precompute heading and FK foot positions."""
    clips = clips or C.CLIPS
    clips = [c for c in clips if os.path.exists(os.path.join(C.DATA_DIR, c + ".pkl"))]
    if not clips:
        raise FileNotFoundError(f"No clips found in {C.DATA_DIR}")

    model = G1Model()
    qpos, clip_id, frame_in_clip, lengths = [], [], [], []
    for cid, name in enumerate(clips):
        q = _load_clip(name)
        qpos.append(q)
        clip_id.append(np.full(len(q), cid))
        frame_in_clip.append(np.arange(len(q)))
        lengths.append(len(q))
        print(f"  [{cid}] {name}: {len(q)} frames")

    qpos = np.concatenate(qpos)
    feet = model.fk_feet(qpos)                       # (N, 2, 3) world
    yaw = quat_wxyz_yaw(qpos[:, 3:7])                 # (N,)

    np.savez_compressed(
        out,
        qpos=qpos.astype(np.float32),
        feet_world=feet.astype(np.float32),
        yaw=yaw.astype(np.float32),
        clip_id=np.concatenate(clip_id).astype(np.int32),
        frame_in_clip=np.concatenate(frame_in_clip).astype(np.int32),
        lengths=np.array(lengths, np.int32),
        clip_names=np.array(clips),
    )
    print(f"Saved library: {qpos.shape[0]} frames, {len(clips)} clips -> {out}")
    return out


def load_library(path=C.LIB_PATH):
    if not os.path.exists(path):
        build_library(out=path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_library()

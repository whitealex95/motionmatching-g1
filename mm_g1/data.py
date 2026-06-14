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


def _label_jump(model, q):
    """Per-frame skill (0 walk, 1 jump) + 5-phase label + jump indices for one jump clip.

    Flight = both feet above C.JUMP_FOOT_THR. Around each flight we carve five phases:
      ready [enter here] | takeoff (push-off) | flight | touchdown | after [exit here].
    skill=1 over the whole ready..after span so locomotion never targets jump frames
    (a jump is entered only via the `ready` run-up). Returns (skill, phase, jumps) where
    each jump is (entry, takeoff, land, continues) as clip-local frame indices.
    """
    READY, TAKEOFF, FLIGHT, TOUCHDOWN, AFTER = 1, 2, 3, 4, 5
    feet = model.fk_feet(q)
    footz = feet[:, :, 2].min(1)
    air = footz > C.JUMP_FOOT_THR
    idx = np.where(air)[0]
    n = len(q)
    skill = np.zeros(n, np.int32)
    phase = np.zeros(n, np.int32)                    # 0 = walk
    jumps = []
    for s in np.split(idx, np.where(np.diff(idx) > 3)[0] + 1) if len(idx) else []:
        if len(s) < 3:
            continue
        t, l = int(s[0]), int(s[-1])                 # take-off (flight start), land (flight end)
        r0 = max(0, t - C.PHASE_TAKEOFF - C.PHASE_READY)        # ready start = jump entry
        a1 = min(n, l + 1 + C.PHASE_TOUCHDOWN + C.PHASE_AFTER)  # after end
        phase[r0:t - C.PHASE_TAKEOFF] = READY
        phase[t - C.PHASE_TAKEOFF:t] = TAKEOFF
        phase[t:l + 1] = FLIGHT
        phase[l + 1:l + 1 + C.PHASE_TOUCHDOWN] = TOUCHDOWN
        phase[l + 1 + C.PHASE_TOUCHDOWN:a1] = AFTER
        skill[r0:a1] = 1
        w0, w1 = min(l + 40, n - 1), min(l + 60, n - 1)        # settled post-landing
        continues = w1 > w0 and np.linalg.norm(q[w1, 0:2] - q[w0, 0:2]) / ((w1 - w0) * C.DT) > 0.5
        jumps.append((r0, t, l, bool(continues)))
    return skill, phase, jumps


def _load_jump_csv(name, data_dir=C.JUMP_DATA_DIR):
    """Load a CAMDM walk->jump->walk CSV (row = [pos, quat_xyzw, 29 joints]) -> (T,36) wxyz."""
    rows = np.genfromtxt(os.path.join(data_dir, name + ".csv"), delimiter=",")
    return csv_to_qpos(rows)


def _gmr_rows(name, data_dir):
    """GMR pickle {root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,29)} -> (T,36) rows in
    the project's [xyz, quat_xyzw, 29 joints] layout (same as the old CSVs)."""
    with open(os.path.join(data_dir, name + ".pkl"), "rb") as f:
        d = pickle.load(f)
    return np.concatenate([d["root_pos"], d["root_rot"], d["dof_pos"]], axis=1)


def _load_clip(name, data_dir=C.DATA_DIR, trim=None):
    """Load a clip and apply its GenoView-matched absolute [start:stop] frame window
    (CLIP_TRIM). This drops T-pose lead-in/out off walk & run and, crucially, isolates the
    short stumble EVENT out of the otherwise-ordinary pushAndStumble clip."""
    rows = _gmr_rows(name, data_dir)
    s, e = trim if trim is not None else C.CLIP_TRIM.get(name, (0, len(rows)))
    rows = rows[s:min(e, len(rows))]
    return csv_to_qpos(rows)  # (T, 36) wxyz; csv_to_qpos reorders quat xyzw -> wxyz


def build_library(clips=None, out=C.LIB_PATH):
    """Concatenate clips into one array; precompute heading and FK foot positions."""
    clips = clips or C.CLIPS
    clips = [c for c in clips if os.path.exists(os.path.join(C.DATA_DIR, c + ".pkl"))]
    if not clips:
        raise FileNotFoundError(f"No clips found in {C.DATA_DIR}")

    model = G1Model()
    # Each base clip is loaded and (GenoView-style) added twice: normal + L/R MIRRORED, for
    # symmetric left/right coverage. Jump clips are phase-labeled per copy so the mirrored
    # jump gets its own correct (entry, takeoff, land) indices that drive the J trigger.
    jump_clips = [c for c in C.JUMP_CLIPS
                  if os.path.exists(os.path.join(C.JUMP_DATA_DIR, c + ".csv"))]
    base = [(c, False) for c in clips] + [(c, True) for c in jump_clips]   # (name, is_jump)
    loaded = []                                      # (name, qpos, is_jump) per concrete clip
    for name, is_jump in base:
        q = _load_jump_csv(name) if is_jump else _load_clip(name)
        loaded.append((name, q, is_jump))
        if C.MIRROR:
            loaded.append((name + "_mirror", model.mirror_qpos(q), is_jump))

    qpos, clip_id, frame_in_clip, lengths, names = [], [], [], [], []
    skill, phase = [], []
    j_entry, j_takeoff, j_land, j_cont = [], [], [], []
    off = 0
    for cid, (name, q, is_jump) in enumerate(loaded):
        if is_jump:
            sk, ph, jumps = _label_jump(model, q)
        else:
            sk, ph, jumps = np.zeros(len(q), np.int32), np.zeros(len(q), np.int32), []
        qpos.append(q); skill.append(sk); phase.append(ph)
        clip_id.append(np.full(len(q), cid))
        frame_in_clip.append(np.arange(len(q)))
        lengths.append(len(q)); names.append(name)
        for e, t, l, cont in jumps:                  # store as GLOBAL frame indices
            j_entry.append(off + e); j_takeoff.append(off + t)
            j_land.append(off + l); j_cont.append(cont)
        off += len(q)
        print(f"  [{cid}] {name}: {len(q)} frames"
              + (f", {len(jumps)} jump(s)" if is_jump else ""))

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
        clip_names=np.array(names),
        skill=np.concatenate(skill).astype(np.int32),
        phase=np.concatenate(phase).astype(np.int32),
        jump_entry=np.array(j_entry, np.int32),
        jump_takeoff=np.array(j_takeoff, np.int32),
        jump_land=np.array(j_land, np.int32),
        jump_continues=np.array(j_cont, bool),
        lib_version=np.array(C.LIB_VERSION),
    )
    print(f"Saved library: {qpos.shape[0]} frames, {len(loaded)} clips "
          f"({len(j_entry)} jumps) -> {out}")
    return out


def load_library(path=C.LIB_PATH):
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        version = int(d["lib_version"]) if "lib_version" in d.files else 0
        if version != C.LIB_VERSION:
            print("Cache is stale (library v%d != v%d); rebuilding..." % (version, C.LIB_VERSION))
            os.remove(path)                          # rebuild with the current heuristics
    if not os.path.exists(path):
        build_library(out=path)
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


if __name__ == "__main__":
    build_library()

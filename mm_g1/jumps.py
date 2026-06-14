"""Jump-skill indexing for the J trigger (ported from ../motiongraph).

A jump is entered ONLY in its `ready` phase (the run-up before take-off) and ridden
through landing/recovery. `jump_entries` returns the `ready` run-up frames of each jump
and maps each to its landing frame, so the controller can transition into a ready frame
that best continues the current pose and then ride take-off / flight / landing.
"""
import numpy as np

READY = 1   # phase code (see config.JUMP_PHASES)


def jump_entries(lib, continuing_only=True):
    """`ready`-phase run-up frames + {frame: land_frame}.

    continuing_only keeps only jumps that resume walking afterwards (so the character
    returns to locomotion rather than stopping). Returns (enter_frames, land_of).
    """
    enter, land_of = [], {}
    if "jump_entry" not in lib or len(lib["jump_entry"]) == 0:
        return np.array([], np.int32), land_of
    cont = lib["jump_continues"] if "jump_continues" in lib \
        else np.ones(len(lib["jump_entry"]), bool)
    phase = lib["phase"] if "phase" in lib else None
    for e, t, l, c in zip(lib["jump_entry"], lib["jump_takeoff"], lib["jump_land"], cont):
        if continuing_only and not c:
            continue
        for f in range(int(e), int(t)):
            if phase is None or phase[f] == READY:      # confine entry to the ready phase
                enter.append(f)
                land_of[f] = int(l)
    return np.array(enter, np.int32), land_of

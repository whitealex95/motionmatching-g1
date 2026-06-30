"""Jump-skill indexing for the obstacle-triggered jump (the EMM "jump bucket").

EMM keeps locomotion and the jump skill in SEPARATE buckets, exactly like
``mm_g1``: the motion-matching search runs over locomotion frames only, and the
jump is a distinct skill entered through its run-up ('ready') frames and ridden
through flight + landing. The only difference from ``mm_g1`` is the trigger: a J
keypress is replaced by an obstacle sitting in the take-off window ahead (see
:meth:`emm_g1.controller.EMMController.step`).

``mm_g1`` annotates jump phases at build time; the EMM feature DB does not carry
that metadata, so here the take-off / landing of each jump clip is detected from
the pelvis-height (``rootPos`` z) hop -- the airborne span where the pelvis lifts
clearly above the clip's own walking baseline. The frames just before take-off are
offered as entries so the controller can transition in from a matching run-up pose.
"""
import numpy as np

READY_LEN = 6           # run-up frames before take-off offered as jump entries
                        # (short -> small, consistent entry->apex travel for clean timing)
HOP_RISE = 0.06         # pelvis-z rise (m) above the clip's walking baseline => airborne
POST_LAND = 25          # frames ridden after landing (mm_g1 PHASE_TOUCHDOWN + PHASE_AFTER + 1)
SKIP_SUBSTR = ("stop",)  # exclude non-continuing jumps (they halt; bad for a lane)


def jump_index(db):
    """Index the jump clips for the trigger.

    Returns ``(enter_frames, land_of, end_of, apex_of)``:
      * ``enter_frames`` (int32): global run-up frames usable as jump entries.
      * ``land_of[frame]``: the global landing frame of that entry's clip.
      * ``end_of[frame]``:  the global frame to ride to before exiting back to
        locomotion (landing + recovery).
      * ``apex_of[frame]``: the global flight-apex frame (peak pelvis height) of
        that entry's clip -- used to auto-time the take-off so the apex lands over
        the obstacle.
    """
    starts, stops = db["starts"], db["stops"]
    cij = np.asarray(db["clip_is_jump"], bool)
    names = db["clip_names"]
    rootZ = db["rootPos"][:, 2]
    enter, land_of, end_of, apex_of = [], {}, {}, {}
    for ci in np.where(cij)[0]:
        if any(s in str(names[ci]) for s in SKIP_SUBSTR):
            continue
        rs, re = int(starts[ci]), int(stops[ci])
        z = rootZ[rs:re]
        base = np.median(z)
        above = z > base + HOP_RISE
        if not above.any():
            continue
        pk = int(z.argmax())
        to = pk
        while to > 0 and above[to - 1]:
            to -= 1
        la = pk
        while la < len(z) - 1 and above[la + 1]:
            la += 1
        takeoff, land, apex = rs + to, rs + la, rs + pk
        end = min(land + POST_LAND, re - 1)
        for f in range(max(rs, takeoff - READY_LEN), takeoff):
            enter.append(f)
            land_of[f] = land
            end_of[f] = end
            apex_of[f] = apex
    return np.array(enter, np.int32), land_of, end_of, apex_of

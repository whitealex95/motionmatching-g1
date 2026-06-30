"""Environment-aware motion-matching search.

Ports ``CrowdMotionMatchingSearchBurst`` / ``CrowdHeightMotionMatchingSearchBurst``
to NumPy as an *exact* branch-and-bound (VarianceFactor=1 => jump=1, full scan):

  * The matched distance is the weighted L2 over the static feature block plus an
    obstacle **penalization** added per trajectory tap (weights 1.0 /
    CrowdSecondTrajectoryWeight=0.4 / CrowdThirdTrajectoryWeight=0.1 for taps
    0/1/2). Because the penalty is always >= 0, the static distance is a lower
    bound on the total, so we iterate ALL valid frames in ascending static-
    distance order and stop once the static distance exceeds the best total found
    (every visited frame's penalty is accumulated with Unity's per-obstacle
    early-out). This is exact: no candidate cap.
  * When no obstacles are near, the penalty is identically 0 and the result is a
    plain nearest neighbour.
"""

import numpy as np

from . import ellipse_geom as eg
from . import g1_model as g1

# Per-tap penalization weights: 1.0, CrowdSecondTrajectoryWeight, CrowdThirdTrajectoryWeight.
CROWD_SECOND_TRAJECTORY_WEIGHT = 0.4
CROWD_THIRD_TRAJECTORY_WEIGHT = 0.1
CROWD_WEIGHTS = np.array([1.0, CROWD_SECOND_TRAJECTORY_WEIGHT, CROWD_THIRD_TRAJECTORY_WEIGHT])


def prepare(db):
    """Precompute per-frame search helpers and cache them on the db dict."""
    if 'valid' in db:
        return db
    Xn, Xmean, Xstd = db['Xn'], db['Xmean'], db['Xstd']
    starts, stops = db['starts'], db['stops']
    T = len(Xn)

    valid = np.zeros(T, bool)
    for rs, re in zip(starts, stops):
        valid[rs:max(rs, re - g1.HORIZONS[-1])] = True

    posTaps = (Xn[:, 0:6] * Xstd[0:6] + Xmean[0:6]).reshape(T, 3, 2)   # local metres
    ell = db['ellipse']                                               # (T,3,3)
    extP = np.linalg.norm(ell[:, :, 0:2], axis=-1)
    extP = np.maximum(extP, 1e-5)
    axisP = ell[:, :, 0:2] / extP[..., None]
    extS = np.maximum(np.abs(ell[:, :, 2]), 1e-5)

    db['valid'] = valid
    db['posTaps'] = posTaps.astype(np.float32)
    db['ellAxis'] = axisP.astype(np.float32)
    db['ellExtP'] = extP.astype(np.float32)
    db['ellExtS'] = extS.astype(np.float32)
    return db


def static_distances(Xn, Xq, weights):
    d = Xn - Xq
    return np.einsum('td,d,td->t', d, weights, d)


def _height_overlap(agent_h, obs_hmin, obs_hmax):
    return not (agent_h[0] > obs_hmax or agent_h[1] < obs_hmin)


def _feature_check(db, i, sqr, best_total, penalty_weight, circ, ell, threshold, height_mode):
    """Accumulate the obstacle penalty onto ``sqr`` for candidate frame ``i``.

    Mirrors Unity ``FeatureCheck``: penalize each tap against its circles then
    ellipses, scaled by the per-tap weight, with the per-obstacle early-out
    (``return best_total`` as soon as the running total exceeds the best). The
    penalty is multiplied by ``penalty_weight`` (Unity ``FeatureWeights[static]``).
    """
    posTaps = db['posTaps'][i]; axisP = db['ellAxis'][i]
    extP = db['ellExtP'][i]; extS = db['ellExtS'][i]
    height = db['height'][i] if height_mode else None
    for k in range(3):
        w = CROWD_WEIGHTS[k]
        center = posTaps[k]
        ext = np.array([extP[k], extS[k]])
        ax = axisP[k]
        sec = np.array([-ax[1], ax[0]])
        c = circ[k]; e = ell[k]
        for j in range(len(c)):
            if height_mode and not _height_overlap(height[k], c[j, 3], c[j, 4]):
                continue
            dist, _ = eg.point_to_ellipse(center, ax, ext, c[j, 0:2][None])
            dist = max(float(dist[0]) - c[j, 2], eg.MAX_INSIDE)
            pen = float(eg.distance_function(dist, threshold)) * w
            sqr += pen * penalty_weight
            if sqr > best_total:
                return best_total
        for j in range(len(e)):
            if height_mode and not _height_overlap(height[k], e[j, 6], e[j, 7]):
                continue
            d, _, _ = eg.ellipse_to_ellipse(center, ax, ext,
                                            e[j, 0:2], e[j, 2:4], e[j, 4:6])
            pen = float(eg.distance_function(d, threshold)) * w
            sqr += pen * penalty_weight
            if sqr > best_total:
                return best_total
    return sqr


def search_env(db, Xq, weights, penalty_weight, circ, ell, threshold,
               height_mode=False, cand_mask=None):
    """Exact branch-and-bound search with obstacle penalization.

    Returns the best frame index. With VarianceFactor=1 Unity scans every valid
    frame; we replicate that exactly by sorting valid frames on static distance
    (a lower bound, since penalty >= 0) and stopping once the static distance is
    no longer below the best total found.

    ``cand_mask`` (bool, T) optionally restricts the candidate pool -- EMM passes
    a locomotion-only mask so the search never matches into a jump clip (jumps are
    a separate skill bucket entered via the obstacle trigger, see controller).
    """
    prepare(db)
    valid = db['valid']
    if cand_mask is not None:
        valid = valid & cand_mask
    sdist = static_distances(db['Xn'], Xq, weights)
    sdist = np.where(valid, sdist, np.inf)

    n_obs = sum(len(c) for c in circ) + sum(len(e) for e in ell)
    if n_obs == 0:
        return int(np.argmin(sdist))

    order = np.argsort(sdist, kind='stable')
    best_idx, best_total = int(order[0]), np.inf
    for i in order:
        s = sdist[i]
        if s >= best_total:        # static distance is a lower bound -> done
            break
        total = _feature_check(db, int(i), float(s), best_total,
                               penalty_weight, circ, ell, threshold, height_mode)
        if total < best_total:
            best_total, best_idx = total, int(i)
    return best_idx

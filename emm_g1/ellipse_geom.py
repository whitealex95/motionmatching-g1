"""Ellipse distance geometry + the obstacle penalty.

Reimplements the parts of the Unity ``UtilitiesBurst`` used by the EMM search:

  * :func:`point_to_ellipse`   — distance from points to an oriented ellipse,
    using Eberly's robust closest-point-on-ellipse algorithm. Unity does NOT
    return a negative distance inside the ellipse; instead it returns a tiny
    positive value lerped between MIN_INSIDE and MAX_INSIDE.
  * :func:`ellipse_to_ellipse` — distance between two oriented ellipses by
    sampling 13 points on the 2nd ellipse (30-degree steps) and taking the min
    point-to-ellipse distance to the 1st (Unity ``DistanceEllipseToEllipse``).
  * :func:`distance_function`  — the penalty kernel
    ``-(thr-d)^4 * log(max(d/thr, MAX_INSIDE))`` for ``d < thr`` (0 otherwise).

All inputs/outputs are 2-D (ground plane). Vectorized over leading batch dims.
"""

import numpy as np

# Unity UtilitiesBurst constants.
MAX_INSIDE = 1e-9          # floor used for log stability / circle distance floor
MIN_INSIDE = 1e-6          # upper end of the tiny positive inside-ellipse lerp


def _root_bisection(r0, z0, z1, g, iters=5):
    """Eberly GetRoot, vectorized: root of (r0 z0/(s+r0))^2 + (z1/(s+1))^2 = 1.

    Unity's search path uses ``maxIterations = 5``.
    """
    n0 = r0 * z0
    s0 = z1 - 1.0
    s1 = np.where(g < 0.0, 0.0, np.hypot(n0, z1) - 1.0)
    s = 0.5 * (s0 + s1)
    with np.errstate(divide='ignore', invalid='ignore'):
        for _ in range(iters):
            s = 0.5 * (s0 + s1)
            ratio0 = n0 / (s + r0)
            ratio1 = z1 / (s + 1.0)
            gg = ratio0 * ratio0 + ratio1 * ratio1 - 1.0
            s0 = np.where(gg > 0.0, s, s0)
            s1 = np.where(gg < 0.0, s, s1)
            # early-exit when the bracket has collapsed (Unity breaks on this).
            done = (np.abs(s - s0) < 1e-9) | (np.abs(s - s1) < 1e-9)
            s0 = np.where(done, s, s0)
            s1 = np.where(done, s, s1)
    return s


def _closest_first_quadrant(e0, e1, y0, y1):
    """Closest point on axis-aligned ellipse (e0>=e1) to first-quadrant (y0,y1>=0)."""
    e0 = np.asarray(e0, float); e1 = np.asarray(e1, float)
    y0 = np.asarray(y0, float); y1 = np.asarray(y1, float)
    x0 = np.zeros_like(y0); x1 = np.zeros_like(y1)

    big = y1 > 1e-12
    # --- y1 > 0 ---
    with np.errstate(divide='ignore', invalid='ignore'):
        z0 = np.where(big, y0 / e0, 0.0)
        z1 = np.where(big, y1 / e1, 0.0)
        g = z0 * z0 + z1 * z1 - 1.0
        r0 = (e0 / e1) ** 2
        sbar = _root_bisection(r0, z0, z1, g)
        xa = r0 * y0 / (sbar + r0)
        xb = y1 / (sbar + 1.0)
    y0pos = y0 > 1e-12
    # y1>0, y0>0 (general); y1>0, y0==0 -> (0, e1)
    x0 = np.where(big & y0pos, xa, np.where(big, 0.0, x0))
    x1 = np.where(big & y0pos, xb, np.where(big, e1, x1))

    # --- y1 == 0 ---
    numer0 = e0 * y0
    denom0 = e0 * e0 - e1 * e1
    inside = numer0 < denom0
    xde = np.where(denom0 != 0.0, numer0 / np.where(denom0 != 0.0, denom0, 1.0), 1.0)
    bx0 = np.where(inside, e0 * xde, e0)
    bx1 = np.where(inside, e1 * np.sqrt(np.clip(1.0 - xde * xde, 0.0, 1.0)), 0.0)
    x0 = np.where(~big, bx0, x0)
    x1 = np.where(~big, bx1, x1)
    return x0, x1


def closest_on_ellipse(a, b, p):
    """Closest point on an axis-aligned ellipse (semi-axes a along x, b along y).

    ``p`` (..., 2). Returns ``(closest(...,2), dist(...))``. Following Unity, when
    ``p`` is inside the ellipse's bounding box the returned distance is a tiny
    POSITIVE value lerped between MIN_INSIDE and MAX_INSIDE by the box-boundary
    distance ratio (never negative).
    """
    p = np.asarray(p, float)
    px, py = p[..., 0], p[..., 1]
    sx = np.sign(px); sx = np.where(sx == 0, 1.0, sx)
    sy = np.sign(py); sy = np.where(sy == 0, 1.0, sy)
    ax, ay = np.abs(px), np.abs(py)

    swap = b > a
    e0 = np.where(swap, b, a); e1 = np.where(swap, a, b)
    q0 = np.where(swap, ay, ax); q1 = np.where(swap, ax, ay)
    c0, c1 = _closest_first_quadrant(e0, e1, q0, q1)
    cx = np.where(swap, c1, c0); cy = np.where(swap, c0, c1)

    closest = np.stack([sx * cx, sy * cy], axis=-1)
    dist = np.linalg.norm(closest - p, axis=-1)

    # Unity: inside-the-bounding-box query points get a tiny positive distance
    # lerped from the *bounding box* boundary distance (NOT the true ellipse eq).
    inside = (q0 < e0) & (q1 < e1)
    boundary = np.minimum(e0 - q0, e1 - q1)
    e0_safe = np.where(e0 > 1e-12, e0, 1.0)
    lerp = boundary / e0_safe
    inside_dist = MIN_INSIDE + (MAX_INSIDE - MIN_INSIDE) * lerp
    dist = np.where(inside, inside_dist, dist)
    return closest, dist


def _to_local(center, axis_p, points):
    """Express ``points`` in the ellipse frame (x along axis_p, y along its perp)."""
    perp = np.stack([-axis_p[..., 1], axis_p[..., 0]], axis=-1)
    rel = points - center
    return np.stack([np.einsum('...d,...d->...', rel, np.broadcast_to(axis_p, rel.shape)),
                     np.einsum('...d,...d->...', rel, np.broadcast_to(perp, rel.shape))], axis=-1)


def point_to_ellipse(center, axis_p, ext, points):
    """Distance + world closest point from ``points`` to an oriented ellipse.

    ``center`` (2,), ``axis_p`` (2,) unit primary axis, ``ext`` (2,)
    (primary, secondary) semi-extents, ``points`` (N, 2). Returns ``(dist(N,),
    closest_world(N, 2))``; ``dist`` is a tiny positive value when ``points`` lie
    inside the ellipse (Unity convention, never negative).
    """
    center = np.asarray(center, float); axis_p = np.asarray(axis_p, float)
    points = np.asarray(points, float)
    perp = np.array([-axis_p[1], axis_p[0]])
    rel = points - center
    loc = np.stack([rel @ axis_p, rel @ perp], axis=-1)
    cl, dist = closest_on_ellipse(ext[0], ext[1], loc)
    world = center + cl[..., 0:1] * axis_p + cl[..., 1:2] * perp
    return dist, world


def generate_point_on_ellipse(center, primary_axis_unit, secondary_axis_unit, ext, angle_deg):
    """A point on an oriented ellipse at parametric ``angle_deg`` (Unity helper)."""
    a = np.radians(angle_deg)
    return (np.asarray(center, float)
            + np.asarray(primary_axis_unit, float) * ext[0] * np.cos(a)
            + np.asarray(secondary_axis_unit, float) * ext[1] * np.sin(a))


def ellipse_to_ellipse(c1, axis1, ext1, c2, axis2, ext2):
    """Distance between two oriented ellipses (Unity ``DistanceEllipseToEllipse``).

    Samples 13 points on ellipse 2 (every 30 degrees, 0..360 inclusive) and
    returns the minimum point-to-ellipse distance to ellipse 1.

    Returns ``(dist, closest1, closest2)`` where ``closest2`` is the sampled
    point on ellipse 2 giving the min and ``closest1`` is its closest point on
    ellipse 1. ``dist`` follows the point_to_ellipse (Unity) convention.
    """
    c1 = np.asarray(c1, float); c2 = np.asarray(c2, float)
    axis1 = np.asarray(axis1, float); axis2 = np.asarray(axis2, float)
    sec2 = np.array([-axis2[1], axis2[0]])

    best_d = np.inf
    best_c1 = c1.copy()
    best_c2 = c2.copy()
    for ang in range(0, 361, 30):
        sample = generate_point_on_ellipse(c2, axis2, sec2, ext2, ang)
        d, cl1 = point_to_ellipse(c1, axis1, ext1, sample[None])
        d = float(d[0])
        if d < best_d:
            best_d, best_c1, best_c2 = d, cl1[0], sample
    return best_d, best_c1, best_c2


def distance_function(d, threshold):
    """EMM penalty kernel. Scalar or array ``d``; 0 beyond ``threshold``."""
    d = np.asarray(d, float)
    ratio = np.maximum(d / threshold, MAX_INSIDE)
    pen = -np.power(np.maximum(threshold - d, 0.0), 4.0) * np.log(ratio)
    return np.where(d > threshold, 0.0, pen)

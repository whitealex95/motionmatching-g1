// Ellipse distance geometry + obstacle penalty kernel.
// A 1:1 (scalar) port of emm_g1/ellipse_geom.py (Eberly closest-point-on-ellipse,
// ellipse-to-ellipse by 13-sample, and the log-barrier penalty). 2-D ground plane.

export const MAX_INSIDE = 1e-9;
export const MIN_INSIDE = 1e-6;

function rootBisection(r0, z0, z1, g, iters = 5) {
  const n0 = r0 * z0;
  let s0 = z1 - 1.0;
  let s1 = g < 0.0 ? 0.0 : Math.hypot(n0, z1) - 1.0;
  let s = 0.5 * (s0 + s1);
  for (let it = 0; it < iters; it++) {
    s = 0.5 * (s0 + s1);
    const ratio0 = n0 / (s + r0);
    const ratio1 = z1 / (s + 1.0);
    const gg = ratio0 * ratio0 + ratio1 * ratio1 - 1.0;
    if (gg > 0.0) s0 = s;
    else if (gg < 0.0) s1 = s;
    if (Math.abs(s - s0) < 1e-9 || Math.abs(s - s1) < 1e-9) break;
  }
  return s;
}

// closest point on axis-aligned ellipse (e0>=e1) to first-quadrant (y0,y1>=0).
function closestFirstQuadrant(e0, e1, y0, y1) {
  let x0, x1;
  if (y1 > 1e-12) {
    const z0 = y0 / e0, z1 = y1 / e1;
    const g = z0 * z0 + z1 * z1 - 1.0;
    const r0 = (e0 / e1) * (e0 / e1);
    const sbar = rootBisection(r0, z0, z1, g);
    if (y0 > 1e-12) { x0 = r0 * y0 / (sbar + r0); x1 = y1 / (sbar + 1.0); }
    else { x0 = 0.0; x1 = e1; }
  } else {
    const numer0 = e0 * y0, denom0 = e0 * e0 - e1 * e1;
    if (numer0 < denom0) {
      const xde = denom0 !== 0.0 ? numer0 / denom0 : 1.0;
      x0 = e0 * xde; x1 = e1 * Math.sqrt(Math.max(1.0 - xde * xde, 0.0));
    } else { x0 = e0; x1 = 0.0; }
  }
  return [x0, x1];
}

// closest point on axis-aligned ellipse (semi-axes a along x, b along y) to p.
// Returns { closest:[x,y], dist } -- inside points get a tiny positive distance.
export function closestOnEllipse(a, b, p) {
  const px = p[0], py = p[1];
  let sx = Math.sign(px); if (sx === 0) sx = 1.0;
  let sy = Math.sign(py); if (sy === 0) sy = 1.0;
  const ax = Math.abs(px), ay = Math.abs(py);
  const swap = b > a;
  const e0 = swap ? b : a, e1 = swap ? a : b;
  const q0 = swap ? ay : ax, q1 = swap ? ax : ay;
  const [c0, c1] = closestFirstQuadrant(e0, e1, q0, q1);
  const cx = swap ? c1 : c0, cy = swap ? c0 : c1;
  const closest = [sx * cx, sy * cy];
  let dist = Math.hypot(closest[0] - px, closest[1] - py);
  if (q0 < e0 && q1 < e1) {                       // inside the bounding box
    const boundary = Math.min(e0 - q0, e1 - q1);
    const lerp = boundary / (e0 > 1e-12 ? e0 : 1.0);
    dist = MIN_INSIDE + (MAX_INSIDE - MIN_INSIDE) * lerp;
  }
  return { closest, dist };
}

// distance + world closest point from a point to an oriented ellipse.
export function pointToEllipse(center, axisP, ext, point) {
  const perp = [-axisP[1], axisP[0]];
  const rx = point[0] - center[0], ry = point[1] - center[1];
  const loc = [rx * axisP[0] + ry * axisP[1], rx * perp[0] + ry * perp[1]];
  const { closest, dist } = closestOnEllipse(ext[0], ext[1], loc);
  const world = [center[0] + closest[0] * axisP[0] + closest[1] * perp[0],
                 center[1] + closest[0] * axisP[1] + closest[1] * perp[1]];
  return { dist, world };
}

function pointOnEllipse(center, primAxis, secAxis, ext, angDeg) {
  const a = angDeg * Math.PI / 180.0, c = Math.cos(a), s = Math.sin(a);
  return [center[0] + primAxis[0] * ext[0] * c + secAxis[0] * ext[1] * s,
          center[1] + primAxis[1] * ext[0] * c + secAxis[1] * ext[1] * s];
}

// distance between two oriented ellipses (13 samples on #2, min dist to #1).
export function ellipseToEllipse(c1, axis1, ext1, c2, axis2, ext2) {
  const sec2 = [-axis2[1], axis2[0]];
  let best = Infinity;
  for (let ang = 0; ang <= 360; ang += 30) {
    const sample = pointOnEllipse(c2, axis2, sec2, ext2, ang);
    const { dist } = pointToEllipse(c1, axis1, ext1, sample);
    if (dist < best) best = dist;
  }
  return best;
}

// EMM penalty kernel: -(thr-d)^4 * log(max(d/thr, MAX_INSIDE)), 0 beyond thr.
export function distanceFunction(d, threshold) {
  if (d > threshold) return 0.0;
  const ratio = Math.max(d / threshold, MAX_INSIDE);
  const base = Math.max(threshold - d, 0.0);
  return -(base * base * base * base) * Math.log(ratio);
}

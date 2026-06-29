// Environment-aware motion-matching search.
// A 1:1 port of emm_g1/search.py: exact branch-and-bound where the matched score
// is the weighted L2 static distance plus an obstacle penalization per trajectory
// tap; the static distance is a lower bound (penalty >= 0), so valid frames are
// scanned in ascending static-distance order and the scan stops once the static
// distance exceeds the best total found. Height gating lets a jump's airborne pose
// pass the obstacle band that a standing pose collides with.

import { pointToEllipse, ellipseToEllipse, distanceFunction, MAX_INSIDE } from './ellipse_geom.js';

const CROWD = [1.0, 0.4, 0.1];   // per-tap penalization weights

function heightOverlap(ah0, ah1, o0, o1) { return !(ah0 > o1 || ah1 < o0); }

// db arrays (typed): Xn (T*27), valid (T), posTaps (T*6), ellAxis (T*6),
// ellExtP (T*3), ellExtS (T*3), height (T*6).
export function staticDistances(Xn, Xq, weights, T) {
  const sd = new Float64Array(T);
  for (let t = 0; t < T; t++) {
    let s = 0; const b = t * 27;
    for (let i = 0; i < 27; i++) { const d = Xn[b + i] - Xq[i]; s += d * weights[i] * d; }
    sd[t] = s;
  }
  return sd;
}

function featureCheck(db, i, sqr, bestTotal, penaltyWeight, circ, ell, threshold, heightMode) {
  const pt = i * 6, ax = i * 6, ep = i * 3, hb = i * 6;
  for (let k = 0; k < 3; k++) {
    const w = CROWD[k];
    const center = [db.posTaps[pt + k * 2], db.posTaps[pt + k * 2 + 1]];
    const axis = [db.ellAxis[ax + k * 2], db.ellAxis[ax + k * 2 + 1]];
    const ext = [db.ellExtP[ep + k], db.ellExtS[ep + k]];
    const ah0 = db.height[hb + k * 2], ah1 = db.height[hb + k * 2 + 1];
    const cs = circ[k], es = ell[k];
    for (let j = 0; j < cs.length; j++) {
      const c = cs[j];
      if (heightMode && !heightOverlap(ah0, ah1, c[3], c[4])) continue;
      let { dist } = pointToEllipse(center, axis, ext, [c[0], c[1]]);
      dist = Math.max(dist - c[2], MAX_INSIDE);
      sqr += distanceFunction(dist, threshold) * w * penaltyWeight;
      if (sqr > bestTotal) return bestTotal;
    }
    for (let j = 0; j < es.length; j++) {
      const e = es[j];
      if (heightMode && !heightOverlap(ah0, ah1, e[6], e[7])) continue;
      const d = ellipseToEllipse(center, axis, ext, [e[0], e[1]], [e[2], e[3]], [e[4], e[5]]);
      sqr += distanceFunction(d, threshold) * w * penaltyWeight;
      if (sqr > bestTotal) return bestTotal;
    }
  }
  return sqr;
}

// candMask (optional Uint8Array, T): restrict the candidate pool. EMM passes a
// locomotion-only mask so the search never matches into a jump clip (jumps are a
// separate skill bucket entered via the obstacle trigger -- see controller.js).
export function searchEnv(db, Xq, weights, penaltyWeight, circ, ell, threshold, heightMode, candMask = null) {
  const T = db.T;
  const sd = staticDistances(db.Xn, Xq, weights, T);
  for (let t = 0; t < T; t++) if (!db.valid[t] || (candMask && !candMask[t])) sd[t] = Infinity;

  let nObs = 0;
  for (let k = 0; k < 3; k++) nObs += circ[k].length + ell[k].length;
  if (nObs === 0) {                                // plain nearest neighbour
    let bi = 0, bv = Infinity;
    for (let t = 0; t < T; t++) if (sd[t] < bv) { bv = sd[t]; bi = t; }
    return bi;
  }

  // indices sorted ascending by static distance (stable not required for correctness)
  const order = Array.from({ length: T }, (_, t) => t);
  order.sort((a, b) => sd[a] - sd[b]);
  let bestIdx = order[0], bestTotal = Infinity;
  for (let oi = 0; oi < T; oi++) {
    const i = order[oi];
    const s = sd[i];
    if (s >= bestTotal) break;                     // static distance is a lower bound
    const total = featureCheck(db, i, s, bestTotal, penaltyWeight, circ, ell, threshold, heightMode);
    if (total < bestTotal) { bestTotal = total; bestIdx = i; }
  }
  return bestIdx;
}

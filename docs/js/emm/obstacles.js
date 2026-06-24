// Obstacle model + the per-tap nearby-obstacle query.
// A 1:1 port of the STATIC-obstacle path of emm_g1/obstacles.py. Obstacles are
// low hurdle walls (oriented ellipses) or circles, each with a height band.
// get_nearby() returns, per trajectory tap, the obstacles in the searching
// character's heading-local frame, after the same per-tap culling.

import { quat } from '../quat.js';

const MAX_ELLIPSE_LENGTH = 0.9;   // matches emm_g1.obstacles.MAXIMUM_ELLIPSE_LENGTH

export class Environment {
  // obstacles: array of meta.obstacles dicts {cx,cy,ax,ay,half_len,half_thick,hmin,hmax}
  // (a circle uses {cx,cy,radius,hmin,hmax}).
  constructor(obstacles, threshold = 0.4) {
    this.threshold = threshold;
    this.obs = obstacles.map((o) => {
      const isEllipse = o.radius === undefined;
      let axis = [o.ax !== undefined ? o.ax : 1.0, o.ay !== undefined ? o.ay : 0.0];
      const n = Math.hypot(axis[0], axis[1]) + 1e-9;
      axis = [axis[0] / n, axis[1] / n];
      return {
        center: [o.cx, o.cy], isEllipse,
        radius: o.radius !== undefined ? o.radius : 0.0,
        axis, ext: [o.half_len !== undefined ? o.half_len : 0.3,
                    o.half_thick !== undefined ? o.half_thick : 0.3],
        h: [o.hmin, o.hmax],
      };
    });
  }

  // Returns { circ:[3][], ell:[3][] }. circ[k]: rows [cx,cy,radius,hmin,hmax];
  // ell[k]: rows [cx,cy,axx,axy,extp,exts,hmin,hmax] -- all in the heading-local frame.
  getNearby(rootXY, rootYaw, tapWorld, horizons) {
    const qh = quat.yaw(rootYaw);
    const cull = MAX_ELLIPSE_LENGTH + this.threshold;
    const toLocalPt = (p) => {
      const v = quat.invMulVec(qh, [p[0] - rootXY[0], p[1] - rootXY[1], 0.0]);
      return [v[0], v[1]];
    };
    const toLocalDir = (d) => {
      const v = quat.invMulVec(qh, [d[0], d[1], 0.0]);
      return [v[0], v[1]];
    };
    const circ = [[], [], []], ell = [[], [], []];
    for (const o of this.obs) {
      for (let k = 0; k < 3; k++) {
        const c = o.center;          // static: future_center == center
        if (Math.hypot(c[0] - tapWorld[k][0], c[1] - tapWorld[k][1]) > cull) continue;
        const cl = toLocalPt(c);
        if (o.isEllipse) {
          let al = toLocalDir(o.axis);
          const n = Math.hypot(al[0], al[1]) + 1e-9; al = [al[0] / n, al[1] / n];
          ell[k].push([cl[0], cl[1], al[0], al[1], o.ext[0], o.ext[1], o.h[0], o.h[1]]);
        } else {
          circ[k].push([cl[0], cl[1], o.radius, o.h[0], o.h[1]]);
        }
      }
    }
    return { circ, ell };
  }
}

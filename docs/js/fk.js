// Forward kinematics for the skeleton renderer -- the exact formula verified against
// MuJoCo in tools/export_web_data.py (worst error ~1e-7 m). Given the model.json body
// tree and a (36,) qpos, returns world position + orientation (wxyz) for every body.

import { quat, v3 } from './quat.js';

export function fk(bodies, qpos) {
  const n = bodies.length;
  const wp = new Array(n), wq = new Array(n);
  wp[0] = [qpos[0], qpos[1], qpos[2]];                 // root body == pelvis
  wq[0] = [qpos[3], qpos[4], qpos[5], qpos[6]];
  for (let i = 1; i < n; i++) {
    const b = bodies[i], p = b.parent;
    wp[i] = v3.add(wp[p], quat.mulVec(wq[p], b.pos));
    let r = quat.mul(wq[p], b.quat);
    if (b.axis) r = quat.mul(r, quat.fromAngleAxis(qpos[b.qadr], b.axis));
    wq[i] = r;
  }
  return { wp, wq };
}

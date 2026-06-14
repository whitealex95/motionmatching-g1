// Minimal quaternion / vec3 helpers (wxyz convention) -- a 1:1 port of the subset of
// mm_g1/quat.py used by the motion matcher. Quats are [w,x,y,z], vecs are [x,y,z].

export const v3 = {
  sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
  add: (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
  scale: (a, s) => [a[0] * s, a[1] * s, a[2] * s],
  norm: (a) => Math.hypot(a[0], a[1], a[2]),
  cross: (a, b) => [a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0]],
};

export const quat = {
  mul(x, y) {
    const [x0, x1, x2, x3] = x, [y0, y1, y2, y3] = y;
    return [
      x0 * y0 - x1 * y1 - x2 * y2 - x3 * y3,
      x0 * y1 + x1 * y0 + x2 * y3 - x3 * y2,
      x0 * y2 - x1 * y3 + x2 * y0 + x3 * y1,
      x0 * y3 + x1 * y2 - x2 * y1 + x3 * y0,
    ];
  },
  inv: (q) => [q[0], -q[1], -q[2], -q[3]],
  mul_inv(x, y) { return quat.mul(x, quat.inv(y)); },
  mulVec(q, x) {                                  // rotate vec x by quat q
    const u = [q[1], q[2], q[3]];
    const t = v3.scale(v3.cross(u, x), 2.0);
    return v3.add(v3.add(x, v3.scale(t, q[0])), v3.cross(u, t));
  },
  invMulVec(q, x) { return quat.mulVec(quat.inv(q), x); },
  fromAngleAxis(angle, axis) {
    const c = Math.cos(angle / 2), s = Math.sin(angle / 2);
    return [c, s * axis[0], s * axis[1], s * axis[2]];
  },
  yaw(theta) { return [Math.cos(theta / 2), 0, 0, Math.sin(theta / 2)]; },
  abs: (q) => (q[0] < 0 ? [-q[0], -q[1], -q[2], -q[3]] : q),
  normalize(q) {
    const n = Math.hypot(q[0], q[1], q[2], q[3]) + 1e-8;
    return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
  },
  toScaledAngleAxis(q, eps = 1e-5) {              // 2 * log(q)
    const len = Math.hypot(q[1], q[2], q[3]);
    const half = len < eps ? 1.0 : Math.atan2(len, q[0]) / len;
    return [2 * half * q[1], 2 * half * q[2], 2 * half * q[3]];
  },
  fromScaledAngleAxis(v, eps = 1e-5) {            // exp(v / 2)
    const h = [v[0] / 2, v[1] / 2, v[2] / 2];
    const half = Math.hypot(h[0], h[1], h[2]);
    const c = half < eps ? 1.0 : Math.cos(half);
    const s = half < eps ? 1.0 : Math.sin(half) / half;
    return [c, s * h[0], s * h[1], s * h[2]];
  },
};

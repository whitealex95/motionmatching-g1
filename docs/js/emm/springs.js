// Critically-damped spring helpers (port of emm_g1/springs.py), using the same
// FastNEgeExp rational approximation of exp(-x) as the Python/Unity reference so
// the predicted trajectories + inertialization decay match.

import { quat } from '../quat.js';

const LN2x4 = 4.0 * 0.69314718056;
const damping = (hl) => LN2x4 / (hl + 1e-5);
const fastNegExp = (x) => 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x);

// Decay an offset (x,v) toward zero (component-wise arrays). Returns [x', v'].
export function decaySpringDamper(x, v, hl, dt) {
  const y = damping(hl) / 2.0, e = fastNegExp(y * dt);
  const xo = new Array(x.length), vo = new Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const j1 = v[i] + x[i] * y;
    xo[i] = e * (x[i] + j1 * dt);
    vo[i] = e * (v[i] - j1 * y * dt);
  }
  return [xo, vo];
}

export function decaySpringDamperScalar(x, v, hl, dt) {
  const y = damping(hl) / 2.0, e = fastNegExp(y * dt);
  const j1 = v + x * y;
  return [e * (x + j1 * dt), e * (v - j1 * y * dt)];
}

// Predict (pos,vel,acc) at time dt given a desired velocity (3-vectors). Returns [P,V,A].
export function trajectorySpringPosition(pos, vel, acc, dvel, hl, dt) {
  const y = damping(hl) / 2.0, e = fastNegExp(y * dt);
  const P = [0, 0, 0], V = [0, 0, 0], A = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const j0 = vel[i] - dvel[i];
    const j1 = acc[i] + j0 * y;
    P[i] = e * (((-j1) / (y * y)) + ((-j0 - j1 * dt) / y)) + (j1 / (y * y)) + j0 / y + dvel[i] * dt + pos[i];
    V[i] = e * (j0 + j1 * dt) + dvel[i];
    A[i] = e * (acc[i] - j1 * y * dt);
  }
  return [P, V, A];
}

// Predict (rot,ang) at time dt given a desired rotation. Returns [q(wxyz), ang(3)].
export function trajectorySpringRotation(rot, ang, dRot, hl, dt) {
  const y = damping(hl) / 2.0, e = fastNegExp(y * dt);
  const j0 = quat.toScaledAngleAxis(quat.abs(quat.mul_inv(rot, dRot)));
  const j1 = [ang[0] + j0[0] * y, ang[1] + j0[1] * y, ang[2] + j0[2] * y];
  const q = quat.mul(quat.fromScaledAngleAxis(
    [e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]), dRot);
  return [q, [e * (ang[0] - j1[0] * y * dt), e * (ang[1] - j1[1] * y * dt), e * (ang[2] - j1[2] * y * dt)]];
}

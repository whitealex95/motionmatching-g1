// Real-time motion matching in the browser -- a 1:1 port of mm_g1/controller.py
// (GenoView smoothed-sim-root architecture) + the J jump skill. Reads the database
// exported by tools/export_web_data.py.

import { quat, v3 } from './quat.js';

const FORWARD = [1, 0, 0];

// ---- DB loader: typed-array views into mm.bin per the mm.json header ----
export function loadDB(meta, buf) {
  const A = {};
  for (const [name, h] of Object.entries(meta.arrays)) {
    const TA = h.dtype === 'int32' ? Int32Array : Float32Array;
    const count = h.shape.reduce((a, b) => a * b, 1);
    A[name] = count ? new TA(buf, h.offset, count) : new TA(0);
  }
  return A;
}

// ---- spring + inertialization helpers (port of mm_g1/springs.py) ----
const damp = (hl) => (4.0 * 0.69314718056) / (hl + 1e-5);

function decayPos(x, v, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt), xo = [], vo = [];
  for (let i = 0; i < x.length; i++) {
    const j1 = v[i] + x[i] * y;
    xo[i] = e * (x[i] + j1 * dt);
    vo[i] = e * (v[i] - j1 * y * dt);
  }
  return [xo, vo];
}
function decayRot(x, v, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt);
  const j0 = quat.toScaledAngleAxis(x);
  const j1 = [v[0] + j0[0] * y, v[1] + j0[1] * y, v[2] + j0[2] * y];
  const q = quat.fromScaledAngleAxis([e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]);
  return [q, [e * (v[0] - j1[0] * y * dt), e * (v[1] - j1[1] * y * dt), e * (v[2] - j1[2] * y * dt)]];
}
function trajPos(pos, vel, acc, dvel, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt), P = [], V = [], Ac = [];
  for (let i = 0; i < 3; i++) {
    const j0 = vel[i] - dvel[i], j1 = acc[i] + j0 * y;
    P[i] = e * ((-j1) / (y * y) + (-j0 - j1 * dt) / y) + j1 / (y * y) + j0 / y + dvel[i] * dt + pos[i];
    V[i] = e * (j0 + j1 * dt) + dvel[i];
    Ac[i] = e * (acc[i] - j1 * y * dt);
  }
  return [P, V, Ac];
}
function trajRot(rot, ang, dRot, hl, dt) {
  const y = damp(hl) / 2, e = Math.exp(-y * dt);
  const j0 = quat.toScaledAngleAxis(quat.abs(quat.mul_inv(rot, dRot)));
  const j1 = [ang[0] + j0[0] * y, ang[1] + j0[1] * y, ang[2] + j0[2] * y];
  const q = quat.mul(quat.fromScaledAngleAxis([e * (j0[0] + j1[0] * dt), e * (j0[1] + j1[1] * dt), e * (j0[2] + j1[2] * dt)]), dRot);
  return [q, [e * (ang[0] - j1[0] * y * dt), e * (ang[1] - j1[1] * y * dt), e * (ang[2] - j1[2] * y * dt)]];
}

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export class MotionMatcher {
  constructor(meta, A) {
    this.A = A;
    this.fps = meta.fps; this.DT = 1 / meta.fps;
    this.H = Math.max(...meta.horizons);
    this.Ttimes = meta.horizons.map((h) => h / meta.fps);
    this.MAX_SPEED = meta.max_speed; this.WALK_SCALE = meta.walk_scale;
    this.SEARCH_TIME = meta.search_time; this.CURRENT_BIAS = meta.current_bias;
    this.INERT = meta.inert_halflife; this.VEL_HL = meta.vel_halflife; this.ROT_HL = meta.rot_halflife;
    this.PHASE_TD = meta.phase_touchdown; this.PHASE_AFTER = meta.phase_after;
    this.clipNames = meta.clip_names;
    this.starts = A.starts; this.stops = A.stops;
    this.X = A.X; this.Xoffset = A.Xoffset; this.Xscale = A.Xscale;
    this.search = Array.from(A.search_clips, (ci) => [ci, A.starts[ci], A.stops[ci]]);
    this.jumpEnter = A.jump_enter; this.jumpLand = A.jump_land;  // parallel arrays
    this.reset();
  }

  // row accessors (return plain arrays)
  _row(arr, d, f) { const b = f * d, r = new Array(d); for (let i = 0; i < d; i++) r[i] = arr[b + i]; return r; }
  dof(f) { return this._row(this.A.dof, 29, f); }
  dofVel(f) { return this._row(this.A.dofVel, 29, f); }
  simPos(f) { return this._row(this.A.simPos, 3, f); }
  simVel(f) { return this._row(this.A.simVel, 3, f); }
  simTheta(f) { return this.A.simTheta[f]; }
  yawRate(f) { return this.A.yawRate[f]; }
  pelvLocalPos(f) { return this._row(this.A.pelvLocalPos, 3, f); }
  pelvLocalVel(f) { return this._row(this.A.pelvLocalVel, 3, f); }
  pelvLocalRot(f) { return this._row(this.A.pelvLocalRot, 4, f); }
  pelvLocalAng(f) { return this._row(this.A.pelvLocalAng, 3, f); }

  _clipOf(f) { let r = 0; for (let i = 0; i < this.starts.length; i++) { if (this.starts[i] <= f) r = i; else break; } return r; }

  reset() {
    const sf = Math.min(this.stops[0] - 1, this.starts[0] + 30);
    this.animRange = this._clipOf(sf);
    this.animFrame = sf;
    this.rootPos = this.simPos(sf);
    this.rootVel = [0, 0, 0]; this.rootAcc = [0, 0, 0]; this.rootAng = [0, 0, 0];
    this.rootYaw = this.simTheta(sf);
    this.rootRot = quat.yaw(this.rootYaw);
    this.desiredDir = quat.mulVec(this.rootRot, FORWARD);
    this.offDof = new Array(29).fill(0); this.offDofVel = new Array(29).fill(0);
    this.offPP = [0, 0, 0]; this.offPPVel = [0, 0, 0];
    this.offPR = [1, 0, 0, 0]; this.offPAng = [0, 0, 0];
    this.searchTimer = 0;
    this.jumpPending = false; this.jumpLocked = 0;
    this.Tpos = [this.rootPos, this.rootPos, this.rootPos];
    this.Tdir = [this.desiredDir, this.desiredDir, this.desiredDir];
  }

  triggerJump() { if (this.jumpLocked === 0) this.jumpPending = true; }
  get jumping() { return this.jumpLocked > 0; }
  get cur() { return this.animFrame; }
  clipName(f) { return this.clipNames[this._clipOf(f)]; }

  _bestJumpEntry() {
    if (this.jumpEnter.length === 0) return -1;
    let bi = -1, bd = Infinity; const ba = this.animFrame * 27, X = this.X;
    for (let e = 0; e < this.jumpEnter.length; e++) {
      const base = this.jumpEnter[e] * 27; let s = 0;
      for (let i = 0; i < 27; i++) { const d = X[base + i] - X[ba + i]; s += d * d; }
      if (s < bd) { bd = s; bi = this.jumpEnter[e]; this._bestJumpIdx = e; }
    }
    return bi;
  }

  _inertInto(b, rng) {
    const a = this.animFrame;
    const da = this.dof(a), db = this.dof(b), va = this.dofVel(a), vb = this.dofVel(b);
    for (let i = 0; i < 29; i++) { this.offDof[i] += da[i] - db[i]; this.offDofVel[i] += va[i] - vb[i]; }
    const pa = this.pelvLocalPos(a), pb = this.pelvLocalPos(b), qa = this.pelvLocalVel(a), qb = this.pelvLocalVel(b);
    for (let i = 0; i < 3; i++) { this.offPP[i] += pa[i] - pb[i]; this.offPPVel[i] += qa[i] - qb[i]; }
    this.offPR = quat.abs(quat.mul_inv(quat.mul(this.offPR, this.pelvLocalRot(a)), this.pelvLocalRot(b)));
    const aa = this.pelvLocalAng(a), ab = this.pelvLocalAng(b);
    for (let i = 0; i < 3; i++) this.offPAng[i] += aa[i] - ab[i];
    this.animRange = rng; this.animFrame = b;
  }

  _runtimeFeatures(qhCtrl) {
    const X = this.X, Xo = this.Xoffset, Xs = this.Xscale, base = this.animFrame * 27, q = new Array(27);
    for (let i = 0; i < 15; i++) q[i] = X[base + i] * Xs[i] + Xo[i];
    for (let k = 0; k < 3; k++) {
      const dp = quat.invMulVec(qhCtrl, v3.sub(this.Tpos[k], this.rootPos));
      q[15 + 2 * k] = dp[0]; q[16 + 2 * k] = dp[1];
      const dd = quat.invMulVec(qhCtrl, this.Tdir[k]);
      q[21 + 2 * k] = dd[0]; q[22 + 2 * k] = dd[1];
    }
    for (let i = 0; i < 27; i++) q[i] = (q[i] - Xo[i]) / Xs[i];
    return q;
  }

  // desiredVel: [x,y,0] m/s (WASD). desiredFace: [x,y,0] unit or [0,0,0] (arrows).
  step(desiredVel, desiredFace) {
    const X = this.X, H = this.H;
    if (v3.norm(desiredFace) > 0.01) this.desiredDir = v3.scale(desiredFace, 1 / v3.norm(desiredFace));
    else if (v3.norm(desiredVel) > 0.01) this.desiredDir = v3.scale(desiredVel, 1 / v3.norm(desiredVel));
    const desiredRot = quat.yaw(Math.atan2(this.desiredDir[1], this.desiredDir[0]));

    // desired-trajectory springs (per horizon)
    this.Tpos = []; this.Tdir = [];
    for (let k = 0; k < 3; k++) {
      const [P] = trajPos(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.VEL_HL, this.Ttimes[k]);
      this.Tpos.push(P);
      const [Q] = trajRot(this.rootRot, this.rootAng, desiredRot, this.ROT_HL, this.Ttimes[k]);
      this.Tdir.push(quat.mulVec(Q, FORWARD));
    }

    // jump trigger
    if (this.jumpPending && this.jumpLocked === 0) {
      this.jumpPending = false;
      const entry = this._bestJumpEntry();
      if (entry >= 0) {
        const rng = this._clipOf(entry);
        this._inertInto(entry, rng);
        const land = this.jumpLand[this._bestJumpIdx];
        const afterEnd = Math.min(land + 1 + this.PHASE_TD + this.PHASE_AFTER, this.stops[rng] - 1);
        this.jumpLocked = Math.max(1, afterEnd - entry);
        this.searchTimer = this.SEARCH_TIME;
      }
    }

    // search (skipped while riding a jump)
    if (this.jumpLocked === 0 && this.searchTimer <= 0) {
      const qhCtrl = quat.yaw(this.rootYaw);
      const Xq = this._runtimeFeatures(qhCtrl);
      let bestRange = this.animRange, bestFrame = this.animFrame, best;
      if (this.animFrame < this.stops[bestRange] - H) {
        let s = 0; const b = this.animFrame * 27;
        for (let i = 0; i < 27; i++) { const d = Xq[i] - X[b + i]; s += d * d; }
        best = Math.sqrt(s) - this.CURRENT_BIAS;
      } else best = Infinity;
      if (best > 0) {
        let bestSq = best * best;
        for (const [ci, rs, re] of this.search) {
          const lim = re - H;
          for (let f = rs; f < lim; f++) {
            const b = f * 27; let s = 0;
            for (let i = 0; i < 27; i++) { const d = Xq[i] - X[b + i]; s += d * d; if (s >= bestSq) { s = -1; break; } }
            if (s >= 0) { bestSq = s; bestRange = ci; bestFrame = f; }
          }
        }
      }
      if (bestRange !== this.animRange || bestFrame !== this.animFrame) this._inertInto(bestFrame, bestRange);
      this.searchTimer = this.SEARCH_TIME;
    }

    // advance playhead
    this.animFrame = clamp(this.animFrame + 1, this.starts[this.animRange], this.stops[this.animRange] - 1);
    this.searchTimer -= this.DT;
    if (this.jumpLocked > 0) { this.jumpLocked -= 1; if (this.jumpLocked === 0) this.searchTimer = 0; }
    else if (this.animFrame >= this.stops[this.animRange] - 2) this.searchTimer = 0;
    const f = this.animFrame;

    // integrate root from the matched clip's smooth root velocity
    const [, , acc] = trajPos(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.ROT_HL, this.DT);
    this.rootAcc = acc;
    const qhClip = quat.yaw(this.simTheta(f));
    const clipVelLocal = quat.invMulVec(qhClip, this.simVel(f));
    this.rootVel = quat.mulVec(this.rootRot, clipVelLocal);
    this.rootAng = [0, 0, this.yawRate(f)];
    this.rootPos = v3.add(this.rootPos, v3.scale(this.rootVel, this.DT));
    this.rootYaw += this.yawRate(f) * this.DT;
    this.rootRot = quat.yaw(this.rootYaw);

    // inertialize joints + pelvis-local offset, reconstruct pose
    [this.offDof, this.offDofVel] = decayPos(this.offDof, this.offDofVel, this.INERT, this.DT);
    [this.offPP, this.offPPVel] = decayPos(this.offPP, this.offPPVel, this.INERT, this.DT);
    [this.offPR, this.offPAng] = decayRot(this.offPR, this.offPAng, this.INERT, this.DT);

    const dof = this.dof(f), dofOut = new Array(29);
    for (let i = 0; i < 29; i++) dofOut[i] = dof[i] + this.offDof[i];
    const plp = v3.add(this.pelvLocalPos(f), this.offPP);
    const plr = quat.mul(this.offPR, this.pelvLocalRot(f));
    const pelvPos = v3.add(this.rootPos, quat.mulVec(this.rootRot, plp));
    const pelvRot = quat.mul(this.rootRot, plr);

    const qpos = new Float64Array(36);
    qpos[0] = pelvPos[0]; qpos[1] = pelvPos[1]; qpos[2] = pelvPos[2];
    qpos[3] = pelvRot[0]; qpos[4] = pelvRot[1]; qpos[5] = pelvRot[2]; qpos[6] = pelvRot[3];
    for (let i = 0; i < 29; i++) qpos[7 + i] = dofOut[i];
    return qpos;
  }
}

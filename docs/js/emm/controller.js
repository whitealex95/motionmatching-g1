// Environment-aware motion-matching controller in the browser.
// A 1:1 port of emm_g1/controller.py (EMMController): spring-predicted trajectory
// -> query vector -> env-aware search (obstacle penalization + height gating) ->
// inertialized pose. step(leftStick, rightStick) returns a world-space qpos(36).
// A low wall makes the G1 hop over it with no trigger.

import { quat, v3 } from '../quat.js';
import { Environment } from './obstacles.js';
import { searchEnv } from './search.js';
import { jumpIndex } from './jumps.js';
import * as eg from './ellipse_geom.js';
import {
  decaySpringDamper, decaySpringDamperScalar,
  trajectorySpringPosition, trajectorySpringRotation,
} from './springs.js';

const FORWARD = [1, 0, 0];
const yaw = (t) => quat.yaw(t);

// typed-array views into emm.bin per the emm.json header.
export function loadDB(meta, buf) {
  const A = {};
  for (const [name, h] of Object.entries(meta.arrays)) {
    const TA = h.dtype === 'int32' ? Int32Array : Float32Array;
    const count = h.shape.reduce((a, b) => a * b, 1);
    A[name] = count ? new TA(buf, h.offset, count) : new TA(0);
  }
  A.T = meta.n_frames;
  return A;
}

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

export class EMMController {
  constructor(meta, A) {
    this.A = A; this.db = A;
    this.fps = meta.fps; this.DT = 1 / meta.fps;
    this.horizons = meta.horizons;
    this.Ttimes = meta.horizons.map((h) => h / meta.fps);
    this.MAX_SPEED = meta.max_speed; this.WALK_SCALE = meta.walk_scale;
    this.SEARCH_TIME = meta.search_time;
    this.INERT = meta.inert_halflife; this.VELH = meta.vel_halflife; this.ROTH = meta.rot_halflife;
    this.penaltyWeight = meta.penalty_weight; this.evasion = meta.evasion;
    this.anticipation = meta.anticipation; this.threshold = meta.threshold;
    this.heightMode = !!meta.height_mode;
    this.clipNames = meta.clip_names;

    this.Xn = A.Xn; this.Xmean = A.Xmean; this.Xstd = A.Xstd;
    this.dof = A.dof; this.dofVel = A.dofVel;
    this.rootPosDB = A.rootPos; this.rootQuatDB = A.rootQuat;
    this.thetaDB = A.theta; this.rootVelDB = A.rootVel; this.yawRateDB = A.yawRate;
    this.starts = A.starts; this.stops = A.stops;
    this.clipIsJump = A.clip_is_jump;

    this.w0 = Array.from(A.weights);          // base 27 weights
    this.startDirW = this.w0[6];
    this.env = new Environment(meta.obstacles, meta.threshold);

    // --- Jump skill bucket (separate from locomotion, like mm_g1): the search runs
    // over locomotion frames only (locoMask); the jump is entered through its run-up
    // when an obstacle crosses triggerDist ahead -- the obstacle replaces the J key. ---
    const ji = jumpIndex(A, this.clipNames);
    this.jumpEnter = ji.enter; this.landOf = ji.landOf; this.endOf = ji.endOf;
    this.locoMask = new Uint8Array(A.T);
    for (let f = 0; f < A.T; f++) this.locoMask[f] = A.clip_is_jump[A.clip_id[f]] === 1 ? 0 : 1;
    this.triggerDist = meta.trigger_dist !== undefined ? meta.trigger_dist : 0.8;

    this.reset();
  }

  _row(arr, d, f) { const b = f * d, r = new Array(d); for (let i = 0; i < d; i++) r[i] = arr[b + i]; return r; }
  _clipOf(f) { let r = 0; for (let i = 0; i < this.starts.length; i++) { if (this.stops[i] <= f) r = i + 1; else break; } return r; }

  reset(startFrame = null) {
    const f0 = startFrame === null ? this.starts[0] + 30 : startFrame;
    this.animClip = this._clipOf(f0);
    this.animFrame = f0;
    this.rootPos = this._row(this.rootPosDB, 3, f0);
    this.rootYaw = this.thetaDB[f0];
    this.rootRot = yaw(this.rootYaw);
    this.rootVel = [0, 0, 0]; this.rootAcc = [0, 0, 0]; this.rootAng = [0, 0, 0];
    this.desiredDir = quat.mulVec(this.rootRot, FORWARD);
    this.offDof = new Array(29).fill(0); this.offDofVel = new Array(29).fill(0);
    this.offH = 0; this.offHVel = 0;
    this.searchTimer = 0; this.targetSpeed = 0;
    this.jumpLocked = 0; this.prevFd = Infinity;
    this.w = this.w0.slice();
    this.Tpos = this.Ttimes.map(() => this.rootPos.slice());
    this.Tdir = this.Ttimes.map(() => this.desiredDir.slice());
  }

  // Spawn helper (matches run_emm.py): place root at xy, facing +x lane.
  spawn(x, y, yawRad = 0) {
    this.rootPos = [x, y, this.rootPos[2]];
    this.rootYaw = yawRad; this.rootRot = yaw(yawRad);
  }

  get cur() { return this.animFrame; }
  clipName() { return this.clipNames[this.animClip]; }
  // True only while riding the jump skill (run-up -> flight -> landing).
  get jumping() { return this.jumpLocked > 0; }

  // -- jump bucket (obstacle-triggered, mm_g1-style run-up + lock) --
  // Nearest forward distance (m) to an obstacle whose height band a standing body
  // overlaps, along the current facing; Infinity if none ahead. Arms the auto-jump.
  _forwardObstacleDist() {
    const fdir = [Math.cos(this.rootYaw), Math.sin(this.rootYaw)];
    const perp = [-fdir[1], fdir[0]];
    let best = Infinity;
    for (const o of this.env.obs) {
      if (o.h[0] > 1.3 || o.h[1] < 0.0) continue;       // not in standing-body band
      const relx = o.center[0] - this.rootPos[0], rely = o.center[1] - this.rootPos[1];
      const fwd = relx * fdir[0] + rely * fdir[1];
      if (fwd <= 0.0) continue;                          // behind us
      const reach = (o.isEllipse ? Math.max(o.ext[0], o.ext[1]) : o.radius) + 0.5;
      if (Math.abs(relx * perp[0] + rely * perp[1]) > reach) continue;   // off to the side
      if (fwd < best) best = fwd;
    }
    return best;
  }

  // Run-up frame whose features best continue the current pose (smooth take-off).
  _bestJumpEntry() {
    if (this.jumpEnter.length === 0) return null;
    const cb = this.animFrame * 27;
    let best = null, bd = Infinity;
    for (const f of this.jumpEnter) {
      const fb = f * 27; let d = 0;
      for (let i = 0; i < 27; i++) { const x = this.Xn[fb + i] - this.Xn[cb + i]; d += x * x; }
      if (d < bd) { bd = d; best = f; }
    }
    return best;
  }

  // Inertialize the pose discontinuity from the current frame into frame b, move there.
  _switchTo(b) {
    const a = this.animFrame;
    const da = this._row(this.dof, 29, a), dbb = this._row(this.dof, 29, b);
    const va = this._row(this.dofVel, 29, a), vb = this._row(this.dofVel, 29, b);
    for (let i = 0; i < 29; i++) { this.offDof[i] += da[i] - dbb[i]; this.offDofVel[i] += va[i] - vb[i]; }
    this.offH += this.rootPosDB[a * 3 + 2] - this.rootPosDB[b * 3 + 2];
    this.animFrame = b; this.animClip = this._clipOf(b);
  }

  _query(qhCtrl) {
    const q = new Array(27), Xs = this.Xstd, Xm = this.Xmean, b = this.animFrame * 27;
    // trajPos (0:6), trajDir (6:12)
    for (let k = 0; k < 3; k++) {
      const dp = quat.invMulVec(qhCtrl, v3.sub(this.Tpos[k], this.rootPos));
      q[2 * k] = dp[0]; q[2 * k + 1] = dp[1];
      const dd = quat.invMulVec(qhCtrl, this.Tdir[k]);
      q[6 + 2 * k] = dd[0]; q[6 + 2 * k + 1] = dd[1];
    }
    // pose (12:27) de-normalized from Xn then re-normalized below
    for (let i = 12; i < 27; i++) q[i] = this.Xn[b + i] * Xs[i] + Xm[i];
    for (let i = 0; i < 27; i++) q[i] = (q[i] - Xm[i]) / Xs[i];
    return q;
  }

  step(leftStick, rightStick) {
    const DT = this.DT;
    const desiredVel = v3.scale(leftStick, this.MAX_SPEED);
    this.targetSpeed = v3.norm(desiredVel);
    if (v3.norm(rightStick) > 0.01) this.desiredDir = v3.scale(rightStick, 1 / v3.norm(rightStick));
    else if (v3.norm(leftStick) > 0.01) this.desiredDir = v3.scale(leftStick, 1 / v3.norm(leftStick));
    const desiredRot = yaw(Math.atan2(this.desiredDir[1], this.desiredDir[0]));

    // desired trajectory springs (per horizon)
    this.Tpos = []; this.Tdir = [];
    for (let k = 0; k < 3; k++) {
      const [P] = trajectorySpringPosition(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.VELH, this.Ttimes[k]);
      this.Tpos.push(P);
      const [Q] = trajectorySpringRotation(this.rootRot, this.rootAng, desiredRot, this.ROTH, this.Ttimes[k]);
      this.Tdir.push(quat.mulVec(Q, FORWARD));
    }

    // nearby obstacles per tap (in heading-local frame)
    const tapWorld = this.Tpos.map((p) => [p[0], p[1]]);
    const { circ, ell } = this.env.getNearby([this.rootPos[0], this.rootPos[1]], this.rootYaw, tapWorld, this.horizons);

    // Jump bucket trigger: arm a jump the instant a wall crosses triggerDist ahead,
    // entered from its best-matching run-up and ridden through flight + landing
    // (search locked), exactly like mm_g1's J key but auto-triggered by the obstacle.
    if (this.jumpLocked === 0) {
      const fd = this._forwardObstacleDist();
      if (this.prevFd > this.triggerDist && this.triggerDist >= fd) {
        const entry = this._bestJumpEntry();
        if (entry !== null) {
          this._switchTo(entry);
          this.jumpLocked = Math.max(1, this.endOf.get(entry) - entry);
          this.searchTimer = this.SEARCH_TIME;
        }
      }
      this.prevFd = fd;
    }

    // Search (rate limited, LOCOMOTION bucket only): env-aware nearest pose.
    if (this.jumpLocked === 0 && this.searchTimer <= 0) {
      const qhCtrl = yaw(this.rootYaw);
      const Xq = this._query(qhCtrl);
      const pw = this.penaltyWeight * this.anticipation * Math.max(0.5, this.targetSpeed);
      const best = searchEnv(this.db, Xq, this.w, pw, circ, ell, this.threshold, this.heightMode, this.locoMask);
      if (best !== this.animFrame) {
        const a = this.animFrame;
        const da = this._row(this.dof, 29, a), dbb = this._row(this.dof, 29, best);
        const va = this._row(this.dofVel, 29, a), vb = this._row(this.dofVel, 29, best);
        for (let i = 0; i < 29; i++) { this.offDof[i] += da[i] - dbb[i]; this.offDofVel[i] += va[i] - vb[i]; }
        this.offH += this.rootPosDB[a * 3 + 2] - this.rootPosDB[best * 3 + 2];
        this.animFrame = best; this.animClip = this._clipOf(best);
      }
      this.searchTimer = this.SEARCH_TIME;
    }

    this._updateEvasion(circ, ell);

    // advance playhead
    this.animFrame = clamp(this.animFrame + 1, this.starts[this.animClip], this.stops[this.animClip] - 1);
    this.searchTimer -= DT;
    if (this.jumpLocked > 0) {
      this.jumpLocked -= 1;
      if (this.jumpLocked === 0) this.searchTimer = 0;     // search out of the jump at once
    } else if (this.animFrame >= this.stops[this.animClip] - 2) {
      this.searchTimer = 0;
    }
    const f = this.animFrame;

    // integrate root from matched clip root velocity
    const [, , acc] = trajectorySpringPosition(this.rootPos, this.rootVel, this.rootAcc, desiredVel, this.ROTH, DT);
    this.rootAcc = acc;
    const qhClip = yaw(this.thetaDB[f]);
    const clipVelLocal = quat.invMulVec(qhClip, this._row(this.rootVelDB, 3, f));
    this.rootVel = quat.mulVec(this.rootRot, clipVelLocal);
    this.rootAng = [0, 0, this.yawRateDB[f]];
    this.rootPos = v3.add(this.rootPos, v3.scale(this.rootVel, DT));
    this.rootYaw += this.yawRateDB[f] * DT;
    this.rootRot = yaw(this.rootYaw);

    // inertialized pose
    [this.offDof, this.offDofVel] = decaySpringDamper(this.offDof, this.offDofVel, this.INERT, DT);
    [this.offH, this.offHVel] = decaySpringDamperScalar(this.offH, this.offHVel, this.INERT, DT);
    const dofRow = this._row(this.dof, 29, f), dofOut = new Array(29);
    for (let i = 0; i < 29; i++) dofOut[i] = dofRow[i] + this.offDof[i];
    const tilt = quat.mul(quat.inv(qhClip), this._row(this.rootQuatDB, 4, f));
    const rootQuatOut = quat.mul(this.rootRot, tilt);

    const qpos = new Float64Array(36);
    qpos[0] = this.rootPos[0]; qpos[1] = this.rootPos[1];
    qpos[2] = this.rootPosDB[f * 3 + 2] + this.offH;
    qpos[3] = rootQuatOut[0]; qpos[4] = rootQuatOut[1]; qpos[5] = rootQuatOut[2]; qpos[6] = rootQuatOut[3];
    for (let i = 0; i < 29; i++) qpos[7 + i] = dofOut[i];
    return qpos;
  }

  // shrink the facing (trajDir) weight when an obstacle is close (port of _update_evasion)
  _updateEvasion(circ, ell) {
    const thr = this.threshold, f = this.animFrame, A = this.A;
    let closest = Infinity;
    for (let k = 0; k < 3; k++) {
      const center = [A.posTaps[f * 6 + k * 2], A.posTaps[f * 6 + k * 2 + 1]];
      const ax = [A.ellAxis[f * 6 + k * 2], A.ellAxis[f * 6 + k * 2 + 1]];
      const ext = [A.ellExtP[f * 3 + k], A.ellExtS[f * 3 + k]];
      for (const c of circ[k]) {
        let { dist } = eg.pointToEllipse(center, ax, ext, [c[0], c[1]]);
        dist = Math.max(dist - c[2], eg.MAX_INSIDE);
        if (dist < thr) closest = Math.min(closest, dist);
      }
      for (const e of ell[k]) {
        const d = eg.ellipseToEllipse(center, ax, ext, [e[0], e[1]], [e[2], e[3]], [e[4], e[5]]);
        if (d < thr) closest = Math.min(closest, d);
      }
    }
    let target = this.startDirW, rate;
    if (isFinite(closest) && closest < thr) {
      const factor = Math.log10(Math.max(closest / thr, 1e-3)) + 1.0;
      target = this.startDirW * Math.max(this.evasion, factor);
      rate = clamp(this.DT * 10.0, 0, 1);
    } else {
      rate = clamp(this.DT * 100.0, 0, 1);
    }
    const cur = this.w[6];
    const nw = cur + (target - cur) * rate;
    for (let i = 6; i < 12; i++) this.w[i] = nw;
  }
}

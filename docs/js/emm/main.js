// Interactive ENVIRONMENT-AWARE G1 motion-matching demo (Three.js).
// Loads the exported EMM database, runs the JS env-aware controller (a 1:1 port of
// emm_g1/controller.py) at a fixed 30 Hz, forward-kinematics the result and draws
// the full G1 mesh. Low hurdle walls sit in the scene; the G1 jumps over them on
// its own -- there is NO jump key. Reuses ../fk.js and ../quat.js unchanged.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EMMController, loadDB } from './controller.js';
import { fk } from '../fk.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);

const DATA = './data';
const hud = document.getElementById('hud');
const setHud = (t) => { hud.textContent = t; };
async function loadJSON(u) { return (await fetch(u)).json(); }
async function loadBin(u) { return (await fetch(u)).arrayBuffer(); }

async function boot() {
  setHud('loading G1 model + EMM database (~20 MB)…');
  const [model, meta, bin, meshMeta, meshBin] = await Promise.all([
    loadJSON(`${DATA}/model.json`), loadJSON(`${DATA}/emm.json`), loadBin(`${DATA}/emm.bin`),
    loadJSON(`${DATA}/mesh.json`), loadBin(`${DATA}/mesh.bin`),
  ]);
  const A = loadDB(meta, bin);
  const mm = new EMMController(meta, A);
  mm.spawn(-2.5, 0.0, 0.0);                 // start a couple of metres before the lane
  start(model.bodies, mm, meta, meshMeta, meshBin);
}

function start(bodies, mm, meta, meshMeta, meshBuf) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x9a9286);
  scene.fog = new THREE.Fog(0x9a9286, 35, 110);

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 200);
  camera.up.set(0, 0, 1);
  // Behind the spawn, looking down the +x hurdle lane, so the camera-relative WASD
  // is intuitive: W walks forward toward the hurdles ahead. (Drag to orbit.)
  camera.position.set(-6.0, 0.0, 2.0);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(-2.5, 0, 0.8);
  controls.enablePan = false;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x554b40, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(4, -6, 8); sun.castShadow = true;
  sun.shadow.camera.top = 10; sun.shadow.camera.bottom = -10;
  sun.shadow.camera.left = -12; sun.shadow.camera.right = 12;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  // checker floor
  const cv = document.createElement('canvas'); cv.width = cv.height = 256;
  const cx = cv.getContext('2d');
  cx.fillStyle = '#7a7165'; cx.fillRect(0, 0, 256, 256);
  cx.fillStyle = '#5e564c'; cx.fillRect(0, 0, 128, 128); cx.fillRect(128, 128, 128, 128);
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(100, 100); tex.anisotropy = 8;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
  floor.receiveShadow = true; scene.add(floor);

  // ---- hurdle walls (the obstacles the G1 auto-jumps) ----
  const wallMat = new THREE.MeshStandardMaterial({ color: 0xd24f2c, metalness: 0.1, roughness: 0.7 });
  for (const o of meta.obstacles) {
    const len = 2 * o.half_len, thick = 2 * o.half_thick, h = Math.max(0.04, o.hmax - o.hmin);
    const box = new THREE.Mesh(new THREE.BoxGeometry(len, thick, h), wallMat);
    box.position.set(o.cx, o.cy, 0.5 * (o.hmin + o.hmax));
    box.rotation.z = Math.atan2(o.ay, o.ax);       // local x (length) -> wall axis
    box.castShadow = true; box.receiveShadow = true;
    scene.add(box);
  }

  // ---- G1 full mesh ----
  const robot = new THREE.Group(); scene.add(robot);
  const bodyGroups = bodies.map(() => { const g = new THREE.Group(); robot.add(g); return g; });
  for (const gm of meshMeta.geoms) {
    const pos = new Float32Array(meshBuf, gm.vstart * 12, gm.vcount * 3);
    const idx = new Uint16Array(meshBuf, meshMeta.idx_byte_offset + gm.istart * 2, gm.icount);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geo.setIndex(new THREE.BufferAttribute(idx, 1));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(gm.rgba[0], gm.rgba[1], gm.rgba[2]),
      metalness: 0.55, roughness: 0.45, flatShading: true });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true; mesh.receiveShadow = true;
    bodyGroups[gm.body].add(mesh);
  }

  // ---- command-trajectory gizmo ----
  const _yAxis = new THREE.Vector3(0, 1, 0);
  const gizmo = new THREE.Group(); scene.add(gizmo);
  const red = new THREE.MeshBasicMaterial({ color: 0xe21818 });
  const gizSph = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 10), red); gizmo.add(m); return m; });
  const gizStk = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), red); gizmo.add(m); return m; });

  // ---- keyboard ----
  const held = new Set(); let shift = false;
  addEventListener('keydown', (e) => {
    shift = e.shiftKey; const k = e.code;
    if (k === 'Space') { mm.reset(); mm.spawn(-2.5, 0, 0); e.preventDefault(); }
    else if (k === 'KeyT') gizmo.visible = !gizmo.visible;
    else held.add(k);
    if (k.startsWith('Arrow')) e.preventDefault();
  });
  addEventListener('keyup', (e) => { shift = e.shiftKey; held.delete(e.code); });
  addEventListener('resize', () => {
    camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
    renderer.setSize(innerWidth, innerHeight);
  });

  function command() {
    const d = new THREE.Vector3(); camera.getWorldDirection(d);
    let fx = d.x, fy = d.y; const fn = Math.hypot(fx, fy) || 1; fx /= fn; fy /= fn;
    const rx = fy, ry = -fx;
    const fwd = [fx, fy, 0], right = [rx, ry, 0];
    const acc = (v, s) => [v[0] + s[0], v[1] + s[1], 0];
    let move = [0, 0, 0], face = [0, 0, 0];
    if (held.has('KeyW')) move = acc(move, fwd);
    if (held.has('KeyS')) move = acc(move, [-fwd[0], -fwd[1], 0]);
    if (held.has('KeyD')) move = acc(move, right);
    if (held.has('KeyA')) move = acc(move, [-right[0], -right[1], 0]);
    if (held.has('ArrowUp')) face = acc(face, fwd);
    if (held.has('ArrowDown')) face = acc(face, [-fwd[0], -fwd[1], 0]);
    if (held.has('ArrowRight')) face = acc(face, right);
    if (held.has('ArrowLeft')) face = acc(face, [-right[0], -right[1], 0]);
    const mN = Math.hypot(move[0], move[1]);
    // leftStick = unit dir * stick magnitude in [0,1] (controller scales by MAX_SPEED)
    let left = [0, 0, 0];
    if (mN > 1e-6) { const s = (shift ? mm.WALK_SCALE : 1) / mN; left = [move[0] * s, move[1] * s, 0]; }
    const fN = Math.hypot(face[0], face[1]);
    const rightStick = fN > 1e-6 ? [face[0] / fN, face[1] / fN, 0] : [0, 0, 0];
    return { left, rightStick, speed: mN > 1e-6 ? mm.MAX_SPEED * (shift ? mm.WALK_SCALE : 1) : 0 };
  }

  function place(qpos) {
    const { wp, wq } = fk(bodies, qpos);
    for (let i = 0; i < bodies.length; i++) {
      bodyGroups[i].position.set(wp[i][0], wp[i][1], wp[i][2]);
      bodyGroups[i].quaternion.set(wq[i][1], wq[i][2], wq[i][3], wq[i][0]);
    }
  }

  const vP = new THREE.Vector3(), vC = new THREE.Vector3(), vMid = new THREE.Vector3(), vDir = new THREE.Vector3();
  function drawGizmo() {
    for (let k = 0; k < 3; k++) {
      const p = mm.Tpos[k], dir = mm.Tdir[k];
      gizSph[k].position.set(p[0], p[1], 0.05);
      const tip = [p[0] + 0.3 * dir[0], p[1] + 0.3 * dir[1], 0.05];
      vP.set(p[0], p[1], 0.05); vC.set(tip[0], tip[1], 0.05);
      vMid.addVectors(vP, vC).multiplyScalar(0.5); gizStk[k].position.copy(vMid);
      vDir.subVectors(vC, vP).normalize(); gizStk[k].quaternion.setFromUnitVectors(_yAxis, vDir);
      gizStk[k].scale.set(1, vP.distanceTo(vC), 1);
    }
  }

  // fixed-timestep loop with render interpolation (same as the GenoView demo)
  const DT = mm.DT;
  const _q0 = new THREE.Quaternion(), _q1 = new THREE.Quaternion(), _qi = new THREE.Quaternion();
  const _rq = new Float64Array(36);
  function interp(a, b, t) {
    for (let i = 0; i < 3; i++) _rq[i] = a[i] + (b[i] - a[i]) * t;
    _q0.set(a[4], a[5], a[6], a[3]); _q1.set(b[4], b[5], b[6], b[3]);
    _qi.slerpQuaternions(_q0, _q1, t);
    _rq[3] = _qi.w; _rq[4] = _qi.x; _rq[5] = _qi.y; _rq[6] = _qi.z;
    for (let i = 7; i < 36; i++) _rq[i] = a[i] + (b[i] - a[i]) * t;
    return _rq;
  }

  let acc = 0, last = performance.now() / 1000, lastSpeed = 0;
  let curQ = mm.step([0, 0, 0], [0, 0, 0]), prevQ = curQ;
  let fps = 0, fpsN = 0, fpsT = last;
  function frame() {
    const now = performance.now() / 1000;
    acc += Math.min(now - last, 0.1); last = now;
    while (acc >= DT) {
      const c = command(); lastSpeed = c.speed;
      prevQ = curQ; curQ = mm.step(c.left, c.rightStick);
      acc -= DT;
    }
    const rq = interp(prevQ, curQ, acc / DT);
    place(rq); drawGizmo();
    controls.target.lerp(new THREE.Vector3(rq[0], rq[1], 0.8), 0.15);
    controls.update();

    fpsN++; if (now - fpsT >= 0.5) { fps = fpsN / (now - fpsT); fpsN = 0; fpsT = now; }
    const gait = mm.jumping ? 'JUMP' : (lastSpeed > mm.MAX_SPEED * 0.7 ? 'RUN' : (lastSpeed > 1e-3 ? 'WALK' : 'IDLE'));
    setHud(`${gait}  ${lastSpeed.toFixed(1)} m/s   (env-aware: jumps obstacles automatically)\n` +
      `clip: ${mm.clipName()}   frame ${mm.cur}\n` +
      `\nrender ${fps.toFixed(0)} fps · sim ${(1 / DT).toFixed(0)} Hz · search every ${(mm.SEARCH_TIME * 1000).toFixed(0)} ms\n` +
      `\nWASD move · arrows face · Shift walk · Space reset · T gizmo · drag/scroll camera\n` +
      `WALK (Shift) into a wall and the G1 hops it by itself — no jump key`);
    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  setHud(''); frame();
}

boot().catch((e) => { setHud('error: ' + e.message); console.error(e); });

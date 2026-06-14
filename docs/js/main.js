// Interactive G1 motion-matching demo (Three.js). Loads the exported database, runs the
// JS motion matcher (mm.js, a verified 1:1 port of the Python controller) at a fixed 30 Hz,
// forward-kinematics the result, and draws the G1 as an articulated capsule skeleton.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { MotionMatcher, loadDB } from './mm.js';
import { fk } from './fk.js';

THREE.Object3D.DEFAULT_UP.set(0, 0, 1);   // MuJoCo is z-up; render in world coords directly

const DATA = './data';
const hud = document.getElementById('hud');
const setHud = (t) => { hud.textContent = t; };

async function loadJSON(u) { return (await fetch(u)).json(); }
async function loadBin(u) { return (await fetch(u)).arrayBuffer(); }

async function boot() {
  setHud('loading motion database (~13 MB)...');
  const [model, meta, bin] = await Promise.all([
    loadJSON(`${DATA}/model.json`), loadJSON(`${DATA}/mm.json`), loadBin(`${DATA}/mm.bin`),
  ]);
  const A = loadDB(meta, bin);
  const mm = new MotionMatcher(meta, A);
  start(model.bodies, mm);
}

function start(bodies, mm) {
  // ---- renderer / scene / camera (z-up) ----
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x8f877c);                  // warm greige (non-blue)
  scene.fog = new THREE.Fog(0x8f877c, 12, 40);

  const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.05, 200);
  camera.up.set(0, 0, 1);
  camera.position.set(2.6, -2.6, 1.7);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.8);
  controls.enablePan = false;

  // ---- lights + floor ----
  scene.add(new THREE.HemisphereLight(0xffffff, 0x554b40, 0.9));
  const sun = new THREE.DirectionalLight(0xffffff, 1.4);
  sun.position.set(4, -6, 8); sun.castShadow = true;
  sun.shadow.camera.top = 8; sun.shadow.camera.bottom = -8;
  sun.shadow.camera.left = -8; sun.shadow.camera.right = 8;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(200, 200),
    new THREE.MeshStandardMaterial({ color: 0x55504a, roughness: 0.95 }));
  floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(200, 200, 0x6b6359, 0x4a443d);
  grid.rotation.x = Math.PI / 2;                                 // GridHelper is xz; rotate to xy
  scene.add(grid);

  // ---- G1 skeleton: a sphere per body + a capsule bone to its parent ----
  const metal = new THREE.MeshStandardMaterial({ color: 0xb9bcc2, metalness: 0.6, roughness: 0.4 });
  const robot = new THREE.Group();
  scene.add(robot);
  const joints = bodies.map(() => {
    const s = new THREE.Mesh(new THREE.SphereGeometry(0.035, 12, 12), metal);
    s.castShadow = true; robot.add(s); return s;
  });
  const bones = bodies.map((b, i) => {
    if (b.parent < 0) return null;
    const c = new THREE.Mesh(new THREE.CylinderGeometry(0.022, 0.022, 1, 10), metal);
    c.castShadow = true; robot.add(c); return c;
  });
  const _yAxis = new THREE.Vector3(0, 1, 0);

  // ---- command-trajectory gizmo (red spheres + facing sticks) ----
  const gizmo = new THREE.Group(); scene.add(gizmo);
  const red = new THREE.MeshBasicMaterial({ color: 0xe21818 });
  const gizSph = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.SphereGeometry(0.05, 10, 10), red); gizmo.add(m); return m; });
  const gizStk = [0, 1, 2].map(() => { const m = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.012, 1, 6), red); gizmo.add(m); return m; });

  // ---- keyboard ----
  const held = new Set();
  let shift = false;
  addEventListener('keydown', (e) => {
    shift = e.shiftKey;
    const k = e.code;
    if (k === 'Space') { mm.reset(); e.preventDefault(); }
    else if (k === 'KeyJ') mm.triggerJump();
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
    // camera-relative ground frame
    const d = new THREE.Vector3(); camera.getWorldDirection(d);
    let fx = d.x, fy = d.y; const fn = Math.hypot(fx, fy) || 1; fx /= fn; fy /= fn;
    const rx = fy, ry = -fx;                                     // right = forward rotated -90deg
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
    if (mN > 1e-6) { const s = mm.MAX_SPEED * (shift ? mm.WALK_SCALE : 1) / mN; move = [move[0] * s, move[1] * s, 0]; }
    else move = [0, 0, 0];
    const fN = Math.hypot(face[0], face[1]);
    face = fN > 1e-6 ? [face[0] / fN, face[1] / fN, 0] : [0, 0, 0];
    return { move, face, speed: mN > 1e-6 ? mm.MAX_SPEED * (shift ? mm.WALK_SCALE : 1) : 0 };
  }

  // ---- bone placement helper ----
  const vP = new THREE.Vector3(), vC = new THREE.Vector3(), vMid = new THREE.Vector3(), vDir = new THREE.Vector3();
  function place(qpos) {
    const { wp, wq } = fk(bodies, qpos);
    for (let i = 0; i < bodies.length; i++) joints[i].position.set(wp[i][0], wp[i][1], wp[i][2]);
    for (let i = 0; i < bodies.length; i++) {
      const c = bones[i]; if (!c) continue;
      const p = bodies[i].parent;
      vP.set(wp[p][0], wp[p][1], wp[p][2]); vC.set(wp[i][0], wp[i][1], wp[i][2]);
      const len = vP.distanceTo(vC);
      if (len < 1e-5) { c.visible = false; continue; }
      c.visible = true;
      vMid.addVectors(vP, vC).multiplyScalar(0.5); c.position.copy(vMid);
      vDir.subVectors(vC, vP).normalize();
      c.quaternion.setFromUnitVectors(_yAxis, vDir);
      c.scale.set(1, len, 1);
    }
  }

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

  // ---- fixed-timestep loop ----
  const DT = mm.DT;
  let acc = 0, last = performance.now() / 1000, lastSpeed = 0, qpos = mm.step([0, 0, 0], [0, 0, 0]);
  function frame() {
    const now = performance.now() / 1000;
    acc += Math.min(now - last, 0.1); last = now;
    while (acc >= DT) {
      const c = command(); lastSpeed = c.speed;
      qpos = mm.step(c.move, c.face);
      acc -= DT;
    }
    place(qpos);
    drawGizmo();

    // follow camera (keep the pelvis centred; user can still orbit/zoom)
    controls.target.lerp(new THREE.Vector3(qpos[0], qpos[1], 0.8), 0.2);
    controls.update();

    const gait = mm.jumping ? 'JUMP' : (lastSpeed > mm.MAX_SPEED * (1 + mm.WALK_SCALE) / 2 ? 'RUN' : (lastSpeed > 1e-3 ? 'WALK' : 'IDLE'));
    const cid = mm._clipOf(mm.cur);
    const fic = mm.cur - mm.starts[cid];
    setHud(`${gait}  ${lastSpeed.toFixed(1)} m/s\nclip [${cid}]: ${mm.clipNames[cid]}\nframe ${fic} (global ${mm.cur})\n\nWASD move · arrows face · Shift walk\nJ jump · Space reset · T gizmo · drag/scroll camera`);

    renderer.render(scene, camera);
    requestAnimationFrame(frame);
  }
  setHud('');
  frame();
}

boot().catch((e) => setHud('error: ' + e.message));

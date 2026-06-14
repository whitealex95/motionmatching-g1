# G1 Motion Matching — interactive web demo

An in-browser, keyboard-controlled motion-matching demo for the Unitree G1, mirroring the
native `run.py` viewer. The whole motion matcher runs **client-side in JavaScript** — a
verified 1:1 port of the Python controller (`mm_g1/controller.py`, GenoView / Holden
"Simple Motion Matching" with the smoothed sim-root + inertialization architecture) — and
the G1 is drawn as an articulated capsule skeleton with [Three.js](https://threejs.org).

**Controls:** `WASD` move (camera-relative) · `Arrow keys` face (independent of travel) ·
`Shift` walk · `J` jump · `Space` reset · `T` toggle gizmo · drag / scroll to orbit.

## Run locally

```bash
cd docs
python3 -m http.server 8099
# open http://localhost:8099/
```

(Any static file server works — the demo is fully self-contained; three.js is vendored
under `docs/vendor/`, so no CDN or network is needed at runtime.)

## Deploy on GitHub Pages

This is the `gh-pages` branch. In the repo: **Settings → Pages → Build and deployment →
Source: “Deploy from a branch” → Branch: `gh-pages`, folder: `/docs`**. The site publishes
to `https://<user>.github.io/motionmatching-g1/`.

## How it works

```
tools/export_web_data.py   (offline, run with the mujoco env)
   ├─ docs/data/model.json  kinematic tree (bodies: parent, local pos/quat, joint axis)
   ├─ docs/data/mm.json     header: array offsets/shapes + matcher hyperparameters
   └─ docs/data/mm.bin      the feature DB + per-frame pose/sim-root arrays (~13 MB)

docs/js/  (runtime, in the browser)
   ├─ quat.js   quaternion/vec helpers (port of mm_g1/quat.py)
   ├─ mm.js     loadDB + MotionMatcher  (port of mm_g1/controller.py + springs.py)
   ├─ fk.js     forward kinematics       (the formula verified vs MuJoCo to ~1e-7 m)
   └─ main.js   Three.js scene, keyboard, fixed-30 Hz loop, capsule-skeleton render
```

Each frame the JS matcher searches the feature DB (brute-force nearest-neighbour — trivial
at this scale), springs the desired trajectory, inertializes the pose transition, integrates
the smoothed root, and emits a 36-D `qpos`; `fk.js` turns that into world body transforms.

**Verified:** the JS matcher reproduces the Python controller to `1.7e-7` (max `|qpos|`
difference over 250 frames of run / turn / stop / jump), and the FK matches MuJoCo to
`8e-8 m`. See `tools/export_web_data.py` (FK self-check) for the model side.

### Regenerate the data

```bash
~/miniconda3/envs/deploy_mujoco/bin/python tools/export_web_data.py
```

(Re-run whenever the clips, trims, mirror, or feature math change. `LIB_VERSION` in
`mm_g1/config.py` controls the native cache; just re-export for the web.)

## Rendering note & roadmap

The G1 is drawn as a **capsule skeleton** (a sphere per body + a bone to its parent) rather
than the full STL meshes — the menagerie meshes total ~35 MB, too heavy to ship as-is. The
motion is identical either way; only the visuals are simplified. Higher-fidelity options:

- **Decimated glTF meshes** (Draco-compressed) per body, loaded by `fk.js`'s body transforms
  instead of capsules — pixel-accurate G1, a few MB.
- **`mujoco_wasm` + Three.js** (what the
  [php-parkour](https://php-parkour.github.io/) demo uses): load `scene.xml` + the STLs into
  a WebAssembly MuJoCo build and set `qpos` each frame. Reuses the model exactly; heavier setup.

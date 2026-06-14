# GMR-retargeted LAFAN1 motions for the Unitree G1

These are LAFAN1 (Ubisoft La Forge) mocap clips retargeted to the Unitree G1 with
[**GMR** (General Motion Retargeting)](https://github.com/YanjieZe/GMR). They replace the
earlier CSV clips as the locomotion library for this project.

## Files

| file | motion | frames @30fps |
|------|--------|---------------|
| `walk1_subject5.pkl`            | walking            | 7839 (~261 s) |
| `run1_subject5.pkl`             | running            | 7134 (~238 s) |
| `pushAndStumble1_subject5.pkl`  | walk + push/stumble recovery | 6800 (~227 s) |

(GMR also produces an `.mp4` preview per clip in its `motion_data/lafan1-g1/`; only the
`.pkl` motion data is bundled here to keep the repo lean.)

## Format (per pickle, a dict)

| key        | shape    | meaning                                              |
|------------|----------|------------------------------------------------------|
| `fps`      | int      | 30                                                   |
| `root_pos` | (T, 3)   | floating-base position (x, y, z) in world metres     |
| `root_rot` | (T, 4)   | floating-base orientation quaternion, **xyzw**       |
| `dof_pos`  | (T, 29)  | the 29 G1 joint angles (radians), G1 menagerie order |

`mm_g1/data.py` loads these into the project's 36-D `qpos` layout
(`[xyz, quat_wxyz, 29 joints]`) — it concatenates the three arrays and reorders the
quaternion from xyzw to MuJoCo's wxyz. The joint order already matches
`assets/unitree_g1/g1.xml` (verified: per-frame FK puts the feet on the ground).

# Microduck RL

<img width="2215" height="884" alt="image" src="https://github.com/user-attachments/assets/5db7cc83-b3ce-4f7c-83f0-0572a63baed7" />


RL training environments for [Microduck](https://github.com/pollen-robotics/microduck) —
a ~800 g, ~25 cm tall bipedal robot — built on
[mjlab](https://github.com/mujocolab/mjlab) (MuJoCo Warp) with PPO.
Policies are trained here at 50 Hz, exported to ONNX, and deployed on the real
robot by the runtime in [pollen-robotics/microduck](https://github.com/pollen-robotics/microduck).

<!-- HERO VIDEO — real robot montage: walking, standup, roulade, roller skating.
     Keep it short (~30 s) and real-robot-first: this is the "why should I care" shot. -->

https://github.com/user-attachments/assets/50c3d537-8db2-4005-9d9c-3472faeec4d0

The repo encodes the full sim2real recipe: [BAM](https://github.com/micro-zoo/bam)
actuator physics, domain randomization, backlash simulation, and the
reward-design lessons that made it work
(see [AGENTS.md](AGENTS.md) for the distilled playbook).

## Quickstart

Requires a CUDA GPU (training runs through MuJoCo Warp) and [uv](https://docs.astral.sh/uv/).

> **On ARM boxes (DGX Spark / GB10, Jetson):** `uv sync` pulls ~2 GB of CUDA
> wheels on first run and uv's default 30 s HTTP timeout can abort mid-download.
> Export `UV_HTTP_TIMEOUT=600` for the first sync. 

```bash
git clone https://github.com/pollen-robotics/microduck_rl
cd microduck_rl

# train the walking policy (uses your GPU; ~1-2 h for a usable gait at 4096 envs)
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096

# DuckEMW tasks: always smoke-test before a full run
uv run train Mjlab-Sprint-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
uv run train Mjlab-Dance-Flat-MicroDuck --env.scene.num-envs 64 --agent.max-iterations 5
uv run train Mjlab-Dance-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 4000

# watch a trained policy in the viewer
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>

# export to ONNX for deployment
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>

# drive the exported policy in CPU MuJoCo with the keyboard
uv run scripts/infer_policy.py --walking output.onnx
```

Resume from a checkpoint:

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.run-name resume --agent.load-checkpoint model_29999.pt --agent.resume True
```

No GPU? Add `--hf-jobs` to any train command to run it on Hugging Face Jobs
instead of locally (see [scripts/hf/README.md](scripts/hf/README.md)).

## Tasks

`uv run list-envs` prints the live registry. Flat/Rough variants exist where noted.

<!-- SHOWCASE GRID — one short GIF per task family (sim or real), 3 per row.
     Priority order if you only record a few: Velocity, VelStand (fall+recover),
     Roulade, SitStand, Rollers/Swizzle, BallKick. -->

| Task id | Terrain | Description |
|---|---|---|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | flat/rough | **The main task**: walking with velocity commands + head-pose commands |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | flat/rough | Walking + fall recovery in one policy |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | flat/rough | Stand up from face-down/face-up/sitting, then hold the stand + body-pose control |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | flat/rough | Commanded sit ↔ stand in one policy, gently, head commandable |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | flat/rough | Crouch and touch the ground with the mouth tip, return to stand |
| `Mjlab-BallKick-Flat-MicroDuck` | flat | Kick a 70 mm / 15 g ball forward (actor is ball-blind) |
| `Mjlab-Roulade-Flat-MicroDuck` | flat | Forward roll over the head, land back on the feet |
| `Mjlab-Sprint-Flat-MicroDuck` | flat | DuckEMW running curriculum with a world-stabilized, forward-looking head |
| `Mjlab-RunningStableHead-Flat-MicroDuck` | flat | Explicit name for the canonical fixed-gaze sprint task |
| `Mjlab-Running-Flat-MicroDuck` | flat | DuckEMW-compatible running recipe without the head constraint |
| `Mjlab-SprintV4-Flat-MicroDuck` | flat | Earlier DuckEMW sprint retained for comparisons |
| `Mjlab-Dance-Flat-MicroDuck` | flat | Beat-conditioned in-place dance: squat, weight shift, head bob, climax, and call-out |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` | flat | Roller-skate velocity tracking (passive wheels under the feet) |
| `Mjlab-Velocity-Swizzle-MicroDuck` | flat | Classic symmetric swizzle skating |
| `Mjlab-RollerCrouch-Flat-MicroDuck` | flat | Crouch while gliding on rollers |
| `Mjlab-RollerSlope-Flat-MicroDuck` | slope | Glide down slopes on rollers |
| `Mjlab-RollerStandUp-Flat-MicroDuck` | flat | Stand up from the ground onto the wheels |
| `Mjlab-Spin-Flat-MicroDuck` | flat | Fast spin in place on rollers |


### Beat-conditioned dance

The dance policy keeps the shared 61-dimensional actor contract and maps beat phase, tempo, and a three-bit move id into the existing six-dimensional body-command slot. Export and validate a checkpoint against a DuckEMW-compatible beat timeline:

```bash
uv run scripts/export.py Mjlab-Dance-Flat-MicroDuck --checkpoint-file <model.pt>
uv run python scripts/dance_to_timeline.py --policy <dance.onnx> \
    --timeline tests/fixtures/all_moves_120.timeline.json --save-csv dance.csv
uv run python scripts/check_beat_align.py dance.csv \
    --timeline tests/fixtures/all_moves_120.timeline.json
```
At deployment the runtime hot-swaps these policies (walk / recover / trick)
behind a shared 61-dimensional observation contract, so any of them can take
over the robot at any moment. `scripts/infer_policy.py` rehearses exactly that:

```bash
uv run scripts/infer_policy.py --walking walk.onnx --standing stand.onnx \
    --sitstand sitstand.onnx --roulade roulade.onnx --new-cmd-obs
```

Keyboard-driven (velocity commands, `G` ground pick, `Y` sit/stand, `R` roulade,
`K`/`L` kicks); `--debug`, `--save-csv`, `--record` support sim2real comparisons.

### DuckEMW sprint

`Mjlab-Sprint-Flat-MicroDuck` uses a forward-progress curriculum with head-camera horizon and angular-rate constraints. Train from random initialization; do not add resume or checkpoint-loading flags.

```bash
MICRODUCK_RUNNING_TARGET_MAX_SPEED=2.5 \
MICRODUCK_RUNNING_SPEED_CAP=2.6 \
MICRODUCK_RUNNING_HIGH_SPEED_STAGE_INTERVAL=750 \
MICRODUCK_RUNNING_FORWARD_PROGRESS_WEIGHT=5.0 \
MICRODUCK_RUNNING_ACTION_RATE_WEIGHT=-0.10 \
MICRODUCK_RUNNING_CURRICULUM_DIVISOR=8 \
uv run train Mjlab-Sprint-Flat-MicroDuck --env.scene.num-envs 4096 \
    --agent.max-iterations 1700

uv run python scripts/evaluate_running_checkpoint.py \
    --checkpoint-file <model.pt> --task-id Mjlab-Sprint-Flat-MicroDuck \
    --speed 2.2 --num-envs 512 --duration-s 10
```

### MimicKit double spin-kick

`Mjlab-Spinkick-Official-BeyondMimic-MicroDuck` is a direct Microduck adapter
around MJLab's stock BeyondMimic tracking task and G1 PPO recipe. It uses all 14 actuated
joints and ten bodies across both legs, trunk, neck, and head. Metre-based
reward widths and reset noise are normalized by the measured UMR
character-to-Microduck height ratio; angular terms and PPO remain unchanged.
Training first uses MJLab's broad 0.25 m exploration guard, then narrows to an
intermediate 0.15 m physical guard.  Final acceptance is morphology-tolerant:
per-frame retarget errors are diagnostic, while the hard requirements are a
complete physically valid rollout, correct spin direction with substantial
rotation coverage, a distinct single-leg kick, broad full-body similarity,
and a settled standing finish. Training keeps the official adaptive
`MotionCommand` resampling and 10 second episode lifecycle. Playback is finite:
after the final standing frame it is held indefinitely instead of looping back
into the kick.

For high-dynamic clips that collapse to a stationary policy, the registered
`Mjlab-Spinkick-Dense-Start-MicroDuck` stage keeps the same joint-position
action space, PPO, compound BAM physics, and nine BeyondMimic rewards, while
adding two bounded non-saturating full-body tracking scores. They use the
complete three-axis angular velocity and averages over every tracked body and
actuated joint; they do not encode a motion phase, preferred rotation axis,
kick leg, keyframe, or target amplitude. Its rollouts start at frame zero so a
finite deployment sequence is learned end-to-end instead of being hidden by
adaptive-RSI performance on later phases.

The converter accepts an official UMR result directly. It repeats the original
60 Hz action at 1.0x to 2.65 seconds, adds the same 0.5 second entry/exit
transitions and 1.0 second standing tail as `g1_spinkick_example`, resamples to
50 Hz, and applies a final hard joint velocity/acceleration/jerk audit. A
robot-level 2.5 mm root-Z calibration removes the sole-mesh floor penetration
without changing the jump excursion, velocity, timing, or pose. The source's
aerial root trajectory is otherwise preserved. For severe morphology changes,
the UMR solve also uses a robot-agnostic source-root orientation prior. This
prevents the free root from rotating the entire target robot sideways merely to
reduce surface-correspondence error; it does not add a Spin-specific pose or RL
reward.

```bash
uv run convert-umr-motion \
    --input artifacts/retarget/mimickit_spinkick_umr60/umr_result.npz \
    --output artifacts/motions/mimickit_spinkick_microduck_umr60_g1pad/motion.npz \
    --duration 2.65 --transition-duration 0.5 --hold-duration 1.0 \
    --output-fps 50

MUJOCO_GL=egl uv run python scripts/render_spinkick_reference.py \
    --motion artifacts/motions/mimickit_spinkick_microduck_umr60_g1pad/motion.npz \
    --output artifacts/videos/mimickit_spinkick_reference.mp4

uv run train Mjlab-Spinkick-Official-BeyondMimic-MicroDuck \
    --env.commands.motion.motion-file \
      artifacts/motions/mimickit_spinkick_microduck_umr60_g1pad/motion.npz \
    --env.scene.num-envs 4096 --agent.max-iterations 20000

# Optional PPO continuation curriculum for preserving a moving initialization.
# This is NOT part of UMR: UMR never changes gravity. A checkpoint produced by
# this task is intermediate-only and cannot be used for final evaluation or a
# deliverable video. Resume it in the Earth-gravity task above; final training,
# evaluation, and video acceptance must all use gravity=9.81 m/s^2.
uv run train Mjlab-Spinkick-Official-ScaledGravity-MicroDuck \
    --env.commands.motion.motion-file \
      artifacts/motions/mimickit_spinkick_microduck_umr60_g1pad/motion.npz \
    --env.scene.num-envs 4096 --agent.max-iterations 20000

MUJOCO_GL=egl uv run evaluate-spinkick \
    --checkpoint-file <model.pt> \
    --motion-file artifacts/motions/mimickit_spinkick_microduck_umr60_g1pad/motion.npz \
    --output-file artifacts/evaluations/spinkick.json \
    --video-file artifacts/evaluations/spinkick.mp4

# For a comparison video's physical-result panel, hide the reference ghost.
MUJOCO_GL=egl uv run evaluate-spinkick \
    --checkpoint-file <earth-gravity-model.pt> \
    --task-id Mjlab-Spinkick-Dense-Start-MicroDuck \
    --no-show-reference \
    --video-file artifacts/evaluations/spinkick_earth_physics_only.mp4

# Diagnostic only: keep simulating after a fall to expose the real remaining
# trajectory.  This mode is deliberately prevented from passing acceptance.
MUJOCO_GL=egl uv run evaluate-spinkick \
    --checkpoint-file <model.pt> \
    --task-id Mjlab-Spinkick-Residual-Nominal-Guided-MicroDuck \
    --diagnostic-no-early-termination
```

### Backlash variants

Every main task has a **Backlash** twin that trains on a model with ±1° of gear
play (2° total) in series with each of the 14 servo joints: insert `-Backlash`
before `MicroDuck` in the task id, e.g. `Mjlab-Velocity-Flat-Backlash-MicroDuck`.

The backlash is modeled properly for sim2real: each servo gets an unactuated
`passive_<joint>_backlash` hinge, and because the real encoder sits on the
output side of the play, both the firmware PD emulation
(`BacklashEncoderBamActuator`) and the `joint_pos`/`joint_vel` observations
read *through* the backlash (`qpos[servo] + qpos[backlash]`). Observation and
action dims are unchanged, so ONNX export and the runtime need no changes.
See `src/mjlab_microduck/tasks/backlash.py`.

## Actuator model

All tasks use the [BAM](https://github.com/micro-zoo/bam) M6 actuator model for
the Dynamixel XL330 (voltage control law, back-EMF, Coulomb/Stribeck/load-dependent
friction), with per-env domain randomization on battery voltage, voltage sag
under load, command delay, and friction magnitude
(`FrictionDRBamActuator` in `src/mjlab_microduck/actuator/`).

At this scale — tiny servos driving a ~800 g biped — actuator fidelity is most
of the sim2real gap, which is why the actuator is modeled down to its voltage
control law instead of an ideal PD.

## Robot models

MJCF models live in `src/mjlab_microduck/robot/microduck/` and are exported
from Onshape with [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot),
one `config_mjcf_*.json` per model:

| XML | Used by |
|---|---|
| `robot_walk.xml` | Velocity (stripped trunk/head contacts — falling is cheap) |
| `robot_allcollisions.xml` | VelStand, StandUp, SitStand, GroundPick, BallKick, Roulade (body can physically lie on the ground) |
| `robot_allcollisions_rollers.xml` | Roller tasks (passive wheels) |
| `robot_*_backlash.xml` | Backlash task variants (generated by `add_backlash.py`) |

`scene*.xml` files wrap the robots with a floor + keyframes (STAND/SIT/FOLD)
for quick viewing and for `infer_policy.py`.

<!-- IMAGE — side-by-side render: walk model vs rollers model (or a collision-geom
     visualization). One image here makes the model-variant story instant. -->

## Project structure

```
src/mjlab_microduck/
├── robot/
│   ├── microduck/                    # MJCF exports, export configs, scenes, add_backlash.py
│   └── microduck_constants.py        # robot cfgs, HOME frame, BAM actuator cfg
├── actuator/friction_dr_bam.py       # BAM + friction DR + backlash encoder feedback
├── tasks/
│   ├── __init__.py                   # task registration (base + backlash variants)
│   ├── mdp.py                        # rewards, events, observations, custom classes
│   ├── backlash.py                   # make_backlash_variant() env-cfg wrapper
│   └── microduck_*_env_cfg.py        # one cfg module per task family
├── train_cli.py                      # `train` entry point (+ --hf-jobs)
└── hf_jobs.py                        # Hugging Face Jobs submission
```

Conventions worth knowing:

- The observation layout is shared across every policy (61-dim actor obs:
  48 proprioception + commands `[twist(3), head_pose(4), body_pose(6)]`), which
  is what makes runtime policy hot-swapping possible. Envs that don't use a
  command slot zero-pad it rather than dropping it.
- Unactuated joints are all named `passive_*` (roller wheels, backlash
  hinges); actuators, joint observations and pose rewards select servo joints
  with `^(?!passive_).*`.
- Domain-randomization toggles are `ENABLE_*` booleans at the top of each
  env cfg file.
- Joint layout (14 servos): 0–4 left leg (hip_yaw, hip_roll, hip_pitch, knee,
  ankle), 5–8 neck/head (neck_pitch, head_pitch, head_yaw, head_roll),
  9–13 right leg.
- The exporter bakes the observation normalizer into the ONNX graph — always
  deploy ONNX produced by `scripts/export.py`, never a hand-converted
  checkpoint, or the policy sees unnormalized observations at runtime.

[AGENTS.md](AGENTS.md) documents the env-building workflow and the reward-design
rules learned across the project (also aimed at AI coding agents working in
this repo).

## Tests

```bash
uv run --with pytest pytest tests/
```

CPU-only config-invariant and reward-function regression tests — they lock in
joint-index mappings, reward sign conventions, and NaN guards.

## Related projects

- [microduck](https://github.com/pollen-robotics/microduck) — the Microduck project home, including the onboard runtime that runs the exported policies
- [mjlab](https://github.com/mujocolab/mjlab) — the training framework (MuJoCo Warp + rsl_rl)
- [BAM](https://github.com/micro-zoo/bam) — better actuator models, by Rhoban

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
3D model files are licensed under Creative Commons BY-SA-NC.

The sprint and beat-conditioned dance tasks are adapted from [DuckEMW](https://github.com/emwstudio/DuckEMW).
Source provenance and licensing are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

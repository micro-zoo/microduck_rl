# Third-party notices

## DuckEMW dance task

The beat-conditioned dance task in `src/mjlab_microduck/tasks/microduck_dance_env_cfg.py`,
the corresponding additions in `src/mjlab_microduck/tasks/mdp.py` and task registry,
the deterministic timeline evaluator (`scripts/dance_to_timeline.py` and
`scripts/check_beat_align.py`), and their tests/fixture are adapted from
[emwstudio/DuckEMW](https://github.com/emwstudio/DuckEMW), commit
[`f0e08bfa8916e4ccee4aaa7960420d77c499a68b`](https://github.com/emwstudio/DuckEMW/tree/f0e08bfa8916e4ccee4aaa7960420d77c499a68b),
whose `third_party/microduck_rl` submodule was at
[`375bdd4396cd1ec3c9e1224c525adfd9e97496b7`](https://github.com/emwstudio/microduck_rl/tree/375bdd4396cd1ec3c9e1224c525adfd9e97496b7).

DuckEMW states that its code is Apache License 2.0 and that its inherited 3D
model assets are CC BY-SA-NC. This integration imports no 3D assets or audio;
it remains covered by this repository's [Apache-2.0 LICENSE](LICENSE).

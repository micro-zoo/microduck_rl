# Third-party notices

## DuckEMW sprint task

The barefoot sprint task in `src/mjlab_microduck/tasks/microduck_sprint_env_cfg.py`,
the corresponding reward/curriculum additions in `src/mjlab_microduck/tasks/mdp.py`
and task registry, the Warp evaluator (`scripts/eval_sprint_speed.py`), and its
configuration tests are adapted from
[emwstudio/DuckEMW](https://github.com/emwstudio/DuckEMW), commit
[`4e4a7c695a83c23f4b00ba7916317cc26b6e636b`](https://github.com/emwstudio/DuckEMW/tree/4e4a7c695a83c23f4b00ba7916317cc26b6e636b),
whose `third_party/microduck_rl` submodule was at
[`c9cf87f085bb461209b20e7ef09c232186a58ec7`](https://github.com/emwstudio/microduck_rl/tree/c9cf87f085bb461209b20e7ef09c232186a58ec7).

The canonical `Mjlab-Sprint-Flat-MicroDuck` task additionally ports the final
Hannes/Vottivott Running recipe and deterministic checkpoint evaluator from
[Vottivott/microduck-playground](https://github.com/Vottivott/microduck-playground),
current source commit
[`748a6a8ad16c5e5796f5960d0957861ac1329810`](https://github.com/Vottivott/microduck-playground/tree/748a6a8ad16c5e5796f5960d0957861ac1329810).

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

# AshCrown-RL

Python-side reinforcement-learning clients, environments and training
pipeline for **AshCrown** (Godot 4.7.2). This repository is mounted as the
`training/` **git submodule** of the game repository.

## Scope

This repository owns everything on the **non-Godot** side of RL:

- the TCP protocol client for the headless `RLEnv` bridge;
- the Gymnasium `AshCrownEnv` and vector-env wrappers;
- scripted baselines, evaluation, replay and state-hash verification;
- PPO/training scripts, curriculum and checkpoints;
- Python tests, benchmarks and training docs.

The Godot side stays in the game repository:

- `res://scenes/rl/rl_env.tscn` and `res://scripts/rl/` — the headless
  environment, observation/reward construction and protocol server;
- the GDScript RL tooling under `res://tools/rl/`;
- the environment contract/spec: `docs/07-rl-headless-env-spec.md`
  (in the game repository).

No game code is duplicated here. The Python side always drives a real Godot
process over the protocol.

## Environment management: uv

This repository is managed with [uv](https://docs.astral.sh/uv/). The host
Python is 3.14; training is pinned to **Python 3.12** for PyTorch /
Stable-Baselines3 compatibility, and uv manages that interpreter.

```bash
# one-time: clone the game repo with its submodules
git clone --recurse-submodules git@github.com:Public-Cat/AshCrown.git
cd AshCrown/training

uv python install 3.12
uv sync --extra dev            # numpy + gymnasium + test/lint tools
# optional, for PPO:
uv sync --extra dev --extra train   # + stable-baselines3 + torch

uv run pytest
uv run ruff check .
```

`uv.lock` is committed; `.venv/` is ignored. Do not commit virtualenvs,
checkpoints, run logs or downloaded models.

## Running against the game

The client launches the Godot RL scene directly (not the normal game shell):

```bash
# from training/
ASHCROWN_PROJECT=.. uv run python -m ashcrown_rl.evaluate --help
```

The launcher defaults `--game-project` to `..` (the parent game checkout) and
`--godot-bin` to `godot` (or the exported `ashcrown.x86_64` with
`--exported-binary`). The exact flags are defined in the game spec:

```text
godot --headless --fixed-fps 60 --disable-vsync --audio-driver Dummy \
  --path <game-project> res://scenes/rl/rl_env.tscn -- \
  --rl-server=127.0.0.1:0 --rl-port-file=<tmp>
```

## Compatibility contract

The game-side handshake is authoritative. The Python client must:

- refuse a `protocol` / `obs_schema` / `action_schema` mismatch;
- build observation slices from the handshake `layout` (never hardcode
  offsets);
- verify the `layout_hash`;
- store `protocol`, `obs_schema` and `layout_hash` in every checkpoint and
  refuse to evaluate a checkpoint against a different layout.

See `compat.json` for the currently targeted versions. Bump it in the same
change that bumps the game-side schema.

## Status

Bootstrap only: package skeleton + uv project. The protocol client, Gymnasium
environment and training scripts land in game tickets **RL-3** / **RL-4**;
the bot/training pipeline is **RL-7**. The authoritative design is
`docs/07-rl-headless-env-spec.md` in the game repository.

## Licence

The AshCrown game and this training client are private project code. The
`LordsOfPain-OldSchoolIsometricAssets` asset pack is **not** redistributed
here; it remains in the game repository under its own licence.

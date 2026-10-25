# AshCrown-RL

Python-side reinforcement-learning clients, environments and training
pipeline for **AshCrown** (Godot 4.7.2). This repository is mounted as the
`training/` **git submodule** of the game repository.

## Scope

This repository owns everything on the **non-Godot** side of RL:

- the TCP protocol client for the headless `RLEnv` bridge (`protocol.py`);
- the process launcher (`launch.py`) and the step benchmark (`bench.py`);
- the Gymnasium `AshCrownEnv` (`env.py`), its layout/action helpers
  (`spaces.py`) and the vector wrapper (`vec.py`);
- the scripted heuristic baseline (`baseline.py`), fixed-seed evaluation
  (`evaluate.py`) and state-hash replay (`replay.py`);
- examples, Python tests and the docs below.

The Godot side stays in the game repository: `res://scenes/rl/rl_env.tscn`,
`res://scripts/rl/`, `res://tools/rl/` and the design contract
`docs/07-rl-headless-env-spec.md`. No game code is duplicated here; every
Python API drives a real headless Godot process over the protocol.

## Environment management: uv

The host Python is 3.14; training is pinned to **Python 3.12** for PyTorch /
Stable-Baselines3 compatibility, and [uv](https://docs.astral.sh/uv/) manages
that interpreter.

```bash
git submodule update --init --recursive      # first checkout
cd training

uv python install 3.12
uv sync --extra dev                          # numpy + gymnasium + pytest/ruff
uv sync --extra dev --extra train            # + stable-baselines3 + torch (RL-7)
```

`uv.lock` is committed; `.venv/`, checkpoints and run logs are ignored.

## Quick start

```python
from ashcrown_rl import AshCrownEnv
from ashcrown_rl.baseline import HeuristicPolicy

env = AshCrownEnv(mode="level", level=1, frame_skip=4)  # launches one Godot process
policy = HeuristicPolicy(env.layout)
obs, info = env.reset(seed=20260101)
while not env.episode_done:
    obs, reward, terminated, truncated, info = env.step(policy(obs, info, env.layout))
print(info["terminal_reason"], info["kills"], info["state_hash"])
env.close()
```

```bash
# one environment, 500 steps, prints the reward components
uv run python examples/random_agent.py --steps 500

# the scripted baseline on one seed
uv run python examples/heuristic_agent.py --seed 5

# fixed-seed evaluation -> JSON artifact
uv run python -m ashcrown_rl.evaluate --policy heuristic --level 1 \
    --seed-start 1 --seed-count 10 --repeat 2 --out /tmp/baseline.json

# record an action log and replay it against a fresh process
uv run python -m ashcrown_rl.replay record --seed 7 --steps 400 --out /tmp/ep.json
uv run python -m ashcrown_rl.replay replay --file /tmp/ep.json

# throughput
uv run python -m ashcrown_rl.bench --steps 1000                 # 1 env
uv run python -m ashcrown_rl.bench --steps 200 --envs 8         # 8 envs aggregate
```

The launcher discovers the game checkout (`ASHCROWN_PROJECT`, default: the
parent of `training/`) and the binary (`ASHCROWN_GODOT`, `GODOT_BIN`, then
`godot` on `PATH`). Every command accepts `--game-project` / `--godot-bin`.

## Environment contract

`AshCrownEnv` is one episode stream over one Godot process (`docs/07` §9.2):

| API | Behaviour |
|---|---|
| `observation_space` | `Box(-1, 1, (obs_len,), float32)` built from the handshake: `obs_len = 3254` at the default config (grid on, radius 10), `608` with `obs_grid=False` |
| `action_space` | `MultiDiscrete([9, 2, 2])` → `[move, attack, interact]` |
| `metadata` | `render_modes`, `obs_layout` (block table, grid geometry, observation config, layout hash), `action_meanings` |
| `reset(seed=None, options={})` | seeds the engine episode; options: `level`, `character`, `max_ticks`, `reward_profile`, `reward_overrides`, `merchant_policy` |
| `step(action)` | one decision window = `frame_skip` physics ticks; returns the Gym 5-tuple |
| auto-reset | **never** — call `reset()`; the vector wrapper auto-resets |
| `close()` | sends `close`, reaps the process, idempotent |

`render_mode` is `None` (headless only). The **observation config** is a
constructor argument (RL-6), passed to the engine at launch so the handshake
layout and the Gym space always agree — and a lifetime key of the process:

| option | default | accepted |
|---|---|---|
| `obs_grid` | `True` | `False` drops the grid block from the vector (no zeros) |
| `obs_grid_radius` | `10` | `4..21` |
| `perception_px` | `900.0` | `100..5000` (`300` = strict / human-visible) |
| `expose_portal` | `True` | `False` zeroes the portal sub-block |
| `obs_blocked` | `False` | adds a physics-query `blocked` grid channel (costly, see the ablation) |

```python
env = AshCrownEnv(obs_grid_radius=21, perception_px=300.0)
print(env.obs_len, env.layout.observation_config)  # 11702, {...}
```

A non-default config has its own `layout_hash`. `compat.json` pins the default
one; a custom config is verified by exact `observation_config` equality.

Determinism (`docs/07` §10): the same seed + the same action sequence produce
the same `info["state_hash"]` sequence across processes — including across
descends, so a run-mode log can be recorded and replayed end to end. The engine
runs one discarded warm-up episode before serving, and every launch uses
`--headless --fixed-fps 60 --disable-vsync`.

### Run mode and the merchant sub-policy (RL-5)

`mode="level"` (default) is one floor: the portal is terminal
(`terminal_reason="level_clear"`). `mode="run"` plays the whole ten-level
descent in one episode (`docs/07` §4.1):

```python
env = AshCrownEnv(mode="run", merchant_policy="auto_heal", frame_skip=4)
obs, info = env.reset(seed=20260101)  # level 1, auto_descend on
while not env.episode_done:
    obs, reward, terminated, truncated, info = env.step(policy(obs, info, env.layout))
    info["level"]  # 1..10, follows the descent
    info["gold"]  # carried across floors (same RunState)
print(info["terminal_reason"])  # "victory" | "death" | "run_died" | "truncated"
env.close()
```

- **Start / cap:** level 1, `max_ticks` defaults to **144,000** (40 game-minutes)
  instead of 18,000; `level` is ignored in run mode (a run always starts at 1).
- **Descends:** entering a cleared floor's portal builds the next level and the
  episode continues; the engine re-acquires the new floor, so `info["level"]`
  and the observation's `level` feature follow it. Gold, gems, kills and the
  seed live on `RunState` and are **preserved** across floors; each floor has a
  fresh `visited` plane.
- **Terminals:** victory = the level-10 demonlord death (`Events.run_ended`),
  death = `Events.player_died` (terminated immediately, no summary screen), and
  `run_ended(false)` = `"run_died"`. A portal is never terminal in run mode;
  `max_ticks` truncates and a terminal on the same tick wins.
- **Merchant:** the economy is **not** in the action space (RL-6 owns the
  observation, RL-5 owns this decision). `merchant_policy` is `"none"`
  (default: no interaction) or `"auto_heal"`: when the player is inside the
  merchant's interaction range, hurt and can afford the 40-gold full heal, the
  engine buys it once per shop/floor visit. `info["events"]["merchant_bought"]`
  and `info["reward_components"]` (via the ordinary heal/spent-gold rules)
  record it, and `info["merchant_policy"]` echoes the active policy.
- **Unknown values** (`mode`, `merchant_policy`) are a `BAD_CONFIG` at reset —
  the handshake publishes the accepted tables as `modes` / `merchant_policies`.

```bash
# honest long rollout of the heuristic through a real run (levels reached,
# terminal, summed reward components, throughput)
uv run python scripts/run_mode_smoke.py --decisions 30000 --seed 1 \
    --json /tmp/run_mode.json

# record and replay a run-mode episode that crosses a descend
uv run python -m ashcrown_rl.replay record --mode run --steps 1200 \
    --merchant-policy auto_heal --out /tmp/run.json
uv run python -m ashcrown_rl.replay replay --file /tmp/run.json
```

Measured on seeds 1–5 (`docs/artifacts/RL/run_mode_heuristic.csv`): the
heuristic clears level 1 on 4/5 seeds and reaches level 3 on seeds 2 and 5
(~250–275 decisions/s); seed 3 is its known level-1 stall. The GUT scripted
clear harness (`tests/integration/test_rl_run_mode.gd`) drives the *real*
environment from level 1 to the **victory** terminal on seed 20260101.

## Observation, action and reward reference

Generated from the handshake layout table for the **default** observation
config (`layout_hash
fbcc9cdae85af53a8b0ee38ad6b81137168bb7fc2639dc1966cca75d01560d15`, grid on,
radius 10, perception 900, portal assist on, blocked off) — the client never
hardcodes an offset, it slices by block name. A different config publishes its
own offsets/`obs_len`/hash; re-run `godot --headless --path . -s
tools/rl/dump_layout.gd -- --write-fixture` in the game repo after a schema
change to refresh `tests/data/layout.json`.

### Blocks

| block | offset | size | shape |
|---|---|---|---|
| `global` | 0 | 31 | (31,) |
| `player` | 31 | 24 | (24,) |
| `enemies` | 55 | 468 | (12, 39) |
| `pickups` | 523 | 36 | (4, 9) |
| `hazards` | 559 | 32 | (4, 8) |
| `merchant` | 591 | 5 | (5,) |
| `events` | 596 | 12 | (12,) |
| `grid` | 608 | 2646 | (6, 21, 21) |

### Global block (0–30, 31 floats)

| # | feature | normalisation |
|---|---|---|
| 1 | `level` | ×10 |
| 2 | `tick` | 0–1 |
| 3 | `game_seconds` | ×600 |
| 4 | `mode_level` | 0–1 |
| 5 | `mode_run` | 0–1 |
| 6 | `mode_combat` | 0–1 |
| 7 | `wave_index` | 0–1 |
| 8 | `wave_active` | 0–1 |
| 9 | `level_cleared` | 0–1 |
| 10 | `enemies_spawned` | ×30 |
| 11 | `enemies_alive` | ×30 |
| 12 | `kills` | ×100 |
| 13 | `gold` | ×500 |
| 14 | `gem_red` | ×50 |
| 15 | `gem_blue` | ×50 |
| 16 | `gem_green` | ×50 |
| 17 | `gem_white` | ×50 |
| 18 | `boss_hp` | 0–1 |
| 19 | `boss_phase` | 0–1 |
| 20 | `boss_aoe_timer` | ×10 |
| 21 | `portal_active` | 0–1 |
| 22 | `portal_rel_x` | ×32 |
| 23 | `portal_rel_y` | ×32 |
| 24 | `portal_distance` | ×32 |
| 25 | `merchant_present` | 0–1 |
| 26 | `merchant_rel_x` | ×32 |
| 27 | `merchant_rel_y` | ×32 |
| 28 | `merchant_distance` | ×32 |
| 29 | `nearest_enemy_distance` | ×32 |
| 30 | `nearest_enemy_bearing_sin` | 0–1 |
| 31 | `nearest_enemy_bearing_cos` | 0–1 |

### Player block (31–54, 24 floats)

| # | feature | normalisation |
|---|---|---|
| 1 | `alive` | 0–1 |
| 2 | `hp_frac` | 0–1 |
| 3 | `hp_abs` | ×400 |
| 4 | `max_hp` | ×400 |
| 5 | `armor` | 0–1 |
| 6 | `damage` | ×80 |
| 7 | `move_speed` | ×500 |
| 8 | `attack_speed_mult` | ×3 |
| 9 | `cell_x` | 0–1 |
| 10 | `cell_y` | 0–1 |
| 11 | `facing_sin` | 0–1 |
| 12 | `facing_cos` | 0–1 |
| 13 | `state_idle` | 0–1 |
| 14 | `state_walk` | 0–1 |
| 15 | `state_attack` | 0–1 |
| 16 | `state_dead` | 0–1 |
| 17 | `attack_progress` | 0–1 |
| 18 | `attack_hitbox_active` | 0–1 |
| 19 | `attack_queued` | 0–1 |
| 20 | `knocked_back` | 0–1 |
| 21 | `attack_reach` | ×100 |
| 22 | `attack_speed_bonus` | ×0.5 |
| 23 | `velocity_x` | ×500 |
| 24 | `velocity_y` | ×500 |

### Enemy slot (55–522, 468 floats)

| # | feature | normalisation |
|---|---|---|
| 1 | `present` | 0–1 |
| 2 | `type_slime_green` | 0–1 |
| 3 | `type_slime_red` | 0–1 |
| 4 | `type_slime_black` | 0–1 |
| 5 | `type_skeleton_normal` | 0–1 |
| 6 | `type_skeleton_armored` | 0–1 |
| 7 | `type_skeleton_swift` | 0–1 |
| 8 | `type_elite_skeleton` | 0–1 |
| 9 | `type_elite_slime` | 0–1 |
| 10 | `type_demonlord` | 0–1 |
| 11 | `rel_x` | ×32 |
| 12 | `rel_y` | ×32 |
| 13 | `distance` | ×32 |
| 14 | `bearing_sin` | 0–1 |
| 15 | `bearing_cos` | 0–1 |
| 16 | `hp_frac` | 0–1 |
| 17 | `armor` | 0–1 |
| 18 | `move_speed` | ×500 |
| 19 | `damage` | ×80 |
| 20 | `state_idle` | 0–1 |
| 21 | `state_chase` | 0–1 |
| 22 | `state_windup` | 0–1 |
| 23 | `state_attack` | 0–1 |
| 24 | `state_recover` | 0–1 |
| 25 | `state_stagger` | 0–1 |
| 26 | `state_dead` | 0–1 |
| 27 | `state_other` | 0–1 |
| 28 | `state_progress` | 0–1 |
| 29 | `windup_remaining` | ×3 |
| 30 | `attack_hitbox_active` | 0–1 |
| 31 | `attack_cooldown` | ×5 |
| 32 | `facing_sin` | 0–1 |
| 33 | `facing_cos` | 0–1 |
| 34 | `is_elite` | 0–1 |
| 35 | `is_boss` | 0–1 |
| 36 | `asleep` | 0–1 |
| 37 | `knocked_back` | 0–1 |
| 38 | `attack_reach` | ×200 |
| 39 | `scale` | ×2 |

### Pickup, hazard and merchant blocks

`pickups`: 4 slots × 9 = 36 floats — `present`, `kind_gold`, `kind_gem_red`, `kind_gem_blue`, `kind_gem_green`, `kind_gem_white`, `rel_x`, `rel_y`, `distance`
`hazards`: 4 slots × 8 = 32 floats — `present`, `rel_x`, `rel_y`, `radius`, `remaining`, `kind`, `telegraph_fraction` (remaining/total), `coverage` (1 − player_distance / (radius + 8 px))

`merchant`: 5 floats — `present`, `rel_x`, `rel_y`, `distance`, `within_interact_range`

`events`: 12 floats — `damage_dealt`, `damage_taken`, `kills`, `gold_gained`, `gems_gained`, `attacks_started`, `hits_landed`, `player_healed`, `level_cleared`, `died`, `portal_entered`, `merchant_bought`

`grid`: `(channels, 2R+1, 2R+1)` channel-major, channels `floor`, `wall`, `solid_prop`, `doorway`, `hazard`, `visited` (+ `blocked` when `obs_blocked=True`); absent entirely when `obs_grid=False`.

### Actions (`MultiDiscrete([9, 2, 2])`)

| move | screen vector | iso grid step |
|---|---|---|
| 0 | (0.0, 0.0) | (0, 0) |
| 1 | (1.0, 0.0) | (1, -1) |
| 2 | (0.893999993801117, 0.446999996900558) | (1, 0) |
| 3 | (0.0, 1.0) | (1, 1) |
| 4 | (-0.893999993801117, 0.446999996900558) | (0, 1) |
| 5 | (-1.0, 0.0) | (-1, 1) |
| 6 | (-0.893999993801117, -0.446999996900558) | (-1, 0) |
| 7 | (0.0, -1.0) | (-1, -1) |
| 8 | (0.893999993801117, -0.446999996900558) | (0, -1) |

`info` (not learned) carries raw units (`hp`, `gold`, `kills`,
`portal_distance`, `game_seconds`), the reward components, the per-window
events, `state_hash`, `terminal_reason` and the layout pins
(`obs_schema`, `layout_hash`, `layout_slices`).

Reward (`reward_profile="combat"`, every coefficient overridable per episode
through `reward_overrides`):

| component | default | units |
|---|---|---|
| `damage_dealt` | +0.01 | per applied HP |
| `damage_taken` | −0.01 | per applied HP |
| `kill_normal` | +1.0 | per kill |
| `kill_elite` | +3.0 | per elite kill |
| `kill_boss` | +20.0 | demonlord |
| `level_clear` | +10.0 | portal activated |
| `portal_entered` | +5.0 | portal entered |
| `victory` | +50.0 | run won |
| `death` | −10.0 | player died |
| `step_penalty` | −0.001 | per decision |
| `gold`, `gem`, `heal`, `whiff` | 0.0 | off by default |

Profiles: `combat` (above), `sparse` (kills/clear/victory/death only), `none`
(all zero, reward computed in Python).

## Vectorised environments

```python
from ashcrown_rl.vec import make_vec_env

envs = make_vec_env(8, seed=20260101, env_kwargs={"level": 1, "frame_skip": 4})
obs, infos = envs.reset()
obs, rewards, terminated, truncated, infos = envs.step(actions)  # (8, ...) batches
envs.close()  # graceful, then reaps every child (also close(terminate=True))
```

- `AsyncVectorEnv`, `multiprocessing_context="spawn"`, `shared_memory=False`;
- one Godot process per worker;
- worker seeds are `SeedSequence(master).spawn(n)` children (deterministic),
  and each worker draws its episode seeds from that stream so even the
  auto-reset episodes are reproducible;
- `autoreset_mode=SAME_STEP`: the terminal step returns the reset observation
  and the real one in `infos["final_obs"]` / `infos["final_info"]`;
- `close(terminate: bool = False)` forwards Gymnasium's flag (its `__del__`
  calls `close(terminate=True)`) and hard-terminates any worker that outlives
  the graceful close, so a dropped vector env leaves no worker and no Godot
  process behind; a constructor that raises reaps the workers it already
  started and closes the parent-side probe env.

## Tests

```bash
uv run pytest                       # unit + integration (needs godot + the game)
uv run pytest -m "not integration"  # unit only, no process
uv run ruff check . && uv run ruff format --check .
```

`integration` tests launch real headless processes (`test_protocol.py`,
`test_spaces.py`, `test_env.py`, `test_vec.py`, `test_baseline.py`,
`test_replay.py`); they skip cleanly when the game checkout or the godot binary
is missing.

## Troubleshooting

**`GodotUnavailable: no game checkout found` / `no godot binary`** — set
`ASHCROWN_PROJECT=/path/to/AshCrown` and `ASHCROWN_GODOT=/path/to/godot`, or
pass `--game-project` / `--godot-bin`. The launcher defaults to the parent of
this repository.

**`godot exited before becoming ready`** — the launcher attaches the engine's
stderr tail (and the exact argv) to the error. The usual causes are a missing
`.godot/` import cache (`godot --headless --import` in the game checkout), a
stale port file (deleted automatically) or another project holding the port.
Leave stale port files alone: each launch uses a fresh temporary file.

**The first reset is slow (~1 s)** — that is the discarded warm-up episode the
engine runs before it accepts a client (`docs/07` §4.2). It is required for
bit-identical replays and happens before the socket is bound.

**`BAD_CONFIG` from `reset`** — `frame_skip` and the five observation-config
keys (`obs_grid`, `obs_grid_radius`, `perception_px`, `expose_portal`,
`obs_blocked`) are fixed for the life of a process; a reset may not change
them. Pass the observation config to `AshCrownEnv(...)` (it goes to the engine
as launch arguments, so the handshake matches), not to `reset()`.

**A v1 handshake / checkpoint / log is refused** — `ashcrown-obs-v1` had a
3,246-float vector (hazards 6 wide, fixed radius 10) and a different
`layout_hash`; `verify_handshake` and `replay` raise a `ProtocolMismatch` /
`ValueError` naming the schema or hash instead of slicing the wrong floats.
Re-train/re-record; see the game repo's `docs/07` §8.5.1 for the migration.

**Replay diverges** — `python -m ashcrown_rl.replay replay` names the first
differing step and prints a bounded state diff of that step (the recorded vs
fresh state line, reduced to the fields that changed). Check that both runs
used the same game build (`layout_hash`) and engine flags; the determinism
promise is per binary/platform (`docs/07` §10.1). A different CPU architecture
or a rebuilt engine may change the last bits.

**Two processes, one machine** — every environment is a full Godot process
(~450 MB RSS). 8–16 fit on this host; cap `make_vec_env(n_envs)` accordingly
and always `close()` it.

**`ALSA` / `V-Sync` warnings** — benign under a headless dummy renderer.

## Status

- **RL-3 (landed)**: protocol client, launcher, benchmark.
- **RL-4 (landed)**: `spaces`, `env`, `vec`, `baseline`, `evaluate`,
  `replay`, examples, tests and this document. Measured on this host:
  **487.7 steps/s** for one environment and **2,370 steps/s** aggregate for
  eight (`docs/artifacts/RL/bench.csv`). The scripted heuristic **meets the
  acceptance bar**: it clears level 1 on **8 of the 10 fixed seeds** (warrior,
  `frame_skip=4`, `max_ticks=18,000`, determinism verified across two repeats)
  — `docs/artifacts/RL/baseline_level1.json` / `.csv`, with the per-seed story
  in `RL4-report.md`. On 60 unseen seeds (11–70) the same configuration clears
  **41/60 (68 %)** — `docs/artifacts/RL/unseen_seeds_11_70.csv` — so the fixed
  list is the friendlier half of the distribution.
- **RL-5 (landed)**: run mode (ten-level descent, per-floor re-acquisition,
  victory/death terminals), the `auto_heal` merchant sub-policy, the Python
  `mode="run"` option, the run-mode smoke script and the GUT scripted clear
  harness. See the game repository's `docs/artifacts/RL/RL5-report.md`.
- **RL-6 (landed)**: `ashcrown-obs-v2` — the layout is resolved from a
  published observation config (grid on/off, radius 4–21, perception,
  portal assist, optional blocked channel), hazard slots gained
  `telegraph_fraction` + `coverage`, and the client derives every slice from
  the handshake. The ablation (`scripts/ablate_obs.py`,
  `docs/artifacts/RL/ablation_obs_v2.csv`) shows the grid and the portal assist
  are what the heuristic needs; the default v2 config keeps the 8/10
  acceptance. See the game repo's `docs/artifacts/RL/RL6-report.md`.
- **RL-7 (next)**: the PPO pipeline.

## Licence

The AshCrown game and this training client are private project code. The
`LordsOfPain-OldSchoolIsometricAssets` asset pack is **not** redistributed
here; it remains in the game repository under its own licence.

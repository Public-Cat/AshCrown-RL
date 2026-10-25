"""Training-pipeline contract: checkpoint metadata, curriculum, curve math.

This module is the **numpy-only half** of the RL-7 training pipeline
(``docs/07`` §9.4), so it imports with the ``dev`` extra alone — no
Stable-Baselines3, no torch:

* the checkpoint metadata document, its ``*.json`` sidecar and the
  refuse-on-mismatch verifier (:class:`MetadataMismatch`,
  :func:`build_metadata`, :func:`write_metadata`, :func:`read_metadata`,
  :func:`verify_metadata`, :func:`checkpoint_frame_stack`);
* :func:`load_policy`, which verifies the sidecar and *then* imports SB3 to
  load the ``*.zip`` (the import is inside the function);
* the curriculum (:class:`LevelSchedule`, :class:`CurriculumEnv`), the
  training-only shaping wrapper (:class:`ShapingWrapper`) and the
  training-curve row math (:func:`_curve_row`).

Checkpoint metadata (``docs/07`` §9.4)
-------------------------------------
Every ``*.zip`` checkpoint written by the pipeline has a ``*.json`` sidecar
with the pins that make the policy loadable *and* refuse-able::

    {"checkpoint": "ppo_level1_300000_steps.zip",
     "protocol": 1,
     "obs_schema": "ashcrown-obs-v2",
     "action_schema": "ashcrown-action-v1",
     "layout_hash": "fbcc9cda…",
     "obs_len": 3254,
     "observation_config": {"obs_grid": true, "obs_grid_radius": 10, …},
     "frame_skip": 4,
     "env": {"level": 1, "mode": "level", "character": "warrior", …},
     "seeds": {"train_master": 7, "tuning": [...], "eval": [...]},
     "git": {"training": "…", "game": "…"},
     "total_steps": 300000,
     "device": "cpu",
     "command": "uv run python examples/train_ppo.py --level 1 …",
     "algorithm": {"name": "PPO", "policy": "MlpPolicy", "hyperparameters": {…}},
     "created": "2026-10-24T…"}

:func:`load_policy` reads the sidecar and refuses a checkpoint whose schema,
observation config or layout hash does not match the live environment, so a
policy is never silently fed a differently-shaped observation.

Optional training extra
-----------------------
The SB3 stack (``Sb3AshCrownVecEnv``, the callbacks, ``AnchoredPPO``,
``make_training_vec_env``, ``train_ppo``) lives in
:mod:`ashcrown_rl.sb3_train` and needs ``stable-baselines3`` + ``torch``
(``uv sync --extra train``). Those names are re-exported **lazily** through the
module-level :func:`__getattr__` below, so ``import ashcrown_rl.train`` —
and everything that only plans, evaluates or verifies checkpoints — works in a
docs/07 §12.3 dev-only environment; touching one of them without the extra
raises ``ImportError`` at that moment, never at import time (regression fixed
in INT-97: the module used to import SB3 at the top).
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np

from .env import AshCrownEnv
from .history import policy_obs_len, resolve_frame_stack

if TYPE_CHECKING:  # pragma: no cover - annotations only, the extra is optional
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices


#: Version of the checkpoint metadata document.
METADATA_VERSION = 1
#: Columns of the training-curve CSV, in order (``--log-csv``).
CURVE_COLUMNS: tuple[str, ...] = (
    "timestep",
    "episodes",
    "episode_reward",
    "success_rate",
    "mean_length",
    "entropy",
    "policy_loss",
    "value_loss",
    "approx_kl",
    "clears",
    "mean_game_seconds",
    "kills",
    "hits_landed",
    "attacks_started",
    "bc_loss",
    "wall_seconds",
)


class MetadataMismatch(RuntimeError):
    """A checkpoint was trained for a different environment contract."""


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def git_commit(path: str | Path) -> str:
    """Short HEAD commit of the repository at ``path`` (``"unknown"`` if none)."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - no git binary
        return "unknown"
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and commit else "unknown"


def metadata_path(checkpoint: str | Path) -> Path:
    """The ``*.json`` sidecar path of a ``*.zip`` checkpoint."""
    return Path(checkpoint).with_suffix(".json")


def build_metadata(
    *,
    checkpoint: str | Path,
    env: AshCrownEnv,
    env_config: Mapping[str, Any],
    seeds: Mapping[str, Any],
    total_steps: int,
    command: str,
    algorithm: Mapping[str, Any],
    device: str,
    git_paths: Mapping[str, str | Path] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Metadata document for one checkpoint (see the module docstring)."""
    paths = dict(git_paths or {})
    payload: dict[str, Any] = {
        "version": METADATA_VERSION,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "checkpoint": Path(checkpoint).name,
        "protocol": env.hello.get("protocol"),
        "obs_schema": env.hello.get("obs_schema") or env.layout.obs_schema,
        "action_schema": env.hello.get("action_schema"),
        "layout_hash": env.layout_hash,
        "obs_len": env.obs_len,
        "observation_config": env.observation_config(),
        "frame_skip": env.frame_skip,
        "env": dict(env_config),
        "seeds": dict(seeds),
        "git": {name: git_commit(path) for name, path in paths.items()},
        "total_steps": int(total_steps),
        "device": str(device),
        "command": str(command),
        "algorithm": dict(algorithm),
    }
    if extra:
        payload.update(dict(extra))
    return payload


def write_metadata(checkpoint: str | Path, metadata: Mapping[str, Any]) -> Path:
    """Write ``metadata`` next to ``checkpoint`` and return the sidecar path."""
    target = metadata_path(checkpoint)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dict(metadata), indent=2) + "\n", encoding="utf-8")
    return target


def read_metadata(checkpoint: str | Path) -> dict[str, Any]:
    """Read a checkpoint's sidecar, raising :class:`MetadataMismatch` if absent."""
    path = metadata_path(checkpoint)
    if not path.is_file():
        raise MetadataMismatch(f"{checkpoint}: missing checkpoint metadata {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def verify_metadata(
    metadata: Mapping[str, Any],
    *,
    protocol: int | None,
    obs_schema: str | None,
    layout_hash: str | None,
    observation_config: Mapping[str, Any] | None,
    frame_stack: int | None = None,
    obs_len: int | None = None,
) -> None:
    """Refuse a checkpoint whose environment contract differs from the live one.

    ``frame_stack``/``obs_len`` guard the observation *history* (``docs/07``
    §5.7): the policy's input is ``frame_stack × obs_len`` floats, so a
    checkpoint trained on a 4-frame stack must not be loaded against a 1-frame
    environment (or against a different stack length). Checkpoints written
    before the stack existed have no ``frame_stack`` key and mean 1.
    """
    if protocol is not None and metadata.get("protocol") != protocol:
        raise MetadataMismatch(
            f"checkpoint protocol {metadata.get('protocol')!r} != live {protocol!r}"
        )
    if obs_schema is not None and metadata.get("obs_schema") != obs_schema:
        raise MetadataMismatch(
            f"checkpoint obs_schema {metadata.get('obs_schema')!r} != live {obs_schema!r}"
        )
    if layout_hash is not None and metadata.get("layout_hash") != layout_hash:
        raise MetadataMismatch(
            f"checkpoint layout_hash {metadata.get('layout_hash')!r} != live {layout_hash!r}"
        )
    if observation_config is not None:
        recorded = dict(metadata.get("observation_config") or {})
        live = dict(observation_config)
        if recorded != live:
            raise MetadataMismatch(f"checkpoint observation_config {recorded!r} != live {live!r}")
    if frame_stack is not None:
        recorded_stack = int(metadata.get("frame_stack", 1) or 1)
        live_stack = resolve_frame_stack(frame_stack)
        if recorded_stack != live_stack:
            raise MetadataMismatch(
                f"checkpoint frame_stack {recorded_stack} != requested {live_stack}"
                " (a stacked policy cannot be evaluated single-frame)"
            )
    if obs_len is not None:
        recorded_len = int(
            metadata.get("policy_obs_len")
            or policy_obs_len(int(metadata.get("obs_len", obs_len)), metadata.get("frame_stack", 1))
        )
        expected = policy_obs_len(int(obs_len), int(metadata.get("frame_stack", 1) or 1))
        if recorded_len != expected:
            raise MetadataMismatch(
                f"checkpoint policy_obs_len {recorded_len} != live {expected}"
                f" ({metadata.get('frame_stack', 1)} frame(s) x {int(obs_len)})"
            )


def load_policy(
    checkpoint: str | Path,
    env: AshCrownEnv | None = None,
    *,
    device: str = "cpu",
    frame_stack: int | None = None,
) -> PPO:
    """Load a PPO checkpoint after verifying its metadata against ``env``.

    SB3 is imported here rather than at module import time: verifying a
    sidecar is numpy-only, and only the actual load needs the ``train`` extra.
    """
    from stable_baselines3 import PPO as Sb3PPO  # noqa: PLC0415 - optional extra

    metadata = read_metadata(checkpoint)
    if env is not None:
        verify_metadata(
            metadata,
            protocol=env.hello.get("protocol"),
            obs_schema=env.hello.get("obs_schema") or env.layout.obs_schema,
            layout_hash=env.layout_hash,
            observation_config=env.observation_config(),
            frame_stack=frame_stack,
            obs_len=env.obs_len,
        )
    return Sb3PPO.load(str(checkpoint), device=device)


def checkpoint_frame_stack(checkpoint: str | Path) -> int:
    """The frame-stack length a checkpoint was trained (or fitted) with."""
    return resolve_frame_stack(read_metadata(checkpoint).get("frame_stack", 1))


# --------------------------------------------------------------------------- #
# SB3 vector-env adapter
# --------------------------------------------------------------------------- #
def _env_info(vector_infos: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Unbatch Gymnasium's dict-of-arrays ``infos`` for one environment.

    Gymnasium 1.x returns the vector info as ``{key: array}`` plus a ``_key``
    presence mask (nested dicts recurse). SB3's rollout loop indexes ``infos``
    as a list of dicts, so the adapter converts it here. ``final_obs`` is
    dropped: it is a second full observation and nothing downstream needs it.
    """
    out: dict[str, Any] = {}
    for key, value in vector_infos.items():
        if key.startswith("_") or key == "final_obs":
            continue
        mask = vector_infos.get(f"_{key}")
        present = bool(mask[index]) if mask is not None else True
        if isinstance(value, Mapping):
            if present:
                out[key] = _env_info(value, index)
            continue
        if present:
            out[key] = value[index]
    return out


def _select(values: list[Any], indices: VecEnvIndices, num_envs: int) -> list[Any]:
    """SB3-style index selection over a full per-worker list."""
    if indices is None:
        return values
    if isinstance(indices, int):
        return [values[indices]]
    selected = [values[int(index)] for index in indices]
    if len(selected) != num_envs and not isinstance(indices, int):
        return selected
    return selected


# --------------------------------------------------------------------------- #
# Curriculum
# --------------------------------------------------------------------------- #
class LevelSchedule:
    """Deterministic per-episode level assignment for ``--curriculum``.

    Levels are handed out round-robin per worker episode so a mixed-level
    training run covers every level roughly equally. The schedule is a pure
    function of the episode index, so a rerun with the same seed set visits the
    same levels.
    """

    def __init__(self, levels: Sequence[int]) -> None:
        """Build the schedule from a non-empty level list."""
        self.levels = [int(level) for level in levels]
        if not self.levels:
            raise ValueError("curriculum needs at least one level")

    def level_for(self, episode_index: int) -> int:
        """Level of episode ``episode_index`` (0-based, per worker)."""
        return self.levels[int(episode_index) % len(self.levels)]


class CurriculumEnv(gym.Wrapper):
    """Worker wrapper that assigns a level on every auto-reset.

    ``AsyncVectorEnv`` auto-resets with ``reset()`` (no options), so the level
    cannot come from the training loop. This wrapper injects one from a
    :class:`LevelSchedule` and leaves an explicit ``options["level"]`` alone.
    """

    def __init__(self, env: Any, levels: Sequence[int]) -> None:
        super().__init__(env)
        self.schedule = LevelSchedule(levels)
        self.episode_index = 0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset with the next scheduled level unless the caller pins one."""
        opts = dict(options or {})
        if "level" not in opts:
            opts["level"] = self.schedule.level_for(self.episode_index)
        self.episode_index += 1
        return self.env.reset(seed=seed, options=opts)


class ShapingWrapper(gym.Wrapper):
    """Training-only intrinsic reward shaping (``docs/07`` §7 allows it).

    The engine reward (``combat`` profile) is dense **once the player meets an
    enemy**, but an untrained network that has not learned to explore never
    meets one — the first RL-7 run spent 44k decisions in empty corners with
    zero damage dealt or taken, so PPO had no gradient at all. This wrapper
    adds two documented terms on top of the engine reward (the engine's own
    components are untouched and still reported in ``info.reward_components``):

    * ``explore_coef`` × newly ``visited`` cells in the local window — the
      ``visited`` grid channel is monotone per floor, and the (2R+1)² window
      slides with the player, so its positive delta is a cheap proxy for
      "moved into unexplored ground";
    * ``approach_coef`` × the decrease of the nearest *perceived* enemy
      distance (a potential, only while an enemy is in the observation).

    Both are small relative to a kill (+1.0), so the engine's objective stays
    dominant; the shaped total is reported as ``info.shaping_reward``.
    """

    def __init__(self, env: Any, *, explore_coef: float = 0.0, approach_coef: float = 0.0) -> None:
        super().__init__(env)
        self.layout = env.unwrapped.layout
        self.explore_coef = float(explore_coef)
        self.approach_coef = float(approach_coef)
        self._visited = 0.0
        self._enemy_distance = 0.0

    def _potential(self, obs: np.ndarray) -> tuple[float, float]:
        visited = 0.0
        if self.layout.grid_enabled:
            grid = self.layout.grid(obs)
            visited = float(np.sum(self.layout.channel(grid, "visited")))
        distance = float(self.layout.nearest_enemy_distance_cells(obs))
        return visited, distance

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._visited, self._enemy_distance = self._potential(obs)
        return obs, info

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        visited, distance = self._potential(obs)
        bonus = self.explore_coef * max(visited - self._visited, 0.0)
        if self.approach_coef and distance > 0.0 and self._enemy_distance > 0.0:
            bonus += self.approach_coef * max(self._enemy_distance - distance, 0.0)
        self._visited = visited
        self._enemy_distance = distance
        info = dict(info)
        info["shaping_reward"] = float(bonus)
        return obs, float(reward) + float(bonus), terminated, truncated, info


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def _curve_row(
    *,
    timestep: int,
    episodes: Sequence[Mapping[str, Any]],
    logger_values: Mapping[str, Any],
    wall_seconds: float,
) -> dict[str, Any]:
    """One training-curve row (means over the episodes of the last rollout)."""
    count = len(episodes) or 1
    clears = sum(1 for episode in episodes if episode.get("terminal_reason") == "level_clear")
    entropy_loss = logger_values.get("train/entropy_loss", math.nan)
    return {
        "timestep": int(timestep),
        "episodes": len(episodes),
        "episode_reward": sum(float(e.get("r", 0.0)) for e in episodes) / count,
        "success_rate": clears / count,
        "mean_length": sum(float(e.get("l", 0)) for e in episodes) / count,
        "entropy": -float(entropy_loss) if entropy_loss is not None else math.nan,
        "policy_loss": float(logger_values.get("train/policy_gradient_loss", math.nan)),
        "value_loss": float(logger_values.get("train/value_loss", math.nan)),
        "approx_kl": float(logger_values.get("train/approx_kl", math.nan)),
        "clears": clears,
        "mean_game_seconds": sum(float(e.get("game_seconds", 0.0)) for e in episodes) / count,
        "kills": sum(float(e.get("kills", 0)) for e in episodes) / count,
        "hits_landed": sum(float(e.get("hits_landed", 0)) for e in episodes) / count,
        "attacks_started": sum(float(e.get("attacks_started", 0)) for e in episodes) / count,
        "bc_loss": float(logger_values.get("train/bc_loss", math.nan)),
        "wall_seconds": round(float(wall_seconds), 3),
    }


#: Names that live in :mod:`ashcrown_rl.sb3_train` (they need the ``train`` extra).
_LAZY_EXPORTS: frozenset[str] = frozenset(
    {
        "AnchoredPPO",
        "MetadataCheckpointCallback",
        "Sb3AshCrownVecEnv",
        "TrainingCurveCallback",
        "make_training_vec_env",
        "train_ppo",
    }
)


def __getattr__(name: str) -> Any:
    """Import an SB3-backed training name on first use (``train`` extra only)."""
    if name in _LAZY_EXPORTS:
        from . import sb3_train

        return getattr(sb3_train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CURVE_COLUMNS",
    "METADATA_VERSION",
    "CurriculumEnv",
    "LevelSchedule",
    "MetadataMismatch",
    "ShapingWrapper",
    "build_metadata",
    "checkpoint_frame_stack",
    "git_commit",
    "load_policy",
    "metadata_path",
    "read_metadata",
    "verify_metadata",
    "write_metadata",
]


if __name__ == "__main__":  # pragma: no cover - module guard
    print("use examples/train_ppo.py as the training entry point", file=sys.stderr)
    raise SystemExit(2)

"""PPO training pipeline and checkpoint metadata (RL-7, ``docs/07`` §9.4).

This module owns the *training* side of the RL pipeline: the Stable-Baselines3
bridge over :func:`ashcrown_rl.vec.make_vec_env`, the incremental training-curve
CSV, and the checkpoint metadata that makes a saved policy usable.

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

``Sb3AshCrownVecEnv`` adapts the Gymnasium ``AsyncVectorEnv`` that
:func:`~ashcrown_rl.vec.make_vec_env` returns to SB3's ``VecEnv`` interface: one
vector step drives every environment, ``SAME_STEP`` autoresets are translated
into SB3's ``dones`` mask plus an ``episode`` stats block, and terminal details
(``terminal_reason``, ``level``) are kept for the training curve.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices

from .env import AshCrownEnv
from .vec import (
    DEFAULT_CONTEXT,
    AshCrownVectorEnv,
    ProbeSink,
    SeededAshCrownEnv,
    _track_probe_env,
    close_envs,
    make_vec_env,
    worker_seeds,
)

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
) -> None:
    """Refuse a checkpoint whose environment contract differs from the live one."""
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


def load_policy(
    checkpoint: str | Path,
    env: AshCrownEnv | None = None,
    *,
    device: str = "cpu",
) -> PPO:
    """Load a PPO checkpoint after verifying its metadata against ``env``."""
    metadata = read_metadata(checkpoint)
    if env is not None:
        verify_metadata(
            metadata,
            protocol=env.hello.get("protocol"),
            obs_schema=env.hello.get("obs_schema") or env.layout.obs_schema,
            layout_hash=env.layout_hash,
            observation_config=env.observation_config(),
        )
    return PPO.load(str(checkpoint), device=device)


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


class Sb3AshCrownVecEnv(VecEnv):
    """SB3 ``VecEnv`` over the Gymnasium ``AsyncVectorEnv`` of :func:`make_vec_env`.

    The Gymnasium vector env already handles spawning, seeding and ``SAME_STEP``
    autoreset; this adapter only adds SB3's ``reset``/``step_async``/
    ``step_wait`` shape and turns terminal ``final_info`` blocks into the
    ``episode`` stats SB3 and the training curve consume.
    """

    def __init__(self, vec: AsyncVectorEnv) -> None:
        self.vec = vec
        super().__init__(
            int(vec.num_envs),
            vec.single_observation_space,
            vec.single_action_space,
        )
        self._pending: np.ndarray | None = None
        #: Finished-episode records since the last :meth:`pop_episodes` call.
        self.finished: list[dict[str, Any]] = []
        self._episode_return = np.zeros(self.num_envs, dtype=np.float64)
        self._episode_length = np.zeros(self.num_envs, dtype=np.int64)

    # SB3 requires these delegations ----------------------------------------
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """Attribute of each environment (Gymnasium's call returns every worker)."""
        values = list(self.vec.get_attr(attr_name))
        return _select(values, indices, self.num_envs)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set an attribute on each environment."""
        del attr_name, value, indices
        raise NotImplementedError("Sb3AshCrownVecEnv does not support set_attr")

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        """Call a method on each environment."""
        values = list(self.vec.call(method_name, *method_args, **method_kwargs))
        return _select(values, indices, self.num_envs)

    def env_is_wrapped(self, wrapper_class: type[Any], indices: VecEnvIndices = None) -> list[bool]:
        """No environment is wrapped in an SB3 wrapper (delegation is not needed)."""
        del wrapper_class, indices
        return [False] * self.num_envs

    # SB3 rollout interface --------------------------------------------------
    def reset(self) -> np.ndarray:
        """Reset every environment and return the batched observations."""
        obs, _infos = self.vec.reset()
        self._pending = None
        self._episode_return[:] = 0.0
        self._episode_length[:] = 0
        return np.asarray(obs, dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        """Queue one batch of actions."""
        self._pending = np.asarray(actions)

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Step every environment and translate autoresets into SB3's contract."""
        if self._pending is None:
            raise RuntimeError("step_wait() called without step_async()")
        actions = self._pending
        self._pending = None
        obs, rewards, terminated, truncated, vector_infos = self.vec.step(actions)
        rewards = np.asarray(rewards, dtype=np.float32)
        dones = np.logical_or(terminated, truncated)
        self._episode_return += rewards
        self._episode_length += 1
        out_infos: list[dict[str, Any]] = []
        for index in range(self.num_envs):
            info = _env_info(vector_infos, index)
            if bool(dones[index]):
                final = dict(info.get("final_info") or {})
                record = {
                    "r": float(self._episode_return[index]),
                    "l": int(self._episode_length[index]),
                    "terminal_reason": str(final.get("terminal_reason", "")),
                    "level": int(final.get("level", 0) or 0),
                    "kills": int(final.get("kills", 0) or 0),
                    "game_seconds": float(final.get("game_seconds", 0.0) or 0.0),
                }
                self.finished.append(record)
                info["episode"] = {"r": record["r"], "l": record["l"]}
                info["TimeLimit.truncated"] = bool(truncated[index])
                info["terminal_reason"] = record["terminal_reason"]
                info["level"] = record["level"]
                self._episode_return[index] = 0.0
                self._episode_length[index] = 0
            out_infos.append(info)
        return (
            np.asarray(obs, dtype=np.float32),
            rewards,
            dones,
            out_infos,
        )

    def close(self) -> None:
        """Close the Gymnasium vector env (and every Godot process behind it)."""
        self.vec.close()

    # Training-curve support -------------------------------------------------
    def pop_episodes(self) -> list[dict[str, Any]]:
        """Return and clear the episodes finished since the last call."""
        finished = self.finished
        self.finished = []
        return finished


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


class CurriculumEnv:
    """Picklable worker wrapper that assigns a level on every auto-reset.

    ``AsyncVectorEnv`` auto-resets with ``reset()`` (no options), so the level
    cannot come from the training loop. This wrapper injects one from a
    :class:`LevelSchedule` and leaves an explicit ``options["level"]`` alone.
    """

    def __init__(self, env: Any, levels: Sequence[int]) -> None:
        self.env = env
        self.schedule = LevelSchedule(levels)
        self.episode_index = 0

    # gymnasium.Env surface used by the vector worker ------------------------
    @property
    def observation_space(self) -> Any:
        """Forward the observation space."""
        return self.env.observation_space

    @property
    def action_space(self) -> Any:
        """Forward the action space."""
        return self.env.action_space

    @property
    def unwrapped(self) -> Any:
        """Forward the unwrapped env."""
        return self.env.unwrapped

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset with the next scheduled level unless the caller pins one."""
        opts = dict(options or {})
        if "level" not in opts:
            opts["level"] = self.schedule.level_for(self.episode_index)
        self.episode_index += 1
        return self.env.reset(seed=seed, options=opts)

    def step(self, action: Any):
        """Step the wrapped environment."""
        return self.env.step(action)

    def close(self) -> None:
        """Close the wrapped environment."""
        self.env.close()

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes (``layout``, ``hello``, …)."""
        return getattr(self.env, name)


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
        "wall_seconds": round(float(wall_seconds), 3),
    }


class TrainingCurveCallback(BaseCallback):
    """Append one CSV row per rollout (crash-safe: flushes every row).

    ``entropy``/``policy_loss``/``value_loss`` come from SB3's logger and
    therefore describe the **previous** completed update (the first row has no
    update yet and is written with the last completed one); the episode columns
    describe the rollout that just finished.
    """

    def __init__(self, path: str | Path, adapter: Sb3AshCrownVecEnv) -> None:
        """Open ``path`` for appending and write the header when it is new."""
        super().__init__()
        self.path = Path(path)
        self.adapter = adapter
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._header = new_file
        self._started = 0.0

    def _on_training_start(self) -> None:
        self._started = time.perf_counter()

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        self._started = time.perf_counter()

    def _on_rollout_end(self) -> None:
        episodes = self.adapter.pop_episodes()
        row = _curve_row(
            timestep=int(self.model.num_timesteps),
            episodes=episodes,
            logger_values=self.model.logger.name_to_value,
            wall_seconds=time.perf_counter() - self._started,
        )
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CURVE_COLUMNS)
            if self._header:
                writer.writeheader()
                self._header = False
            writer.writerow(row)
            handle.flush()


class MetadataCheckpointCallback(BaseCallback):
    """Save the model plus its metadata sidecar every ``save_freq`` steps."""

    def __init__(
        self,
        save_dir: str | Path,
        *,
        save_freq: int,
        adapter: Sb3AshCrownVecEnv,
        metadata_builder: Callable[[int], dict[str, Any]],
        prefix: str = "ckpt",
    ) -> None:
        """Remember the destination, the cadence and the metadata factory."""
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_freq = max(int(save_freq), 1)
        self.adapter = adapter
        self.metadata_builder = metadata_builder
        self.prefix = prefix
        self.saved: list[Path] = []
        self._last_saved = 0

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self._last_saved < self.save_freq:
            return True
        self._last_saved = int(self.model.num_timesteps)
        self.save_now(self._last_saved)
        return True

    def save_now(self, timestep: int) -> Path:
        """Save one checkpoint and its metadata, returning the zip path."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        target = self.save_dir / f"{self.prefix}_{int(timestep)}_steps.zip"
        self.model.save(str(target))
        write_metadata(target, self.metadata_builder(int(timestep)))
        self.saved.append(target)
        return target


# --------------------------------------------------------------------------- #
# Training entry point
# --------------------------------------------------------------------------- #
def make_training_vec_env(
    *,
    n_envs: int,
    seed: int,
    env_kwargs: Mapping[str, Any],
    levels: Sequence[int] | None = None,
) -> Sb3AshCrownVecEnv:
    """Build the SB3 vector env (optionally level-scheduled) for training."""
    kwargs = dict(env_kwargs)
    if not levels:
        return Sb3AshCrownVecEnv(make_vec_env(int(n_envs), seed=int(seed), env_kwargs=kwargs))
    # Mixed-level training: every auto-reset draws the next level of the
    # schedule, so the level cannot be pinned by the rollout loop.
    env_fns = [
        _curriculum_worker_factory(dict(kwargs), int(worker_seed), list(levels))
        for worker_seed in worker_seeds(int(seed), int(n_envs))
    ]
    probe = ProbeSink()
    env_fns[0] = _track_probe_env(env_fns[0], probe)
    try:
        vec = AshCrownVectorEnv(
            env_fns,
            shared_memory=False,
            copy=True,
            context=DEFAULT_CONTEXT,
            daemon=True,
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    except BaseException:
        close_envs(probe.envs)
        raise
    probe.envs.clear()
    return Sb3AshCrownVecEnv(vec)


def _curriculum_worker_factory(kwargs: dict[str, Any], worker_seed: int, levels: list[int]) -> Any:
    """Picklable ``env_fn`` building a level-scheduled worker env."""
    seed_sequence = np.random.SeedSequence(worker_seed)

    def _factory() -> Any:
        seeder = SeededAshCrownEnv(AshCrownEnv(**kwargs), seed_sequence)
        return CurriculumEnv(seeder, levels)

    return _factory


def train_ppo(
    *,
    n_envs: int,
    total_steps: int,
    seed: int,
    env_kwargs: Mapping[str, Any],
    checkpoint_dir: str | Path,
    log_csv: str | Path,
    command: str,
    algorithm: Mapping[str, Any],
    hyperparameters: Mapping[str, Any] | None = None,
    device: str = "cpu",
    levels: Sequence[int] | None = None,
    save_freq: int = 100_000,
    eval_seeds: Mapping[str, Any] | None = None,
    git_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Train PPO and write checkpoints, their metadata and the curve CSV."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    vec = make_training_vec_env(
        n_envs=int(n_envs), seed=int(seed), env_kwargs=env_kwargs, levels=levels
    )
    metadata_factory: Callable[[int], dict[str, Any]]
    try:
        # Read the pins from a worker env (no extra Godot process): the
        # handshake/layout attributes are plain members, the observation config
        # is a method.
        hello = vec.get_attr("hello")[0]
        pins = {
            "protocol": hello.get("protocol"),
            "obs_schema": hello.get("obs_schema"),
            "action_schema": hello.get("action_schema"),
            "layout_hash": vec.get_attr("layout_hash")[0],
            "obs_len": int(vec.get_attr("obs_len")[0]),
            "observation_config": vec.env_method("observation_config")[0],
        }

        def metadata_factory(steps: int) -> dict[str, Any]:
            return {
                "version": METADATA_VERSION,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "checkpoint": "",
                **pins,
                "frame_skip": int(env_kwargs.get("frame_skip", 4)),
                "env": dict(env_kwargs),
                "seeds": dict(eval_seeds or {}),
                "git": {name: git_commit(path) for name, path in (git_paths or {}).items()},
                "total_steps": int(steps),
                "device": str(device),
                "command": str(command),
                "algorithm": dict(algorithm),
            }

        hp = dict(hyperparameters or {})
        net_arch = hp.pop("net_arch", None)
        if net_arch:
            hp["policy_kwargs"] = {"net_arch": list(net_arch)}
        model = PPO(
            "MlpPolicy",
            vec,
            device=device,
            seed=int(seed),
            verbose=0,
            tensorboard_log=None,
            **hp,
        )
        curve = TrainingCurveCallback(log_csv, vec)
        checkpoint = MetadataCheckpointCallback(
            checkpoint_dir,
            save_freq=int(save_freq),
            adapter=vec,
            metadata_builder=metadata_factory,
        )
        model.learn(
            total_timesteps=int(total_steps),
            callback=[curve, checkpoint],
            progress_bar=False,
        )
        if checkpoint.saved and int(model.num_timesteps) == checkpoint._last_saved:
            final = checkpoint.saved[-1]
        else:
            final = checkpoint.save_now(int(model.num_timesteps))
        return {
            "started": started,
            "total_steps": int(model.num_timesteps),
            "checkpoints": [str(path) for path in checkpoint.saved],
            "final": str(final),
            "pins": pins,
            "curve": str(log_csv),
        }
    finally:
        vec.close()


__all__ = [
    "CURVE_COLUMNS",
    "METADATA_VERSION",
    "CurriculumEnv",
    "LevelSchedule",
    "MetadataCheckpointCallback",
    "MetadataMismatch",
    "Sb3AshCrownVecEnv",
    "TrainingCurveCallback",
    "build_metadata",
    "git_commit",
    "load_policy",
    "make_training_vec_env",
    "metadata_path",
    "read_metadata",
    "train_ppo",
    "verify_metadata",
    "write_metadata",
]


if __name__ == "__main__":  # pragma: no cover - module guard
    print("use examples/train_ppo.py as the training entry point", file=sys.stderr)
    raise SystemExit(2)

"""Vectorised AshCrown environments (RL-4, ``docs/07`` §9.3).

One Godot process per environment (``docs/07`` §2.1), so the vector env is
:class:`gymnasium.vector.AsyncVectorEnv` with ``multiprocessing_context="spawn"``
and ``shared_memory=False``:

* every worker constructs its own :class:`~ashcrown_rl.env.AshCrownEnv`, which
  launches its own headless Godot process;
* worker seeds are assigned deterministically from
  ``SeedSequence(master_seed).spawn(n_envs)``, and each worker derives its
  episode seeds from that stream — so a vector rollout is reproducible across
  processes even though the engine draws its own seed per episode;
* ``autoreset_mode=SAME_STEP`` gives Gymnasium's standard autoreset: the
  terminal step returns the reset observation plus the real
  ``info["final_obs"]``/``info["final_info"]``;
* :meth:`AshCrownVectorEnv.close` also kills any worker process that outlives
  the graceful close, so no Godot process is left behind.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from .env import AshCrownEnv

#: ``AsyncVectorEnv`` worker function type (kept as a permissive alias).
DEFAULT_CONTEXT = "spawn"


class SeededAshCrownEnv(gym.Wrapper):
    """An :class:`AshCrownEnv` whose episodes are seeded deterministically.

    ``reset(seed=...)`` (an explicit worker seed from the vector env) is passed
    through. ``reset()`` without a seed — Gymnasium's autoreset path — draws the
    next seed from this worker's ``SeedSequence`` instead of letting the engine
    pick a random one, so an auto-reset episode is reproducible too.
    """

    def __init__(self, env: AshCrownEnv, seed_sequence: np.random.SeedSequence) -> None:
        super().__init__(env)
        self._seed_sequence = seed_sequence
        self.episode_seeds: list[int] = []

    @property
    def base_seed(self) -> int:
        """The worker's master entropy (the seed ``make_vec_env`` handed out)."""
        return int(self._seed_sequence.entropy)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = int(self._seed_sequence.spawn(1)[0].generate_state(1)[0])
        self.episode_seeds.append(int(seed))
        return self.env.reset(seed=int(seed), options=options)

    @property
    def ashcrown(self) -> AshCrownEnv:
        """The underlying :class:`AshCrownEnv` (layout, obs_len, …)."""
        return self.env  # type: ignore[return-value]


class AshCrownVectorEnv(AsyncVectorEnv):
    """``AsyncVectorEnv`` that also reaps its workers (and their Godot processes)."""

    def close(self) -> None:
        """Graceful close, then hard-terminate any surviving worker process."""
        try:
            super().close()
        finally:
            self._reap_workers()

    def _reap_workers(self) -> None:
        for process in getattr(self, "processes", []) or []:
            try:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                if process.is_alive():  # pragma: no cover - a wedged worker
                    process.kill()
                    process.join(timeout=5)
            except (OSError, ValueError):  # pragma: no cover - already reaped
                continue


def make_vec_env(
    n_envs: int,
    seed: int | None = None,
    *,
    env_kwargs: dict[str, Any] | None = None,
    context: str = DEFAULT_CONTEXT,
    daemon: bool = True,
) -> AshCrownVectorEnv:
    """Build ``n_envs`` AshCrown environments, one Godot process each.

    ``env_kwargs`` is forwarded to :class:`AshCrownEnv` (``level``,
    ``frame_skip``, ``character``, ``project``, …). ``seed=None`` draws one
    master seed from OS entropy; pass an ``int`` for a reproducible rollout.
    """
    if int(n_envs) < 1:
        raise ValueError(f"n_envs={n_envs!r} must be >= 1")
    kwargs = dict(env_kwargs or {})
    if seed is None:
        master = int(np.random.SeedSequence().entropy)
    else:
        master = int(seed)
    children = np.random.SeedSequence(master).spawn(int(n_envs))
    env_fns = [_make_worker_env(dict(kwargs), child) for child in children]
    return AshCrownVectorEnv(
        env_fns,
        shared_memory=False,
        copy=True,
        context=context,
        daemon=daemon,
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def _make_worker_env(kwargs: dict[str, Any], seed_sequence: np.random.SeedSequence) -> Any:
    """Picklable ``env_fn`` for one worker (spawn-safe: no closures over envs)."""

    def _factory() -> SeededAshCrownEnv:
        return SeededAshCrownEnv(AshCrownEnv(**kwargs), seed_sequence)

    return _factory


def worker_seeds(seed: int, n_envs: int) -> list[int]:
    """The per-worker master entropies used by :func:`make_vec_env`."""
    return [int(child.entropy) for child in np.random.SeedSequence(int(seed)).spawn(n_envs)]


__all__ = [
    "DEFAULT_CONTEXT",
    "AshCrownVectorEnv",
    "SeededAshCrownEnv",
    "_make_worker_env",
    "make_vec_env",
    "worker_seeds",
]

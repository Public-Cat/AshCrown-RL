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
* :meth:`AshCrownVectorEnv.close` forwards Gymnasium's ``terminate`` flag and
  then kills any worker process that outlives the close, so neither a
  ``close(terminate=True)`` nor the ``__del__`` path (which passes
  ``terminate=True``) leaves a Godot process behind;
* a constructor that raises (a worker env that refuses its config, a space
  mismatch, …) reaps the workers it already started, and :func:`make_vec_env`
  closes the parent-side probe environment Gymnasium builds from
  ``env_fns[0]`` — a failed ``make_vec_env`` leaks nothing.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode

from .env import AshCrownEnv
from .history import resolve_frame_stack, wrap_frame_stack

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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Build the vector env, reaping every started worker if it fails.

        ``AsyncVectorEnv.__init__`` starts the worker processes before it
        validates their spaces, so a raising constructor would otherwise leave
        live workers (and an open pipe per worker) behind.
        """
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._discard_partial_init()
            raise

    def close(self, terminate: bool = False) -> None:
        """Close every worker, then hard-terminate any that survived.

        ``terminate`` is Gymnasium's flag (``AsyncVectorEnv.__del__`` calls
        ``close(terminate=True)``), so it must be accepted — otherwise a
        garbage-collected vector env raises ``TypeError`` instead of closing
        and leaks one Godot process per worker.
        """
        try:
            super().close(terminate=terminate)
        finally:
            self._reap_workers()

    def _discard_partial_init(self) -> None:
        """Reap what a failed ``__init__`` already started (workers + pipes)."""
        self._reap_workers()
        for pipe in getattr(self, "parent_pipes", None) or []:
            if pipe is None:
                continue
            try:
                pipe.close()
            except (OSError, ValueError):  # pragma: no cover - already closed
                continue
        # Marks the instance closed so the inherited ``__del__`` does not try a
        # second close on a half-built object.
        self.closed = True

    def _reap_workers(self) -> None:
        for process in getattr(self, "processes", None) or []:
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
    frame_stack: int = 1,
) -> AshCrownVectorEnv:
    """Build ``n_envs`` AshCrown environments, one Godot process each.

    ``env_kwargs`` is forwarded to :class:`AshCrownEnv` (``level``,
    ``frame_skip``, ``character``, ``project``, …). ``seed=None`` draws one
    master seed from OS entropy; pass an ``int`` for a reproducible rollout.
    ``frame_stack > 1`` stacks the last ``k`` observations **inside each worker**
    (``docs/07`` §5.7), so both the training policy and the collectors see the
    same ``(k, obs_len)`` observation the evaluator rebuilds.
    """
    if int(n_envs) < 1:
        raise ValueError(f"n_envs={n_envs!r} must be >= 1")
    stack = resolve_frame_stack(frame_stack)
    kwargs = dict(env_kwargs or {})
    if seed is None:
        master = int(np.random.SeedSequence().entropy)
    else:
        master = int(seed)
    env_fns = [
        _make_worker_env(dict(kwargs), np.random.SeedSequence(worker_seed), stack)
        for worker_seed in worker_seeds(master, int(n_envs))
    ]
    # ``AsyncVectorEnv.__init__`` builds one env in *this* process from
    # ``env_fns[0]`` to read the spaces. Track it so a failing constructor can
    # close it: Gymnasium only closes it on its own success path.
    probe = ProbeSink()
    env_fns[0] = _track_probe_env(env_fns[0], probe)
    try:
        vec = AshCrownVectorEnv(
            env_fns,
            shared_memory=False,
            copy=True,
            context=context,
            daemon=daemon,
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    except BaseException:
        close_envs(probe.envs)
        raise
    probe.envs.clear()  # the probe env is closed by the constructor
    return vec


def _make_worker_env(
    kwargs: dict[str, Any],
    seed_sequence: np.random.SeedSequence,
    frame_stack: int = 1,
) -> Any:
    """Picklable ``env_fn`` for one worker (spawn-safe: no closures over envs)."""
    stack = resolve_frame_stack(frame_stack)

    def _factory() -> Any:
        env: Any = SeededAshCrownEnv(AshCrownEnv(**kwargs), seed_sequence)
        # Outermost on purpose: the stack must be the last wrapper so the
        # published observation is the stacked one and no other wrapper has to
        # know about it.
        return wrap_frame_stack(env, stack)

    return _factory


class ProbeSink:
    """Collects the parent-side probe env Gymnasium builds from ``env_fns[0]``.

    ``AsyncVectorEnv.__init__`` constructs one env in the *calling* process to
    read its spaces and only closes it on its own success path, so a raising
    constructor would leak that env (and its Godot process). The sink makes it
    reachable from :func:`make_vec_env`.

    It pickles as an **empty** sink: the env fns are shipped to the workers with
    ``cloudpickle``, and a worker must not inherit a reference to the parent's
    env (which holds a live subprocess and is not picklable).
    """

    def __init__(self) -> None:
        self.envs: list[Any] = []

    def __reduce__(self) -> tuple[type[ProbeSink], tuple[()]]:  # type: ignore[valid-type]
        return (ProbeSink, ())


def _track_probe_env(env_fn: Any, sink: ProbeSink) -> Any:
    """Wrap ``env_fn`` so the env it builds in this process lands in ``sink``."""

    def _factory() -> Any:
        env = env_fn()
        sink.envs.append(env)
        return env

    return _factory


def close_envs(envs: list[Any]) -> None:
    """Close every env in ``envs``, swallowing teardown errors (best effort)."""
    for env in envs:
        try:
            env.close()
        except Exception:  # teardown must never mask the constructor's cause
            continue
    envs.clear()


def worker_seeds(seed: int, n_envs: int) -> list[int]:
    """The per-worker seed entropies used by :func:`make_vec_env`.

    ``SeedSequence(master).spawn(n)`` gives every worker a distinct child
    sequence; the child's ``entropy`` is the *parent's* entropy (identical for
    all children), so the worker seed is the first 32-bit state of the child.
    The mapping is deterministic for a given master seed.
    """
    children = np.random.SeedSequence(int(seed)).spawn(int(n_envs))
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


__all__ = [
    "DEFAULT_CONTEXT",
    "AshCrownVectorEnv",
    "ProbeSink",
    "SeededAshCrownEnv",
    "_make_worker_env",
    "close_envs",
    "make_vec_env",
    "worker_seeds",
]

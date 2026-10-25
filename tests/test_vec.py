"""Vector-environment tests (RL-4, ``docs/07`` §9.3).

One Godot process per worker, so every test here is ``integration``-marked and
skips cleanly when the game checkout or the godot binary is missing.
"""

from __future__ import annotations

import gc
import subprocess
import time
import weakref
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from ashcrown_rl.vec import AshCrownVectorEnv, make_vec_env, worker_seeds


@pytest.fixture()
def vec_env(env_kwargs: dict[str, Any]) -> AshCrownVectorEnv:
    env = make_vec_env(
        2,
        seed=20260101,
        env_kwargs={**env_kwargs, "level": 1, "max_ticks": 120},
    )
    try:
        yield env
    finally:
        env.close()


@pytest.mark.integration
def test_two_envs_have_batched_spaces(vec_env: AshCrownVectorEnv) -> None:
    assert vec_env.num_envs == 2
    assert isinstance(vec_env.single_observation_space, gym.spaces.Box)
    assert vec_env.observation_space.shape == (2, vec_env.single_observation_space.shape[0])
    assert vec_env.observation_space.dtype == np.float32
    assert vec_env.single_action_space.nvec.tolist() == [9, 2, 2]
    assert vec_env.action_space.shape == (2, 3)


@pytest.mark.integration
def test_workers_are_seeded_deterministically(vec_env: AshCrownVectorEnv) -> None:
    """``SeedSequence(master).spawn(n)`` assigns one stable seed per worker."""
    expected = worker_seeds(20260101, 2)
    actual = list(vec_env.call("base_seed"))
    assert actual == expected
    assert len(set(actual)) == 2
    # Same master seed → same worker seeds (and different from another master).
    assert worker_seeds(20260101, 2) == expected
    assert worker_seeds(20260102, 2) != expected


@pytest.mark.integration
def test_reset_and_step_are_batched_and_autoreset_leaves_the_final_obs(
    vec_env: AshCrownVectorEnv,
) -> None:
    obs, infos = vec_env.reset()
    assert obs.shape == (2, 3254)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert "obs_schema" in infos

    episode_seeds = vec_env.call("episode_seeds")
    assert all(len(seeds) == 1 for seeds in episode_seeds)
    assert episode_seeds[0] != episode_seeds[1], "the two workers run different seeds"

    actions = np.array([[0, 0, 0], [0, 0, 0]], dtype=np.int64)
    saw_terminal = False
    for _ in range(60):
        obs, rewards, terminated, truncated, infos = vec_env.step(actions)
        assert obs.shape == (2, 3254)
        assert rewards.shape == (2,)
        assert terminated.shape == (2,) and truncated.shape == (2,)
        if np.any(terminated | truncated):
            saw_terminal = True
            # Gymnasium's SAME_STEP autoreset puts the terminal observation in
            # ``final_obs`` and the new episode's observation in ``obs``.
            assert "final_obs" in infos
            finals = [value for value in infos["final_obs"] if value is not None]
            assert finals, "the autoreset step must expose the final observation"
            assert all(np.asarray(value).shape == (3254,) for value in finals)
            break
    assert saw_terminal, "a 120-tick episode must end within 60 decisions"
    # Both workers auto-reset independently and keep stepping.
    obs, *_ = vec_env.step(actions)
    assert np.all(np.isfinite(obs))


@pytest.mark.integration
def test_close_stops_every_child_process(vec_env: AshCrownVectorEnv) -> None:
    vec_env.reset()
    processes = list(vec_env.processes)
    assert len(processes) == 2
    vec_env.close()
    vec_env.close()  # idempotent
    assert all(not process.is_alive() for process in processes)


# --------------------------------------------------------------------------- #
# Process-leak regressions (review findings on #94)
# --------------------------------------------------------------------------- #
def _engine_pids() -> set[int]:
    """PIDs of live ``rl_env.tscn`` engine processes on this host."""
    result = subprocess.run(
        ["pgrep", "-f", "scenes/rl/rl_env.tscn"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {int(line) for line in result.stdout.split() if line.strip()}


def _wait_for(predicate: Any, timeout: float = 30.0) -> bool:
    """Poll ``predicate`` until true (or ``timeout`` elapses); return the result."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return bool(predicate())


@pytest.mark.integration
def test_close_accepts_the_gymnasium_terminate_flag(env_kwargs: dict[str, Any]) -> None:
    """``AsyncVectorEnv.__del__`` calls ``close(terminate=True)`` (docs §9.3)."""
    vec = make_vec_env(2, seed=20260106, env_kwargs={**env_kwargs, "max_ticks": 120})
    vec.reset()
    processes = list(vec.processes)
    assert len(processes) == 2
    vec.close(terminate=True)
    assert _wait_for(lambda: not any(process.is_alive() for process in processes))
    vec.close(terminate=True)  # idempotent


@pytest.mark.integration
def test_garbage_collection_closes_the_workers(env_kwargs: dict[str, Any]) -> None:
    """A dropped vector env must not leave live workers (nor engines) behind."""
    baseline = _engine_pids()
    vec = make_vec_env(2, seed=20260107, env_kwargs={**env_kwargs, "max_ticks": 120})
    vec.reset()
    processes = list(vec.processes)
    ref = weakref.ref(vec)
    del vec
    gc.collect()
    assert ref() is None, "the vector env is still referenced"
    assert _wait_for(lambda: not any(process.is_alive() for process in processes))
    assert _wait_for(lambda: not (_engine_pids() - baseline))


@pytest.mark.integration
def test_a_failed_construction_leaks_no_worker(
    env_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A raising ``__init__`` reaps the workers it already started."""
    baseline = _engine_pids()

    def _boom(self: Any) -> None:
        raise RuntimeError("space check failed")

    monkeypatch.setattr(AshCrownVectorEnv, "_check_spaces", _boom)
    with pytest.raises(RuntimeError, match="space check failed"):
        make_vec_env(1, seed=20260108, env_kwargs={**env_kwargs, "max_ticks": 120})
    assert _wait_for(lambda: not (_engine_pids() - baseline)), "a worker engine survived"


@pytest.mark.integration
def test_a_failed_construction_closes_the_parent_probe_env(
    env_kwargs: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe env Gymnasium builds from ``env_fns[0]`` is closed on failure."""
    import gymnasium.vector.async_vector_env as async_vector_env

    baseline = _engine_pids()

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("batch_space failed")

    monkeypatch.setattr(async_vector_env, "batch_space", _boom)
    with pytest.raises(RuntimeError, match="batch_space failed"):
        make_vec_env(2, seed=20260109, env_kwargs={**env_kwargs, "max_ticks": 120})
    assert _wait_for(lambda: not (_engine_pids() - baseline)), "the probe env survived"

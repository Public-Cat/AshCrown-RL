"""PPO training-pipeline tests that need the ``train`` extra (RL-7, INT-97).

The Stable-Baselines3 stack (``AnchoredPPO`` and its anchor-loss update,
``make_training_vec_env`` and ``train_ppo``) plus the ``integration`` runs that
train a real (tiny) PPO model on two headless environments: the checkpoint
loads under a matching contract, a tampered metadata file is refused, a PPO
episode records and replays in a fresh process, and the parallel evaluator
aggregates seeds.

``stable-baselines3`` and ``torch`` are the optional ``train`` extra
(``uv sync --extra dev --extra train``), so the whole module is skipped when
they are missing instead of failing collection — the documented dev-only gate
(``docs/07`` §12.3) must stay green. :mod:`ashcrown_rl.train` itself imports
with the ``dev`` extra alone; only these SB3 entry points need the extra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

pytest.importorskip("torch", reason="the PPO pipeline needs the train extra")
pytest.importorskip("stable_baselines3", reason="the PPO pipeline needs the train extra")

from ashcrown_rl.evaluate import evaluate_ppo
from ashcrown_rl.train import (
    CURVE_COLUMNS,
    MetadataMismatch,
    load_policy,
    make_training_vec_env,
    metadata_path,
    read_metadata,
    train_ppo,
)


# --------------------------------------------------------------------------- #
# Anchored PPO (behaviour-cloning anchor loss, RL-7 round 2)
# --------------------------------------------------------------------------- #
class _TinyObsEnv(gym.Env):
    """Random-observation env with the AshCrown action space (no Godot)."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([9, 2, 2])

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        return self.observation_space.sample().astype(np.float32), {}

    def step(self, action: Any):
        observation = self.observation_space.sample().astype(np.float32)
        return observation, 0.0, False, False, {}


def test_anchored_ppo_records_the_anchor_loss() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ashcrown_rl.train import AnchoredPPO

    kwargs: dict[str, Any] = {
        "n_steps": 64,
        "batch_size": 32,
        "n_epochs": 2,
        "device": "cpu",
        "policy_kwargs": {"net_arch": [16, 16]},
    }
    anchor = PPO("MlpPolicy", DummyVecEnv([_TinyObsEnv]), **kwargs)
    model = AnchoredPPO(
        "MlpPolicy",
        DummyVecEnv([_TinyObsEnv]),
        anchor=anchor,
        bc_coef=0.5,
        bc_steps=2,
        **kwargs,
    )
    model.learn(total_timesteps=128, progress_bar=False)
    assert model.bc_loss > 0.0
    assert "train/bc_loss" in model.logger.name_to_value


def test_anchored_ppo_without_anchor_keeps_the_plain_update() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import DummyVecEnv

    from ashcrown_rl.train import AnchoredPPO

    model = AnchoredPPO(
        "MlpPolicy",
        DummyVecEnv([_TinyObsEnv]),
        bc_coef=0.0,
        n_steps=64,
        batch_size=32,
        n_epochs=2,
        device="cpu",
        policy_kwargs={"net_arch": [16, 16]},
    )
    model.learn(total_timesteps=128, progress_bar=False)
    assert model.bc_loss == 0.0


# --------------------------------------------------------------------------- #
# Integration tests (real headless processes)
# --------------------------------------------------------------------------- #
def _train_tiny(tmp_path: Path, env_kwargs: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    """Train a couple of rollouts on two environments and return the result."""
    kwargs: dict[str, Any] = {
        "n_envs": 2,
        "total_steps": 512,
        "seed": 11,
        "env_kwargs": {**env_kwargs, "level": 1, "frame_skip": 4, "max_ticks": 600},
        "checkpoint_dir": tmp_path / "ckpt",
        "log_csv": tmp_path / "curve.csv",
        "command": "pytest tiny",
        "algorithm": {"name": "PPO", "policy": "MlpPolicy"},
        "hyperparameters": {"n_steps": 256, "batch_size": 64, "net_arch": [32, 32]},
        "device": "cpu",
        "save_freq": 256,
    }
    kwargs.update(overrides)
    return train_ppo(**kwargs)


@pytest.mark.integration
def test_training_writes_a_checkpoint_with_matching_metadata(
    env_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    from ashcrown_rl.env import AshCrownEnv

    result = _train_tiny(tmp_path, env_kwargs)
    checkpoints = [Path(path) for path in result["checkpoints"]]
    assert checkpoints and checkpoints[0].is_file()
    metadata = read_metadata(checkpoints[-1])
    env = AshCrownEnv(level=1, frame_skip=4, **env_kwargs)
    try:
        assert metadata["protocol"] == env.hello.get("protocol")
        assert metadata["obs_schema"] == env.hello.get("obs_schema")
        assert metadata["layout_hash"] == env.layout_hash
        assert metadata["observation_config"] == env.observation_config()
        assert metadata["total_steps"] >= 1
        assert metadata["command"] == "pytest tiny"
        model = load_policy(checkpoints[-1], env)
        observation = env.reset(seed=1)[0]
        action, _state = model.predict(observation, deterministic=True)
        assert len(list(action)) == 3
    finally:
        env.close()
    # The curve CSV exists with the documented header and at least one row.
    curve = (tmp_path / "curve.csv").read_text(encoding="utf-8").splitlines()
    assert curve[0].split(",") == list(CURVE_COLUMNS)
    assert len(curve) >= 2


@pytest.mark.integration
def test_training_with_a_bc_anchor_records_it_in_metadata(
    env_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    pytest.importorskip("torch")
    anchor = Path(_train_tiny(tmp_path, env_kwargs)["checkpoints"][-1])
    result = _train_tiny(
        tmp_path / "anchored",
        env_kwargs,
        checkpoint_dir=tmp_path / "anchored" / "ckpt",
        log_csv=tmp_path / "anchored" / "curve.csv",
        bc_anchor=str(anchor),
        bc_coef=0.5,
        bc_steps=2,
        save_freq=10_000,
    )
    metadata = read_metadata(Path(result["checkpoints"][-1]))
    assert metadata["bc_anchor"] == str(anchor)
    assert metadata["bc_coef"] == 0.5
    assert metadata["bc_steps"] == 2


@pytest.mark.integration
def test_load_policy_refuses_a_tampered_metadata_file(
    env_kwargs: dict[str, Any], tmp_path: Path
) -> None:
    from ashcrown_rl.env import AshCrownEnv

    result = _train_tiny(tmp_path, env_kwargs)
    checkpoint = Path(result["checkpoints"][-1])
    metadata = json.loads(metadata_path(checkpoint).read_text(encoding="utf-8"))
    metadata["layout_hash"] = "0" * 64
    metadata_path(checkpoint).write_text(json.dumps(metadata), encoding="utf-8")
    env = AshCrownEnv(level=1, frame_skip=4, **env_kwargs)
    try:
        with pytest.raises(MetadataMismatch):
            load_policy(checkpoint, env)
    finally:
        env.close()


@pytest.mark.integration
def test_make_training_vec_env_curriculum_assigns_levels(env_kwargs: dict[str, Any]) -> None:
    vec = make_training_vec_env(
        n_envs=2,
        seed=3,
        env_kwargs={**env_kwargs, "level": 1, "frame_skip": 4, "max_ticks": 120},
        levels=[1, 2],
    )
    try:
        obs = vec.reset()
        assert obs.shape == (2, 3254)
    finally:
        vec.close()


@pytest.mark.integration
def test_evaluate_ppo_runs_parallel_seed_chunks(env_kwargs: dict[str, Any], tmp_path: Path) -> None:
    result = _train_tiny(tmp_path, env_kwargs)
    checkpoint = result["checkpoints"][-1]
    report = evaluate_ppo(
        checkpoint=checkpoint,
        seeds=[901, 902],
        env_kwargs={**env_kwargs, "level": 1, "frame_skip": 4, "max_ticks": 240},
        max_steps=5,
        workers=2,
        csv_path=tmp_path / "eval.csv",
    )
    assert report["workers"] == 2
    assert [episode["seed"] for episode in report["episodes"]] == [901, 902]
    assert report["aggregates"]["episodes"] == 2
    assert report["pins"]["layout_hash"]
    assert (tmp_path / "eval.csv").is_file()


@pytest.mark.integration
def test_ppo_replay_log_round_trip_and_gzip(env_kwargs: dict[str, Any], tmp_path: Path) -> None:
    from ashcrown_rl.env import AshCrownEnv
    from ashcrown_rl.replay import default_policy, read_log, record, replay, write_log

    result = _train_tiny(tmp_path, env_kwargs)
    checkpoint = result["checkpoints"][-1]
    env = AshCrownEnv(level=1, frame_skip=4, max_ticks=240, **env_kwargs)
    try:
        policy = default_policy("ppo", 7, env.layout, checkpoint=checkpoint, env=env)
        log = record(env, seed=902, steps=12, policy=policy, policy_name="ppo")
    finally:
        env.close()
    path = write_log(tmp_path / "ppo.json.gz", log)
    assert path.suffix == ".gz"
    loaded = read_log(path)
    assert len(loaded["actions"]) == len(log["actions"])
    verify = AshCrownEnv(level=1, frame_skip=4, max_ticks=240, **env_kwargs)
    try:
        outcome = replay(verify, loaded)
    finally:
        verify.close()
    assert outcome["ok"]

"""Training-pipeline unit tests: metadata, curriculum, curve math (RL-7).

Everything here is numpy-only, so this module runs with the ``dev`` extra alone
— it is the half of the RL-7 suite that must stay green in the documented
``docs/07`` §12.3 gate (``uv sync --locked --extra dev && uv run pytest``). The
Stable-Baselines3 half (``AnchoredPPO``, ``make_training_vec_env``,
``train_ppo`` and the integration runs) lives in ``tests/test_train_ppo.py``,
which skips cleanly when the ``train`` extra is absent (INT-97).

Unit coverage: the evaluated-seed helpers (``expand_seed_ranges``,
``_seed_line``, ``_write_partial``, ``split_seeds``, ``resolve_policy``), the
curriculum level schedule and wrapper, the training-curve row math, and the
checkpoint metadata contract (write/read and refuse-on-mismatch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from ashcrown_rl.evaluate import (
    EpisodeMetrics,
    _seed_line,
    _write_partial,
    expand_seed_ranges,
    resolve_policy,
    split_seeds,
)
from ashcrown_rl.train import (
    CURVE_COLUMNS,
    CurriculumEnv,
    LevelSchedule,
    MetadataMismatch,
    _curve_row,
    metadata_path,
    read_metadata,
    verify_metadata,
    write_metadata,
)


# --------------------------------------------------------------------------- #
# Unit tests (no process)
# --------------------------------------------------------------------------- #
def test_expand_seed_ranges_accepts_ranges_and_single_values() -> None:
    assert expand_seed_ranges("1000-1002,7") == [1000, 1001, 1002, 7]
    assert expand_seed_ranges(" 5 ") == [5]
    assert expand_seed_ranges("") == []
    assert expand_seed_ranges(None) == []


def test_seed_line_is_the_stdout_record() -> None:
    metrics = EpisodeMetrics(seed=9000, level=1, terminal_reason="death", steps=10)
    line = _seed_line(metrics.as_dict())
    assert line["seed"] == 9000
    assert line["reason"] == "death"
    assert set(line) == {
        "seed",
        "reason",
        "decisions",
        "seconds",
        "swings",
        "hits",
        "requests",
        "hits_per_swing",
        "hits_per_request",
        "dmg_dealt",
        "dmg_taken",
        "kills",
        "hp",
    }


def test_write_partial_marks_pending_seeds_and_partial(tmp_path: Path) -> None:
    metrics = EpisodeMetrics(
        seed=1, level=1, terminal_reason="level_clear", terminated=True, policy="ppo"
    )
    out = tmp_path / "eval.json"
    csv = tmp_path / "eval.csv"
    _write_partial(
        out,
        csv,
        started_wall="2026-10-25T00:00:00+00:00",
        seeds=[1, 2, 3],
        episodes=[metrics],
        pins={"protocol": 1},
        config={"policy": "ppo"},
        failures=[{"kind": "timeout", "seed": 3}],
        elapsed=1.5,
        workers=2,
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["partial"] is True
    assert payload["seeds_pending"] == [2, 3]
    assert payload["aggregates"]["clears"] == 1
    assert payload["failures"] == [{"kind": "timeout", "seed": 3}]
    assert csv.read_text(encoding="utf-8").splitlines()[1].startswith("1,ppo,1,level_clear,1,")


def test_level_schedule_is_round_robin() -> None:
    schedule = LevelSchedule([1, 2, 3])
    assert [schedule.level_for(index) for index in range(7)] == [1, 2, 3, 1, 2, 3, 1]


def test_level_schedule_rejects_an_empty_level_list() -> None:
    with pytest.raises(ValueError):
        LevelSchedule([])


class _StubEnv(gym.Env):
    """Minimal gymnasium env recording the options of every reset."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)
    action_space = gym.spaces.MultiDiscrete([9, 2, 2])

    def __init__(self) -> None:
        self.options: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.options.append(dict(options or {}))
        return None, {"seed": seed}

    def step(self, action: Any):
        return None, 0.0, False, False, {}

    def close(self) -> None:
        return None


def test_curriculum_env_injects_a_level_per_episode() -> None:
    stub = _StubEnv()
    env = CurriculumEnv(stub, [1, 2])
    env.reset(seed=1)
    env.reset(seed=2)
    env.reset(seed=3)
    assert [options["level"] for options in stub.options] == [1, 2, 1]


def test_curriculum_env_keeps_an_explicit_level() -> None:
    stub = _StubEnv()
    env = CurriculumEnv(stub, [1, 2, 3, 4])
    env.reset(seed=1, options={"level": 3})
    assert stub.options[-1] == {"level": 3}
    # The schedule still advanced: the next auto-reset takes level 2.
    env.reset(seed=2)
    assert stub.options[-1]["level"] == 2


def test_curve_row_aggregates_episodes_and_logger_values() -> None:
    row = _curve_row(
        timestep=4096,
        episodes=[
            {
                "r": 10.0,
                "l": 100,
                "terminal_reason": "level_clear",
                "game_seconds": 30.0,
                "kills": 4,
                "hits_landed": 20,
                "attacks_started": 22,
            },
            {
                "r": 0.0,
                "l": 300,
                "terminal_reason": "death",
                "game_seconds": 90.0,
                "kills": 0,
                "hits_landed": 2,
                "attacks_started": 10,
            },
        ],
        logger_values={
            "train/entropy_loss": -2.5,
            "train/policy_gradient_loss": -0.01,
            "train/value_loss": 0.5,
            "train/approx_kl": 0.02,
        },
        wall_seconds=4.0,
    )
    assert tuple(row) == CURVE_COLUMNS
    assert row["episodes"] == 2
    assert row["episode_reward"] == pytest.approx(5.0)
    assert row["success_rate"] == pytest.approx(0.5)
    assert row["mean_length"] == pytest.approx(200.0)
    assert row["entropy"] == pytest.approx(2.5)
    assert row["clears"] == 1
    assert row["mean_game_seconds"] == pytest.approx(60.0)
    # The RL-7 round-2 acceptance gate reads these: a policy that stops landing
    # hits is reward hacking, not learning.
    assert row["kills"] == pytest.approx(2.0)
    assert row["hits_landed"] == pytest.approx(11.0)
    assert row["attacks_started"] == pytest.approx(16.0)


def test_curve_row_keeps_the_columns_when_nothing_finished() -> None:
    row = _curve_row(timestep=0, episodes=[], logger_values={}, wall_seconds=0.0)
    assert row["episodes"] == 0
    assert row["episode_reward"] == 0.0
    assert row["success_rate"] == 0.0


def test_metadata_round_trip_and_mismatch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_100_steps.zip"
    metadata = {
        "version": 1,
        "protocol": 1,
        "obs_schema": "ashcrown-obs-v2",
        "layout_hash": "abc",
        "observation_config": {"obs_grid": True},
        "total_steps": 100,
    }
    write_metadata(checkpoint, metadata)
    assert metadata_path(checkpoint).is_file()
    assert read_metadata(checkpoint)["total_steps"] == 100
    verify_metadata(
        metadata,
        protocol=1,
        obs_schema="ashcrown-obs-v2",
        layout_hash="abc",
        observation_config={"obs_grid": True},
    )
    with pytest.raises(MetadataMismatch):
        verify_metadata(
            metadata,
            protocol=2,
            obs_schema="ashcrown-obs-v2",
            layout_hash="abc",
            observation_config={"obs_grid": True},
        )
    with pytest.raises(MetadataMismatch):
        verify_metadata(
            metadata,
            protocol=1,
            obs_schema="ashcrown-obs-v1",
            layout_hash="abc",
            observation_config={"obs_grid": True},
        )
    with pytest.raises(MetadataMismatch):
        verify_metadata(
            metadata,
            protocol=1,
            obs_schema="ashcrown-obs-v2",
            layout_hash="other",
            observation_config={"obs_grid": True},
        )
    with pytest.raises(MetadataMismatch):
        verify_metadata(
            metadata,
            protocol=1,
            obs_schema="ashcrown-obs-v2",
            layout_hash="abc",
            observation_config={"obs_grid": True, "obs_grid_radius": 4},
        )


def test_read_metadata_refuses_a_checkpoint_without_a_sidecar(tmp_path: Path) -> None:
    with pytest.raises(MetadataMismatch):
        read_metadata(tmp_path / "missing.zip")


def test_split_seeds_is_contiguous_and_balanced() -> None:
    assert split_seeds(list(range(10)), 3) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert split_seeds([5], 8) == [[5]]
    assert split_seeds([], 4) == []


def test_resolve_ppo_policy_needs_a_checkpoint() -> None:
    with pytest.raises(ValueError, match="checkpoint"):
        resolve_policy("ppo", 1)

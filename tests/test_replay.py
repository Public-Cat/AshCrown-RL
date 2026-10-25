"""Action-log replay tests (RL-4, ``docs/07`` §10.3/§10.4).

The round-trip test records a short episode on one real process and replays it
on a **fresh** one, comparing the per-step state-hash sequences — that is the
determinism contract. The unit tests drive a scripted stub so the comparison and
error paths run without Godot.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ashcrown_rl.baseline import HeuristicPolicy
from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.protocol import StepResult
from ashcrown_rl.replay import LOG_VERSION, ReplayMismatch, main, record, replay


class ScriptedEnv:
    """Minimal env surface ``replay()`` needs (``layout_hash``/reset/step)."""

    layout_hash = "b" * 64

    def __init__(self, hashes: list[str]) -> None:
        self.hashes = hashes
        self.seeds: list[int] = []
        self.actions: list[list[int]] = []
        self.index = 0

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        assert seed is not None
        self.seeds.append(int(seed))
        self.index = 0
        return np.zeros(4, dtype=np.float32), {"state_hash": "reset"}

    def step(self, action: Any) -> StepResult:
        self.actions.append([int(value) for value in action])
        info = {"state_hash": self.hashes[self.index], "terminal_reason": ""}
        self.index += 1
        return StepResult(
            obs=np.zeros(4, dtype=np.float32),
            reward=0.0,
            terminated=False,
            truncated=False,
            info=info,
        )


def _log(hashes: list[str], *, seed: int = 77, actions: int = 2) -> dict[str, Any]:
    return {
        "version": LOG_VERSION,
        "pins": {"layout_hash": "b" * 64},
        "config": {"level": 1, "character": "warrior", "frame_skip": 4, "max_ticks": 600},
        "seed": seed,
        "policy": "heuristic",
        "actions": [[1, 0, 0]] * actions,
        "hashes": hashes,
        "rewards": [0.0] * actions,
        "terminal": {},
    }


# --------------------------------------------------------------------------- #
# Unit tests (stub env)
# --------------------------------------------------------------------------- #
def test_replay_accepts_a_matching_log() -> None:
    env = ScriptedEnv(["h1", "h2"])
    result = replay(env, _log(["h1", "h2"]))
    assert result["ok"] and result["steps"] == 2
    assert env.seeds == [77]
    assert env.actions == [[1, 0, 0], [1, 0, 0]]


def test_replay_reports_the_first_divergence() -> None:
    env = ScriptedEnv(["h1", "nope"])
    with pytest.raises(ReplayMismatch) as caught:
        replay(env, _log(["h1", "h2"]))
    assert caught.value.step == 1
    assert "nope" in str(caught.value)


def test_replay_rejects_a_malformed_log() -> None:
    env = ScriptedEnv([])
    with pytest.raises(ValueError, match="actions but"):
        replay(env, _log(["h1"]))
    with pytest.raises(ValueError, match="version"):
        replay(env, {"version": 99})


def test_replay_rejects_a_layout_mismatch() -> None:
    env = ScriptedEnv(["h1", "h2"])
    log = _log(["h1", "h2"])
    log["pins"]["layout_hash"] = "c" * 64
    with pytest.raises(ValueError, match="layout_hash"):
        replay(env, log)


# --------------------------------------------------------------------------- #
# Integration: record on one process, replay on a fresh one
# --------------------------------------------------------------------------- #
@pytest.mark.integration
def test_record_and_replay_round_trip_across_processes(env_kwargs: dict[str, Any]) -> None:
    env = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        policy = HeuristicPolicy(env.layout)
        log = record(env, seed=20260103, steps=40, policy=policy, policy_name="heuristic")
    finally:
        env.close()
    assert log["version"] == LOG_VERSION
    assert len(log["actions"]) == len(log["hashes"]) == 40
    assert log["pins"]["layout_hash"] == env.layout_hash
    assert all(len(value) == 64 for value in log["hashes"])

    fresh = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        result = replay(fresh, log)
    finally:
        fresh.close()
    assert result["ok"] and result["steps"] == 40
    assert result["layout_hash"] == log["pins"]["layout_hash"]


@pytest.mark.integration
def test_replay_detects_a_tampered_action_log(env_kwargs: dict[str, Any]) -> None:
    env = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        policy = HeuristicPolicy(env.layout)
        log = record(env, seed=20260104, steps=10, policy=policy, policy_name="heuristic")
    finally:
        env.close()
    tampered = dict(log)
    tampered["actions"] = [list(action) for action in log["actions"]]
    tampered["actions"][-1] = [5, 0, 0]
    fresh = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        with pytest.raises(ReplayMismatch):
            replay(fresh, tampered)
    finally:
        fresh.close()


@pytest.mark.integration
def test_replay_cli_writes_and_verifies_a_log(env_kwargs: dict[str, Any], tmp_path: Any) -> None:
    out = tmp_path / "episode.json"
    assert (
        main(
            [
                "record",
                "--seed",
                "20260105",
                "--steps",
                "8",
                "--max-ticks",
                "600",
                "--policy",
                "zero",
                "--godot-bin",
                env_kwargs["godot_bin"],
                "--game-project",
                env_kwargs["project"],
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert out.is_file()
    assert (
        main(
            [
                "replay",
                "--file",
                str(out),
                "--godot-bin",
                env_kwargs["godot_bin"],
                "--game-project",
                env_kwargs["project"],
            ]
        )
        == 0
    )

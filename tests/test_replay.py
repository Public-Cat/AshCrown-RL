"""Action-log replay tests (RL-4, ``docs/07`` §10.3/§10.4).

The round-trip test records a short episode on one real process and replays it
on a **fresh** one, comparing the per-step state-hash sequences — that is the
determinism contract. The unit tests drive a scripted stub so the comparison and
error paths run without Godot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ashcrown_rl
from ashcrown_rl.baseline import HeuristicPolicy
from ashcrown_rl.env import AshCrownEnv
from ashcrown_rl.protocol import StepResult
from ashcrown_rl.replay import (
    LOG_VERSION,
    MAX_DIFF_CHARS,
    ReplayMismatch,
    diff_state_lines,
    main,
    record,
    replay,
    state_line,
)

#: The training package root (the ``cwd``/``PYTHONPATH`` a ``python -m`` run needs).
PACKAGE_ROOT = Path(ashcrown_rl.__file__).resolve().parent.parent


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


# --------------------------------------------------------------------------- #
# The promised first-diff state diff (``docs/07`` §10.4, review finding on #94)
# --------------------------------------------------------------------------- #
def test_state_line_is_bounded_and_covers_the_canonical_quantities() -> None:
    line = state_line(
        {
            "tick": 48,
            "level": 1,
            "hp": 94.25,
            "kills": 3,
            "enemies_alive": 10,
            "gold": 12,
            "player_cell": (12.25, 4.5),
            "player_position": (1234.5, 678.25),
            "portal_active": True,
            "portal_distance": 512.0,
            "boss_hp": 0.0,
            "boss_phase": 0,
            "wave": -1,
            "stuck": False,
        }
    )
    for fragment in ("tick=48", "hp=94.25", "kills=3", "alive=10", "cell=12.25,4.50"):
        assert fragment in line
    assert line.count("|") == 11, "one field per canonical quantity"
    assert len(line) <= 240


def test_diff_state_lines_reports_only_the_changed_fields() -> None:
    recorded = "tick=8|hp=94.20|kills=1|alive=12"
    fresh = "tick=8|hp=90.40|kills=1|alive=12"
    diff = diff_state_lines(recorded, fresh)
    assert diff == "hp 94.20 -> 90.40"
    assert diff_state_lines(recorded, recorded) == ""
    assert diff_state_lines("", fresh) == ""


def test_diff_state_lines_is_bounded() -> None:
    recorded = "|".join(f"f{index}=0.00" for index in range(200))
    fresh = "|".join(f"f{index}=1.00" for index in range(200))
    assert len(diff_state_lines(recorded, fresh)) <= MAX_DIFF_CHARS


def test_replay_mismatch_carries_the_first_diff_state_lines() -> None:
    env = ScriptedEnv(["h1", "nope"])
    log = _log(["h1", "h2"])
    log["states"] = [
        "tick=4|level=1|hp=94.20|kills=1|alive=12",
        "tick=8|level=1|hp=94.20|kills=1|alive=12",
    ]
    with pytest.raises(ReplayMismatch) as caught:
        replay(env, log)
    mismatch = caught.value
    assert mismatch.step == 1
    assert mismatch.recorded_state.startswith("tick=8")
    assert mismatch.live_state.startswith("tick=")
    assert "hp 94.20 ->" in mismatch.diff
    assert "state diff (recorded -> fresh)" in str(mismatch)
    assert "last agreeing state" in str(mismatch)
    assert len(str(mismatch)) <= MAX_DIFF_CHARS + 400


def test_replay_mismatch_without_states_still_names_the_step() -> None:
    env = ScriptedEnv(["h1", "nope"])
    with pytest.raises(ReplayMismatch, match="no state lines") as caught:
        replay(env, _log(["h1", "h2"]))
    assert caught.value.diff == ""


@pytest.mark.integration
def test_record_stores_bounded_state_lines(env_kwargs: dict[str, Any]) -> None:
    env = AshCrownEnv(level=1, max_ticks=600, **env_kwargs)
    try:
        policy = HeuristicPolicy(env.layout)
        log = record(env, seed=20260111, steps=12, policy=policy, policy_name="heuristic")
    finally:
        env.close()
    assert len(log["states"]) == len(log["actions"]) == 12
    assert log["states"][0].startswith("tick=")
    assert all(len(line) <= 240 for line in log["states"])
    assert "hp=" in log["states"][-1]


# --------------------------------------------------------------------------- #
# CLI as a module (``python -m ashcrown_rl.replay``)
# --------------------------------------------------------------------------- #
def _cli(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run ``python -m ashcrown_rl.replay <args>`` in a fresh interpreter."""
    environment = {**os.environ, "PYTHONPATH": str(PACKAGE_ROOT)}
    return subprocess.run(
        [sys.executable, "-m", "ashcrown_rl.replay", *args],
        cwd=str(PACKAGE_ROOT),
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


@pytest.mark.integration
def test_replay_module_cli_records_and_replays(env_kwargs: dict[str, Any], tmp_path: Any) -> None:
    """``__main__`` guard (review finding on #94): the CLI is not a no-op."""
    out = tmp_path / "episode.json"
    recorded = _cli(
        [
            "record",
            "--seed",
            "20260112",
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
    assert recorded.returncode == 0, recorded.stderr
    assert out.is_file() and out.stat().st_size > 0
    assert "REPLAY_RECORD" in recorded.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["states"] and payload["hashes"]

    played = _cli(
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
    assert played.returncode == 0, played.stderr
    line = next(text for text in played.stdout.splitlines() if text.startswith("REPLAY "))
    result = json.loads(line[len("REPLAY ") :])
    assert result["ok"] is True
    assert result["steps"] == len(payload["actions"])

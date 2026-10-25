"""``AshCrownEnv`` tests: Gymnasium API, config, terminal semantics, determinism.

Unit tests drive a scripted stub client (no Godot). The ``integration`` tests
launch the real headless environment and run ``gymnasium.utils.env_checker``
against it, exactly as ``docs/07`` §12.2 requires; they skip cleanly when the
game checkout or the godot binary is missing.
"""

from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ashcrown_rl.baseline import HeuristicPolicy
from ashcrown_rl.env import (
    DEFAULT_LEVEL_MAX_TICKS,
    DEFAULT_RUN_MAX_TICKS,
    MERCHANT_POLICIES,
    MODES,
    RESET_OPTIONS,
    AshCrownEnv,
)
from ashcrown_rl.protocol import GodotError, ProtocolError, ResetResult, StepResult
from ashcrown_rl.spaces import LayoutError, fixture_path, load_fixture_layout


# --------------------------------------------------------------------------- #
# Stub client (no process)
# --------------------------------------------------------------------------- #
def stub_hello() -> dict[str, Any]:
    """A handshake payload identical in shape to the live engine's."""
    with fixture_path().open(encoding="utf-8") as handle:
        layout = json.load(handle)
    return {
        "ok": True,
        "cmd": "hello",
        "protocol": layout["protocol"],
        "obs_schema": layout["obs_schema"],
        "action_schema": layout["action_schema"],
        "layout_hash": "a" * 64,
        "obs_len": layout["obs_len"],
        "obs_clamp": [-1.0, 1.0],
        "frame_skip_default": 4,
        "physics_ticks_per_second": 60,
        "character_ids": ["warrior", "knight", "fighter"],
        "enemy_types": layout["enemy_types"],
        "layout": layout,
        "action_directions": layout["action_directions"],
        "reward_profile_defaults": {
            "default": "combat",
            "profiles": {"combat": {}, "sparse": {}, "none": {}},
        },
    }


class StubClient:
    """Scripted protocol client: records requests, returns queued replies."""

    def __init__(self, *, steps: int = 3, terminal_at: int | None = None) -> None:
        self.hello = stub_hello()
        self.obs_len = int(self.hello["obs_len"])
        self.steps = int(steps)
        self.terminal_at = terminal_at
        self.reset_calls: list[dict[str, Any]] = []
        self.step_calls: list[tuple[int, int, int]] = []
        self.closed = 0
        self.tick = 0
        self.episode = 0
        self.obs = np.zeros(self.obs_len, dtype=np.float32)
        self.obs[0] = 0.1

    def reset(self, **kwargs: Any) -> ResetResult:
        self.reset_calls.append(dict(kwargs))
        self.tick = 0
        self.episode += 1
        info = {
            "tick": 0,
            "episode": self.episode,
            "level": kwargs.get("level", 1),
            "seed": kwargs.get("seed", 0),
            "game_seconds": 0.0,
            "hp": 100.0,
            "state_hash": "r" * 64,
            "layout_hash": self.hello["layout_hash"],
            "terminal_reason": "",
            "reward_components": {},
            "events": {},
        }
        return ResetResult(obs=self.obs.copy(), info=info)

    def step(self, move: int, attack: int = 0, interact: int = 0) -> StepResult:
        if len(self.step_calls) >= self.steps:
            raise GodotError("EPISODE_DONE", "stub exhausted")
        self.step_calls.append((int(move), int(attack), int(interact)))
        self.tick += 4
        index = len(self.step_calls)
        terminated = self.terminal_at is not None and index >= self.terminal_at
        info = {
            "tick": self.tick,
            "episode": self.episode,
            "level": 1,
            "game_seconds": self.tick / 60.0,
            "hp": 100.0,
            "state_hash": f"{index:064x}",
            "terminal_reason": "level_clear" if terminated else "",
            "reward_components": {"damage_dealt": 0.2, "step_penalty": -0.001},
            "events": {"damage_dealt": 20.0},
            "action": [move, attack, interact],
            "truncated": False,
        }
        return StepResult(
            obs=self.obs.copy(),
            reward=0.199,
            terminated=terminated,
            truncated=False,
            info=info,
        )

    def close(self) -> None:
        self.closed += 1


def make_stub_env(**overrides: Any) -> tuple[AshCrownEnv, StubClient]:
    """Build an env backed by a fresh stub client."""
    client = StubClient(**overrides.pop("client_kwargs", {}))
    env = AshCrownEnv(client_factory=lambda: client, **overrides)
    return env, client


# --------------------------------------------------------------------------- #
# Unit tests (stub client)
# --------------------------------------------------------------------------- #
def test_spaces_come_from_the_handshake() -> None:
    env, _ = make_stub_env()
    try:
        layout = load_fixture_layout()
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (layout.obs_len,)
        assert env.observation_space.dtype == np.float32
        assert float(env.observation_space.low[0]) == -1.0
        assert float(env.observation_space.high[0]) == 1.0
        assert isinstance(env.action_space, gym.spaces.MultiDiscrete)
        assert env.action_space.nvec.tolist() == [9, 2, 2]
        assert env.layout.obs_len == layout.obs_len
    finally:
        env.close()


def test_the_box_bounds_come_from_the_handshake_clamp() -> None:
    """``docs/07`` §5 publishes ``obs_clamp``; the Box must follow it."""
    client = StubClient()
    client.hello = {**client.hello, "obs_clamp": [-2.0, 2.0]}
    env = AshCrownEnv(client_factory=lambda: client)
    try:
        assert env.obs_bound == (-2.0, 2.0)
        assert float(env.observation_space.low[0]) == -2.0
        assert float(env.observation_space.high[0]) == 2.0
    finally:
        env.close()


def test_a_handshake_that_disagrees_on_obs_len_is_refused() -> None:
    """The top-level ``obs_len`` and the layout table must agree."""
    client = StubClient()
    client.hello = {**client.hello, "obs_len": int(client.hello["obs_len"]) + 1}
    with pytest.raises(LayoutError, match="obs_len"):
        AshCrownEnv(client_factory=lambda: client)


def test_metadata_publishes_the_layout_and_action_meanings() -> None:
    env, _ = make_stub_env()
    try:
        metadata = env.metadata
        assert metadata["render_modes"] == []
        layout_meta = metadata["obs_layout"]
        assert layout_meta["obs_len"] == env.obs_len
        assert layout_meta["blocks"]["global"] == [0, 31]
        assert layout_meta["grid"]["radius"] == 10
        meanings = metadata["action_meanings"]
        assert len(meanings) == 11
        assert meanings[0].startswith("0:stay")
        assert meanings[1].startswith("1:screen")
    finally:
        env.close()


def test_reset_forwards_the_config_and_returns_a_valid_observation() -> None:
    env, client = make_stub_env()
    try:
        obs, info = env.reset(seed=1234)
        assert isinstance(obs, np.ndarray) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        call = client.reset_calls[-1]
        assert call["seed"] == 1234
        assert call["mode"] == "level"
        assert call["level"] == 1
        assert call["character"] == "warrior"
        assert call["frame_skip"] == 4
        assert call["max_ticks"] == DEFAULT_LEVEL_MAX_TICKS
        assert call["reward_profile"] == "combat"
        assert call["hash_every_ticks"] == 60
        assert call["merchant_policy"] is None
        assert info["seed"] == 1234
        assert info["obs_schema"] == "ashcrown-obs-v1"
        assert info["layout_hash"] == "a" * 64
        assert info["layout_slices"]["player"] == [31, 24]
        assert info["episode_steps"] == 0
        assert env.episode_seed == 1234
    finally:
        env.close()


def test_reset_options_override_the_episode_config() -> None:
    env, client = make_stub_env()
    try:
        env.reset(
            seed=1,
            options={
                "level": 4,
                "character": "knight",
                "max_ticks": 600,
                "reward_profile": "sparse",
                "reward_overrides": {"kill_normal": 2.0},
                "merchant_policy": "none",
            },
        )
        call = client.reset_calls[-1]
        assert call["level"] == 4
        assert call["character"] == "knight"
        assert call["max_ticks"] == 600
        assert call["reward_profile"] == "sparse"
        assert call["reward_overrides"] == {"kill_normal": 2.0}
        assert call["merchant_policy"] == "none"
    finally:
        env.close()


def test_unknown_reset_option_is_rejected() -> None:
    env, _ = make_stub_env()
    try:
        with pytest.raises(ValueError, match="unknown reset option"):
            env.reset(seed=1, options={"level_index": 2})
        assert RESET_OPTIONS == {
            "level",
            "character",
            "max_ticks",
            "reward_profile",
            "reward_overrides",
            "merchant_policy",
        }
    finally:
        env.close()


def test_reset_without_a_seed_draws_and_returns_one() -> None:
    env, client = make_stub_env()
    try:
        _, first = env.reset(seed=99)
        _, second = env.reset()
        drawn = client.reset_calls[-1]["seed"]
        assert isinstance(drawn, int) and 0 <= drawn < 2**31
        assert second["seed"] == drawn != first["seed"]
        # The same explicit seed is forwarded verbatim (reproducible episodes).
        env.reset(seed=99)
        assert client.reset_calls[-1]["seed"] == 99
    finally:
        env.close()


def test_step_returns_the_gym_tuple_and_never_auto_resets() -> None:
    env, client = make_stub_env(client_kwargs={"steps": 5, "terminal_at": 2})
    try:
        env.reset(seed=3)
        obs, reward, terminated, truncated, info = env.step([2, 1, 0])
        assert env.observation_space.contains(obs)
        assert reward == pytest.approx(0.199)
        assert (terminated, truncated) == (False, False)
        assert info["episode_steps"] == 1
        assert info["episode_return"] == pytest.approx(0.199)
        assert client.step_calls[-1] == (2, 1, 0)

        _, reward, terminated, _, info = env.step(np.array([0, 1, 0]))
        assert terminated
        assert info["terminal_reason"] == "level_clear"
        assert env.episode_done

        # A step after the terminal raises instead of hitting the engine
        # (the engine would answer EPISODE_DONE, docs/07 §4.4).
        with pytest.raises(RuntimeError, match="call reset"):
            env.step([0, 0, 0])
        assert len(client.step_calls) == 2

        # A reset starts the next episode.
        env.reset(seed=4)
        assert not env.episode_done
        assert env.episode_steps == 0
    finally:
        env.close()


def test_step_before_reset_is_rejected() -> None:
    env, _ = make_stub_env()
    try:
        with pytest.raises(RuntimeError, match="reset"):
            env.step([0, 0, 0])
    finally:
        env.close()


def test_invalid_actions_are_rejected_before_the_engine_sees_them() -> None:
    env, client = make_stub_env()
    try:
        env.reset(seed=1)
        for bad in ([9, 0, 0], [0, 2, 0], [0, 0, 2], [-1, 0, 0], [0, 0], "nope"):
            with pytest.raises((ValueError, TypeError)):
                env.step(bad)
        assert client.step_calls == []
    finally:
        env.close()


def test_close_is_idempotent_and_blocks_later_use() -> None:
    env, client = make_stub_env()
    env.reset(seed=1)
    env.close()
    env.close()
    assert client.closed == 1
    assert env.closed
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=1)


def test_config_validation() -> None:
    with pytest.raises(ValueError, match="mode"):
        AshCrownEnv(mode="combat", client_factory=StubClient)
    with pytest.raises(ValueError, match="merchant_policy"):
        AshCrownEnv(mode="run", merchant_policy="buy_all", client_factory=StubClient)
    with pytest.raises(ValueError, match="obs_grid"):
        AshCrownEnv(obs_grid=False, client_factory=StubClient)
    with pytest.raises(ValueError, match="frame_skip"):
        AshCrownEnv(frame_skip=0, client_factory=StubClient)
    with pytest.raises(ValueError, match="render_mode"):
        AshCrownEnv(render_mode="human", client_factory=StubClient)
    with pytest.raises(ValueError, match="level"):
        AshCrownEnv(level=11, client_factory=StubClient)
    with pytest.raises(ValueError, match="character"):
        AshCrownEnv(character="wizard", client_factory=StubClient)
    with pytest.raises(ValueError, match="reward_profile"):
        AshCrownEnv(reward_profile="imaginary", client_factory=StubClient)
    with pytest.raises(ValueError, match="obs_grid_radius"):
        AshCrownEnv(obs_grid_radius=4, client_factory=StubClient)


def test_run_mode_defaults_to_the_run_cap_and_level_1() -> None:
    """``docs/07`` §4.1/§4.2: run mode starts level 1 with a 144,000-tick cap."""
    env, client = make_stub_env(mode="run", level=5)
    try:
        assert MODES == ("level", "run")
        assert MERCHANT_POLICIES == ("none", "auto_heal")
        assert env.mode == "run"
        assert env.max_ticks == DEFAULT_RUN_MAX_TICKS
        obs, info = env.reset(seed=20260101)
        assert env.observation_space.contains(obs)
        call = client.reset_calls[-1]
        assert call["mode"] == "run"
        assert call["level"] == 1, "a run always starts at level 1"
        assert call["max_ticks"] == DEFAULT_RUN_MAX_TICKS
        assert call["merchant_policy"] is None, "the engine default policy passes through"
        assert info["mode"] == "run"
    finally:
        env.close()


def test_run_mode_forwards_the_merchant_policy() -> None:
    env, client = make_stub_env(mode="run", merchant_policy="auto_heal")
    try:
        env.reset(seed=3)
        assert client.reset_calls[-1]["merchant_policy"] == "auto_heal"
        env.reset(seed=3, options={"merchant_policy": "none"})
        assert client.reset_calls[-1]["merchant_policy"] == "none"
        with pytest.raises(ValueError, match="merchant_policy"):
            env.reset(seed=3, options={"merchant_policy": "buy_all"})
    finally:
        env.close()


def test_engine_errors_propagate() -> None:
    class FailingClient(StubClient):
        def step(self, move: int, attack: int = 0, interact: int = 0) -> StepResult:
            raise GodotError("BAD_ACTION", "engine rejected it")

    env = AshCrownEnv(client_factory=FailingClient)
    try:
        env.reset(seed=1)
        with pytest.raises(GodotError) as caught:
            env.step([0, 0, 0])
        assert caught.value.code == "BAD_ACTION"
    finally:
        env.close()


def test_render_is_a_no_op_without_a_renderer() -> None:
    env, _ = make_stub_env()
    try:
        assert env.render() is None
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# Integration tests (real headless process)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def live_env(env_kwargs: dict[str, Any]) -> AshCrownEnv:
    env = AshCrownEnv(**env_kwargs)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.integration
def test_check_env_passes_on_the_real_environment(live_env: AshCrownEnv) -> None:
    """Gymnasium's API checker against the real process (docs/07 §12.2)."""
    check_env(live_env, skip_render_check=True)


@pytest.mark.integration
def test_live_reset_and_step_shapes(live_env: AshCrownEnv) -> None:
    obs, info = live_env.reset(seed=20260101)
    assert obs.shape == (3246,)
    assert obs.dtype == np.float32
    assert live_env.observation_space.contains(obs)
    assert info["episode"] >= 1, "the engine's reset counter is reported on reset"
    assert info["layout_hash"] == live_env.layout_hash
    assert live_env.layout.layout_hash == live_env.layout_hash

    obs, reward, terminated, truncated, info = live_env.step(np.array([0, 0, 0]))
    assert live_env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert not (terminated or truncated)
    assert info["tick"] == 4
    assert info["episode_steps"] == 1
    assert "episode" not in info, "the reset counter stays out of the step info"
    assert info["episode_seed"] == 20260101
    assert info["obs_schema"] == "ashcrown-obs-v1"


@pytest.mark.integration
def test_live_episode_is_deterministic_for_a_seed(live_env: AshCrownEnv) -> None:
    """Same seed + same actions → identical state-hash sequence (docs/07 §10.1)."""
    actions = [[1, 0, 0]] * 5 + [[3, 1, 0]] * 5 + [[0, 0, 0]] * 5
    sequences: list[list[str]] = []
    for _ in range(2):
        live_env.reset(seed=20260102)
        hashes = []
        for action in actions:
            _, _, terminated, truncated, info = live_env.step(action)
            hashes.append(str(info["state_hash"]))
            if terminated or truncated:
                break
        sequences.append(hashes)
    assert sequences[0] == sequences[1]
    assert len(sequences[0]) == len(actions)


@pytest.mark.integration
def test_live_level_episode_terminates_or_truncates(env_kwargs: dict[str, Any]) -> None:
    """A short ``max_ticks`` episode truncates cleanly and only then (docs/07 §4.4)."""
    env = AshCrownEnv(max_ticks=120, **env_kwargs)
    try:
        env.reset(seed=7)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 100:
            _, _, terminated, truncated, info = env.step([0, 0, 0])
            steps += 1
        assert truncated and not terminated
        assert info["terminal_reason"] in ("", "truncated")
        assert info["tick"] == 120
    finally:
        env.close()


@pytest.mark.integration
def test_live_run_mode_reports_level_progression_metadata(env_kwargs: dict[str, Any]) -> None:
    """A short run-mode episode over the TCP bridge (``docs/07`` §4.1, §7.3).

    The full ten-level descent is exercised by the GDScript scripted clear
    harness (``tests/integration/test_rl_run_mode.gd``); this test pins the
    contract the Python client depends on: the mode is accepted and validated,
    the cap defaults to 144,000 ticks, the layout is unchanged and every step
    reports the live level as both an `info` field and the observation feature.
    """
    env = AshCrownEnv(mode="run", max_ticks=64, **env_kwargs)
    try:
        assert env.mode == "run"
        assert env.max_ticks == 64
        obs, info = env.reset(seed=20260101)
        assert info["mode"] == "run"
        assert info["level"] == 1
        assert info["merchant_policy"] == "none"
        assert env.layout.level(obs) == 1
        assert env.layout.value(obs, "global", "mode_run") >= 0.5
        assert env.layout.value(obs, "global", "mode_level") < 0.5
        levels = [info["level"]]
        terminated = truncated = False
        for _ in range(64):
            obs, _, terminated, truncated, info = env.step([0, 0, 0])
            levels.append(info["level"])
            if terminated or truncated:
                break
        assert truncated and not terminated
        assert info["terminal_reason"] == "truncated"
        assert set(levels) == {1}, "no descend happens without clearing the floor"
        assert info["mode"] == "run"
    finally:
        env.close()


@pytest.mark.integration
def test_live_run_mode_descends_over_the_bridge(env_kwargs: dict[str, Any]) -> None:
    """Clear level 1 with the heuristic and watch the bridge follow the descend.

    This is the TCP-side run-mode transition test (``docs/07`` §4.1): ``reset``
    with ``mode="run"``, step through the real action space until ``info``
    reports level 2, and check the run state carried over (kills/gold) and that
    the observation's ``level``/``mode_run`` features followed the floor.
    """
    env = AshCrownEnv(mode="run", **env_kwargs)
    try:
        policy = HeuristicPolicy(env.layout)
        obs, info = env.reset(seed=20260101)
        assert info["level"] == 1
        levels = {1}
        for _ in range(1400):
            obs, _, terminated, truncated, info = env.step(policy(obs, info, env.layout))
            assert env.observation_space.contains(obs)
            levels.add(int(info["level"]))
            if int(info["level"]) >= 2 or terminated or truncated:
                break
        assert 2 in levels, f"the bridge never reported the descend (levels={sorted(levels)})"
        assert not (terminated or truncated), "level 2 was entered, not a terminal"
        assert info["kills"] > 0, "kills carried over from level 1"
        assert env.layout.level(obs) == int(info["level"])
        assert env.layout.value(obs, "global", "mode_run") >= 0.5
        assert env.layout.value(obs, "global", "portal_active") < 0.5, "a fresh floor"
    finally:
        env.close()


@pytest.mark.integration
def test_live_run_mode_default_cap_is_144k(env_kwargs: dict[str, Any]) -> None:
    env = AshCrownEnv(mode="run", **env_kwargs)
    try:
        assert env.max_ticks == DEFAULT_RUN_MAX_TICKS
        _, info = env.reset(seed=5)
        assert info["mode"] == "run"
        assert info["level"] == 1
    finally:
        env.close()


@pytest.mark.integration
def test_live_close_twice_leaves_no_process(env_kwargs: dict[str, Any]) -> None:
    env = AshCrownEnv(**env_kwargs)
    env.reset(seed=1)
    assert env.process is not None
    env.close()
    env.close()
    assert env.closed
    with pytest.raises(RuntimeError):
        env.reset(seed=1)
    # The launcher reaped the child and removed its port file.
    assert env.process is None


@pytest.mark.integration
def test_live_bad_action_does_not_break_the_episode(live_env: AshCrownEnv) -> None:
    live_env.reset(seed=11)
    with pytest.raises(GodotError) as caught:
        # Bypass the env's own validation to reach the engine's BAD_ACTION path.
        live_env._client.step(9, 0, 0)
    assert caught.value.bad_action
    obs, _, _, _, info = live_env.step([0, 0, 0])
    assert info["tick"] == 4
    assert obs.shape == (3246,)


def test_protocol_error_type_is_exported() -> None:
    """Sanity: the env's error handling uses the protocol's exception types."""
    assert issubclass(GodotError, ProtocolError)

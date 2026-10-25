"""Observation-history (frame stack) tests (RL-7 round 4, ``docs/07`` §5.7).

The stack exists in three places at once — the Gymnasium wrapper inside the
training/collection worker, the index arithmetic used at fit time, and the
evaluator's deque — so the tests pin them against **each other**:

* :func:`ashcrown_rl.history.stack_indices` must reproduce the wrapper's window
  byte for byte, including the "reset value repeated" padding at an episode's
  first decision;
* :class:`ashcrown_rl.history.ObsStack` must produce the same ``(k, obs_len)``
  array the wrapper produces;
* ``frame_stack == 1`` must be the identity (no wrapper, raw rows);
* a checkpoint's sidecar must be refused when the requested stack length or the
  effective policy input size disagrees with it — the failure mode this whole
  mechanism exists to prevent (a 4-frame policy evaluated single-frame).

No Godot process is started: the "env" is a deterministic dummy that encodes the
step counter in every observation, so any off-by-one or cross-episode leak shows
up as a wrong value rather than as a subtle behavioural difference.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from ashcrown_rl.history import (
    DEFAULT_FRAME_STACK,
    FRAME_STACK_MAX,
    ObsStack,
    episode_marker,
    policy_obs_len,
    resolve_frame_stack,
    stack_indices,
    stack_observations,
    wrap_frame_stack,
)
from ashcrown_rl.train import MetadataMismatch, verify_metadata

OBS_LEN = 5
EPISODE_STEPS = 4


class CountingEnv(gym.Env):
    """Dummy env whose observation is ``[episode, step, 0, 0, 0]``."""

    observation_space = gym.spaces.Box(-1.0, 1.0, (OBS_LEN,), np.float32)
    action_space = gym.spaces.Discrete(2)

    def __init__(self) -> None:
        self.episode = 0
        self.step_index = 0

    def _obs(self) -> np.ndarray:
        return np.array([self.episode, self.step_index, 0, 0, 0], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del seed, options
        return self._obs(), {"episode": self.episode, "tick": 0, "episode_steps": 0}

    def step(self, action: Any):
        del action
        self.step_index += 1
        terminated = self.step_index >= EPISODE_STEPS
        if terminated:
            self.episode += 1
            self.step_index = 0
            info = {"episode": self.episode, "tick": 0, "episode_steps": 0}
            return self._obs(), 0.0, True, False, info
        info = {"episode": self.episode, "tick": self.step_index, "episode_steps": self.step_index}
        return self._obs(), 0.0, False, False, info


def _roll(k: int, episodes: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll the dummy env and return ``(stacked_rows, raw_rows, episode_ids)``."""
    env = wrap_frame_stack(CountingEnv(), k)
    stacked: list[np.ndarray] = []
    raw: list[np.ndarray] = []
    ids: list[int] = []
    obs, info = env.reset()
    while len(stacked) < episodes * EPISODE_STEPS:
        stacked.append(np.asarray(obs, dtype=np.float32).copy())
        raw.append(np.asarray(obs, dtype=np.float32)[-1].copy())
        ids.append(int(info["episode"]))
        obs, _reward, terminated, _truncated, info = env.step(0)
        if terminated:
            # SAME_STEP-style auto-reset (the vector env calls reset() itself,
            # which is what makes the stack start a fresh window).
            obs, info = env.reset()
    return np.stack(stacked), np.stack(raw), np.asarray(ids, dtype=np.int64)


def test_frame_stack_shape_is_k_times_obs_len() -> None:
    env = wrap_frame_stack(CountingEnv(), 4)
    obs, _info = env.reset()
    assert env.observation_space.shape == (4, OBS_LEN)
    assert obs.shape == (4, OBS_LEN)
    assert policy_obs_len(OBS_LEN, 4) == 20


def test_single_frame_stack_is_the_identity() -> None:
    base = CountingEnv()
    assert wrap_frame_stack(base, 1) is base
    assert policy_obs_len(OBS_LEN, 1) == OBS_LEN


def test_resolve_frame_stack_rejects_bad_values() -> None:
    assert resolve_frame_stack(None) == DEFAULT_FRAME_STACK
    assert resolve_frame_stack("4") == 4
    for bad in (0, -1, FRAME_STACK_MAX + 1, True, "four"):
        with pytest.raises(ValueError):
            resolve_frame_stack(bad)


def test_stack_indices_reproduce_the_gymnasium_wrapper() -> None:
    """The fit-time window equals the wrapper's window, padding included."""
    stacked, raw, ids = _roll(4)
    rebuilt = stack_observations(raw, stack_indices(ids, 4))
    assert rebuilt.shape == stacked.shape
    np.testing.assert_array_equal(rebuilt, stacked)
    # The first decision of every episode repeats its own frame k times: that is
    # FrameStackObservation(padding_type="reset"), and the reason a stack must
    # be rebuilt per episode rather than as one flat sliding window.
    first = 0
    np.testing.assert_array_equal(rebuilt[first], np.stack([raw[first]] * 4))
    assert not np.array_equal(rebuilt[first], np.stack([raw[first], raw[first + 1]] * 2))


def test_stack_indices_never_cross_an_episode_boundary() -> None:
    _stacked, raw, ids = _roll(3)
    indices = stack_indices(ids, 3)
    for row, episode in enumerate(ids.tolist()):
        assert set(ids[indices[row]].tolist()) == {episode}
        assert indices[row, -1] == row


def test_stack_indices_handle_empty_and_single_frame_inputs() -> None:
    assert stack_indices([], 4).shape == (0, 4)
    single = stack_indices([0, 0, 1, 1], 1)
    np.testing.assert_array_equal(single.reshape(-1), np.arange(4))


def test_obs_stack_matches_the_wrapper() -> None:
    """The evaluator's deque must produce exactly the wrapper's arrays."""
    stacked, raw, ids = _roll(4, episodes=2)
    stack = ObsStack(4)
    rebuilt = []
    for row, episode in enumerate(ids.tolist()):
        if row == 0 or ids[row - 1] != episode:
            rebuilt.append(stack.reset(raw[row]))
        else:
            rebuilt.append(stack.push(raw[row]))
    np.testing.assert_array_equal(np.stack(rebuilt), stacked)


def test_obs_stack_newest_frame_is_last() -> None:
    stack = ObsStack(3)
    first = stack.reset(np.zeros(OBS_LEN, dtype=np.float32))
    np.testing.assert_array_equal(first, np.zeros((3, OBS_LEN), dtype=np.float32))
    window = stack.push(np.ones(OBS_LEN, dtype=np.float32))
    np.testing.assert_array_equal(window[-1], np.ones(OBS_LEN, dtype=np.float32))
    np.testing.assert_array_equal(window[0], np.zeros(OBS_LEN, dtype=np.float32))


def test_episode_marker_detects_the_first_decision() -> None:
    assert episode_marker({"episode": 3, "tick": 0}) == (3, True)
    assert episode_marker({"episode": 3, "tick": 12}) == (3, False)
    assert episode_marker({}) == (None, True)


def _metadata(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocol": 1,
        "obs_schema": "ashcrown-obs-v2",
        "layout_hash": "abc",
        "obs_len": OBS_LEN,
        "frame_stack": 4,
        "policy_obs_len": policy_obs_len(OBS_LEN, 4),
        "observation_config": {"obs_grid": True},
    }
    payload.update(overrides)
    return payload


def _verify(metadata: dict[str, Any], **kwargs: Any) -> None:
    verify_metadata(
        metadata,
        protocol=1,
        obs_schema="ashcrown-obs-v2",
        layout_hash="abc",
        observation_config={"obs_grid": True},
        **kwargs,
    )


def test_verify_metadata_accepts_a_matching_stack() -> None:
    _verify(_metadata(), frame_stack=4, obs_len=OBS_LEN)


def test_verify_metadata_refuses_a_stack_mismatch() -> None:
    """A 4-frame checkpoint must be refused by a single-frame evaluator."""
    with pytest.raises(MetadataMismatch, match="frame_stack"):
        _verify(_metadata(), frame_stack=1, obs_len=OBS_LEN)


def test_verify_metadata_refuses_a_policy_input_size_mismatch() -> None:
    broken = _metadata(policy_obs_len=policy_obs_len(OBS_LEN, 2))
    with pytest.raises(MetadataMismatch, match="policy_obs_len"):
        _verify(broken, frame_stack=4, obs_len=OBS_LEN)


def test_verify_metadata_treats_a_missing_stack_key_as_single_frame() -> None:
    legacy = _metadata()
    del legacy["frame_stack"]
    del legacy["policy_obs_len"]
    _verify(legacy, frame_stack=1, obs_len=OBS_LEN)
    with pytest.raises(MetadataMismatch, match="frame_stack"):
        _verify(legacy, frame_stack=4, obs_len=OBS_LEN)


def test_verify_metadata_without_a_stack_request_still_checks_the_pins() -> None:
    _verify(_metadata())
    with pytest.raises(MetadataMismatch, match="layout_hash"):
        verify_metadata(
            _metadata(),
            protocol=1,
            obs_schema="ashcrown-obs-v2",
            layout_hash="other",
            observation_config={"obs_grid": True},
        )


def test_checkpoint_frame_stack_reads_the_sidecar(tmp_path: Path) -> None:
    from ashcrown_rl.train import checkpoint_frame_stack

    checkpoint = tmp_path / "ckpt.zip"
    checkpoint.write_bytes(b"not a real zip")
    checkpoint.with_suffix(".json").write_text(json.dumps(_metadata()), encoding="utf-8")
    assert checkpoint_frame_stack(checkpoint) == 4
    checkpoint.with_suffix(".json").write_text(json.dumps({"obs_len": OBS_LEN}), encoding="utf-8")
    assert checkpoint_frame_stack(checkpoint) == 1


def test_dataset_keeps_episode_ids_unique_across_rounds() -> None:
    from ashcrown_rl.dagger import SOURCE_DAGGER, SOURCE_DEMO, append_samples

    first = append_samples(
        None,
        np.zeros((3, OBS_LEN), dtype=np.float32),
        np.zeros((3, 3), dtype=np.int64),
        source=SOURCE_DEMO,
        episode_ids=np.array([0, 0, 0]),
    )
    merged = append_samples(
        first,
        np.zeros((2, OBS_LEN), dtype=np.float32),
        np.zeros((2, 3), dtype=np.int64),
        source=SOURCE_DAGGER,
        episode_ids=np.array([0, 1]),
    )
    assert merged.episode_ids is not None
    np.testing.assert_array_equal(merged.episode_ids, np.array([0, 0, 0, 1, 2]))
    # The stack therefore never blends two collection rounds.
    indices = stack_indices(merged.episode_ids, 2)
    assert set(merged.episode_ids[indices[3]].tolist()) == {1}

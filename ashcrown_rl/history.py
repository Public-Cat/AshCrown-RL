"""Observation history (frame stacking) for the RL-7 policies (``docs/07`` §5.7).

A single 3 254-float frame cannot carry the demonstrator's *route plan*: the
heuristic walks a ``NavMap`` sweep ring, so the same frame maps to "north" only
because of what it decided a moment earlier (RL-7 round 3 measured this: 39 % of
the expert's labels on student states are "north"/"north-east"). ``docs/07``
§5.7 allows the sanctioned fix — the policy may see the last ``k`` observations.
The engine observation and ``ashcrown-obs-v2`` are **not** touched: the stack is
built on the Python side only, so the layout hash and every protocol pin stay
valid.

One implementation, four consumers, and they must agree exactly:

* training — :func:`wrap_frame_stack` wraps **each worker env** in Gymnasium's
  :class:`~gymnasium.wrappers.FrameStackObservation` (``padding_type="reset"``),
  so the SB3 vector env publishes ``(k, obs_len)`` observations and the policy's
  flattening feed-forward net sees ``k × obs_len`` floats;
* collection — the same wrapper (the collector stores the *raw* newest frame
  plus an episode id per row and rebuilds the stack with
  :func:`stack_indices`, because storing the stack would multiply the 192 k-row
  aggregate by ``k``);
* fitting — :func:`stack_indices` reproduces the wrapper's window exactly,
  including the "reset value repeated" padding at an episode's first decision;
* evaluation — :class:`ObsStack` is a per-episode deque with the same layout,
  reset from ``info["episode"]`` (or ``tick == 0``) so a new episode never sees
  the previous one's frames.

``tests/test_history.py`` pins the equivalence: a fake env driven through
:func:`wrap_frame_stack`, :func:`stack_indices` and :class:`ObsStack` must
produce byte-identical stacks.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import FrameStackObservation

#: Default history length: the current decision plus the three before it.
DEFAULT_FRAME_STACK = 4
#: Hard cap, so ``--frame-stack 4096`` cannot silently allocate a huge buffer.
FRAME_STACK_MAX = 32


def resolve_frame_stack(value: Any) -> int:
    """Validate a ``--frame-stack`` value (``None`` → :data:`DEFAULT_FRAME_STACK`)."""
    if value is None:
        return DEFAULT_FRAME_STACK
    if isinstance(value, bool) or not isinstance(value, (int, np.integer, str)):
        raise ValueError(f"frame_stack={value!r} is not an integer")
    try:
        stack = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"frame_stack={value!r} is not an integer") from exc
    if stack < 1:
        raise ValueError(f"frame_stack={stack} must be >= 1 (1 disables stacking)")
    if stack > FRAME_STACK_MAX:
        raise ValueError(f"frame_stack={stack} exceeds the {FRAME_STACK_MAX}-frame cap")
    return stack


def policy_obs_len(obs_len: int, frame_stack: int) -> int:
    """Effective number of floats one stacked observation carries."""
    return int(obs_len) * resolve_frame_stack(frame_stack)


def wrap_frame_stack(env: gym.Env, frame_stack: int) -> gym.Env:
    """Wrap ``env`` in Gymnasium's frame stack (identity when ``frame_stack == 1``)."""
    stack = resolve_frame_stack(frame_stack)
    if stack == 1:
        return env
    return FrameStackObservation(env, stack, padding_type="reset")


def stack_indices(episode_ids: Sequence[int] | np.ndarray, frame_stack: int) -> np.ndarray:
    """Row indices of the stack window ending at each row: shape ``(N, k)``.

    Row ``i`` of episode ``e`` gets ``[i-k+1 .. i]`` clamped to ``e``'s first row,
    repeated — the same window :class:`~gymnasium.wrappers.FrameStackObservation`
    builds with ``padding_type="reset"`` (the reset observation is repeated ``k``
    times at the first decision of an episode). Rows must be in collection order;
    that is how :func:`~ashcrown_rl.dagger.save_dataset` stores them.
    """
    stack = resolve_frame_stack(frame_stack)
    ids = np.asarray(episode_ids, dtype=np.int64).reshape(-1)
    count = int(ids.size)
    if count == 0:
        return np.zeros((0, stack), dtype=np.int64)
    # First row of every episode (`starts[i]` = the row's own episode start).
    starts = np.zeros(count, dtype=np.int64)
    last_start = 0
    for index in range(1, count):
        if ids[index] != ids[index - 1]:
            last_start = index
        starts[index] = last_start
    rows = np.arange(count, dtype=np.int64)[:, None] - np.arange(stack - 1, -1, -1)[None, :]
    return np.maximum(rows, starts[:, None])


def stack_observations(obs: np.ndarray, indices: np.ndarray | None = None) -> np.ndarray:
    """Gather ``obs`` into stacked observations using :func:`stack_indices` output."""
    values = np.asarray(obs, dtype=np.float32)
    if indices is None:
        return values
    return np.asarray(values[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


class ObsStack:
    """A single-environment frame stack for the closed-loop evaluator.

    ``push(obs)`` returns ``(k, obs_len)`` — oldest frame first, newest last,
    matching the wrapper the policy was trained with. ``reset(obs)`` seeds the
    window with ``k`` copies of ``obs``.
    """

    def __init__(self, frame_stack: int = DEFAULT_FRAME_STACK) -> None:
        """Create an empty stack of ``frame_stack`` frames."""
        self.stack_size = resolve_frame_stack(frame_stack)
        self._queue: deque[np.ndarray] = deque(maxlen=self.stack_size)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        """Start a new episode from ``obs`` and return its stacked observation."""
        frame = np.asarray(obs, dtype=np.float32).reshape(-1)
        self._queue.clear()
        for _ in range(self.stack_size):
            self._queue.append(frame)
        return self.current()

    def push(self, obs: np.ndarray) -> np.ndarray:
        """Append ``obs`` and return the stacked window."""
        self._queue.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        while len(self._queue) < self.stack_size:  # pragma: no cover - after reset()
            self._queue.appendleft(np.zeros_like(self._queue[-1]))
        return self.current()

    def current(self) -> np.ndarray:
        """The current stacked window as ``(k, obs_len)`` (zeros when empty)."""
        if not self._queue:  # pragma: no cover - policy always resets first
            raise RuntimeError("ObsStack.current() before reset()")
        return np.stack(list(self._queue), axis=0)


def episode_marker(info: Mapping[str, Any]) -> tuple[Any, bool]:
    """``(episode id, is first decision)`` for one engine ``info`` block."""
    episode = info.get("episode")
    first = int(info.get("tick", 0) or 0) == 0 or int(info.get("episode_steps", 1) or 0) == 0
    return (int(episode) if episode is not None else None, first)


__all__ = [
    "DEFAULT_FRAME_STACK",
    "FRAME_STACK_MAX",
    "ObsStack",
    "episode_marker",
    "policy_obs_len",
    "resolve_frame_stack",
    "stack_indices",
    "stack_observations",
    "wrap_frame_stack",
]

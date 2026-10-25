"""Fixed-seed evaluation harness (RL-4, ``docs/07`` §9.4 / §12.2).

Runs a policy over a fixed seed list on one real environment process, records
per-episode metrics and aggregates, and writes the JSON artifact that the
``docs/artifacts/RL/baseline_level1.json`` gate names::

    uv run python -m ashcrown_rl.evaluate \\
        --policy heuristic --level 1 --seed-start 1 --seed-count 10 \\
        --out ../docs/artifacts/RL/baseline_level1.json

Each seed is run ``--repeat`` times (default 2) and the per-step state-hash
sequences must match across repeats — that is the determinism proof for the
slice (``docs/07`` §10.1/§10.4).

Episode metrics: ``level``, ``terminated``, ``truncated``, ``terminal_reason``,
``steps``, ``game_seconds``, ``return`` and the summed ``components`` (the
engine's reward components). Aggregates: success rate, mean time, damage taken,
stuck rate, terminal-reason counts and the protocol/obs-schema/layout pins.

Combat instrumentation (RL-4 round 3): every episode also records the engine's
``events.attacks_started`` / ``events.hits_landed`` counters and the number of
decision windows in which the policy *requested* an attack, so the two ratios
that were previously conflated are both reported:

* ``hits_per_swing``  = ``hits_landed / attacks_started`` (the true land rate; a
  started swing is one committed animation, ``attack_count``);
* ``hits_per_request`` = ``hits_landed / attack_requests`` (diluted by every
  held/queued attack request that could not start a new swing).

``attack_requests`` is several times ``attacks_started`` whenever the policy
holds the attack button: a swing locks the body for ~0.7 s ≈ 10 decision
windows at ``frame_skip=4`` while each window produces one request, so a
request-based ratio under-reports the land rate by design.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np

from .baseline import DEFAULT_CONFIG as HEURISTIC_CONFIG
from .baseline import HeuristicPolicy, heuristic_action
from .env import AshCrownEnv
from .history import DEFAULT_FRAME_STACK, ObsStack, episode_marker, resolve_frame_stack
from .protocol import ProtocolError
from .spaces import Layout

#: Window over which the ``stuck`` diagnostic is measured (game seconds).
STUCK_WINDOW_S = 2.0
#: Movement below this many pixels inside one window counts as "not moving".
STUCK_MIN_TRAVEL_PX = 8.0

Policy = Callable[[np.ndarray, Mapping[str, Any], Layout], Sequence[int]]


def heuristic_policy(obs: np.ndarray, info: Mapping[str, Any], layout: Layout) -> np.ndarray:
    """The scripted baseline as an evaluation policy."""
    return heuristic_action(obs, info, layout, HEURISTIC_CONFIG)


def zero_policy(obs: np.ndarray, info: Mapping[str, Any], layout: Layout) -> list[int]:
    """Stand still and do nothing (a floor for the reward scale)."""
    return [0, 0, 0]


def random_policy_factory(seed: int) -> Policy:
    """Deterministic uniform-random policy (seeded outside the env)."""
    rng = np.random.default_rng(int(seed))

    def _policy(obs: np.ndarray, info: Mapping[str, Any], layout: Layout) -> list[int]:
        return [int(rng.integers(0, 9)), int(rng.integers(0, 2)), int(rng.integers(0, 2))]

    return _policy


POLICIES: Mapping[str, Policy] = {
    "zero": zero_policy,
    "heuristic_core": heuristic_policy,
}


def ppo_model_policy(model: Any, frame_stack: int = 1) -> Policy:
    """Wrap a loaded Stable-Baselines3 model as an evaluation policy.

    ``frame_stack > 1`` reproduces the training observation exactly: a
    per-episode :class:`~ashcrown_rl.history.ObsStack` whose window is reset from
    ``info`` (``episode`` id / ``tick == 0``) before the first prediction of an
    episode, so no frame leaks across episodes — the same "reset value repeated"
    padding Gymnasium's ``FrameStackObservation`` applies.
    """
    stack_size = resolve_frame_stack(frame_stack)
    state: dict[str, Any] = {
        "marker": None,
        "stack": ObsStack(stack_size) if stack_size > 1 else None,
    }

    def _policy(obs: np.ndarray, info: Mapping[str, Any], layout: Layout) -> list[int]:
        stack = state["stack"]
        if stack is not None:
            marker = episode_marker(info)
            if state["marker"] is None or marker[1] or marker[0] != state["marker"]:
                stacked = stack.reset(obs)
                state["marker"] = marker
            else:
                stacked = stack.push(obs)
            action, _state = model.predict(stacked, deterministic=True)
        else:
            action, _state = model.predict(obs, deterministic=True)
        return [int(value) for value in action]

    return _policy


def resolve_policy(
    name: str,
    seed: int,
    layout: Layout | None = None,
    *,
    checkpoint: str | None = None,
    env: AshCrownEnv | None = None,
    model: Any = None,
    frame_stack: int | None = None,
) -> Policy:
    """Policy callable for ``--policy``.

    ``heuristic`` (the default) is :class:`~ashcrown_rl.baseline.HeuristicPolicy`
    — the pure core plus the anti-stuck guard — and needs the layout, which
    :func:`run_episode` supplies on the first call. ``random`` is seeded per
    episode; ``zero`` stands still. ``ppo`` needs a trained checkpoint
    (``--checkpoint``) whose metadata is verified against ``env``.
    """
    if name == "ppo":
        trained = model
        if checkpoint and frame_stack is None:
            # Default to the stack the checkpoint was trained with: a stacked
            # policy evaluated single-frame is the failure mode the metadata
            # check exists to prevent.
            from .train import checkpoint_frame_stack  # noqa: PLC0415

            frame_stack = checkpoint_frame_stack(checkpoint)
        if trained is None:
            if not checkpoint:
                raise ValueError("--policy ppo needs --checkpoint <model.zip>")
            from .train import load_policy  # noqa: PLC0415 - keeps numpy-only imports light

            trained = load_policy(checkpoint, env, frame_stack=frame_stack)
        return ppo_model_policy(trained, resolve_frame_stack(frame_stack or 1))
    if name == "heuristic":
        if layout is None:
            raise ValueError("the heuristic policy needs the env layout")
        return HeuristicPolicy(layout)
    if name in POLICIES:
        return POLICIES[name]
    if name == "random":
        return random_policy_factory(seed)
    raise ValueError(
        f"unknown policy {name!r}; choose from {sorted([*POLICIES, 'heuristic', 'random'])}"
    )


# --------------------------------------------------------------------------- #
# Episode loop
# --------------------------------------------------------------------------- #
@dataclass
class EpisodeMetrics:
    """One measured episode."""

    seed: int
    level: int
    terminated: bool = False
    truncated: bool = False
    terminal_reason: str = ""
    steps: int = 0
    ticks: int = 0
    game_seconds: float = 0.0
    episode_return: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    damage_taken: float = 0.0
    damage_dealt: float = 0.0
    kills: int = 0
    player_hp: float = 0.0
    attacks_started: int = 0
    hits_landed: int = 0
    attack_requests: int = 0
    state_hash: str = ""
    hashes: list[str] = field(default_factory=list)
    stuck_windows: int = 0
    windows: int = 0
    policy: str = ""

    @property
    def cleared(self) -> bool:
        """True when the episode ended on the level-clear terminal."""
        return self.terminated and self.terminal_reason in ("level_clear", "portal_entered")

    @property
    def stuck(self) -> bool:
        """True when at least one 2 s window made no progress."""
        return self.stuck_windows > 0

    @property
    def hits_per_swing(self) -> float:
        """Landed hits per committed swing (``0`` when no swing started)."""
        return self.hits_landed / self.attacks_started if self.attacks_started else 0.0

    @property
    def hits_per_request(self) -> float:
        """Landed hits per attack *request* window (``0`` when none)."""
        return self.hits_landed / self.attack_requests if self.attack_requests else 0.0

    def as_dict(self, *, include_hashes: bool = False) -> dict[str, Any]:
        """JSON-ready episode record."""
        payload: dict[str, Any] = {
            "seed": self.seed,
            "level": self.level,
            "policy": self.policy,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "terminal_reason": self.terminal_reason,
            "cleared": self.cleared,
            "steps": self.steps,
            "ticks": self.ticks,
            "game_seconds": round(self.game_seconds, 3),
            "return": round(self.episode_return, 6),
            "components": {key: round(value, 6) for key, value in self.components.items()},
            "damage_taken": round(self.damage_taken, 3),
            "damage_dealt": round(self.damage_dealt, 3),
            "kills": self.kills,
            "player_hp": round(self.player_hp, 3),
            "attacks_started": self.attacks_started,
            "hits_landed": self.hits_landed,
            "attack_requests": self.attack_requests,
            "hits_per_swing": round(self.hits_per_swing, 4),
            "hits_per_request": round(self.hits_per_request, 4),
            "stuck_windows": self.stuck_windows,
            "windows": self.windows,
            "state_hash": self.state_hash,
        }
        if include_hashes:
            payload["hashes"] = list(self.hashes)
        return payload


def run_episode(
    env: AshCrownEnv,
    *,
    seed: int,
    policy: Policy,
    policy_name: str = "",
    max_steps: int | None = None,
    level: int | None = None,
    options: Mapping[str, Any] | None = None,
    record_hashes: bool = True,
) -> EpisodeMetrics:
    """Drive one episode and return its metrics (no auto-reset)."""
    reset_options = dict(options or {})
    if level is not None:
        reset_options.setdefault("level", int(level))
    obs, info = env.reset(seed=int(seed), options=reset_options)
    layout = env.layout
    metrics = EpisodeMetrics(
        seed=int(seed),
        level=int(info.get("level", reset_options.get("level", 1))),
        policy=policy_name,
    )
    components: dict[str, float] = {}
    travel = 0.0
    window_start = 0.0
    window_damage = 0.0
    frame_skip = env.frame_skip
    while True:
        action = policy(obs, info, layout)
        if int(action[1]) == 1:
            metrics.attack_requests += 1
        obs, reward, terminated, truncated, info = env.step(action)
        metrics.steps += 1
        metrics.episode_return += float(reward)
        metrics.ticks = int(info.get("tick", metrics.ticks))
        metrics.game_seconds = float(info.get("game_seconds", 0.0))
        for key, value in (info.get("reward_components") or {}).items():
            components[str(key)] = components.get(str(key), 0.0) + float(value)
        events = info.get("events") or {}
        metrics.damage_taken += float(events.get("damage_taken", 0.0))
        metrics.damage_dealt += float(events.get("damage_dealt", 0.0))
        metrics.attacks_started += int(events.get("attacks_started", 0))
        metrics.hits_landed += int(events.get("hits_landed", 0))
        window_damage += float(events.get("damage_dealt", 0.0)) + float(
            events.get("damage_taken", 0.0)
        )
        player = layout.player_features(obs)
        travel += float(np.hypot(*player.velocity_px)) * frame_skip / 60.0
        if record_hashes:
            metrics.hashes.append(str(info.get("state_hash", "")))
        if metrics.game_seconds - window_start >= STUCK_WINDOW_S:
            metrics.windows += 1
            if "stuck" in info:
                # The engine's own 2 s window (docs/07 §5.7).
                metrics.stuck_windows += 1 if bool(info["stuck"]) else 0
            elif travel < STUCK_MIN_TRAVEL_PX and window_damage <= 0.0:
                metrics.stuck_windows += 1
            travel = 0.0
            window_damage = 0.0
            window_start = metrics.game_seconds
        if terminated or truncated:
            metrics.terminated = bool(terminated)
            metrics.truncated = bool(truncated)
            metrics.terminal_reason = str(
                info.get("terminal_reason")
                or ("truncated" if truncated else ("victory" if terminated else ""))
            )
            metrics.kills = int(info.get("kills", 0))
            metrics.player_hp = float(info.get("hp", 0.0))
            metrics.state_hash = str(info.get("state_hash", ""))
            break
        if max_steps is not None and metrics.steps >= int(max_steps):
            metrics.truncated = True
            metrics.terminal_reason = "max_steps"
            metrics.state_hash = str(info.get("state_hash", ""))
            break
    metrics.components = components
    return metrics


# --------------------------------------------------------------------------- #
# Fixed-seed evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    *,
    seeds: Sequence[int],
    policy_name: str = "heuristic",
    env_kwargs: Mapping[str, Any] | None = None,
    level: int | None = None,
    options: Mapping[str, Any] | None = None,
    repeat: int = 2,
    max_steps: int | None = None,
    with_hashes: bool = False,
    csv_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate ``policy_name`` on every seed ``repeat`` times, twice-checked."""
    started_wall = datetime.now(UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    env = AshCrownEnv(**dict(env_kwargs or {}))
    try:
        runs: list[list[EpisodeMetrics]] = []
        for index in range(max(int(repeat), 1)):
            run: list[EpisodeMetrics] = []
            for seed in seeds:
                policy = resolve_policy(policy_name, int(seed) + index, env.layout)
                run.append(
                    run_episode(
                        env,
                        seed=int(seed),
                        policy=policy,
                        policy_name=policy_name,
                        level=level,
                        options=options,
                        max_steps=max_steps,
                        record_hashes=True,
                    )
                )
            runs.append(run)
        pins = {
            "protocol": env.hello.get("protocol"),
            "obs_schema": env.hello.get("obs_schema"),
            "action_schema": env.hello.get("action_schema"),
            "layout_hash": env.layout_hash,
            "obs_len": env.obs_len,
            "observation_config": env.observation_config(),
            "physics_ticks_per_second": env.hello.get("physics_ticks_per_second"),
            "frame_skip": env.frame_skip,
        }
        config = {
            "level": level if level is not None else env.level,
            "character": env.character,
            "mode": env.mode,
            "frame_skip": env.frame_skip,
            "max_ticks": env.max_ticks,
            "reward_profile": env.reward_profile,
            **env.observation_config(),
            "policy": policy_name,
            "seeds": [int(seed) for seed in seeds],
            "repeat": int(repeat),
        }
        env_summary = _episode_summary(runs[0])
        determinism = _determinism(runs)
        if csv_path is not None:
            write_csv(csv_path, runs[0])
    finally:
        env.close()
    elapsed = time.perf_counter() - clock
    return {
        "generated": started_wall,
        "elapsed_seconds": round(elapsed, 3),
        "pins": pins,
        "config": config,
        "episodes": [metrics.as_dict(include_hashes=with_hashes) for metrics in runs[0]],
        "aggregates": env_summary,
        "determinism": determinism,
    }


def _determinism(runs: Sequence[Sequence[EpisodeMetrics]]) -> dict[str, Any]:
    """Compare the per-step hash sequences of every repeat."""
    if len(runs) <= 1:
        return {"repeats": len(runs), "identical": True, "mismatched_seeds": []}
    reference = runs[0]
    mismatched: list[int] = []
    for index in range(1, len(runs)):
        for base, other in zip(reference, runs[index], strict=True):
            if base.hashes != other.hashes:
                mismatched.append(int(base.seed))
    return {
        "repeats": len(runs),
        "identical": not mismatched,
        "mismatched_seeds": sorted(set(mismatched)),
        "compared_hashes": sum(len(metrics.hashes) for metrics in reference),
    }


def _episode_summary(episodes: Sequence[EpisodeMetrics]) -> dict[str, Any]:
    """Aggregates over the measured episodes."""
    count = len(episodes) or 1
    clears = sum(1 for metrics in episodes if metrics.cleared)
    reasons: dict[str, int] = {}
    for metrics in episodes:
        reason = metrics.terminal_reason or "none"
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "episodes": len(episodes),
        "clears": clears,
        "success_rate": round(clears / count, 4),
        "mean_game_seconds": round(sum(metrics.game_seconds for metrics in episodes) / count, 3),
        "mean_steps": round(sum(metrics.steps for metrics in episodes) / count, 2),
        "mean_return": round(sum(metrics.episode_return for metrics in episodes) / count, 4),
        "mean_damage_taken": round(sum(metrics.damage_taken for metrics in episodes) / count, 3),
        "mean_damage_dealt": round(sum(metrics.damage_dealt for metrics in episodes) / count, 3),
        "mean_kills": round(sum(metrics.kills for metrics in episodes) / count, 2),
        "stuck_rate": round(sum(1 for metrics in episodes if metrics.stuck) / count, 4),
        "stuck_windows": sum(metrics.stuck_windows for metrics in episodes),
        "terminal_reasons": dict(sorted(reasons.items())),
    }


# --------------------------------------------------------------------------- #
# Pins, config and the parallel PPO path
# --------------------------------------------------------------------------- #
def _pins(env: AshCrownEnv) -> dict[str, Any]:
    """The environment pins recorded next to an evaluation result."""
    return {
        "protocol": env.hello.get("protocol"),
        "obs_schema": env.hello.get("obs_schema") or env.layout.obs_schema,
        "action_schema": env.hello.get("action_schema"),
        "layout_hash": env.layout_hash,
        "obs_len": env.obs_len,
        "observation_config": env.observation_config(),
        "physics_ticks_per_second": env.hello.get("physics_ticks_per_second"),
        "frame_skip": env.frame_skip,
    }


def _config(
    env: AshCrownEnv,
    *,
    policy_name: str,
    seeds: Sequence[int],
    repeat: int,
    level: int | None,
    checkpoint: str | None = None,
) -> dict[str, Any]:
    """The run configuration recorded next to an evaluation result."""
    config = {
        "level": level if level is not None else env.level,
        "character": env.character,
        "mode": env.mode,
        "frame_skip": env.frame_skip,
        "max_ticks": env.max_ticks,
        "reward_profile": env.reward_profile,
        **env.observation_config(),
        "policy": policy_name,
        "seeds": [int(seed) for seed in seeds],
        "repeat": int(repeat),
    }
    if checkpoint:
        config["checkpoint"] = str(checkpoint)
    return config


#: Seconds a worker may spend on ONE seed before the watchdog kills the env.
DEFAULT_SEED_TIMEOUT = 180.0
#: Seconds a worker may spend starting its Godot process and loading the model.
DEFAULT_STARTUP_TIMEOUT = 120.0


class EvaluationTimeout(RuntimeError):
    """A worker did not finish its chunk inside the wall-clock deadline."""


def _ppo_chunk(payload: Mapping[str, Any], out: Any) -> None:  # pragma: no cover - worker
    """Evaluate one seed chunk of a PPO checkpoint, streaming one message per seed.

    Each worker owns one Godot process and loads the model once. Three layers
    keep a wedged environment from hanging the run (the RL-7 crash: the parent
    slept forever with seven idle Godot children):

    1. a per-seed :class:`threading.Timer` closes the env, which kills the
       Godot process and unblocks the socket read with a protocol error;
    2. an env that raised is discarded and rebuilt for the next seed, so one
       bad episode costs one seed and not the whole chunk;
    3. every seed result is put on ``out`` immediately, so the parent keeps the
       seeds that already finished if the worker itself has to be killed.
    """
    chunk = int(payload.get("chunk", 0))
    per_seed = float(payload.get("seed_timeout") or 0.0)
    # Torch's default thread pool is sized for one process, not for four
    # workers on a 32-core host: without this the workers thrash each other and
    # a seed that takes 4 s alone needs >100 s (measured, RL-7).
    try:
        import torch  # noqa: PLC0415 - optional, training extra only
    except ImportError:  # pragma: no cover - numpy-only install
        pass
    else:
        torch.set_num_threads(1)
    env: AshCrownEnv | None = None
    policy: Policy | None = None
    try:
        for seed in payload["seeds"]:
            started = time.perf_counter()
            watchdog: threading.Timer | None = None
            try:
                if env is None or policy is None:
                    env = AshCrownEnv(**dict(payload["env_kwargs"]))
                    policy = resolve_policy(
                        "ppo",
                        0,
                        env.layout,
                        checkpoint=str(payload["checkpoint"]),
                        env=env,
                        frame_stack=int(payload.get("frame_stack") or DEFAULT_FRAME_STACK),
                    )
                if per_seed > 0.0:
                    watchdog = threading.Timer(per_seed, env.close)
                    watchdog.daemon = True
                    watchdog.start()
                metrics = run_episode(
                    env,
                    seed=int(seed),
                    policy=policy,
                    policy_name="ppo",
                    level=payload.get("level"),
                    options=payload.get("options"),
                    max_steps=payload.get("max_steps"),
                    record_hashes=False,
                )
                message: dict[str, Any] = {
                    "kind": "seed",
                    "chunk": chunk,
                    "seed": int(seed),
                    "metrics": metrics,
                    "wall_seconds": round(time.perf_counter() - started, 3),
                    "pins": _pins(env),
                    "config": _config(
                        env,
                        policy_name="ppo",
                        seeds=payload["seeds"],
                        repeat=1,
                        level=payload.get("level"),
                        checkpoint=str(payload["checkpoint"]),
                    ),
                }
                out.put(message)
            except (ProtocolError, OSError, RuntimeError, ValueError) as exc:
                timed_out = env is not None and env.closed
                out.put(
                    {
                        "kind": "timeout" if timed_out else "error",
                        "chunk": chunk,
                        "seed": int(seed),
                        "error": f"{type(exc).__name__}: {exc}",
                        "wall_seconds": round(time.perf_counter() - started, 3),
                    }
                )
                if env is not None:
                    env.close()
                env = None
                policy = None
            finally:
                if watchdog is not None:
                    watchdog.cancel()
    finally:
        if env is not None:
            env.close()
        out.put({"kind": "done", "chunk": chunk})


def _children(pid: int) -> list[int]:
    """Live PIDs whose parent is ``pid`` (Linux ``/proc`` scan; ``[]`` elsewhere)."""
    found: list[int] = []
    try:
        entries = os.listdir("/proc")
    except OSError:  # pragma: no cover - non-Linux hosts
        return found
    for entry in entries:
        if not entry.isdigit():
            continue
        try:
            stat = Path(f"/proc/{entry}/stat").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fields = stat[stat.rfind(")") + 2 :].split()
        if len(fields) > 1 and fields[1].isdigit() and int(fields[1]) == int(pid):
            found.append(int(entry))
    return found


def _kill(pid: int) -> None:
    """Best-effort ``SIGKILL`` of ``pid`` (already-dead pids are ignored)."""
    try:
        os.kill(int(pid), 9)
    except (OSError, ProcessLookupError):
        return


def _terminate_worker(process: Any) -> list[int]:
    """Kill a stuck worker **and its Godot children**, then reap it.

    A worker that is killed outright leaves its Godot child orphaned (reparented
    to init), which is exactly the stray-process pile-up the RL-7 crash left
    behind, so the children are killed first, while the worker still owns them.
    """
    killed = _children(process.pid)
    for child in killed:
        _kill(child)
    process.terminate()
    process.join(10.0)
    if process.is_alive():  # pragma: no cover - SIGTERM ignored
        for child in _children(process.pid):
            _kill(child)
        process.kill()
        process.join(5.0)
    return killed


def _run_worker_pool(
    payloads: Sequence[Mapping[str, Any]],
    *,
    workers: int,
    startup_timeout: float,
    seed_timeout: float,
    on_message: Callable[[dict[str, Any]], None],
) -> list[dict[str, Any]]:
    """Run ``payloads`` over at most ``workers`` processes with hard deadlines.

    The parent never blocks on a worker without a deadline: it polls the shared
    queue, so a seed result is reported the moment it arrives, and a worker that
    blows ``startup_timeout + seed_timeout × len(chunk)`` is killed together
    with its Godot children. Seeds that never reported come back as ``timeout``
    records instead of vanishing.
    """
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    pending = deque(payloads)
    running: dict[int, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < max(int(workers), 1):
            payload = pending.popleft()
            chunk = int(payload.get("chunk", 0))
            process = context.Process(target=_ppo_chunk, args=(payload, queue), daemon=True)
            process.start()
            running[chunk] = {
                "process": process,
                "payload": payload,
                "reported": set(),
                "deadline": time.monotonic()
                + float(startup_timeout)
                + float(seed_timeout) * len(payload["seeds"]),
            }
        try:
            message = queue.get(timeout=0.25)
        except Empty:
            message = None
        if message is not None:
            chunk = int(message.get("chunk", -1))
            entry = running.get(chunk)
            if entry is not None:
                if message.get("kind") == "done":
                    entry["process"].join(10.0)
                    running.pop(chunk, None)
                else:
                    entry["reported"].add(int(message.get("seed", -1)))
            if message.get("kind") in ("timeout", "error"):
                failures.append(dict(message))
            on_message(dict(message))
        for chunk, entry in list(running.items()):
            if time.monotonic() < entry["deadline"]:
                continue
            killed = _terminate_worker(entry["process"])
            missing = [
                int(seed)
                for seed in entry["payload"]["seeds"]
                if int(seed) not in entry["reported"]
            ]
            for seed in missing:
                failures.append(
                    {
                        "kind": "timeout",
                        "chunk": chunk,
                        "seed": seed,
                        "error": f"worker killed after {entry['deadline'] - 0:.0f}s deadline",
                        "godot_children_killed": len(killed),
                    }
                )
            running.pop(chunk, None)
    return failures


def split_seeds(seeds: Sequence[int], workers: int) -> list[list[int]]:
    """Contiguous, balanced chunks of ``seeds`` (order preserved on merge)."""
    count = max(int(workers), 1)
    total = len(seeds)
    if total == 0:
        return []
    size = max(1, (total + count - 1) // count)
    return [list(seeds[index : index + size]) for index in range(0, total, size)]


def evaluate_ppo(
    *,
    checkpoint: str,
    seeds: Sequence[int],
    env_kwargs: Mapping[str, Any] | None = None,
    level: int | None = None,
    options: Mapping[str, Any] | None = None,
    max_steps: int | None = None,
    workers: int = 1,
    csv_path: str | Path | None = None,
    out_path: str | Path | None = None,
    seed_timeout: float = DEFAULT_SEED_TIMEOUT,
    startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
    frame_stack: int | None = None,
    progress: bool = True,
) -> dict[str, Any]:
    """Fixed-seed PPO evaluation, optionally over ``workers`` processes.

    ``out_path`` is rewritten after **every** seed, so an interrupted run still
    leaves a usable artifact (marked ``"partial": true`` and listing the seeds
    that have not finished yet). ``seed_timeout`` bounds one seed and
    ``startup_timeout`` bounds process start plus model load; a worker that
    exceeds them is killed with its Godot children and its seeds are reported as
    ``timeout`` failures instead of hanging the run.
    """
    started_wall = datetime.now(UTC).isoformat(timespec="seconds")
    clock = time.perf_counter()
    seeds = [int(seed) for seed in seeds]
    if frame_stack is None:
        from .train import checkpoint_frame_stack  # noqa: PLC0415 - numpy-only path

        frame_stack = checkpoint_frame_stack(checkpoint)
    stack_size = resolve_frame_stack(frame_stack)
    failures: list[dict[str, Any]] = []
    if int(workers) <= 1:
        env = AshCrownEnv(**dict(env_kwargs or {}))
        try:
            policy = resolve_policy(
                "ppo", 0, env.layout, checkpoint=checkpoint, env=env, frame_stack=stack_size
            )
            episodes = [
                run_episode(
                    env,
                    seed=seed,
                    policy=policy,
                    policy_name="ppo",
                    level=level,
                    options=options,
                    max_steps=max_steps,
                    record_hashes=False,
                )
                for seed in seeds
            ]
            pins = _pins(env)
            config = _config(
                env,
                policy_name="ppo",
                seeds=seeds,
                repeat=1,
                level=level,
                checkpoint=str(checkpoint),
            )
        finally:
            env.close()
    else:
        chunks = split_seeds(seeds, int(workers))
        payloads = [
            {
                "chunk": index,
                "checkpoint": str(checkpoint),
                "env_kwargs": dict(env_kwargs or {}),
                "seeds": chunk,
                "level": level,
                "options": dict(options or {}),
                "max_steps": max_steps,
                "seed_timeout": float(seed_timeout),
                "frame_stack": stack_size,
            }
            for index, chunk in enumerate(chunks)
        ]
        episodes = []
        pins: dict[str, Any] = {}
        config: dict[str, Any] = {}

        def _on_message(message: dict[str, Any]) -> None:
            nonlocal pins, config
            if message.get("kind") == "seed":
                metrics = message["metrics"]
                episodes.append(metrics)
                pins = pins or dict(message.get("pins") or {})
                config = config or dict(message.get("config") or {})
                if progress:
                    print("EVAL_SEED " + json.dumps(_seed_line(metrics.as_dict())), flush=True)
                    _write_partial(
                        out_path,
                        csv_path,
                        started_wall=started_wall,
                        seeds=seeds,
                        episodes=episodes,
                        pins=pins,
                        config=config,
                        failures=failures,
                        elapsed=time.perf_counter() - clock,
                        workers=int(workers),
                    )
            elif progress and message.get("kind") in ("timeout", "error"):
                print(
                    "EVAL_WARN " + json.dumps({k: v for k, v in message.items() if k != "metrics"}),
                    flush=True,
                )

        failures = _run_worker_pool(
            payloads,
            workers=int(workers),
            startup_timeout=float(startup_timeout),
            seed_timeout=float(seed_timeout),
            on_message=_on_message,
        )
        episodes.sort(key=lambda metrics: metrics.seed)
        if config:
            config["seeds"] = list(seeds)
    elapsed = time.perf_counter() - clock
    summary = _episode_summary(episodes)
    if csv_path is not None:
        write_csv(csv_path, episodes)
    report = {
        "generated": started_wall,
        "elapsed_seconds": round(elapsed, 3),
        "partial": False,
        "pins": pins,
        "config": config,
        "episodes": [metrics.as_dict() for metrics in episodes],
        "aggregates": summary,
        "failures": list(failures),
        "timed_out_seeds": sorted(
            int(failure["seed"]) for failure in failures if failure.get("kind") == "timeout"
        ),
        "determinism": {
            "repeats": 1,
            "identical": True,
            "mismatched_seeds": [],
            "compared_hashes": 0,
            "note": "determinism check runs on --repeat > 1 (sequential path)",
        },
        "workers": int(workers),
    }
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return report


def _write_partial(
    out_path: str | Path | None,
    csv_path: str | Path | None,
    *,
    started_wall: str,
    seeds: Sequence[int],
    episodes: Sequence[EpisodeMetrics],
    pins: Mapping[str, Any],
    config: Mapping[str, Any],
    failures: Sequence[Mapping[str, Any]],
    elapsed: float,
    workers: int,
) -> None:
    """Rewrite the JSON/CSV artifacts with the seeds measured so far."""
    if csv_path is not None:
        write_csv(csv_path, episodes)
    if out_path is None:
        return
    done = {int(metrics.seed) for metrics in episodes}
    report = {
        "generated": started_wall,
        "elapsed_seconds": round(float(elapsed), 3),
        "partial": True,
        "seeds_pending": [int(seed) for seed in seeds if int(seed) not in done],
        "pins": dict(pins),
        "config": dict(config),
        "episodes": [metrics.as_dict() for metrics in episodes],
        "aggregates": _episode_summary(episodes),
        "failures": list(failures),
        "workers": int(workers),
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _seed_line(episode: Mapping[str, Any]) -> dict[str, Any]:
    """The one-line per-seed stdout record both evaluation paths print."""
    return {
        "seed": episode["seed"],
        "reason": episode["terminal_reason"],
        "decisions": episode["steps"],
        "seconds": episode["game_seconds"],
        "swings": episode["attacks_started"],
        "hits": episode["hits_landed"],
        "requests": episode["attack_requests"],
        "hits_per_swing": episode["hits_per_swing"],
        "hits_per_request": episode["hits_per_request"],
        "dmg_dealt": episode["damage_dealt"],
        "dmg_taken": episode["damage_taken"],
        "kills": episode["kills"],
        "hp": episode["player_hp"],
    }


# --------------------------------------------------------------------------- #
# CSV artifact
# --------------------------------------------------------------------------- #
#: Column order of the per-episode combat CSV (``--csv``).
CSV_COLUMNS: tuple[str, ...] = (
    "seed",
    "policy",
    "level",
    "terminal_reason",
    "cleared",
    "decisions",
    "game_seconds",
    "attacks_started",
    "hits_landed",
    "attack_requests",
    "hits_per_swing",
    "hits_per_request",
    "damage_dealt",
    "damage_taken",
    "kills",
    "hp_left",
    "return",
    "stuck_windows",
    "windows",
)


def write_csv(path: str | Path, episodes: Sequence[EpisodeMetrics]) -> Path:
    """Write the per-episode combat table (one row per episode)."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(CSV_COLUMNS)]
    for metrics in episodes:
        row = {
            "seed": metrics.seed,
            "policy": metrics.policy,
            "level": metrics.level,
            "terminal_reason": metrics.terminal_reason,
            "cleared": int(metrics.cleared),
            "decisions": metrics.steps,
            "game_seconds": f"{metrics.game_seconds:.3f}",
            "attacks_started": metrics.attacks_started,
            "hits_landed": metrics.hits_landed,
            "attack_requests": metrics.attack_requests,
            "hits_per_swing": f"{metrics.hits_per_swing:.4f}",
            "hits_per_request": f"{metrics.hits_per_request:.4f}",
            "damage_dealt": f"{metrics.damage_dealt:.1f}",
            "damage_taken": f"{metrics.damage_taken:.1f}",
            "kills": metrics.kills,
            "hp_left": f"{metrics.player_hp:.1f}",
            "return": f"{metrics.episode_return:.3f}",
            "stuck_windows": metrics.stuck_windows,
            "windows": metrics.windows,
        }
        lines.append(",".join(str(row[column]) for column in CSV_COLUMNS))
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def expand_seed_ranges(text: str | None) -> list[int]:
    """``"1000-1004,7"`` -> ``[1000, 1001, 1002, 1003, 1004, 7]``."""
    seeds: list[int] = []
    if not text:
        return seeds
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk[1:]:
            first, _, last = chunk.partition("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return expand_seed_ranges(args.seeds)
    return [int(args.seed_start) + offset for offset in range(int(args.seed_count))]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m ashcrown_rl.evaluate``."""
    parser = argparse.ArgumentParser(
        prog="python -m ashcrown_rl.evaluate",
        description="Fixed-seed evaluation of a scripted policy on the real headless environment.",
    )
    parser.add_argument("--policy", default="heuristic", help="heuristic | random | zero | ppo")
    parser.add_argument("--checkpoint", default=None, help="PPO checkpoint (.zip) for --policy ppo")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel evaluation processes for --policy ppo (default 1)",
    )
    parser.add_argument("--level", type=int, default=1, help="level to evaluate (default 1)")
    parser.add_argument("--character", default="warrior", help="character id")
    parser.add_argument("--frame-skip", type=int, default=4, help="physics ticks per step")
    parser.add_argument("--max-ticks", type=int, default=18_000, help="episode tick cap")
    parser.add_argument("--reward-profile", default="combat", help="engine reward profile")
    parser.add_argument("--seeds", default=None, help="explicit seed list, e.g. 1,2,3")
    parser.add_argument("--seed-start", type=int, default=1, help="first seed of the list")
    parser.add_argument("--seed-count", type=int, default=10, help="number of seeds")
    parser.add_argument("--repeat", type=int, default=2, help="runs per seed (determinism)")
    parser.add_argument("--max-steps", type=int, default=None, help="optional step cap")
    parser.add_argument(
        "--seed-timeout",
        type=float,
        default=DEFAULT_SEED_TIMEOUT,
        help="wall-clock seconds one seed may take before the env is killed (--policy ppo)",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=DEFAULT_STARTUP_TIMEOUT,
        help="wall-clock seconds a worker may spend starting Godot + loading the model",
    )
    parser.add_argument("--with-hashes", action="store_true", help="store hash sequences")
    parser.add_argument("--game-project", default=None, help="game checkout (default: auto)")
    parser.add_argument("--godot-bin", default=None, help="godot binary (default: auto)")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--csv", default=None, help="write the per-episode combat CSV here")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    seeds = _parse_seeds(args)
    env_kwargs: dict[str, Any] = {
        "level": args.level,
        "character": args.character,
        "frame_skip": args.frame_skip,
        "max_ticks": args.max_ticks,
        "reward_profile": args.reward_profile,
        "godot_bin": args.godot_bin,
        "project": args.game_project,
    }
    try:
        if args.policy == "ppo":
            report = evaluate_ppo(
                checkpoint=str(args.checkpoint or ""),
                seeds=seeds,
                env_kwargs=env_kwargs,
                level=None,
                max_steps=args.max_steps,
                workers=args.workers,
                csv_path=args.csv,
                out_path=args.out,
                seed_timeout=args.seed_timeout,
                startup_timeout=args.startup_timeout,
            )
        else:
            report = evaluate(
                seeds=seeds,
                policy_name=args.policy,
                env_kwargs=env_kwargs,
                repeat=args.repeat,
                max_steps=args.max_steps,
                with_hashes=args.with_hashes,
                csv_path=args.csv,
            )
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"EVAL_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    summary = {
        "policy": report["config"].get("policy", args.policy),
        "level": report["config"].get("level", args.level),
        "seeds": report["config"].get("seeds", seeds),
        "clears": report["aggregates"]["clears"],
        "episodes": report["aggregates"]["episodes"],
        "success_rate": report["aggregates"]["success_rate"],
        "mean_game_seconds": report["aggregates"]["mean_game_seconds"],
        "deterministic": report["determinism"]["identical"],
        "failures": len(report.get("failures") or []),
        "timed_out_seeds": report.get("timed_out_seeds") or [],
        "out": args.out,
        "csv": args.csv,
    }
    print("EVAL " + json.dumps(summary, sort_keys=True))
    for episode in report["episodes"]:
        print("EVAL_SEED " + json.dumps(_seed_line(episode), sort_keys=True))
    if summary["timed_out_seeds"] or summary["failures"]:
        return 3
    return 0 if report["determinism"]["identical"] else 2


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())

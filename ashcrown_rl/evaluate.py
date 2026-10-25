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
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .baseline import DEFAULT_CONFIG as HEURISTIC_CONFIG
from .baseline import HeuristicPolicy, heuristic_action
from .env import AshCrownEnv
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


def resolve_policy(name: str, seed: int, layout: Layout | None = None) -> Policy:
    """Policy callable for ``--policy``.

    ``heuristic`` (the default) is :class:`~ashcrown_rl.baseline.HeuristicPolicy`
    — the pure core plus the anti-stuck guard — and needs the layout, which
    :func:`run_episode` supplies on the first call. ``random`` is seeded per
    episode; ``zero`` stands still.
    """
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
            "policy": policy_name,
            "seeds": [int(seed) for seed in seeds],
            "repeat": int(repeat),
        }
        env_summary = _episode_summary(runs[0])
        determinism = _determinism(runs)
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
# CLI
# --------------------------------------------------------------------------- #
def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(value) for value in str(args.seeds).replace(",", " ").split()]
    return [int(args.seed_start) + offset for offset in range(int(args.seed_count))]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``python -m ashcrown_rl.evaluate``."""
    parser = argparse.ArgumentParser(
        prog="python -m ashcrown_rl.evaluate",
        description="Fixed-seed evaluation of a scripted policy on the real headless environment.",
    )
    parser.add_argument("--policy", default="heuristic", help="heuristic | random | zero")
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
    parser.add_argument("--with-hashes", action="store_true", help="store hash sequences")
    parser.add_argument("--game-project", default=None, help="game checkout (default: auto)")
    parser.add_argument("--godot-bin", default=None, help="godot binary (default: auto)")
    parser.add_argument("--out", default=None, help="write the JSON report here")
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
        report = evaluate(
            seeds=seeds,
            policy_name=args.policy,
            env_kwargs=env_kwargs,
            repeat=args.repeat,
            max_steps=args.max_steps,
            with_hashes=args.with_hashes,
        )
    except (ProtocolError, OSError, ValueError) as exc:
        print(f"EVAL_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    summary = {
        "policy": report["config"]["policy"],
        "level": report["config"]["level"],
        "seeds": report["config"]["seeds"],
        "clears": report["aggregates"]["clears"],
        "episodes": report["aggregates"]["episodes"],
        "success_rate": report["aggregates"]["success_rate"],
        "mean_game_seconds": report["aggregates"]["mean_game_seconds"],
        "deterministic": report["determinism"]["identical"],
        "out": args.out,
    }
    print("EVAL " + json.dumps(summary, sort_keys=True))
    return 0 if report["determinism"]["identical"] else 2


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())

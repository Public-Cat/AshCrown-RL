"""Protocol benchmark: the RL-3 definition-of-done harness.

Connects to a **real** headless process (from source or an exported binary),
resets one level episode, steps it ``--steps`` times, asserts the observation
contract on every step, measures steps/s, and closes cleanly::

    uv run python -m ashcrown_rl.bench --steps 1000 --csv ../docs/artifacts/RL/bench.csv

The CSV columns are ``env,mode,steps,seconds,steps_per_s,obs_len,seed,commit``
(game spec ``docs/07`` §11). A terminated/truncated episode is reset with the
same seed and the steps keep counting, so the number is a steady-state step
rate; the number of episodes is reported in the JSON summary.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .launch import (
    GodotProcess,
    GodotProcessError,
    GodotUnavailable,
    LaunchConfig,
    default_game_project,
    find_godot,
)
from .protocol import DEFAULT_TIMEOUT, GodotError

CSV_COLUMNS = (
    "env",
    "mode",
    "env_count",
    "steps",
    "seconds",
    "steps_per_s",
    "obs_len",
    "seed",
    "commit",
)
#: Deterministic action stream: four move directions, attack every other step.
MOVE_CYCLE = (1, 3, 5, 7)


@dataclass
class BenchResult:
    """One benchmark run."""

    env: str
    mode: str
    steps: int
    seconds: float
    obs_len: int
    seed: int
    commit: str
    episodes: int
    startup_seconds: float
    env_count: int = 1

    @property
    def steps_per_s(self) -> float:
        return self.steps / self.seconds if self.seconds > 0 else float("inf")

    def row(self) -> dict[str, Any]:
        """CSV row (values already formatted)."""
        return {
            "env": self.env,
            "mode": self.mode,
            "env_count": self.env_count,
            "steps": self.steps,
            "seconds": f"{self.seconds:.3f}",
            "steps_per_s": f"{self.steps_per_s:.1f}",
            "obs_len": self.obs_len,
            "seed": self.seed,
            "commit": self.commit,
        }

    def summary(self) -> dict[str, Any]:
        return {
            "env": self.env,
            "mode": self.mode,
            "env_count": self.env_count,
            "steps": self.steps,
            "seconds": round(self.seconds, 3),
            "steps_per_s": round(self.steps_per_s, 1),
            "obs_len": self.obs_len,
            "seed": self.seed,
            "commit": self.commit,
            "episodes": self.episodes,
            "startup_seconds": round(self.startup_seconds, 3),
        }


def build_config(args: argparse.Namespace) -> LaunchConfig:
    """Launch config from CLI arguments (source or exported binary)."""
    if args.exported_binary:
        return LaunchConfig(
            exported_binary=Path(args.exported_binary),
            timeout_sec=args.timeout,
            seed=args.seed,
        )
    project = Path(args.game_project).expanduser() if args.game_project else default_game_project()
    godot = find_godot(args.godot_bin)
    if project is None or not (Path(project) / "project.godot").is_file():
        raise GodotUnavailable("no game checkout (pass --game-project or set ASHCROWN_PROJECT)")
    if godot is None:
        raise GodotUnavailable("no godot binary (pass --godot-bin or set ASHCROWN_GODOT)")
    return LaunchConfig(game_project=Path(project), godot_bin=godot, timeout_sec=args.timeout)


def game_commit(project: Path | None) -> str:
    """Short git commit of the game checkout (``unknown`` outside a repo)."""
    if project is None:
        return "unknown"
    try:
        completed = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def run_benchmark(
    config: LaunchConfig,
    *,
    steps: int,
    seed: int,
    level: int = 1,
    character: str = "warrior",
    frame_skip: int = 4,
    env_label: str | None = None,
    mode: str = "level",
    timeout: float | None = None,
) -> BenchResult:
    """Run one benchmark against a real process and return its measurements."""
    startup_started = time.perf_counter()
    process = GodotProcess(config)
    try:
        process.start()
        client = process.connect(timeout)
        startup_seconds = time.perf_counter() - startup_started
        obs, info = client.reset(
            seed=seed,
            mode=mode,
            level=level,
            character=character,
            frame_skip=frame_skip,
            max_ticks=1_000_000,
        )
        obs_len = int(client.hello["obs_len"]) if client.hello else obs.size
        _check_observation(obs, obs_len, "reset")
        episodes = 0
        last_tick = int(info["tick"])
        started = time.perf_counter()
        for index in range(steps):
            move = MOVE_CYCLE[index % len(MOVE_CYCLE)]
            attack = 1 if index % 2 == 0 else 0
            result = client.step(move, attack, 0)
            _check_observation(result.obs, obs_len, f"step {index}")
            tick = int(result.info["tick"])
            if result.terminated or result.truncated:
                episodes += 1
                obs, info = client.reset(
                    seed=seed,
                    mode=mode,
                    level=level,
                    character=character,
                    frame_skip=frame_skip,
                    max_ticks=1_000_000,
                )
                _check_observation(obs, obs_len, f"reset {episodes}")
                last_tick = int(info["tick"])
            else:
                expected = last_tick + frame_skip
                if tick != expected:
                    raise AssertionError(f"tick {tick} != expected {expected} at step {index}")
                last_tick = tick
        elapsed = time.perf_counter() - started
        client.close()
        project = config.game_project or default_game_project()
        return BenchResult(
            env=env_label or ("exported" if config.exported_binary else "source"),
            mode=mode,
            steps=steps,
            seconds=elapsed,
            obs_len=obs_len,
            seed=seed,
            commit=game_commit(Path(project) if project else None),
            episodes=episodes,
            startup_seconds=startup_seconds,
        )
    finally:
        process.close()


def run_vec_benchmark(
    *,
    n_envs: int,
    steps: int,
    seed: int,
    level: int = 1,
    character: str = "warrior",
    frame_skip: int = 4,
    env_label: str = "source",
    game_project: Path | None = None,
    godot_bin: str | None = None,
) -> BenchResult:
    """Measure aggregate steps/s of ``n_envs`` AshCrown processes (RL-4).

    ``steps`` is counted per environment (so the aggregate is
    ``steps * n_envs / seconds``); process startup and the first reset are
    outside the measured window.
    """
    from .vec import make_vec_env

    configuration = {"level": level, "character": character, "frame_skip": frame_skip}
    if game_project is not None:
        configuration["project"] = str(game_project)
    if godot_bin is not None:
        configuration["godot_bin"] = godot_bin
    started = time.perf_counter()
    env = make_vec_env(int(n_envs), int(seed), env_kwargs=configuration)
    try:
        obs, _ = env.reset()
        startup_seconds = time.perf_counter() - started
        obs_len = int(env.single_observation_space.shape[0])
        move_cycle = np.array(MOVE_CYCLE, dtype=np.int64)
        actions = np.zeros((int(n_envs), 3), dtype=np.int64)
        measured = time.perf_counter()
        for index in range(int(steps)):
            actions[:, 0] = move_cycle[index % len(move_cycle)]
            actions[:, 1] = 1 if index % 2 == 0 else 0
            obs, rewards, terminated, truncated, _infos = env.step(actions)
            if obs.shape != (n_envs, obs_len):
                raise AssertionError(f"batched observation shape {obs.shape} is wrong")
            if not np.all(np.isfinite(obs)):
                raise AssertionError(f"step {index}: observation holds a non-finite value")
            del rewards, terminated, truncated
        elapsed = time.perf_counter() - measured
    finally:
        env.close()
    return BenchResult(
        env=env_label,
        mode="level",
        steps=int(steps) * int(n_envs),
        seconds=elapsed,
        obs_len=obs_len,
        seed=int(seed),
        commit=game_commit(game_project),
        episodes=1,
        startup_seconds=startup_seconds,
        env_count=int(n_envs),
    )


def _check_observation(obs: np.ndarray, obs_len: int, label: str) -> None:
    if obs.size != obs_len:
        raise AssertionError(f"{label}: observation has {obs.size} floats, expected {obs_len}")
    if obs.dtype != np.float32:
        raise AssertionError(f"{label}: observation dtype is {obs.dtype}, expected float32")
    if not np.all(np.isfinite(obs)):
        raise AssertionError(f"{label}: observation holds a non-finite value")


def append_csv(path: Path, result: BenchResult) -> None:
    """Append one row to ``bench.csv`` (header written when the file is new)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists() or path.stat().st_size == 0
    row = result.row()
    with path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write(",".join(CSV_COLUMNS) + "\n")
        handle.write(",".join(str(row[column]) for column in CSV_COLUMNS) + "\n")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m ashcrown_rl.bench",
        description="Step a real headless AshCrown environment and measure steps/s.",
    )
    parser.add_argument("--steps", type=int, default=1000, help="steps to run (default 1000)")
    parser.add_argument("--seed", type=int, default=20260101, help="episode seed")
    parser.add_argument("--level", type=int, default=1, help="level to reset (default 1)")
    parser.add_argument("--character", default="warrior", help="character id")
    parser.add_argument("--frame-skip", type=int, default=4, help="physics ticks per step")
    parser.add_argument("--mode", default="level", help="episode mode (level only in RL-2/3)")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="socket/wait timeout"
    )
    parser.add_argument("--game-project", default=None, help="game checkout (default: auto)")
    parser.add_argument("--godot-bin", default=None, help="godot binary (default: auto)")
    parser.add_argument("--exported-binary", default=None, help="run an exported Linux binary")
    parser.add_argument("--env-label", default=None, help="CSV env label (source|exported|...)")
    parser.add_argument(
        "--envs",
        type=int,
        default=1,
        help="parallel environments (1 = single-process benchmark, RL-3)",
    )
    parser.add_argument("--csv", default=None, help="append the row to this CSV file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = parse_args(argv)
    if args.envs > 1:
        try:
            vector_result = run_vec_benchmark(
                n_envs=args.envs,
                steps=args.steps,
                seed=args.seed,
                level=args.level,
                character=args.character,
                frame_skip=args.frame_skip,
                env_label=args.env_label or "source",
                game_project=(
                    Path(args.game_project).expanduser()
                    if args.game_project
                    else default_game_project()
                ),
                godot_bin=args.godot_bin,
            )
        except (GodotUnavailable, GodotProcessError, GodotError, AssertionError) as exc:
            print(f"BENCH_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1
        if args.csv:
            append_csv(Path(args.csv), vector_result)
        print("BENCH " + json.dumps(vector_result.summary(), sort_keys=True))
        return 0
    config = build_config(args)
    try:
        result = run_benchmark(
            config,
            steps=args.steps,
            seed=args.seed,
            level=args.level,
            character=args.character,
            frame_skip=args.frame_skip,
            env_label=args.env_label,
            mode=args.mode,
            timeout=args.timeout,
        )
    except (GodotUnavailable, GodotProcessError, GodotError, AssertionError) as exc:
        print(f"BENCH_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.csv:
        append_csv(Path(args.csv), result)
    print("BENCH " + json.dumps(result.summary(), sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI guard
    raise SystemExit(main())

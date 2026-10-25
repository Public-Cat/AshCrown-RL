#!/usr/bin/env python3
"""Observation-config ablation for ``ashcrown-obs-v2`` (RL-6).

Runs the shipped heuristic (:class:`~ashcrown_rl.baseline.HeuristicPolicy`) on
level 1 over a fixed seed list for every observation variant and writes one CSV
row per variant to ``docs/artifacts/RL/ablation_obs_v2.csv`` (path configurable):

    1. ``grid_on``      default: grid on, radius 10, portal assist, 900 px
    2. ``grid_off``     ``obs_grid=false`` (the grid block is absent)
    3. ``radius_21``    ``obs_grid_radius=21`` (43x43 window)
    4. ``portal_off``   ``expose_portal=false`` (no portal bearing)
    5. ``perception_300`` ``perception_px=300`` (strict, human-visible)
    6. ``blocked_on``   ``obs_blocked=true`` (physics point query channel; its
       steps/s against ``grid_on`` is the per-observation cost measurement)

Every variant is a **separate process**, because ``frame_skip``, the grid
geometry and the perception/portal/blocked flags are lifetime keys of one
environment (``docs/07`` §5.1). The heuristic is the same in all variants: it
slices the published layout by name, so a different radius or a grid-less
vector is transparent (grid-less falls back to the bearing-only policy).

Usage::

    cd training
    uv run python scripts/ablate_obs.py            # all variants, seeds 1..10
    uv run python scripts/ablate_obs.py --seeds 1,2,3 --out /tmp/ablation.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Run from a source checkout without installing the package (like
# `scripts/bench_protocol.py`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashcrown_rl.baseline import HeuristicPolicy  # noqa: E402
from ashcrown_rl.env import AshCrownEnv  # noqa: E402
from ashcrown_rl.evaluate import EpisodeMetrics, run_episode  # noqa: E402

#: Variants in report order: name → the ``AshCrownEnv`` observation kwargs.
VARIANTS: tuple[tuple[str, Mapping[str, Any]], ...] = (
    ("grid_on", {}),
    ("grid_off", {"obs_grid": False}),
    ("radius_21", {"obs_grid_radius": 21}),
    ("portal_off", {"expose_portal": False}),
    ("perception_300", {"perception_px": 300.0}),
    ("blocked_on", {"obs_blocked": True}),
)

#: Seeds of the RL-4 acceptance gate (``tests/test_baseline.py``).
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(1, 11))

CSV_FIELDS = (
    "variant",
    "obs_len",
    "grid_radius",
    "perception_px",
    "expose_portal",
    "obs_blocked",
    "seeds",
    "clears",
    "clear_rate",
    "mean_kills",
    "mean_game_seconds",
    "mean_damage_taken",
    "mean_steps",
    "steps_per_s",
    "layout_hash",
)


@dataclass
class VariantResult:
    """Aggregated outcome of one variant over the seed list."""

    name: str
    config: dict[str, Any]
    obs_len: int = 0
    grid_radius: int = 0
    layout_hash: str = ""
    seeds: int = 0
    clears: int = 0
    kills: list[int] = field(default_factory=list)
    game_seconds: list[float] = field(default_factory=list)
    damage_taken: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    steps_per_s: float = 0.0

    def row(self) -> dict[str, Any]:
        seconds = self.steps_per_s or 0.0
        return {
            "variant": self.name,
            "obs_len": self.obs_len,
            "grid_radius": self.grid_radius,
            "perception_px": float(self.config.get("perception_px", 900.0)),
            "expose_portal": bool(self.config.get("expose_portal", True)),
            "obs_blocked": bool(self.config.get("obs_blocked", False)),
            "seeds": self.seeds,
            "clears": self.clears,
            "clear_rate": round(self.clears / self.seeds, 3) if self.seeds else 0.0,
            "mean_kills": round(statistics.fmean(self.kills), 2) if self.kills else 0.0,
            "mean_game_seconds": (
                round(statistics.fmean(self.game_seconds), 1) if self.game_seconds else 0.0
            ),
            "mean_damage_taken": (
                round(statistics.fmean(self.damage_taken), 1) if self.damage_taken else 0.0
            ),
            "mean_steps": round(statistics.fmean(self.steps), 1) if self.steps else 0.0,
            "steps_per_s": round(seconds, 1),
            "layout_hash": self.layout_hash,
        }


def _cleared(metrics: EpisodeMetrics) -> bool:
    return bool(metrics.terminated) and metrics.terminal_reason == "level_clear"


def run_variant(
    name: str,
    config: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    level: int,
    max_ticks: int,
    **env_kwargs: Any,
) -> VariantResult:
    """Run one variant over ``seeds`` in a fresh process and aggregate it."""
    result = VariantResult(name=name, config=dict(config))
    env = AshCrownEnv(level=level, max_ticks=max_ticks, **config, **env_kwargs)
    try:
        result.obs_len = env.obs_len
        result.grid_radius = env.layout.grid_radius
        result.layout_hash = env.layout_hash
        started = time.perf_counter()
        for seed in seeds:
            metrics = run_episode(
                env,
                seed=int(seed),
                policy=HeuristicPolicy(env.layout),
                policy_name="heuristic",
                record_hashes=False,
            )
            result.seeds += 1
            result.clears += 1 if _cleared(metrics) else 0
            result.kills.append(int(metrics.kills))
            result.game_seconds.append(float(metrics.game_seconds))
            result.damage_taken.append(float(metrics.damage_taken))
            result.steps.append(int(metrics.steps))
            print(
                f"  {name} seed={seed} {metrics.terminal_reason or 'running':10s} "
                f"kills={metrics.kills:2d} t={metrics.game_seconds:6.1f}s "
                f"dmg={metrics.damage_taken:5.1f} steps={metrics.steps}",
                flush=True,
            )
        elapsed = time.perf_counter() - started
        total_steps = sum(result.steps)
        result.steps_per_s = total_steps / elapsed if elapsed > 0 else 0.0
    finally:
        env.close()
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="comma-separated seed list (default: the 10 acceptance seeds)",
    )
    parser.add_argument("--level", type=int, default=1, help="level to play (default 1)")
    parser.add_argument("--max-ticks", type=int, default=18_000, help="episode cap")
    parser.add_argument(
        "--variants",
        default=",".join(name for name, _ in VARIANTS),
        help="comma-separated variant names to run",
    )
    parser.add_argument(
        "--out",
        default=str(
            Path(__file__).resolve().parents[2]
            / "docs"
            / "artifacts"
            / "RL"
            / "ablation_obs_v2.csv"
        ),
        help="CSV output path",
    )
    parser.add_argument("--godot-bin", default=None, help="godot binary override")
    parser.add_argument("--project", default=None, help="game checkout override")
    args = parser.parse_args(argv)

    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        parser.error("--seeds is empty")
    wanted = {value.strip() for value in args.variants.split(",") if value.strip()}
    variants = [(name, config) for name, config in VARIANTS if name in wanted]
    unknown = wanted - {name for name, _ in variants}
    if unknown:
        parser.error(f"unknown variant(s) {sorted(unknown)}; known: {[n for n, _ in VARIANTS]}")
    env_kwargs: dict[str, Any] = {}
    if args.godot_bin:
        env_kwargs["godot_bin"] = args.godot_bin
    if args.project:
        env_kwargs["project"] = args.project

    results: list[VariantResult] = []
    for name, config in variants:
        print(f"[ablate] {name}: {dict(config) or 'default config'}", flush=True)
        results.append(
            run_variant(
                name,
                config,
                seeds=seeds,
                level=args.level,
                max_ticks=args.max_ticks,
                **env_kwargs,
            )
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for result in results:
            writer.writerow(result.row())
    print(f"[ablate] wrote {out}", flush=True)
    for result in results:
        row = result.row()
        print(
            f"{row['variant']:15s} obs_len={row['obs_len']:5d} "
            f"clears={row['clears']}/{row['seeds']} "
            f"kills={row['mean_kills']:5.2f} time={row['mean_game_seconds']:6.1f}s "
            f"dmg={row['mean_damage_taken']:6.1f} steps/s={row['steps_per_s']:6.1f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Train a PPO policy on the headless AshCrown environment (RL-7).

Example (level 1 warrior, 16 processes, the recipe the RL-7 report uses)::

    uv run python examples/train_ppo.py --level 1 --character warrior \
        --frame-skip 4 --n-envs 16 --total-steps 2000000 --seed 7 \
        --checkpoint-dir runs/ppo_level1 --log-csv runs/ppo_level1/curve.csv

Every checkpoint (``<checkpoint-dir>/ckpt_<steps>_steps.zip`` plus ``final.zip``)
gets a ``*.json`` metadata sidecar with the protocol/obs-schema/layout-hash
pins, the observation config, the seed set, the git commits, the total steps,
the device and the exact command — ``ashcrown_rl.evaluate --policy ppo
--checkpoint <zip>`` refuses a checkpoint whose contract does not match the
live environment.

The level schedule (``--curriculum 4`` = levels 1..4) is handed out per episode
round-robin, so one training run covers every level.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ashcrown_rl.train import train_ppo  # noqa: E402


def _parse_overrides(text: str | None) -> dict[str, float]:
    """``"kill_normal=2,step_penalty=-0.002"`` -> ``{...}``."""
    overrides: dict[str, float] = {}
    if not text:
        return overrides
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, _, value = chunk.partition("=")
        overrides[key.strip()] = float(value)
    return overrides


def _parse_levels(text: str | None) -> list[int]:
    """``"4"`` -> ``[1, 2, 3, 4]``; ``"1,3,5"`` -> that list."""
    if not text:
        return []
    text = str(text).strip()
    if "," not in text:
        return list(range(1, int(text) + 1))
    return [int(value) for value in text.replace(",", " ").split()]


def _parse_seeds(text: str | None) -> list[int]:
    """Whitespace/comma separated seed list."""
    if not text:
        return []
    return [int(value) for value in str(text).replace(",", " ").split()]


def _parse_net_arch(text: str | None) -> list[int]:
    """``"256,256"`` -> ``[256, 256]``."""
    if not text:
        return [256, 256]
    return [int(value) for value in str(text).replace(",", " ").split()]


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI parser for ``examples/train_ppo.py``."""
    parser = argparse.ArgumentParser(
        prog="python examples/train_ppo.py",
        description="Train PPO (Stable-Baselines3) on the headless AshCrown environment.",
    )
    parser.add_argument("--n-envs", type=int, default=16, help="parallel environments")
    parser.add_argument("--total-steps", type=int, default=2_000_000, help="PPO timesteps")
    parser.add_argument("--seed", type=int, default=7, help="master seed (workers + PPO)")
    parser.add_argument("--level", type=int, default=1, help="level (mode=level)")
    parser.add_argument("--mode", default="level", help="level | run")
    parser.add_argument("--character", default="warrior", help="warrior | knight | fighter")
    parser.add_argument("--frame-skip", type=int, default=4, help="physics ticks per decision")
    parser.add_argument("--max-ticks", type=int, default=None, help="episode tick cap")
    parser.add_argument("--reward-profile", default="combat", help="combat | sparse | none")
    parser.add_argument("--reward-overrides", default=None, help="k=v,k=v reward coefficients")
    parser.add_argument(
        "--curriculum",
        default=None,
        help="mixed-level training: N for levels 1..N, or an explicit list (1,2,3)",
    )
    parser.add_argument("--merchant-policy", default=None, help="none | auto_heal (run mode)")
    parser.add_argument("--obs-grid", default=None, help="true | false")
    parser.add_argument("--obs-grid-radius", type=int, default=10)
    parser.add_argument("--perception-px", type=float, default=900.0)
    parser.add_argument("--expose-portal", default=None, help="true | false")
    parser.add_argument("--obs-blocked", action="store_true", help="add the blocked grid channel")
    parser.add_argument("--n-steps", type=int, default=2048, help="rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=512, help="PPO minibatch size")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--net-arch", default="256,256", help="hidden layers, e.g. 256,256")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto (policy update)")
    parser.add_argument(
        "--explore-coef",
        type=float,
        default=0.02,
        help="training-only reward per newly visited grid cell (0 disables shaping)",
    )
    parser.add_argument(
        "--approach-coef",
        type=float,
        default=0.05,
        help="training-only reward per cell of reduced nearest-enemy distance",
    )
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="warm-start policy weights (examples/bc_pretrain.py output)",
    )
    parser.add_argument("--checkpoint-dir", default="runs/ppo", help="checkpoint directory")
    parser.add_argument(
        "--log-csv", default=None, help="training-curve CSV (default: dir/curve.csv)"
    )
    parser.add_argument("--save-freq", type=int, default=100_000, help="checkpoint cadence")
    parser.add_argument("--tuning-seeds", default="1-10", help="seeds used for tuning (metadata)")
    parser.add_argument("--eval-seeds", default="1000-1099", help="held-out seeds (metadata)")
    parser.add_argument("--game-project", default=None, help="game checkout (default: auto)")
    parser.add_argument("--godot-bin", default=None, help="godot binary (default: auto)")
    return parser


def _bool(value: str | None) -> bool | None:
    """Parse an optional CLI boolean."""
    if value is None:
        return None
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _expand_ranges(text: str | None) -> list[int]:
    """``"1-10"`` -> ``[1..10]``; ``"1000-1004,7"`` -> that list."""
    seeds: list[int] = []
    if not text:
        return seeds
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk:
            first, _, last = chunk.partition("-")
            seeds.extend(range(int(first), int(last) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_arg_parser().parse_args(argv)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_csv = Path(args.log_csv) if args.log_csv else checkpoint_dir / "curve.csv"
    env_kwargs: dict[str, Any] = {
        "mode": args.mode,
        "level": args.level,
        "character": args.character,
        "frame_skip": args.frame_skip,
        "reward_profile": args.reward_profile,
        "reward_overrides": _parse_overrides(args.reward_overrides),
        "obs_grid_radius": args.obs_grid_radius,
        "perception_px": args.perception_px,
        "obs_blocked": bool(args.obs_blocked),
        "godot_bin": args.godot_bin,
        "project": args.game_project,
    }
    if args.max_ticks is not None:
        env_kwargs["max_ticks"] = args.max_ticks
    if args.merchant_policy:
        env_kwargs["merchant_policy"] = args.merchant_policy
    if _bool(args.obs_grid) is not None:
        env_kwargs["obs_grid"] = _bool(args.obs_grid)
    if _bool(args.expose_portal) is not None:
        env_kwargs["expose_portal"] = _bool(args.expose_portal)
    levels = _parse_levels(args.curriculum)
    net_arch = _parse_net_arch(args.net_arch)
    algorithm = {
        "name": "PPO",
        "policy": "MlpPolicy",
        "hyperparameters": {
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "ent_coef": args.ent_coef,
            "clip_range": args.clip_range,
            "n_epochs": args.n_epochs,
            "net_arch": net_arch,
        },
    }
    command = "uv run python examples/train_ppo.py " + " ".join(
        sys.argv[1:] if argv is None else argv
    )
    result = train_ppo(
        n_envs=args.n_envs,
        total_steps=args.total_steps,
        seed=args.seed,
        env_kwargs=env_kwargs,
        checkpoint_dir=checkpoint_dir,
        log_csv=log_csv,
        command=command,
        algorithm=algorithm,
        hyperparameters=algorithm["hyperparameters"],
        init_checkpoint=args.init_checkpoint,
        device=args.device,
        levels=levels,
        shaping={
            "explore_coef": args.explore_coef,
            "approach_coef": args.approach_coef,
        },
        save_freq=args.save_freq,
        eval_seeds={
            "train_master": int(args.seed),
            "curriculum_levels": levels,
            "tuning": _expand_ranges(args.tuning_seeds),
            "eval": _expand_ranges(args.eval_seeds),
        },
        git_paths={
            "training": Path(__file__).resolve().parents[1],
            "game": Path(args.game_project)
            if args.game_project
            else Path(__file__).resolve().parents[2],
        },
    )
    print("TRAIN " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - example CLI
    raise SystemExit(main())

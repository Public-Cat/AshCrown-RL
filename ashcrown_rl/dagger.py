"""DAgger data aggregation and per-round bookkeeping (RL-7 round 3).

The round-2 behaviour-cloning warm start ran *two* DAgger rounds inside one
process and kept the aggregate in memory, so it could not answer the question
the round actually needs answered: **does round *n* of DAgger clear more levels
than round *n-1*?** This module is the missing bookkeeping — the part of a
DAgger loop that is pure data manipulation and therefore unit-testable without a
Godot process:

* :class:`Dataset` — the aggregate ``(observation, expert action)`` set with the
  provenance of every row (round-0 demonstrations versus round-*n* relabelled
  student states), persisted as a single ``.npz`` so an interrupted run resumes
  from disk instead of from memory;
* :func:`head_report` — per-head held-out accuracy plus the label rate and the
  *predicted* rate of each rare class. Accuracy alone hid the round-2 failure
  (the attack head was 97.5 % accurate and never fired); the two rates together
  expose it;
* :func:`round_row` / :func:`write_rounds` — the per-round CSV row the report
  table is built from.

Nothing here touches Stable-Baselines3 at import time: :func:`head_report`
imports torch lazily, like :mod:`ashcrown_rl.evaluate` does for the loader.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Row provenance: a round-0 demonstration state (the expert's own manifold).
SOURCE_DEMO = 0
#: Row provenance: a state the *student* visited and the expert relabelled.
SOURCE_DAGGER = 1

#: Column order of the per-round CSV (``rounds.csv``).
ROUND_COLUMNS: tuple[str, ...] = (
    "iteration",
    "samples",
    "demo_samples",
    "dagger_samples",
    "dagger_episodes",
    "dagger_clears",
    "acc_move",
    "acc_attack",
    "acc_interact",
    "exact_match",
    "attack_label_rate",
    "attack_pred_rate",
    "eval_clears",
    "eval_seeds",
    "eval_kills",
    "eval_steps",
    "collect_s",
    "fit_s",
    "eval_s",
    "wall_s",
)


@dataclass
class Dataset:
    """An aggregate DAgger dataset (observations, expert actions, provenance)."""

    obs: np.ndarray
    actions: np.ndarray
    sources: np.ndarray

    def __post_init__(self) -> None:
        self.obs = np.asarray(self.obs, dtype=np.float32)
        self.actions = np.asarray(self.actions, dtype=np.int64)
        self.sources = np.asarray(self.sources, dtype=np.int8)
        if self.obs.shape[0] != self.actions.shape[0]:
            raise ValueError(
                f"obs rows {self.obs.shape[0]} != action rows {self.actions.shape[0]}"
            )
        if self.obs.shape[0] != self.sources.shape[0]:
            raise ValueError(
                f"obs rows {self.obs.shape[0]} != source rows {self.sources.shape[0]}"
            )

    @property
    def samples(self) -> int:
        """Number of ``(observation, action)`` rows."""
        return int(self.obs.shape[0])

    @property
    def demo_samples(self) -> int:
        """Rows collected from the demonstrator's own rollouts (round 0)."""
        return int(np.count_nonzero(self.sources == SOURCE_DEMO))

    @property
    def dagger_samples(self) -> int:
        """Rows the student visited and the demonstrator relabelled."""
        return int(np.count_nonzero(self.sources == SOURCE_DAGGER))

    def counts(self) -> dict[str, int]:
        """JSON-ready provenance summary."""
        return {
            "samples": self.samples,
            "demo_samples": self.demo_samples,
            "dagger_samples": self.dagger_samples,
            "obs_len": int(self.obs.shape[1]),
        }


def save_dataset(path: str | Path, dataset: Dataset) -> Path:
    """Write ``dataset`` to ``path`` (``.npz``) and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        obs=dataset.obs,
        actions=dataset.actions,
        sources=dataset.sources,
    )
    return target


def load_dataset(path: str | Path) -> Dataset:
    """Read an aggregate dataset written by :func:`save_dataset`."""
    with np.load(Path(path)) as payload:
        return Dataset(
            obs=payload["obs"],
            actions=payload["actions"],
            sources=payload["sources"],
        )


def append_samples(
    dataset: Dataset | None,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    source: int,
) -> Dataset:
    """Aggregate ``(obs, actions)`` onto ``dataset`` (DAgger aggregation step).

    The aggregate is what separates DAgger from plain cloning: the new round's
    student-visited states are *added* to every earlier round, so retraining
    keeps the expert's manifold covered while the student's own distribution is
    corrected. Passing ``dataset=None`` starts a fresh aggregate.
    """
    new_obs = np.asarray(obs, dtype=np.float32)
    new_actions = np.asarray(actions, dtype=np.int64)
    new_sources = np.full(new_obs.shape[0], int(source), dtype=np.int8)
    if dataset is None:
        return Dataset(obs=new_obs, actions=new_actions, sources=new_sources)
    if new_obs.shape[1:] != dataset.obs.shape[1:]:
        raise ValueError(
            f"observation shape {new_obs.shape[1:]} != aggregate {dataset.obs.shape[1:]}"
        )
    return Dataset(
        obs=np.concatenate([dataset.obs, new_obs]),
        actions=np.concatenate([dataset.actions, new_actions]),
        sources=np.concatenate([dataset.sources, new_sources]),
    )


def _head_dists(model: Any, obs: np.ndarray, device: str) -> list[Any]:
    """Per-head ``Categorical`` distributions for ``obs`` (SB3 policy heads)."""
    import torch as th  # noqa: PLC0415 - keep torch out of the import graph

    with th.no_grad():
        tensor = th.as_tensor(obs, dtype=th.float32, device=device)
        return list(model.policy.get_distribution(tensor).distribution)


def head_report(
    model: Any,
    obs: np.ndarray,
    actions: np.ndarray,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Held-out per-head accuracy, predicted and label rates, exact match.

    ``attack_label_rate`` and ``attack_pred_rate`` are the fraction of rows with
    class ``1`` on the attack head (head index 1 of ``[move, attack, interact]``)
    — the pair that catches a head which is accurate because it never fires.
    """
    import torch as th  # noqa: PLC0415 - keep torch out of the import graph

    heads = _head_dists(model, np.asarray(obs, dtype=np.float32), device)
    labels = np.asarray(actions, dtype=np.int64)
    predicted = np.stack(
        [head.probs.detach().cpu().numpy().argmax(axis=1) for head in heads], axis=1
    )
    per_head = []
    for index, head in enumerate(heads):
        probs = head.probs.detach().cpu().numpy()
        correct = predicted[:, index] == labels[:, index]
        per_head.append(
            {
                "head": index,
                "classes": int(probs.shape[1]),
                "accuracy": float(correct.mean()) if correct.size else 0.0,
                "label_counts": [
                    int(value) for value in np.bincount(labels[:, index], minlength=probs.shape[1])
                ],
                "predicted_counts": [
                    int(value)
                    for value in np.bincount(predicted[:, index], minlength=probs.shape[1])
                ],
            }
        )
    exact = np.all(predicted == labels, axis=1)
    log_prob = None
    if len(heads) > 1:
        with th.no_grad():
            tensor = th.as_tensor(labels, dtype=th.long, device=device)
            log_prob = float(
                np.mean(
                    [
                        float(head.log_prob(tensor[:, index]).mean().item())
                        for index, head in enumerate(heads)
                    ]
                )
            )
    report: dict[str, Any] = {
        "rows": int(labels.shape[0]),
        "exact_match": float(exact.mean()) if exact.size else 0.0,
        "mean_log_prob": log_prob,
        "heads": per_head,
    }
    attack = per_head[1]
    report["attack_accuracy"] = attack["accuracy"]
    report["attack_label_rate"] = (
        attack["label_counts"][1] / sum(attack["label_counts"])
        if len(attack["label_counts"]) > 1 and sum(attack["label_counts"])
        else 0.0
    )
    report["attack_pred_rate"] = sum(attack["predicted_counts"][1:]) / sum(
        attack["predicted_counts"]
    )
    return report


def round_row(
    *,
    iteration: int,
    dataset: Dataset,
    report: dict[str, Any] | None,
    dagger_stats: dict[str, Any] | None = None,
    eval_report: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
) -> dict[str, Any]:
    """One row of :data:`ROUND_COLUMNS` from a round's measurements."""
    heads = {entry["head"]: entry for entry in (report or {}).get("heads", [])}
    aggregates = dict((eval_report or {}).get("aggregates") or {})
    row: dict[str, Any] = {
        "iteration": int(iteration),
        "samples": dataset.samples,
        "demo_samples": dataset.demo_samples,
        "dagger_samples": dataset.dagger_samples,
        "dagger_episodes": int((dagger_stats or {}).get("episodes", 0)),
        "dagger_clears": int((dagger_stats or {}).get("clears", 0)),
        "acc_move": float(heads.get(0, {}).get("accuracy", 0.0)),
        "acc_attack": float(heads.get(1, {}).get("accuracy", 0.0)),
        "acc_interact": float(heads.get(2, {}).get("accuracy", 0.0)),
        "exact_match": float((report or {}).get("exact_match", 0.0)),
        "attack_label_rate": float((report or {}).get("attack_label_rate", 0.0)),
        "attack_pred_rate": float((report or {}).get("attack_pred_rate", 0.0)),
        "eval_clears": int(aggregates.get("clears", 0)),
        "eval_seeds": int(aggregates.get("episodes", 0)),
        "eval_kills": float(aggregates.get("mean_kills", 0.0)),
        "eval_steps": float(aggregates.get("mean_steps", 0.0)),
        "collect_s": round(float((timings or {}).get("collect_s", 0.0)), 1),
        "fit_s": round(float((timings or {}).get("fit_s", 0.0)), 1),
        "eval_s": round(float((timings or {}).get("eval_s", 0.0)), 1),
        "wall_s": round(float((timings or {}).get("wall_s", 0.0)), 1),
    }
    return row


def write_rounds(rows: Sequence[dict[str, Any]], path: str | Path) -> Path:
    """(Re)write the per-round CSV and its JSON twin; returns the CSV path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(ROUND_COLUMNS)]
    for row in rows:
        lines.append(
            ",".join(_format_cell(row.get(column)) for column in ROUND_COLUMNS)
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.with_suffix(".json").write_text(
        json.dumps(list(rows), indent=2) + "\n", encoding="utf-8"
    )
    return target


def read_rounds(path: str | Path) -> list[dict[str, Any]]:
    """Read the JSON twin written by :func:`write_rounds` (``[]`` when absent)."""
    twin = Path(path).with_suffix(".json")
    if not twin.is_file():
        return []
    return list(json.loads(twin.read_text(encoding="utf-8")))


def _format_cell(value: Any) -> str:
    """Stable CSV formatting (floats keep 4 decimals, ints stay ints)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

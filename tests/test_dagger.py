"""DAgger bookkeeping tests (RL-7 round 3).

Everything here is pure data manipulation, so no Godot process is started: the
aggregate round-trips through ``.npz`` with its provenance intact, the provenance
split is what the report table reads, the per-round CSV is stable, and a tiny
in-memory policy is enough to check that :func:`head_report` exposes the
label-versus-predicted-rate pair that detects a rare head which never fires.

The in-memory policy is a torch tensor policy and the diagnostics come from
``scripts/bc_diagnose.py``, so this module needs the optional ``train`` extra
(``torch`` + ``stable-baselines3``): it skips as a whole instead of failing
collection in the documented dev-only gate (``docs/07`` §12.3, INT-97).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch", reason="the DAgger diagnostics need the train extra")
pytest.importorskip("stable_baselines3", reason="the DAgger diagnostics need the train extra")

import torch as th
from stable_baselines3.common.distributions import MultiCategoricalDistribution

from ashcrown_rl.dagger import (
    ROUND_COLUMNS,
    SOURCE_DAGGER,
    SOURCE_DEMO,
    Dataset,
    append_samples,
    head_report,
    load_dataset,
    read_rounds,
    round_row,
    save_dataset,
    write_rounds,
)
from scripts.bc_diagnose import head_metrics


class _FakePolicy:
    """A policy whose action heads emit fixed logits (no network needed)."""

    def __init__(self, logits: th.Tensor) -> None:
        self.logits = logits

    def get_distribution(self, obs: th.Tensor) -> object:
        distribution = MultiCategoricalDistribution([9, 2, 2]).proba_distribution(
            self.logits[: obs.shape[0]]
        )

        class _Out:
            pass

        out = _Out()
        out.distribution = distribution.distribution
        return out


class _FakeModel:
    def __init__(self, logits: th.Tensor) -> None:
        self.policy = _FakePolicy(logits)


def _dataset(rows: int, obs_len: int = 4) -> Dataset:
    return Dataset(
        obs=np.zeros((rows, obs_len), dtype=np.float32),
        actions=np.zeros((rows, 3), dtype=np.int64),
        sources=np.full(rows, SOURCE_DEMO, dtype=np.int8),
    )


def test_append_aggregates_and_keeps_provenance() -> None:
    first = _dataset(3)
    second = _dataset(2)
    second.actions[:] = 1
    aggregate = append_samples(first, second.obs, second.actions, source=SOURCE_DAGGER)
    assert aggregate.samples == 5
    assert aggregate.demo_samples == 3
    assert aggregate.dagger_samples == 2
    assert aggregate.counts() == {
        "samples": 5,
        "demo_samples": 3,
        "dagger_samples": 2,
        "obs_len": 4,
    }
    assert aggregate.actions[-1].tolist() == [1, 1, 1]


def test_append_from_none_starts_the_aggregate() -> None:
    aggregate = append_samples(None, np.ones((2, 4)), np.ones((2, 3)), source=SOURCE_DAGGER)
    assert aggregate.samples == 2
    assert aggregate.dagger_samples == 2


def test_append_rejects_a_mismatched_observation_width() -> None:
    aggregate = _dataset(2, obs_len=4)
    with pytest.raises(ValueError, match="observation shape"):
        append_samples(aggregate, np.zeros((1, 5)), np.zeros((1, 3)), source=SOURCE_DAGGER)


def test_dataset_rejects_ragged_arrays() -> None:
    with pytest.raises(ValueError, match="obs rows"):
        Dataset(obs=np.zeros((2, 4)), actions=np.zeros((1, 3)), sources=np.zeros(2))


def test_dataset_round_trips_through_npz(tmp_path: Path) -> None:
    original = _dataset(4)
    original.actions[2] = [3, 1, 0]
    original.sources[2] = SOURCE_DAGGER
    path = save_dataset(tmp_path / "aggregate.npz", original)
    restored = load_dataset(path)
    assert np.array_equal(restored.obs, original.obs)
    assert np.array_equal(restored.actions, original.actions)
    assert np.array_equal(restored.sources, original.sources)
    assert restored.dagger_samples == 1


def test_head_report_flags_a_collapsed_attack_head() -> None:
    # 9-way move head is right; the attack head always predicts 0 even though
    # half the labels are 1 — accurate-looking, and visibly collapsed.
    rows = 4
    logits = th.tensor(
        [
            [0.0] * 9 + [4.0, 0.0, 4.0, 0.0],
            [0.0] * 9 + [4.0, 0.0, 4.0, 0.0],
            [0.0] * 9 + [4.0, 0.0, 4.0, 0.0],
            [0.0] * 9 + [4.0, 0.0, 4.0, 0.0],
        ]
    )
    actions = np.zeros((rows, 3), dtype=np.int64)
    actions[:, 0] = 0
    actions[:2, 1] = 1
    report = head_report(_FakeModel(logits), np.zeros((rows, 4)), actions)
    assert report["rows"] == rows
    assert report["heads"][1]["accuracy"] == pytest.approx(0.5)
    assert report["attack_label_rate"] == pytest.approx(0.5)
    assert report["attack_pred_rate"] == pytest.approx(0.0)
    assert report["exact_match"] == pytest.approx(0.5)
    assert report["mean_log_prob"] < 0.0


def test_head_metrics_confusion_matrix_is_row_normalisable() -> None:
    labels = np.array([0, 0, 1, 1, 2])
    logits = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],  # 0 misread as 1
            [4.0, 0.0, 0.0],  # 1 misread as 0
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    metrics = head_metrics("move", labels, logits, confusion=True)
    assert metrics["accuracy"] == pytest.approx(0.6)
    matrix = metrics["confusion"]
    assert [sum(row) for row in matrix] == [2, 2, 1]
    assert matrix[0][1] == 1
    assert matrix[1][0] == 1


def test_round_row_fills_every_column() -> None:
    dataset = _dataset(6)
    report = {
        "exact_match": 0.5,
        "attack_label_rate": 0.02,
        "attack_pred_rate": 0.03,
        "heads": [
            {"head": 0, "accuracy": 0.7},
            {"head": 1, "accuracy": 0.9},
            {"head": 2, "accuracy": 1.0},
        ],
    }
    eval_report = {
        "aggregates": {"clears": 3, "episodes": 10, "mean_kills": 8.5, "mean_steps": 1200}
    }
    row = round_row(
        iteration=2,
        dataset=dataset,
        report=report,
        dagger_stats={"episodes": 5, "clears": 1},
        eval_report=eval_report,
        timings={"collect_s": 12.34, "fit_s": 5.0, "eval_s": 60.0, "wall_s": 80.0},
    )
    assert set(row) == set(ROUND_COLUMNS)
    assert row["iteration"] == 2
    assert row["samples"] == 6
    assert row["dagger_clears"] == 1
    assert row["acc_attack"] == pytest.approx(0.9)
    assert row["eval_clears"] == 3
    assert row["fit_s"] == pytest.approx(5.0)


def test_rounds_csv_is_written_and_reread(tmp_path: Path) -> None:
    path = tmp_path / "rounds.csv"
    rows = [
        round_row(iteration=0, dataset=_dataset(4), report=None),
        round_row(iteration=1, dataset=_dataset(8), report=None),
    ]
    write_rounds(rows, path)
    header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == list(ROUND_COLUMNS)
    assert read_rounds(path) == rows
    assert read_rounds(tmp_path / "absent.csv") == []

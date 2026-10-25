"""AshCrown-RL: Python clients and training pipeline for AshCrown.

RL-3 provides the ``ashcrown-rl-protocol-v1`` client (:mod:`ashcrown_rl.protocol`)
and the headless launch wrapper (:mod:`ashcrown_rl.launch`); the benchmark that
drives a real process is :mod:`ashcrown_rl.bench`. RL-4 adds the Gymnasium
environment and vector wrapper, RL-7 the training pipeline. See the game
repository's ``docs/07-rl-headless-env-spec.md`` for the contract.
"""

from .protocol import (
    AshCrownClient,
    GodotError,
    ProtocolError,
    ProtocolMismatch,
    Timeout,
)

__version__ = "0.0.0"

__all__ = [
    "AshCrownClient",
    "GodotError",
    "ProtocolError",
    "ProtocolMismatch",
    "Timeout",
    "__version__",
]

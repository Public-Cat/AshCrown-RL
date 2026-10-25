"""Gymnasium environment for one headless AshCrown process (RL-4).

One :class:`AshCrownEnv` owns exactly one Godot process (game spec ``docs/07``
§2.1, §9.2): the process runs ``res://scenes/rl/rl_env.tscn`` headless under
``--fixed-fps 60``, performs its own discarded warm-up episode before serving
(§4.2), and then steps in **lockstep** with this client — ``frame_skip``
physics ticks per :meth:`step`, blocking while it waits for the next action.

::

    from ashcrown_rl import AshCrownEnv

    env = AshCrownEnv(mode="level", level=1, frame_skip=4)
    obs, info = env.reset(seed=20260101)
    obs, reward, terminated, truncated, info = env.step([2, 1, 0])
    env.close()

Contract (``docs/07`` §9.2):

* ``observation_space`` is ``Box(-1, 1, (obs_len,), float32)`` and
  ``action_space`` is ``MultiDiscrete([9, 2, 2])`` — both built from the live
  handshake, never hardcoded;
* ``metadata`` carries ``render_modes``, the observation block table
  (``obs_layout``) and the action meanings;
* :meth:`reset` accepts ``options`` overriding ``level``, ``character``,
  ``max_ticks``, ``reward_profile``, ``reward_overrides`` and
  ``merchant_policy``;
* ``mode`` is ``"level"`` (one floor, portal terminal) or ``"run"`` (the full
  ten-level descent, ``max_ticks`` default 144,000, `docs/07` §4.1);
* :meth:`step` **never auto-resets** (the engine rejects a step after a
  terminal with ``EPISODE_DONE``; the vector wrapper auto-resets instead);
* :meth:`close` is idempotent and leaves no process behind.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .launch import (
    GodotProcess,
    GodotUnavailable,
    LaunchConfig,
    default_game_project,
)
from .protocol import DEFAULT_TIMEOUT, GodotError, ProtocolError
from .spaces import Layout, LayoutError

#: Default episode cap in physics ticks: 5 game-minutes for one level
#: (``docs/07`` §4.2 / Appendix C).
DEFAULT_LEVEL_MAX_TICKS = 18_000
#: Default episode cap for a full run: 40 game-minutes (``docs/07`` §4.2).
DEFAULT_RUN_MAX_TICKS = 144_000
#: Episode modes the engine implements (``docs/07`` §4.1). ``run`` descends
#: level 1 -> 10 in one episode; ``combat`` stays reserved.
MODES: tuple[str, ...] = ("level", "run")
#: Merchant sub-policies the engine implements (``docs/07`` §7.3). The economy
#: stays out of the action space: ``auto_heal`` buys the full heal whenever the
#: player is in range, hurt and able to pay.
MERCHANT_POLICIES: tuple[str, ...] = ("none", "auto_heal")
#: Reset options ``docs/07`` §9.2 allows the caller to override.
RESET_OPTIONS: frozenset[str] = frozenset(
    {"level", "character", "max_ticks", "reward_profile", "reward_overrides", "merchant_policy"}
)


class AshCrownEnv(gym.Env):
    """One AshCrown episode stream backed by one headless Godot process."""

    metadata = {"render_modes": [], "render_fps": 15}

    def __init__(
        self,
        *,
        mode: str = "level",
        level: int = 1,
        character: str = "warrior",
        frame_skip: int = 4,
        obs_grid: bool = True,
        obs_grid_radius: int = 10,
        reward_profile: str = "combat",
        reward_overrides: Mapping[str, float] | None = None,
        max_ticks: int | None = None,
        merchant_policy: str | None = None,
        hash_every_ticks: int = 60,
        godot_bin: str | None = None,
        project: str | None = None,
        exported_binary: str | None = None,
        extra_args: Sequence[str] = (),
        timeout: float = DEFAULT_TIMEOUT,
        startup_timeout: float | None = None,
        render_mode: str | None = None,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        """Launch the environment process and build the Gym spaces.

        ``client_factory`` replaces the real launcher with a stub that already
        exposes ``hello`` plus ``reset``/``step``/``close`` (unit tests only;
        no Godot process is spawned then).
        """
        if render_mode is not None:
            raise ValueError("render_mode=None only: RL-4 has no renderer (docs/07 §9.2)")
        if mode not in MODES:
            raise ValueError(f"mode={mode!r} is not implemented: choose one of {MODES}")
        if not 1 <= int(frame_skip) <= 64:
            raise ValueError(f"frame_skip={frame_skip!r} must be inside 1..64")
        if not obs_grid:
            raise ValueError("obs_grid=false is not implemented; the local grid stays on")
        if int(hash_every_ticks) < 1:
            raise ValueError(f"hash_every_ticks={hash_every_ticks!r} must be >= 1")
        if not 1 <= int(level) <= 10:
            raise ValueError(f"level={level!r} must be inside 1..10")
        if max_ticks is not None and not 1 <= int(max_ticks) <= 1_000_000:
            raise ValueError(f"max_ticks={max_ticks!r} must be inside 1..1,000,000")
        if merchant_policy is not None and merchant_policy not in MERCHANT_POLICIES:
            raise ValueError(
                f"merchant_policy={merchant_policy!r} is not implemented: "
                f"choose one of {MERCHANT_POLICIES}"
            )

        self.mode = str(mode)
        self.level = int(level)
        self.character = str(character)
        self.frame_skip = int(frame_skip)
        self.obs_grid = bool(obs_grid)
        self.obs_grid_radius = int(obs_grid_radius)
        self.reward_profile = str(reward_profile)
        self.reward_overrides = dict(reward_overrides or {})
        # The cap default follows the mode (``docs/07`` §4.2): 5 game-minutes for
        # one level, 40 for a full run. An explicit ``max_ticks`` wins.
        if max_ticks is None:
            self.max_ticks = int(
                DEFAULT_RUN_MAX_TICKS if self.mode == "run" else DEFAULT_LEVEL_MAX_TICKS
            )
        else:
            self.max_ticks = int(max_ticks)
        self.merchant_policy = merchant_policy
        self.hash_every_ticks = int(hash_every_ticks)
        self.render_mode = render_mode
        self.godot_bin = godot_bin
        self.project = project
        self.exported_binary = exported_binary
        self.extra_args = tuple(extra_args)
        self.timeout = float(timeout)
        self.startup_timeout = startup_timeout

        self._closed = False
        self._process: GodotProcess | None = None
        self._client: Any | None = None
        self._episode_done = True
        self._episode_steps = 0
        self._episode_return = 0.0
        self._episode_seed: int | None = None

        self._start(client_factory)
        hello = self.hello
        self._layout = Layout.from_handshake(hello)
        self.obs_len = int(self._layout.obs_len)
        self._layout_slices = self._layout.slices()
        self._validate_config_against_handshake()
        # The published clamp (``docs/07`` §5, ``RLProtocol.OBS_CLAMP``) is the
        # authority for the Box bounds: a build that widens or narrows it must
        # not be silently coerced by the client.
        self.obs_bound = self._obs_clamp(hello)
        self.observation_space = spaces.Box(
            low=self.obs_bound[0],
            high=self.obs_bound[1],
            shape=(self.obs_len,),
            dtype=np.float32,
        )
        self.action_space = spaces.MultiDiscrete(np.array([9, 2, 2], dtype=np.int64))
        self.metadata = {
            "render_modes": [],
            "render_fps": int(hello.get("physics_ticks_per_second", 60)) // max(self.frame_skip, 1),
            "obs_layout": {
                "obs_schema": self._layout.obs_schema,
                "layout_hash": self.layout_hash,
                "obs_len": self.obs_len,
                "rel_cell_scale": self._layout.rel_cell_scale,
                "blocks": self._layout_slices,
                "grid": {
                    "radius": self._layout.grid_radius,
                    "size": self._layout.grid_size,
                    "channels": list(self._layout.grid_channels),
                },
            },
            "action_meanings": self._layout.action_meanings(),
            "action_space_shape": [9, 2, 2],
        }

    # -- construction helpers ----------------------------------------------- #
    def _start(self, client_factory: Callable[[], Any] | None) -> None:
        """Start the stub client or a real headless Godot process."""
        if client_factory is not None:
            client = client_factory()
            if getattr(client, "hello", None) is None:
                raise TypeError("client_factory must return a connected client with `hello`")
            self._client = client
            self.hello = dict(client.hello)
            return
        config = LaunchConfig(
            game_project=default_game_project() if self.project is None else self.project,
            godot_bin=self.godot_bin,
            exported_binary=None if self.exported_binary is None else self.exported_binary,
            timeout_sec=self.timeout,
            extra_args=self.extra_args,
        )
        if self.startup_timeout is not None:
            config.startup_timeout_sec = float(self.startup_timeout)
        process = GodotProcess(config)
        try:
            process.start()
            client = process.connect(self.timeout)
        except BaseException:
            process.close()
            raise
        self._process = process
        self._client = client
        self.hello = dict(client.hello or {})

    def _obs_clamp(self, hello: Mapping[str, Any]) -> tuple[float, float]:
        """``(low, high)`` of the observation space, from the handshake clamp."""
        clamp = hello.get("obs_clamp")
        if not isinstance(clamp, Sequence) or len(clamp) != 2:
            raise LayoutError(f"handshake obs_clamp {clamp!r} is not a [low, high] pair")
        low, high = float(clamp[0]), float(clamp[1])
        if not low < high:
            raise LayoutError(f"handshake obs_clamp {clamp!r} is empty")
        return low, high

    def _validate_config_against_handshake(self) -> None:
        """Refuse config the live build cannot honour (engine would say BAD_CONFIG)."""
        characters = self.hello.get("character_ids") or []
        if characters and self.character not in characters:
            raise ValueError(f"character={self.character!r} is not in {characters}")
        profiles = (self.hello.get("reward_profile_defaults") or {}).get("profiles") or {}
        if profiles and self.reward_profile not in profiles:
            raise ValueError(f"reward_profile={self.reward_profile!r} is not in {sorted(profiles)}")
        modes = self.hello.get("modes")
        if modes and self.mode not in modes:
            raise ValueError(f"mode={self.mode!r} is not in the engine's modes {modes}")
        policies = self.hello.get("merchant_policies")
        if policies and self.merchant_policy is not None and self.merchant_policy not in policies:
            raise ValueError(
                f"merchant_policy={self.merchant_policy!r} is not in the engine's "
                f"policies {policies}"
            )
        radius = self._layout.grid_radius
        if radius and self.obs_grid_radius != radius:
            raise ValueError(
                f"obs_grid_radius={self.obs_grid_radius} but this build fixes it at {radius}"
            )
        # ``obs_len`` is published twice: at the top level of the handshake and
        # inside the ``layout`` table. A disagreement means the client would
        # slice observations against a different table than the engine emits.
        published_len = self.hello.get("obs_len")
        if published_len is not None and int(published_len) != self.obs_len:
            raise LayoutError(
                f"handshake obs_len {published_len} disagrees with the layout "
                f"table ({self.obs_len})"
            )

    # -- Gym API ------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode and return ``(obs, info)``.

        ``seed`` seeds the engine episode (``RunState.seed``, the global RNG and
        therefore floor generation/loot/boss timers, ``docs/07`` §10.1). With
        ``seed=None`` a seed is drawn from this env's Gym RNG and returned in
        ``info["seed"]``, so the episode is still reproducible from the record.
        """
        self._assert_open()
        super().reset(seed=seed)
        overrides = dict(options or {})
        unknown = set(overrides) - RESET_OPTIONS
        if unknown:
            raise ValueError(
                f"unknown reset option(s) {sorted(unknown)}; "
                f"docs/07 §9.2 allows {sorted(RESET_OPTIONS)}"
            )
        level = int(overrides.get("level", self.level))
        if not 1 <= level <= 10:
            raise ValueError(f"level={level!r} must be inside 1..10")
        if self.mode == "run":
            # A run always starts the descent at level 1 (``docs/07`` §4.1); the
            # engine ignores any other level, the client mirrors it so the request
            # states what really happens.
            level = 1
        character = str(overrides.get("character", self.character))
        max_ticks = int(overrides.get("max_ticks", self.max_ticks))
        reward_profile = str(overrides.get("reward_profile", self.reward_profile))
        reward_overrides = overrides.get("reward_overrides", self.reward_overrides)
        merchant_policy = overrides.get("merchant_policy", self.merchant_policy)
        if merchant_policy is not None and str(merchant_policy) not in MERCHANT_POLICIES:
            raise ValueError(
                f"merchant_policy={merchant_policy!r} is not implemented: "
                f"choose one of {MERCHANT_POLICIES}"
            )

        engine_seed = self._resolve_seed(seed)
        obs, info = self._client.reset(
            seed=engine_seed,
            mode=self.mode,
            level=level,
            character=character,
            frame_skip=self.frame_skip,
            max_ticks=max_ticks,
            obs_grid=self.obs_grid,
            obs_grid_radius=self.obs_grid_radius,
            reward_profile=reward_profile,
            reward_overrides=dict(reward_overrides or {}),
            hash_every_ticks=self.hash_every_ticks,
            merchant_policy=None if merchant_policy is None else str(merchant_policy),
        )
        self._episode_seed = engine_seed
        self._episode_steps = 0
        self._episode_return = 0.0
        self._episode_done = False
        enriched = self._enrich(info)
        enriched["episode_steps"] = 0
        enriched["episode_return"] = 0.0
        return np.asarray(obs, dtype=np.float32), enriched

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one decision window (``frame_skip`` physics ticks).

        ``action`` is ``[move, attack, interact]`` (``np.ndarray`` or sequence).
        The env does **not** auto-reset: after ``terminated``/``truncated`` call
        :meth:`reset` (the engine would answer ``EPISODE_DONE``).
        """
        self._assert_open()
        if self._episode_done:
            raise RuntimeError(
                "the episode is over (terminated/truncated); call reset() — "
                "AshCrownEnv never auto-resets (docs/07 §4.4)"
            )
        moves = self._validate_action(action)
        result = self._client.step(int(moves[0]), int(moves[1]), int(moves[2]))
        self._episode_steps += 1
        self._episode_return += float(result.reward)
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)
        if terminated or truncated:
            self._episode_done = True
        info = self._enrich(result.info)
        # The engine's ``episode`` counter grows with every reset, so leaving it
        # in the step info would make two identical seeded episodes report
        # different infos (Gymnasium's ``check_env`` determinism check compares
        # them). It is reported on reset, where it belongs.
        info.pop("episode", None)
        info["episode_seed"] = self._episode_seed
        info["episode_steps"] = self._episode_steps
        info["episode_return"] = self._episode_return
        return (
            np.asarray(result.obs, dtype=np.float32),
            float(result.reward),
            terminated,
            truncated,
            info,
        )

    def render(self) -> None:
        """No renderer in v1 (``render_mode=None``); returns ``None``."""
        if self.render_mode is not None:
            raise NotImplementedError("AshCrownEnv has no renderer (docs/07 §9.2)")
        return None

    def close(self) -> None:
        """Close the protocol session and reap the process. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        client = self._client
        self._process = None
        self._client = None
        self._episode_done = True
        try:
            if process is not None:
                process.close()
            elif client is not None:
                client.close()
        except (OSError, ProtocolError, GodotError):
            # Cleanup must never mask the error that made the caller close.
            pass

    # -- introspection ------------------------------------------------------ #
    @property
    def layout(self) -> Layout:
        """Parsed observation layout (named slices, action table)."""
        return self._layout

    @property
    def layout_hash(self) -> str:
        """Layout hash pinned by the handshake (and ``compat.json``)."""
        return self._layout.layout_hash or str(self.hello.get("layout_hash", ""))

    @property
    def process(self) -> GodotProcess | None:
        """The launcher handle (``None`` for a stub client)."""
        return self._process

    @property
    def closed(self) -> bool:
        """True once :meth:`close` ran."""
        return self._closed

    @property
    def episode_done(self) -> bool:
        """True between a terminal step and the next :meth:`reset`."""
        return self._episode_done

    @property
    def episode_seed(self) -> int | None:
        """Seed of the current (or last) episode."""
        return self._episode_seed

    @property
    def episode_steps(self) -> int:
        """Decision windows stepped in the current episode."""
        return self._episode_steps

    @property
    def episode_return(self) -> float:
        """Summed reward of the current episode."""
        return self._episode_return

    # -- internals ---------------------------------------------------------- #
    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("this AshCrownEnv is closed")

    def _resolve_seed(self, seed: int | None) -> int:
        """Engine episode seed: the caller's, or drawn from the Gym RNG."""
        if seed is not None:
            value = int(seed)
            if value < 0:
                raise ValueError(f"seed={seed!r} must be a non-negative integer")
            return value
        rng = self.np_random
        if rng is None:  # pragma: no cover - Gym always seeds it in reset()
            raise RuntimeError("no Gym RNG: call reset(seed=...) at least once")
        return int(rng.integers(0, 2**31 - 1))

    def _validate_action(self, action: Any) -> np.ndarray:
        try:
            moves = np.asarray(action, dtype=np.int64).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"action must be [move, attack, interact], got {action!r}") from exc
        if moves.size != 3:
            raise ValueError(f"action must have 3 components, got {moves.size}")
        if not bool(self.action_space.contains(moves)):
            raise ValueError(
                f"action {moves.tolist()} is outside MultiDiscrete([9, 2, 2])"
                " (move 0-8, attack/interact 0-1)"
            )
        return moves

    def _enrich(self, info: Mapping[str, Any]) -> dict[str, Any]:
        """Copy the engine info and add the layout metadata pins."""
        enriched = dict(info)
        enriched["obs_schema"] = self._layout.obs_schema
        enriched["action_schema"] = self._layout.action_schema
        enriched["layout_hash"] = self.layout_hash
        enriched["obs_len"] = self.obs_len
        # Fresh objects per call: Gymnasium rejects infos that share objects
        # between calls (a caller may keep and mutate them).
        enriched["layout_slices"] = {
            name: list(bounds) for name, bounds in self._layout_slices.items()
        }
        enriched["frame_skip"] = self.frame_skip
        enriched["mode"] = self.mode
        return enriched

    # -- convenience -------------------------------------------------------- #
    def is_level_cleared(self, obs: np.ndarray) -> bool:
        """``level_cleared`` from the observation (portal spawned)."""
        return self._layout.level_cleared(obs)


def make_env(**kwargs: Any) -> Callable[[], AshCrownEnv]:
    """Factory suitable for ``gymnasium.vector`` ``env_fns``."""

    def _factory() -> AshCrownEnv:
        return AshCrownEnv(**kwargs)

    return _factory


__all__ = [
    "DEFAULT_LEVEL_MAX_TICKS",
    "DEFAULT_RUN_MAX_TICKS",
    "MERCHANT_POLICIES",
    "MODES",
    "RESET_OPTIONS",
    "AshCrownEnv",
    "GodotUnavailable",
    "make_env",
]

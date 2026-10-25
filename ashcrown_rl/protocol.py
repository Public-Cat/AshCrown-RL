"""TCP protocol client for the AshCrown headless RL environment (RL-3).

The wire format is ``ashcrown-rl-protocol-v1`` (game spec ``docs/07`` §8.1)::

    u32le total_len      # bytes after this field
    u8    msg_type       # 0x01 request, 0x02 response
    u32le json_len
    bytes json_utf8
    u32le bin_len        # 0 when absent
    bytes float32le      # observation payload (optional)

The server sends a ``hello`` response on accept. The client verifies the
handshake against ``compat.json`` (protocol / observation schema / action
schema), verifies ``layout_hash`` and slices observations by block name — no
offset is ever hardcoded here. See :class:`AshCrownClient`.

Typical use::

    from ashcrown_rl.launch import GodotProcess

    with GodotProcess() as process:
        client = process.connect()
        obs, info = client.reset(seed=20260101)
        obs, reward, terminated, truncated, info = client.step(0, 0, 0)
        client.close()
"""

from __future__ import annotations

import json
import socket
import struct
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

#: Protocol generation the client speaks (``docs/07`` §8.5).
PROTOCOL = 1
OBS_SCHEMA = "ashcrown-obs-v2"
ACTION_SCHEMA = "ashcrown-act-v1"

MSG_REQUEST = 0x01
MSG_RESPONSE = 0x02

#: Hard frame cap (``docs/07`` §8.1); a larger frame fails the connection.
MAX_FRAME_BYTES = 32 * 1024 * 1024

#: ``u32le total_len`` prefix.
_PREFIX = struct.Struct("<I")
#: ``u32le json_len`` / ``u32le bin_len``.
_LENGTH = struct.Struct("<I")
#: Smallest legal ``total_len``: msg type + json_len + bin_len.
MIN_TOTAL_LEN = 1 + 4 + 4

#: Reset defaults (``docs/07`` Appendix C) used when the caller omits a field.
DEFAULT_FRAME_SKIP = 4
DEFAULT_MAX_TICKS = 18_000
DEFAULT_OBS_GRID = True
DEFAULT_OBS_GRID_RADIUS = 10
#: Accepted radius window of the local grid (``docs/07`` §5.1).
GRID_RADIUS_MIN = 4
GRID_RADIUS_MAX = 21
#: Enemy-slot perception radius in pixels (default = the game's cull boundary).
DEFAULT_PERCEPTION_PX = 900.0
#: The strict, human-visible preset (``--rl-perception-px=300``).
STRICT_PERCEPTION_PX = 300.0
PERCEPTION_PX_MIN = 100.0
PERCEPTION_PX_MAX = 5000.0
#: Portal bearing assist (``docs/07`` §5.7) and the optional blocked channel.
DEFAULT_EXPOSE_PORTAL = True
DEFAULT_OBS_BLOCKED = False
DEFAULT_HASH_EVERY_TICKS = 60
DEFAULT_CHARACTER = "warrior"
DEFAULT_MODE = "level"
DEFAULT_REWARD_PROFILE = "combat"
#: Socket timeout for one request/response round trip.
DEFAULT_TIMEOUT = 60.0


class ProtocolError(RuntimeError):
    """A malformed frame, an unexpected response or a broken connection."""


class ProtocolMismatch(ProtocolError):
    """The live handshake does not match the compatibility contract."""


class Timeout(ProtocolError):
    """The server did not answer within the socket timeout."""


class GodotError(ProtocolError):
    """The engine answered ``{ok: false, code: …}`` (``docs/07`` §8.4)."""

    def __init__(self, code: str, message: str = "") -> None:
        self.code = code
        self.message = message
        detail = f"{code}: {message}" if message else code
        super().__init__(detail)

    @property
    def episode_done(self) -> bool:
        """True for ``EPISODE_DONE`` (call ``reset`` for a new episode)."""
        return self.code == "EPISODE_DONE"

    @property
    def no_episode(self) -> bool:
        """True for ``NO_EPISODE`` (no ``reset`` happened yet)."""
        return self.code == "NO_EPISODE"

    @property
    def bad_action(self) -> bool:
        """True for ``BAD_ACTION`` (the episode was not mutated)."""
        return self.code == "BAD_ACTION"


class ResetResult(NamedTuple):
    """``reset`` reply: the first observation and its info block."""

    obs: np.ndarray
    info: dict[str, Any]


class StepResult(NamedTuple):
    """``step`` reply (Gymnasium-shaped, minus auto-reset)."""

    obs: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class Frame:
    """One decoded frame."""

    msg_type: int
    payload: dict[str, Any]
    binary: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


def encode_frame(msg_type: int, payload: Mapping[str, Any], binary: Any | None = None) -> bytes:
    """Encode one frame.

    ``payload`` must be JSON-serialisable and finite (NaN/Inf raise, matching
    the engine's contract); ``binary`` is encoded as little-endian float32.
    """
    json_bytes = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    bin_bytes = b""
    if binary is not None:
        array = np.asarray(binary, dtype="<f4").reshape(-1)
        bin_bytes = array.tobytes()
    total_len = 1 + 4 + len(json_bytes) + 4 + len(bin_bytes)
    if total_len > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame of {total_len} bytes exceeds the {MAX_FRAME_BYTES} byte limit")
    if not 0 <= msg_type <= 0xFF:
        raise ProtocolError(f"message type {msg_type} does not fit in one byte")
    parts = [
        _PREFIX.pack(total_len),
        bytes((msg_type,)),
        _LENGTH.pack(len(json_bytes)),
        json_bytes,
        _LENGTH.pack(len(bin_bytes)),
        bin_bytes,
    ]
    return b"".join(parts)


class FrameDecoder:
    """Incremental frame decoder (partial reads and coalesced frames).

    Feed arbitrary chunks; every completed frame is returned. A malformed or
    oversized frame raises :class:`ProtocolError` **and kills the decoder**:
    the byte stream can no longer be trusted, so every later read raises too
    (the engine closes the connection for the same reason, ``docs/07`` §8.1).
    """

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._error: str | None = None

    def feed(self, data: bytes) -> list[Frame]:
        if self._error is not None:
            raise ProtocolError(
                f"the decoder is dead after '{self._error}'; the stream is no longer trustworthy"
            )
        self._buffer.extend(data)
        frames: list[Frame] = []
        try:
            while len(self._buffer) >= _PREFIX.size:
                (total_len,) = _PREFIX.unpack_from(self._buffer)
                if total_len > MAX_FRAME_BYTES:
                    raise ProtocolError(
                        f"frame of {total_len} bytes exceeds the {MAX_FRAME_BYTES} byte limit"
                    )
                if total_len < MIN_TOTAL_LEN:
                    raise ProtocolError(f"frame length {total_len} is below the minimum")
                if len(self._buffer) < _PREFIX.size + total_len:
                    break
                body = bytes(self._buffer[_PREFIX.size : _PREFIX.size + total_len])
                del self._buffer[: _PREFIX.size + total_len]
                frames.append(_parse_body(body))
        except ProtocolError as exc:
            # Dead decoder: drop the garbage so no later read can resynchronise
            # on a frame boundary inside a corrupt stream.
            self._error = str(exc)
            self._buffer.clear()
            raise
        return frames

    @property
    def dead(self) -> bool:
        """True once a bad frame made every further read fail."""
        return self._error is not None

    @property
    def error(self) -> str | None:
        """Why the decoder died (``None`` while healthy)."""
        return self._error

    @property
    def buffered_bytes(self) -> int:
        """Bytes of a not-yet-complete frame held by the decoder."""
        return len(self._buffer)


def _parse_body(body: bytes) -> Frame:
    """Decode one frame body (everything after ``total_len``)."""
    if len(body) < MIN_TOTAL_LEN:
        raise ProtocolError(f"frame body of {len(body)} bytes is too short")
    msg_type = body[0]
    (json_len,) = _LENGTH.unpack_from(body, 1)
    json_start = 5
    bin_len_offset = json_start + json_len
    if bin_len_offset + _LENGTH.size > len(body):
        raise ProtocolError("json_len runs past the end of the frame")
    (bin_len,) = _LENGTH.unpack_from(body, bin_len_offset)
    if bin_len % 4 != 0:
        raise ProtocolError(f"binary payload of {bin_len} bytes is not float32-aligned")
    if bin_len_offset + _LENGTH.size + bin_len != len(body):
        raise ProtocolError("bin_len does not match the frame length")
    try:
        payload = json.loads(body[json_start:bin_len_offset].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"frame JSON is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError(f"frame JSON must be an object, got {type(payload).__name__}")
    binary = np.frombuffer(body, dtype="<f4", count=bin_len // 4, offset=bin_len_offset + 4)
    return Frame(msg_type=msg_type, payload=payload, binary=np.array(binary, dtype=np.float32))


# --------------------------------------------------------------------------- #
# Compatibility contract
# --------------------------------------------------------------------------- #
def compat_path() -> Path:
    """Path of the submodule's ``compat.json``."""
    return Path(__file__).resolve().parents[1] / "compat.json"


def load_compat(path: Path | str | None = None) -> dict[str, Any]:
    """Load ``compat.json`` (protocol / schema versions)."""
    target = Path(path) if path is not None else compat_path()
    try:
        with target.open(encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ProtocolMismatch(f"compat file '{target}' is missing") from exc
    if not isinstance(data, dict):
        raise ProtocolMismatch(f"compat file '{target}' is not a JSON object")
    return data


def verify_handshake(
    hello: Mapping[str, Any],
    compat: Mapping[str, Any] | None = None,
    expected_layout_hash: str | None = None,
    expected_obs_config: Mapping[str, Any] | None = None,
) -> None:
    """Verify the engine handshake against the compatibility contract.

    Raises :class:`ProtocolMismatch` on any difference and :class:`ProtocolError`
    when the payload is not a handshake at all. ``compat=None`` loads
    ``compat.json``; pass ``{}`` to skip the version checks (evaluation only).

    ``layout_hash`` is pinned by ``compat.json`` unless ``expected_layout_hash``
    overrides it explicitly (e.g. to evaluate an intentionally older build) —
    or unless the handshake's ``observation_config`` is not the default one the
    compat file pins (RL-6: every observation config has its own layout hash).
    ``expected_obs_config`` is what the caller asked the engine to publish; the
    live ``observation_config`` must equal it exactly.
    The layout block table is cross-checked against ``obs_len`` and the
    published observation clamp must be ``[-1, 1]``.
    """
    if not isinstance(hello, Mapping):
        raise ProtocolError("handshake is not a JSON object")
    if hello.get("ok") is not True or hello.get("cmd") != "hello":
        raise ProtocolError(f"the first frame is not a hello response (cmd={hello.get('cmd')!r})")
    contract = load_compat() if compat is None else compat
    for key in ("protocol", "obs_schema", "action_schema"):
        if key not in contract:
            continue
        expected = contract[key]
        actual = hello.get(key)
        if actual != expected:
            raise ProtocolMismatch(f"handshake {key}={actual!r} but compat.json pins {expected!r}")
    if hello.get("obs_schema") != OBS_SCHEMA:
        raise ProtocolMismatch(
            f"handshake obs_schema={hello.get('obs_schema')!r}, client speaks {OBS_SCHEMA!r}"
        )
    if hello.get("action_schema") != ACTION_SCHEMA:
        raise ProtocolMismatch(
            f"handshake action_schema={hello.get('action_schema')!r}, "
            f"client speaks {ACTION_SCHEMA!r}"
        )
    layout_hash = hello.get("layout_hash")
    if not isinstance(layout_hash, str) or len(layout_hash) != 64:
        raise ProtocolMismatch(f"handshake layout_hash={layout_hash!r} is not a SHA-256 hex")
    if expected_obs_config is not None:
        missing = [key for key in OBS_CONFIG_KEYS if key not in expected_obs_config]
        if missing:
            raise ProtocolMismatch(
                "expected observation_config is missing "
                f"{', '.join(repr(key) for key in missing)}: a handshake cannot be verified "
                "against a partial config (pass all five keys, docs/07 §5.1)"
            )
        actual_config = hello.get("observation_config")
        if not isinstance(actual_config, Mapping):
            actual_config = (hello.get("layout") or {}).get("observation_config")
        wanted = {key: expected_obs_config[key] for key in OBS_CONFIG_KEYS}
        if not isinstance(actual_config, Mapping):
            raise ProtocolMismatch("handshake carries no observation_config")
        published = {key: actual_config.get(key) for key in OBS_CONFIG_KEYS}
        for key, value in wanted.items():
            if published[key] != value:
                raise ProtocolMismatch(
                    f"handshake observation_config[{key!r}]={published[key]!r} "
                    f"but {value!r} was requested"
                )
    pinned_hash = expected_layout_hash
    if pinned_hash is None:
        contract_hash = contract.get("layout_hash")
        contract_config = contract.get("observation_config")
        actual_config = hello.get("observation_config")
        if not isinstance(actual_config, Mapping):
            actual_config = (hello.get("layout") or {}).get("observation_config")
        default_config = (
            contract_config is None
            or actual_config is None
            or dict(actual_config) == dict(contract_config)
        )
        pinned_hash = contract_hash if (isinstance(contract_hash, str) and default_config) else None
    if pinned_hash is not None and layout_hash != pinned_hash:
        raise ProtocolMismatch(f"handshake layout_hash={layout_hash} but {pinned_hash} is required")
    clamp = _verify_clamp(hello.get("obs_clamp"))
    if "obs_clamp" in contract and list(contract["obs_clamp"]) != list(clamp):
        raise ProtocolMismatch(
            f"handshake obs_clamp={list(clamp)} but compat.json pins {contract['obs_clamp']}"
        )
    obs_len = hello.get("obs_len")
    if not isinstance(obs_len, int) or obs_len <= 0:
        raise ProtocolMismatch(f"handshake obs_len={obs_len!r} is not a positive integer")
    layout = hello.get("layout")
    if not isinstance(layout, Mapping) or not isinstance(layout.get("blocks"), list):
        raise ProtocolMismatch("handshake carries no layout block table")
    _verify_observation_config(hello, layout)
    if layout.get("obs_len") is not None and layout["obs_len"] != obs_len:
        raise ProtocolMismatch(
            f"layout obs_len={layout['obs_len']!r} does not match the handshake's {obs_len}"
        )
    offset = 0
    for block in layout["blocks"]:
        if not isinstance(block, Mapping):
            raise ProtocolMismatch("layout block is not an object")
        if block.get("offset") != offset:
            raise ProtocolMismatch(
                f"layout block {block.get('name')!r} starts at {block.get('offset')!r}, "
                f"expected {offset}"
            )
        size = block.get("size")
        if not isinstance(size, int) or size <= 0:
            raise ProtocolMismatch(f"layout block {block.get('name')!r} has size {size!r}")
        offset += size
    if offset != obs_len:
        raise ProtocolMismatch(f"layout blocks cover {offset} floats but obs_len is {obs_len}")
    for key in ("frame_skip_default", "physics_ticks_per_second"):
        value = hello.get(key)
        if not isinstance(value, int) or value <= 0:
            raise ProtocolMismatch(f"handshake {key}={value!r} is not a positive integer")


#: Keys every ``ashcrown-obs-v2`` handshake publishes in ``observation_config``.
OBS_CONFIG_KEYS = (
    "obs_grid",
    "obs_grid_radius",
    "perception_px",
    "expose_portal",
    "obs_blocked",
)


def _verify_observation_config(hello: Mapping[str, Any], layout: Mapping[str, Any]) -> None:
    """The layout's observation config must agree with the blocks it publishes.

    ``ashcrown-obs-v2`` resolves the layout from the observation config, so a
    table that claims ``obs_grid=false`` while publishing a grid block (or the
    reverse) would make the client slice the wrong floats.
    """
    config = hello.get("observation_config", layout.get("observation_config"))
    if config is None:
        return
    if not isinstance(config, Mapping):
        raise ProtocolMismatch(f"handshake observation_config={config!r} is not an object")
    missing = [key for key in OBS_CONFIG_KEYS if key not in config]
    if missing:
        raise ProtocolMismatch(f"observation_config is missing {missing}")
    has_grid_block = any(
        isinstance(block, Mapping) and block.get("name") == "grid" for block in layout["blocks"]
    )
    if bool(config["obs_grid"]) != has_grid_block:
        raise ProtocolMismatch(
            f"observation_config.obs_grid={config['obs_grid']!r} but the block table "
            f"{'has' if has_grid_block else 'has no'} grid block"
        )
    layout_config = layout.get("observation_config")
    if isinstance(layout_config, Mapping) and dict(layout_config) != dict(config):
        raise ProtocolMismatch("the handshake observation_config disagrees with the layout's copy")
    grid = layout.get("grid")
    if isinstance(grid, Mapping) and "enabled" in grid:
        if bool(grid["enabled"]) != bool(config["obs_grid"]):
            raise ProtocolMismatch("layout.grid.enabled disagrees with observation_config")


def _verify_clamp(value: Any) -> tuple[float, float]:
    """The published ``[-1, 1]`` observation clamp, or :class:`ProtocolMismatch`."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ProtocolMismatch(f"handshake obs_clamp={value!r} is not a [min, max] pair")
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError) as exc:
        raise ProtocolMismatch(f"handshake obs_clamp={value!r} is not numeric") from exc
    if (low, high) != (-1.0, 1.0):
        raise ProtocolMismatch(f"handshake obs_clamp=[{low}, {high}] is not the documented [-1, 1]")
    return low, high


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class AshCrownClient:
    """Blocking protocol client for one headless Godot process.

    The engine freezes the simulation while it waits for a request, so a
    request/response pair is one lockstep decision. ``reset``/``step`` raise
    :class:`GodotError` for protocol errors (``EPISODE_DONE``, ``NO_EPISODE``,
    ``BAD_ACTION``) and :class:`Timeout` when the server does not answer.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = DEFAULT_TIMEOUT,
        *,
        compat: Mapping[str, Any] | None = None,
        expected_layout_hash: str | None = None,
        expected_obs_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self.hello: dict[str, Any] | None = None
        self._compat = compat
        self._expected_layout_hash = expected_layout_hash
        self._expected_obs_config = expected_obs_config
        self._socket: socket.socket | None = None
        self._decoder = FrameDecoder()
        self._pending: list[Frame] = []

    # -- lifecycle ---------------------------------------------------------- #
    def connect(self, timeout: float | None = None) -> dict[str, Any]:
        """Connect, read and verify the handshake. Returns the hello payload."""
        if self._socket is not None:
            raise ProtocolError("client is already connected")
        effective = self.timeout if timeout is None else float(timeout)
        try:
            sock = socket.create_connection((self.host, self.port), timeout=effective)
        except OSError as exc:
            raise ProtocolError(f"cannot connect to {self.host}:{self.port}: {exc}") from exc
        sock.settimeout(effective)
        self._socket = sock
        self._decoder = FrameDecoder()
        self._pending = []
        try:
            frame = self._receive()
            verify_handshake(
                frame.payload,
                self._compat,
                self._expected_layout_hash,
                self._expected_obs_config,
            )
        except ProtocolError:
            self._socket.close()
            self._socket = None
            raise
        self.hello = frame.payload
        return self.hello

    def close(self) -> None:
        """Send ``close`` (the engine exits 0) and drop the socket. Idempotent."""
        sock = self._socket
        if sock is None:
            return
        try:
            self._send({"cmd": "close"})
            self._receive()
        except (OSError, ProtocolError):
            pass
        finally:
            try:
                sock.close()
            finally:
                self._socket = None

    def __enter__(self) -> AshCrownClient:
        if self._socket is None:
            self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- requests ----------------------------------------------------------- #
    def reset(
        self,
        *,
        seed: int | None = None,
        mode: str = DEFAULT_MODE,
        level: int = 1,
        character: str = DEFAULT_CHARACTER,
        frame_skip: int | None = None,
        max_ticks: int | None = None,
        obs_grid: bool | None = None,
        obs_grid_radius: int | None = None,
        perception_px: float | None = None,
        expose_portal: bool | None = None,
        obs_blocked: bool | None = None,
        reward_profile: str = DEFAULT_REWARD_PROFILE,
        reward_overrides: Mapping[str, float] | None = None,
        hash_every_ticks: int = DEFAULT_HASH_EVERY_TICKS,
        merchant_policy: str | None = None,
    ) -> ResetResult:
        """Start a new episode and return ``(obs, info)``.

        ``seed=None`` lets the engine draw one (it is returned in ``info``).
        ``frame_skip=None`` uses the handshake's ``frame_skip_default``;
        ``max_ticks=None`` omits the field so the engine applies its level
        default (18,000 ticks, ``docs/07`` §4.2). The observation config fields
        default to the **handshake's** ``observation_config`` (the process was
        launched with it and treats it as a lifetime key, ``docs/07`` §5.1).
        """
        hello = self._hello()
        if frame_skip is None:
            frame_skip = int(hello["frame_skip_default"])
        published = hello.get("observation_config") or {}
        if obs_grid is None:
            obs_grid = bool(published.get("obs_grid", DEFAULT_OBS_GRID))
        if obs_grid_radius is None:
            obs_grid_radius = int(published.get("obs_grid_radius", DEFAULT_OBS_GRID_RADIUS))
        if perception_px is None:
            perception_px = float(published.get("perception_px", DEFAULT_PERCEPTION_PX))
        if expose_portal is None:
            expose_portal = bool(published.get("expose_portal", DEFAULT_EXPOSE_PORTAL))
        if obs_blocked is None:
            obs_blocked = bool(published.get("obs_blocked", DEFAULT_OBS_BLOCKED))
        payload: dict[str, Any] = {
            "cmd": "reset",
            "mode": mode,
            "level": int(level),
            "character": character,
            "frame_skip": int(frame_skip),
            "obs_grid": bool(obs_grid),
            "obs_grid_radius": int(obs_grid_radius),
            "perception_px": float(perception_px),
            "expose_portal": bool(expose_portal),
            "obs_blocked": bool(obs_blocked),
            "reward_profile": reward_profile,
            "reward_overrides": dict(reward_overrides or {}),
            "hash_every_ticks": int(hash_every_ticks),
        }
        if seed is not None:
            payload["seed"] = int(seed)
        if max_ticks is not None:
            payload["max_ticks"] = int(max_ticks)
        if merchant_policy is not None:
            payload["merchant_policy"] = merchant_policy
        reply = self._exchange(payload, expected_cmd="reset", expect_observation=True)
        return ResetResult(obs=reply.binary, info=reply.payload.get("info", {}))

    def step(self, move: int, attack: int = 0, interact: int = 0) -> StepResult:
        """Advance one decision window (``frame_skip`` physics ticks)."""
        reply = self._exchange(
            {"cmd": "step", "move": move, "attack": attack, "interact": interact},
            expected_cmd="step",
            expect_observation=True,
        )
        payload = reply.payload
        return StepResult(
            obs=reply.binary,
            reward=float(payload["reward"]),
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
            info=dict(payload.get("info", {})),
        )

    def ping(self) -> dict[str, Any]:
        """Liveness probe; allowed in every server state (``docs/07`` §8.4)."""
        return dict(self._exchange({"cmd": "ping"}, expected_cmd="ping").payload)

    # -- internals ---------------------------------------------------------- #
    def _hello(self) -> dict[str, Any]:
        if self.hello is None:
            raise ProtocolError("client is not connected")
        return self.hello

    def _exchange(
        self, payload: Mapping[str, Any], expected_cmd: str, *, expect_observation: bool = False
    ) -> Frame:
        self._send(payload)
        reply = self._receive()
        if reply.msg_type != MSG_RESPONSE:
            raise ProtocolError(f"expected a response frame, got type {reply.msg_type}")
        body = reply.payload
        if body.get("ok") is not True:
            raise GodotError(str(body.get("code", "UNKNOWN")), str(body.get("message", "")))
        if body.get("cmd") != expected_cmd:
            raise ProtocolError(f"expected a {expected_cmd!r} response, got {body.get('cmd')!r}")
        if expect_observation:
            self._check_observation(reply)
        return reply

    def _check_observation(self, reply: Frame) -> None:
        hello = self._hello()
        obs_len = int(hello["obs_len"])
        if reply.binary.size != obs_len:
            raise ProtocolError(
                f"observation has {reply.binary.size} floats, handshake promises {obs_len}"
            )
        declared = reply.payload.get("obs_len")
        if declared is not None and int(declared) != obs_len:
            raise ProtocolError(
                f"response obs_len={declared} does not match the handshake's {obs_len}"
            )
        if not np.all(np.isfinite(reply.binary)):
            raise ProtocolError("observation holds a non-finite value")

    def _send(self, payload: Mapping[str, Any]) -> None:
        sock = self._socket
        if sock is None:
            raise ProtocolError("client is not connected")
        try:
            sock.sendall(encode_frame(MSG_REQUEST, payload))
        except OSError as exc:
            raise ProtocolError(f"send failed: {exc}") from exc

    def _receive(self) -> Frame:
        sock = self._socket
        if sock is None:
            raise ProtocolError("client is not connected")
        while True:
            if self._pending:
                return self._pending.pop(0)
            try:
                chunk = sock.recv(65536)
            except TimeoutError as exc:
                raise Timeout(f"no response within {self.timeout:.1f} s") from exc
            except OSError as exc:
                raise ProtocolError(f"receive failed: {exc}") from exc
            if not chunk:
                raise ProtocolError("the engine closed the connection")
            self._pending.extend(self._decoder.feed(chunk))


__all__ = [
    "ACTION_SCHEMA",
    "AshCrownClient",
    "DEFAULT_CHARACTER",
    "DEFAULT_FRAME_SKIP",
    "DEFAULT_HASH_EVERY_TICKS",
    "DEFAULT_MAX_TICKS",
    "DEFAULT_MODE",
    "DEFAULT_EXPOSE_PORTAL",
    "DEFAULT_OBS_BLOCKED",
    "DEFAULT_OBS_GRID",
    "DEFAULT_OBS_GRID_RADIUS",
    "DEFAULT_PERCEPTION_PX",
    "DEFAULT_REWARD_PROFILE",
    "GRID_RADIUS_MAX",
    "GRID_RADIUS_MIN",
    "OBS_CONFIG_KEYS",
    "PERCEPTION_PX_MAX",
    "PERCEPTION_PX_MIN",
    "STRICT_PERCEPTION_PX",
    "DEFAULT_TIMEOUT",
    "Frame",
    "FrameDecoder",
    "GodotError",
    "MAX_FRAME_BYTES",
    "MSG_REQUEST",
    "MSG_RESPONSE",
    "MIN_TOTAL_LEN",
    "OBS_SCHEMA",
    "PROTOCOL",
    "ProtocolError",
    "ProtocolMismatch",
    "ResetResult",
    "StepResult",
    "Timeout",
    "compat_path",
    "encode_frame",
    "load_compat",
    "verify_handshake",
]

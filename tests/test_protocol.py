"""RL-3 protocol tests: framing, handshake, client behaviour and the live engine.

Unit tests run without Godot. The ``integration``-marked tests launch the real
headless game process (``ashcrown_rl.launch``) and drive the real
``RLEnv`` bridge; they skip cleanly when no Godot binary or game checkout is
available. Run them with ``uv run pytest -m integration``.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ashcrown_rl.launch import (
    GodotProcess,
    GodotProcessError,
    LaunchConfig,
    default_game_project,
    find_godot,
)
from ashcrown_rl.protocol import (
    ACTION_SCHEMA,
    MAX_FRAME_BYTES,
    MIN_TOTAL_LEN,
    MSG_REQUEST,
    MSG_RESPONSE,
    OBS_SCHEMA,
    PROTOCOL,
    AshCrownClient,
    Frame,
    FrameDecoder,
    GodotError,
    ProtocolError,
    ProtocolMismatch,
    Timeout,
    encode_frame,
    load_compat,
    verify_handshake,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _hello(*, obs_len: int = 6, layout_hash: str | None = None) -> dict[str, Any]:
    """Minimal but valid handshake payload for unit tests."""
    return {
        "ok": True,
        "cmd": "hello",
        "protocol": PROTOCOL,
        "obs_schema": OBS_SCHEMA,
        "action_schema": ACTION_SCHEMA,
        "layout_hash": layout_hash or "a" * 64,
        "obs_len": obs_len,
        "obs_clamp": [-1.0, 1.0],
        "frame_skip_default": 4,
        "physics_ticks_per_second": 60,
        "character_ids": ["warrior"],
        "enemy_types": ["slime_green"],
        "layout": {"blocks": [{"name": "global", "offset": 0, "size": obs_len}]},
        "action_directions": [[0.0, 0.0]],
        "reward_profile_defaults": {"default": "combat", "profiles": {}},
    }


class _FakeEngine:
    """In-process engine implementing just enough of the protocol for units.

    Sends ``hello`` on accept, then answers each request with the scripted
    ``(payload, binary)``. ``None`` means "stay silent" (used for timeouts).
    """

    def __init__(self, responses: list[tuple[dict[str, Any], Any] | None]) -> None:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self.port = self._server.getsockname()[1]
        self._responses = responses
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            connection, _ = self._server.accept()
        except OSError:
            return
        with connection:
            connection.sendall(encode_frame(MSG_RESPONSE, _hello()))
            decoder = FrameDecoder()
            for response in self._responses:
                while True:
                    data = connection.recv(65536)
                    if not data:
                        return
                    if decoder.feed(data):
                        break
                if response is None:
                    while connection.recv(65536):
                        pass
                    return
                payload, binary = response
                connection.sendall(encode_frame(MSG_RESPONSE, payload, binary))

    def close(self) -> None:
        self._server.close()


def _obs_payload(obs_len: int = 6, *, tick: int = 4) -> tuple[dict[str, Any], Any]:
    return (
        {
            "ok": True,
            "cmd": "step",
            "reward": 0.5,
            "terminated": False,
            "truncated": False,
            "obs_len": obs_len,
            "binary_len": obs_len * 4,
            "info": {"tick": tick},
        },
        np.zeros(obs_len, dtype=np.float32),
    )


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #
def test_encode_frame_matches_the_documented_layout() -> None:
    payload = {"ok": True}
    json_bytes = json.dumps(payload, separators=(",", ":")).encode()
    frame = encode_frame(MSG_RESPONSE, payload, np.array([1.0, -2.5], dtype=np.float32))
    (total_len,) = struct.unpack("<I", frame[:4])
    assert total_len == 1 + 4 + len(json_bytes) + 4 + 8
    assert frame[4] == MSG_RESPONSE
    assert struct.unpack("<I", frame[5:9])[0] == len(json_bytes)
    assert frame[9 : 9 + len(json_bytes)] == json_bytes
    bin_offset = 9 + len(json_bytes)
    assert struct.unpack("<I", frame[bin_offset : bin_offset + 4])[0] == 8
    assert frame[bin_offset + 4 :].hex() == np.array([1.0, -2.5], dtype="<f4").tobytes().hex()
    assert len(frame) == 4 + total_len


def test_round_trip_keeps_json_and_binary() -> None:
    decoder = FrameDecoder()
    frames = decoder.feed(
        encode_frame(MSG_REQUEST, {"cmd": "step", "move": 2}, np.array([0.25, -1.0, 3.5]))
    )
    assert len(frames) == 1
    frame = frames[0]
    assert frame.msg_type == MSG_REQUEST
    assert frame.payload == {"cmd": "step", "move": 2}
    assert frame.payload["move"] == 2
    assert frame.binary.dtype == np.float32
    assert frame.binary.tolist() == pytest.approx([0.25, -1.0, 3.5])


def test_partial_reads_are_reassembled() -> None:
    data = encode_frame(MSG_REQUEST, {"cmd": "ping"})
    decoder = FrameDecoder()
    for index in range(len(data) - 1):
        assert decoder.feed(data[index : index + 1]) == []
    assert decoder.buffered_bytes == len(data) - 1
    frames = decoder.feed(data[-1:])
    assert [frame.payload for frame in frames] == [{"cmd": "ping"}]
    assert decoder.buffered_bytes == 0


def test_coalesced_frames_are_all_decoded() -> None:
    decoder = FrameDecoder()
    first = encode_frame(MSG_REQUEST, {"cmd": "ping"})
    second = encode_frame(MSG_REQUEST, {"cmd": "close"}, np.array([9.0]))
    frames = decoder.feed(first + second)
    assert [frame.payload["cmd"] for frame in frames] == ["ping", "close"]
    assert frames[1].binary.tolist() == pytest.approx([9.0])


def test_oversized_frame_is_rejected() -> None:
    decoder = FrameDecoder()
    with pytest.raises(ProtocolError, match="exceeds"):
        decoder.feed(struct.pack("<I", MAX_FRAME_BYTES + 1))


def test_short_frame_length_is_rejected() -> None:
    decoder = FrameDecoder()
    with pytest.raises(ProtocolError, match="below the minimum"):
        decoder.feed(struct.pack("<I", MIN_TOTAL_LEN - 1))


def test_inconsistent_binary_length_is_rejected() -> None:
    frame = bytearray(encode_frame(MSG_RESPONSE, {"cmd": "reset"}, np.array([1.0])))
    json_len = struct.unpack("<I", frame[5:9])[0]
    struct.pack_into("<I", frame, 9 + json_len, 8)
    with pytest.raises(ProtocolError, match="bin_len"):
        FrameDecoder().feed(bytes(frame))


def test_malformed_json_is_rejected() -> None:
    json_bytes = b"{not json"
    total = 1 + 4 + len(json_bytes) + 4
    body = bytes((MSG_REQUEST,)) + struct.pack("<I", len(json_bytes)) + json_bytes
    body += struct.pack("<I", 0)
    with pytest.raises(ProtocolError, match="JSON"):
        FrameDecoder().feed(struct.pack("<I", total) + body)


def test_non_finite_payload_is_rejected() -> None:
    with pytest.raises(ValueError):
        encode_frame(MSG_RESPONSE, {"reward": float("nan")})


def test_decoder_stays_dead_after_an_oversized_frame() -> None:
    decoder = FrameDecoder()
    with pytest.raises(ProtocolError, match="exceeds"):
        decoder.feed(struct.pack("<I", MAX_FRAME_BYTES + 1))
    assert decoder.dead, "a bad frame kills the decoder"
    assert decoder.error is not None
    with pytest.raises(ProtocolError, match="dead"):
        decoder.feed(encode_frame(MSG_REQUEST, {"cmd": "ping"}))
    with pytest.raises(ProtocolError, match="dead"):
        decoder.feed(b"")


def test_decoder_stays_dead_after_a_malformed_frame() -> None:
    json_bytes = b"{not json"
    total = 1 + 4 + len(json_bytes) + 4
    body = bytes((MSG_REQUEST,)) + struct.pack("<I", len(json_bytes)) + json_bytes
    body += struct.pack("<I", 0)
    decoder = FrameDecoder()
    with pytest.raises(ProtocolError, match="JSON"):
        decoder.feed(struct.pack("<I", total) + body)
    assert decoder.dead, "the decoder is dead, matching the engine closing the socket"
    assert decoder.buffered_bytes == 0, "the corrupt bytes are dropped"
    with pytest.raises(ProtocolError, match="dead"):
        decoder.feed(struct.pack("<I", total) + body)


# --------------------------------------------------------------------------- #
# Handshake verification
# --------------------------------------------------------------------------- #
def test_verify_handshake_accepts_a_valid_hello() -> None:
    verify_handshake(_hello(), compat={})
    verify_handshake(_hello(), compat={}, expected_layout_hash="a" * 64)


def test_verify_handshake_rejects_a_protocol_mismatch() -> None:
    hello = _hello()
    hello["protocol"] = PROTOCOL + 1
    with pytest.raises(ProtocolMismatch, match="protocol"):
        verify_handshake(hello, compat={"protocol": PROTOCOL})


def test_verify_handshake_rejects_a_schema_mismatch() -> None:
    hello = _hello()
    hello["obs_schema"] = "ashcrown-obs-v0"
    with pytest.raises(ProtocolMismatch, match="obs_schema"):
        verify_handshake(hello, compat={})


def test_verify_handshake_rejects_a_layout_hash_change() -> None:
    with pytest.raises(ProtocolMismatch, match="layout_hash"):
        verify_handshake(_hello(), compat={}, expected_layout_hash="b" * 64)


def test_verify_handshake_uses_the_compat_layout_hash_pin() -> None:
    verify_handshake(_hello(), compat={"layout_hash": "a" * 64})
    with pytest.raises(ProtocolMismatch, match="layout_hash"):
        verify_handshake(_hello(), compat={"layout_hash": "c" * 64})
    # The explicit argument overrides the pin (evaluation of an older build).
    verify_handshake(_hello(), compat={"layout_hash": "c" * 64}, expected_layout_hash="a" * 64)


def test_verify_handshake_requires_the_documented_clamp() -> None:
    wide = _hello()
    wide["obs_clamp"] = [-2.0, 2.0]
    with pytest.raises(ProtocolMismatch, match="obs_clamp"):
        verify_handshake(wide, compat={})
    missing = _hello()
    del missing["obs_clamp"]
    with pytest.raises(ProtocolMismatch, match="obs_clamp"):
        verify_handshake(missing, compat={})
    with pytest.raises(ProtocolMismatch, match="obs_clamp"):
        verify_handshake(_hello(), compat={"obs_clamp": [-1.0, 2.0]})


def test_verify_handshake_cross_checks_the_layout_obs_len() -> None:
    hello = _hello()
    hello["layout"]["obs_len"] = 7
    with pytest.raises(ProtocolMismatch, match="layout obs_len"):
        verify_handshake(hello, compat={})
    blocked = _hello()
    blocked["layout"]["blocks"][0]["size"] = 0
    with pytest.raises(ProtocolMismatch, match="size"):
        verify_handshake(blocked, compat={})


def test_verify_handshake_rejects_a_broken_layout_table() -> None:
    hello = _hello()
    hello["obs_len"] = 7
    with pytest.raises(ProtocolMismatch, match="layout blocks"):
        verify_handshake(hello, compat={})
    broken = _hello()
    broken["layout"] = {"blocks": [{"name": "global", "offset": 3, "size": 3}]}
    with pytest.raises(ProtocolMismatch, match="starts at"):
        verify_handshake(broken, compat={})


def test_verify_handshake_rejects_a_non_handshake() -> None:
    with pytest.raises(ProtocolError, match="hello"):
        verify_handshake({"ok": True, "cmd": "reset"}, compat={})


def test_load_compat_reads_the_submodule_contract() -> None:
    contract = load_compat()
    assert contract["protocol"] == PROTOCOL
    assert contract["obs_schema"] == OBS_SCHEMA
    assert contract["action_schema"] == ACTION_SCHEMA
    layout_hash = contract["layout_hash"]
    assert isinstance(layout_hash, str) and len(layout_hash) == 64
    assert contract["obs_clamp"] == [-1.0, 1.0]


# --------------------------------------------------------------------------- #
# Client against a scripted in-process engine (no Godot)
# --------------------------------------------------------------------------- #
def test_client_raises_godot_error_with_code() -> None:
    engine = _FakeEngine([({"ok": False, "code": "BAD_ACTION", "message": "nope"}, None)])
    try:
        with AshCrownClient(port=engine.port, timeout=5.0, compat={}) as client:
            with pytest.raises(GodotError) as caught:
                client.step(9, 0, 0)
            assert caught.value.code == "BAD_ACTION"
            assert caught.value.bad_action
            assert "nope" in str(caught.value)
    finally:
        engine.close()


def test_client_rejects_a_short_observation() -> None:
    engine = _FakeEngine(
        [
            (
                {"ok": True, "cmd": "step", "reward": 0.0, "terminated": False, "truncated": False},
                np.zeros(2, dtype=np.float32),
            )
        ]
    )
    try:
        with AshCrownClient(port=engine.port, timeout=5.0, compat={}) as client:
            with pytest.raises(ProtocolError, match="floats"):
                client.step(0, 0, 0)
    finally:
        engine.close()


def test_client_times_out_when_the_engine_does_not_answer() -> None:
    engine = _FakeEngine([None])
    try:
        with AshCrownClient(port=engine.port, timeout=0.3, compat={}) as client:
            with pytest.raises(Timeout):
                client.ping()
    finally:
        engine.close()


def test_client_reports_a_dead_endpoint() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    client = AshCrownClient(port=port, timeout=1.0, compat={})
    with pytest.raises(ProtocolError, match="cannot connect"):
        client.connect()


def test_reset_omits_max_ticks_when_not_given() -> None:
    """``reset()`` without ``max_ticks`` lets the engine apply its default."""
    captured: dict[str, Any] = {}
    client = AshCrownClient(port=1, compat={}, expected_layout_hash="a" * 64)
    client.hello = _hello()

    def _capture(payload: Mapping[str, Any], expected_cmd: str, **_kwargs: Any) -> Frame:
        captured.clear()
        captured.update(payload)
        return Frame(
            msg_type=MSG_RESPONSE,
            payload={"ok": True, "cmd": expected_cmd, "info": {}},
            binary=np.zeros(6, dtype=np.float32),
        )

    client._exchange = _capture  # type: ignore[method-assign]
    client.reset(seed=1)
    assert "max_ticks" not in captured
    client.reset(seed=1, max_ticks=600)
    assert captured["max_ticks"] == 600


# --------------------------------------------------------------------------- #
# Launcher fail-safety (no Godot needed: a fake binary exits immediately)
# --------------------------------------------------------------------------- #
def _write_fake_binary(directory: Path, stderr_line: str = "") -> Path:
    """Stand-in for an exported binary: prints one stderr line and exits 1."""
    path = directory / "ashcrown.x86_64"
    lines = ["#!/bin/sh"]
    if stderr_line:
        lines.append(f"echo {stderr_line!r} >&2")
    lines.append("exit 1")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


class _ExitedChild:
    """Minimal stand-in for an exited engine process."""

    returncode = 1

    def poll(self) -> int:
        return 1

    def wait(self, timeout: float | None = None) -> int:
        return 1

    def terminate(self) -> None:
        return None

    def kill(self) -> None:
        return None


def test_start_failure_removes_the_scene_override(tmp_path: Path) -> None:
    """A failed retry must not leave ``override.cfg`` or a zombie behind."""
    binary = _write_fake_binary(tmp_path, "path overrides are disabled in this template")
    process = GodotProcess(
        LaunchConfig(exported_binary=binary, startup_timeout_sec=30, shutdown_timeout_sec=5)
    )
    with pytest.raises(GodotProcessError) as caught:
        process.start()
    assert caught.value.returncode == 1, caught.value.stderr_tail
    assert "path overrides" in caught.value.stderr_tail, "the retry path was exercised"
    assert not (tmp_path / "override.cfg").exists(), "the written override.cfg was removed"
    assert process.ready is None
    assert process._process is None, "the child was killed and reaped"
    assert process._readers == [], "the reader threads were joined"
    assert binary.is_file(), "the binary itself is untouched"
    port_file = Path(process.config.port_file or "")
    process.close()
    process.close()
    assert not port_file.exists(), "close() twice is clean"
    assert not (tmp_path / "override.cfg").exists()


def test_start_ignores_a_stale_port_file(tmp_path: Path) -> None:
    """A port file from an earlier run must not make a dead process look ready."""
    binary = _write_fake_binary(tmp_path)
    port_file = tmp_path / "rl.port"
    port_file.write_text(
        json.dumps({"port": 1, "pid": 0, "protocol": PROTOCOL, "obs_schema": OBS_SCHEMA}),
        encoding="utf-8",
    )
    process = GodotProcess(
        LaunchConfig(exported_binary=binary, port_file=port_file, startup_timeout_sec=30)
    )
    try:
        with pytest.raises(GodotProcessError) as caught:
            process.start()
        assert caught.value.returncode == 1
        assert process.ready is None, "the stale port was not used as a readiness signal"
        assert not port_file.exists(), "the stale port file is removed before the spawn"
    finally:
        process.close()


def test_connect_failure_reports_the_stderr_tail() -> None:
    """A refused session carries the engine log, not just the socket error."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    process = GodotProcess(LaunchConfig())
    process._process = _ExitedChild()  # type: ignore[assignment]
    process._ready = {"port": port}
    process._stderr.append("engine said: schema mismatch")
    try:
        with pytest.raises(GodotProcessError) as caught:
            process.connect()
        assert "schema mismatch" in caught.value.stderr_tail
        assert caught.value.returncode == 1
        assert "RL session" in str(caught.value)
    finally:
        process.close()


def test_close_without_start_is_a_no_op() -> None:
    process = GodotProcess(LaunchConfig())
    process.close()
    process.close()
    assert process.override_restore_error == ""
    assert process.ready is None


# --------------------------------------------------------------------------- #
# Live engine (integration)
# --------------------------------------------------------------------------- #
def _launch_config(**overrides: Any) -> LaunchConfig:
    project = default_game_project()
    godot = find_godot()
    if project is None or godot is None:
        pytest.skip("godot binary or game checkout unavailable")
    return LaunchConfig(game_project=project, godot_bin=godot, **overrides)


@pytest.fixture(scope="module")
def live_process() -> Any:
    process = GodotProcess(_launch_config())
    process.start()
    try:
        yield process
    finally:
        process.close()


@pytest.fixture(scope="module")
def live_client(live_process: GodotProcess) -> AshCrownClient:
    return live_process.connect()


@pytest.mark.integration
def test_live_handshake_matches_the_compat_contract(live_client: AshCrownClient) -> None:
    hello = live_client.hello
    assert hello is not None
    verify_handshake(hello, load_compat())
    assert hello["layout_hash"] == load_compat()["layout_hash"]
    assert hello["obs_clamp"] == [-1.0, 1.0]
    assert hello["obs_len"] == 3246
    assert hello["character_ids"] == ["warrior", "knight", "fighter"]
    assert hello["physics_ticks_per_second"] == 60
    assert hello["frame_skip_default"] == 4
    blocks = hello["layout"]["blocks"]
    assert [block["name"] for block in blocks][-1] == "grid"
    assert sum(block["size"] for block in blocks) == hello["obs_len"]
    defaults = hello["reward_profile_defaults"]
    assert defaults["default"] == "combat"
    assert set(defaults["profiles"]) == {"combat", "sparse", "none"}


@pytest.mark.integration
def test_live_reset_step_and_ping(live_client: AshCrownClient) -> None:
    obs, info = live_client.reset(seed=20260101)
    assert obs.shape == (3246,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["seed"] == 20260101
    assert info["level"] == 1
    assert info["tick"] == 0
    assert info["episode"] == 1
    hello = live_client.hello
    assert hello is not None
    assert info["layout_hash"] == hello["layout_hash"]

    result = live_client.step(0, 0, 0)
    assert result.obs.shape == (3246,)
    assert np.isfinite(result.reward)
    assert not result.terminated
    assert not result.truncated
    assert result.info["tick"] == 4
    assert result.info["step_ticks"] == 4
    assert result.info["episode"] == 1
    assert len(result.info["state_hash"]) == 64

    assert live_client.ping()["tick"] == 4


@pytest.mark.integration
def test_live_first_reset_after_warmup_accepts_a_custom_config() -> None:
    """The discarded warm-up must not leak its config into episode 1."""
    with GodotProcess(_launch_config()) as process:
        client = process.connect()
        obs, info = client.reset(seed=20260110, frame_skip=8, max_ticks=600)
        assert obs.shape == (3246,), "the first client reset is accepted (used to be BAD_CONFIG)"
        assert info["episode"] == 1
        result = client.step(0, 0, 0)
        assert result.info["step_ticks"] == 8, "the requested decision window is used"
        assert not result.truncated


@pytest.mark.integration
def test_live_first_reset_without_max_ticks_uses_the_level_default() -> None:
    """A first ``reset`` without ``max_ticks`` must not cap at the 240-tick warm-up."""
    with GodotProcess(_launch_config()) as process:
        client = process.connect()
        client.reset(seed=20260111)
        result = client.step(0, 0, 0)
        for _step in range(60):
            assert not result.truncated, f"truncated at tick {result.info['tick']}"
            result = client.step(0, 0, 0)
        assert result.info["tick"] > 240, "the episode runs past the warm-up cap (18,000 default)"
        assert not result.truncated


@pytest.mark.integration
def test_live_second_connection_is_rejected() -> None:
    process = GodotProcess(_launch_config())
    try:
        client = process.connect()
        client.reset(seed=20260112, max_ticks=60)
        assert process.ready is not None
        port = int(process.ready["port"])
        second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        second.settimeout(5.0)
        try:
            try:
                second.connect((process.config.host, port))
            except OSError:
                pass  # refused: the listener is closed after the first accept
            else:
                # A connection queued in the closing backlog is dropped, so the
                # peer sees an immediate close/reset instead of a hang.
                try:
                    assert second.recv(4096) == b""
                except OSError:
                    pass
        finally:
            second.close()
        assert client.ping()["ok"] is True, "the first client is unaffected"
    finally:
        process.close()


@pytest.mark.integration
def test_live_bad_action_does_not_mutate_the_episode(live_client: AshCrownClient) -> None:
    live_client.reset(seed=20260102)
    live_client.step(1, 0, 0)
    before = live_client.ping()["tick"]
    with pytest.raises(GodotError) as caught:
        live_client.step(9, 0, 0)
    assert caught.value.code == "BAD_ACTION"
    assert live_client.ping()["tick"] == before


@pytest.mark.integration
def test_live_episode_done_and_reset(live_client: AshCrownClient) -> None:
    live_client.reset(seed=20260103, max_ticks=4)
    result = live_client.step(0, 0, 0)
    assert result.truncated or result.terminated
    with pytest.raises(GodotError) as caught:
        live_client.step(0, 0, 0)
    assert caught.value.code == "EPISODE_DONE"
    obs, info = live_client.reset(seed=20260103, max_ticks=600)
    assert obs.shape == (3246,)
    assert info["tick"] == 0


@pytest.mark.integration
def test_live_step_before_reset_reports_no_episode() -> None:
    with GodotProcess(_launch_config()) as process:
        client = process.connect()
        with pytest.raises(GodotError) as caught:
            client.step(0, 0, 0)
        assert caught.value.code == "NO_EPISODE"
        assert client.ping()["ok"] is True


@pytest.mark.integration
def test_live_close_exits_zero() -> None:
    process = GodotProcess(_launch_config())
    port_file: Path | None = None
    try:
        client = process.connect()
        client.reset(seed=20260104, max_ticks=60)
        assert process.config.port_file is not None
        port_file = Path(process.config.port_file)
        assert port_file.is_file()
        client.close()
        assert process.wait(timeout=30) == 0
    finally:
        process.close()
    assert port_file is not None and not port_file.exists()


@pytest.mark.integration
def test_live_disconnect_exits_two_with_a_stderr_tail() -> None:
    process = GodotProcess(_launch_config())
    try:
        client = process.connect()
        client.reset(seed=20260105, max_ticks=60)
        assert client._socket is not None
        client._socket.close()  # abrupt: no `close` command
        client._socket = None
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.returncode is None:
            time.sleep(0.05)
        assert process.returncode == 2, process.stderr_tail
        assert "client gone" in process.stderr_tail
    finally:
        process.close()


@pytest.mark.integration
def test_live_idle_client_times_out_and_exits_two() -> None:
    process = GodotProcess(_launch_config(timeout_sec=2))
    try:
        process.connect()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and process.returncode is None:
            time.sleep(0.05)
        assert process.returncode == 2, process.stderr_tail
        assert "TIMEOUT" in process.stderr_tail
    finally:
        process.close()


@pytest.mark.integration
def test_launch_reports_early_exit_with_a_stderr_tail() -> None:
    project = default_game_project()
    godot = find_godot()
    if project is None or godot is None:
        pytest.skip("godot binary or game checkout unavailable")
    config = LaunchConfig(
        game_project=project,
        godot_bin=godot,
        scene="res://scenes/rl/does_not_exist.tscn",
        startup_timeout_sec=30,
    )
    process = GodotProcess(config)
    try:
        with pytest.raises(GodotProcessError) as caught:
            process.start()
        assert caught.value.returncode != 0
        assert caught.value.stderr_tail
    finally:
        process.close()


@pytest.mark.integration
def test_exported_binary_handshake_and_step() -> None:
    binary = os.environ.get("ASHCROWN_EXPORTED_BINARY")
    if not binary:
        pytest.skip("set ASHCROWN_EXPORTED_BINARY to the exported Linux binary")
    config = LaunchConfig(exported_binary=Path(binary))
    with GodotProcess(config) as process:
        client = process.connect()
        assert client.hello is not None
        assert client.hello["obs_len"] == 3246
        obs, info = client.reset(seed=20260106, max_ticks=600)
        assert obs.shape == (3246,)
        assert info["seed"] == 20260106
        result = client.step(0, 0, 0)
        assert result.obs.shape == (3246,)
        client.close()

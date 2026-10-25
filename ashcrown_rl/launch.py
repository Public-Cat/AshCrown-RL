"""Launch a real headless AshCrown process and connect the protocol client (RL-3).

The launch contract is game spec ``docs/07`` §3. From source::

    godot --headless --fixed-fps 60 --disable-vsync --audio-driver Dummy \
      --path <game-project> res://scenes/rl/rl_env.tscn -- \
      --rl-server=127.0.0.1:0 --rl-port-file=<tmp> --rl-timeout-sec=60

From an exported Linux binary (same trailing args, scene path positional)::

    ./ashcrown.x86_64 --headless --fixed-fps 60 --disable-vsync \
      res://scenes/rl/rl_env.tscn -- --rl-server=127.0.0.1:0

The launcher waits for the port file (preferred) or the ``RL_READY`` stdout
line, captures the stderr tail and raises :class:`GodotProcessError` with that
tail when the process dies first. ``close()`` sends the protocol ``close`` (the
engine exits 0), reaps the process and removes the port file — no zombies.

Discovery defaults (``docs/07`` Appendix C)::

    ASHCROWN_PROJECT      game checkout       (default: the parent of training/)
    ASHCROWN_GODOT        godot binary        (default: $GODOT_BIN, then `godot` on PATH)
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .protocol import DEFAULT_TIMEOUT, OBS_SCHEMA, PROTOCOL, AshCrownClient, ProtocolError

#: Scene the RL environment lives in (packed into exports too).
RL_SCENE = "res://scenes/rl/rl_env.tscn"
DEFAULT_HOST = "127.0.0.1"
#: Engine flags the determinism contract depends on (``docs/07`` §3.1).
ENGINE_FLAGS: tuple[str, ...] = ("--headless", "--fixed-fps", "60", "--disable-vsync")
#: Seconds to wait for the port file / ``RL_READY`` line.
DEFAULT_STARTUP_TIMEOUT = 120.0
#: Seconds to wait for a graceful exit after ``close`` before killing.
DEFAULT_SHUTDOWN_TIMEOUT = 20.0
#: Number of captured stdout/stderr lines kept for error reports.
LOG_TAIL_LINES = 200


class GodotUnavailable(RuntimeError):
    """No usable Godot binary or game checkout (tests skip on this)."""


class GodotProcessError(RuntimeError):
    """The engine failed to start, died early or never became ready."""

    def __init__(
        self,
        message: str,
        returncode: int | None = None,
        stderr_tail: str = "",
        command: tuple[str, ...] = (),
    ) -> None:
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.command = command
        detail = message
        if returncode is not None:
            detail += f" (exit code {returncode})"
        if command:
            detail += f"\ncommand: {' '.join(command)}"
        if stderr_tail.strip():
            detail += f"\nstderr tail:\n{stderr_tail.rstrip()}"
        super().__init__(detail)


def default_game_project() -> Path | None:
    """The game checkout to launch, or ``None`` when there is none.

    ``ASHCROWN_PROJECT`` wins; otherwise the parent directory of this
    repository (``training/`` lives inside the game checkout).
    """
    override = os.environ.get("ASHCROWN_PROJECT")
    candidate = Path(override).expanduser() if override else Path(__file__).resolve().parents[2]
    candidate = candidate.resolve()
    return candidate if (candidate / "project.godot").is_file() else None


def find_godot(binary: str | None = None) -> str | None:
    """Resolve the godot executable: argument, ``ASHCROWN_GODOT``, ``GODOT_BIN``, PATH."""
    candidates = [binary, os.environ.get("ASHCROWN_GODOT"), os.environ.get("GODOT_BIN"), "godot"]
    for candidate in candidates:
        if not candidate:
            continue
        found = shutil.which(str(candidate))
        if found:
            return found
    return None


@dataclass
class LaunchConfig:
    """Everything needed to start one headless environment process."""

    game_project: Path | None = None
    godot_bin: str | None = None
    exported_binary: Path | None = None
    host: str = DEFAULT_HOST
    port: int = 0
    timeout_sec: float = DEFAULT_TIMEOUT
    startup_timeout_sec: float = DEFAULT_STARTUP_TIMEOUT
    shutdown_timeout_sec: float = DEFAULT_SHUTDOWN_TIMEOUT
    scene: str = RL_SCENE
    port_file: Path | None = None
    seed: int | None = None
    extra_args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    scene_override_fallback: bool = True

    def resolved_project(self) -> Path:
        """The game checkout, or raise :class:`GodotUnavailable`."""
        project = self.game_project or default_game_project()
        if project is None:
            raise GodotUnavailable(
                "no game checkout found (set ASHCROWN_PROJECT or pass game_project=)"
            )
        project = Path(project).expanduser().resolve()
        if not (project / "project.godot").is_file():
            raise GodotUnavailable(f"'{project}' is not a Godot project (no project.godot)")
        return project

    def user_args(self) -> list[str]:
        """Arguments after the ``--`` separator (``docs/07`` §3.1)."""
        args = [
            f"--rl-server={self.host}:{int(self.port)}",
            f"--rl-timeout-sec={int(self.timeout_sec)}",
        ]
        if self.port_file is not None:
            args.insert(1, f"--rl-port-file={Path(self.port_file)}")
        if self.seed is not None:
            args.append(f"--rl-seed={int(self.seed)}")
        args.extend(self.extra_args)
        return args

    def build_command(self, *, use_scene_override: bool = False) -> list[str]:
        """The exact argv the launcher runs.

        Official export templates are built with path overrides disabled, so an
        exported binary refuses the positional scene path (``docs/07`` §3.2
        assumes a template that allows it). ``use_scene_override`` drops the
        positional scene and relies on an ``override.cfg`` next to the binary
        that points ``run/main_scene`` at the RL scene instead.
        """
        if self.exported_binary is not None:
            binary = Path(self.exported_binary).expanduser()
            if not binary.is_file():
                raise GodotUnavailable(f"exported binary '{binary}' does not exist")
            head = [str(binary), *ENGINE_FLAGS]
            if not use_scene_override:
                head.append(self.scene)
        else:
            godot = find_godot(self.godot_bin)
            if godot is None:
                raise GodotUnavailable(
                    "no godot binary found (set ASHCROWN_GODOT or pass godot_bin=)"
                )
            project = self.resolved_project()
            head = [
                godot,
                *ENGINE_FLAGS,
                "--audio-driver",
                "Dummy",
                "--path",
                str(project),
                self.scene,
            ]
        return [*head, "--", *self.user_args()]


class GodotProcess:
    """One headless environment process plus its protocol client.

    ::

        with GodotProcess(LaunchConfig(seed=1)) as process:
            client = process.connect()
            obs, info = client.reset(seed=1)
        # the process exited 0 and the port file is gone
    """

    def __init__(self, config: LaunchConfig | None = None) -> None:
        self.config = config or LaunchConfig()
        self.command: tuple[str, ...] = ()
        self._process: subprocess.Popen[str] | None = None
        self._client: AshCrownClient | None = None
        self._ready: dict[str, Any] | None = None
        self._stdout: collections.deque[str] = collections.deque(maxlen=LOG_TAIL_LINES)
        self._stderr: collections.deque[str] = collections.deque(maxlen=LOG_TAIL_LINES)
        self._readers: list[threading.Thread] = []
        self._temporary_port_file: Path | None = None
        self._override_file: Path | None = None
        self._override_backup: bytes | None = None
        self._override_restore_error: str = ""

    # -- lifecycle ---------------------------------------------------------- #
    def start(self) -> GodotProcess:
        """Spawn the engine and wait until it is listening.

        Fail-safe: any failure after the first spawn restores ``override.cfg``
        and kills/reaps the child (and its reader threads) before re-raising,
        so a retry never inherits a half-started process or a stray override.
        """
        if self._process is not None:
            return self
        if self.config.port_file is None:
            handle, path = tempfile.mkstemp(prefix="ashcrown-rl-", suffix=".port")
            os.close(handle)
            os.unlink(path)  # the server creates it atomically; never reuse a stale file
            self._temporary_port_file = Path(path)
            self.config = dataclasses.replace(self.config, port_file=self._temporary_port_file)
        else:
            # A caller-supplied path may still hold the port of an older run;
            # reading it would make a dead process look ready.
            Path(self.config.port_file).unlink(missing_ok=True)
        self._spawn(use_scene_override=False)
        try:
            try:
                self._ready = self._wait_ready()
            except GodotProcessError as exc:
                if not self._can_retry_with_scene_override(exc):
                    raise
                self._write_scene_override()
                self._spawn(use_scene_override=True)
                self._ready = self._wait_ready()
        except BaseException:
            self._restore_scene_override()
            self._kill_and_reap(graceful=False)
            raise
        return self

    def connect(self, timeout: float | None = None) -> AshCrownClient:
        """Start if needed and return a connected, handshake-verified client.

        A connect/handshake failure is reported as :class:`GodotProcessError`
        carrying the captured stderr tail: the interesting cause (a refused
        port, a dying engine, a schema mismatch) is almost always in the
        engine log, not in the socket error.
        """
        self.start()
        if self._client is not None:
            return self._client
        assert self._ready is not None
        client = AshCrownClient(
            host=self.config.host,
            port=int(self._ready["port"]),
            timeout=self.config.timeout_sec,
        )
        try:
            client.connect(timeout)
        except (ProtocolError, OSError) as exc:
            client.close()
            raise GodotProcessError(
                f"could not open the RL session: {exc}",
                returncode=self.returncode,
                stderr_tail=self.stderr_tail,
                command=self.command,
            ) from exc
        self._client = client
        return client

    def close(self) -> None:
        """Close the protocol session, reap the process, remove the port file.

        Idempotent: safe when nothing was started, after a failed ``start()``
        and when called twice.
        """
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None
        self._kill_and_reap()
        self._restore_scene_override()
        port_file = self.config.port_file
        if port_file is not None:
            Path(port_file).unlink(missing_ok=True)
        self._temporary_port_file = None

    def __enter__(self) -> GodotProcess:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @property
    def override_restore_error(self) -> str:
        """Why the last ``override.cfg`` restore failed, or ``""``."""
        return self._override_restore_error

    def _restore_scene_override(self) -> None:
        """Undo :meth:`_write_scene_override` (remove the file or restore it).

        Idempotent and safe when no override was written: the recorded path and
        backup are cleared before the filesystem is touched. A filesystem error
        is recorded in :attr:`override_restore_error` instead of raising, so a
        cleanup never hides the error that triggered it.
        """
        path = self._override_file
        backup = self._override_backup
        self._override_file = None
        self._override_backup = None
        if path is None:
            return
        try:
            if backup is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(backup)
        except OSError as exc:
            self._override_restore_error = f"{path}: {exc}"

    # -- introspection ------------------------------------------------------ #
    @property
    def ready(self) -> dict[str, Any] | None:
        """The ``RL_READY``/port-file payload (``port``, ``pid``, ``protocol``…)."""
        return self._ready

    @property
    def returncode(self) -> int | None:
        """Exit code of the engine process, or ``None`` while it runs."""
        return None if self._process is None else self._process.poll()

    @property
    def stdout_tail(self) -> str:
        """Captured stdout lines (bounded ring)."""
        return "\n".join(self._stdout)

    @property
    def stderr_tail(self) -> str:
        """Captured stderr lines (bounded ring)."""
        return "\n".join(self._stderr)

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the engine to exit and return its code."""
        if self._process is None:
            raise GodotProcessError("the process was not started")
        return self._process.wait(timeout=timeout)

    # -- internals ---------------------------------------------------------- #
    def _kill_and_reap(self, *, graceful: bool = True) -> None:
        """Stop the child (if any) and join its reader threads.

        ``graceful`` waits for the child to exit on its own (the ``close``
        path, after the protocol ``close`` command); otherwise it terminates
        immediately (the fail-safe path of :meth:`start`, where the child is
        known to be unusable). Leaves no zombie behind and is safe to call
        when nothing was spawned.
        """
        process = self._process
        self._process = None
        if process is not None:
            if graceful:
                try:
                    process.wait(timeout=self.config.shutdown_timeout_sec)
                except subprocess.TimeoutExpired:
                    pass
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        self._drain_readers()

    def _drain_readers(self, timeout: float = 1.0) -> None:
        """Join the stdout/stderr pumps so their tails are complete.

        The readers end when the pipes close (the child exited), so this makes
        an error report deterministic instead of racing the pumps.
        """
        for reader in self._readers:
            reader.join(timeout=timeout)
        self._readers = []

    def _spawn(self, *, use_scene_override: bool) -> None:
        """Start the engine process with the resolved argv."""
        self._process = None
        self._readers = []
        self.command = tuple(self.config.build_command(use_scene_override=use_scene_override))
        environment = dict(os.environ)
        environment.update(self.config.env)
        try:
            self._process = subprocess.Popen(
                list(self.command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                env=environment,
            )
        except OSError as exc:
            raise GodotProcessError(f"could not launch godot: {exc}", command=self.command) from exc
        self._start_readers()

    def _can_retry_with_scene_override(self, error: GodotProcessError) -> bool:
        """True when the export template refuses a positional scene path."""
        return (
            self.config.scene_override_fallback
            and self.config.exported_binary is not None
            and self._override_file is None
            and "path overrides" in error.stderr_tail
        )

    def _write_scene_override(self) -> None:
        """Point the exported project's main scene at the RL scene.

        The file lives next to the exported binary (Godot only reads
        ``override.cfg`` from the project directory). An existing file is
        backed up and restored by :meth:`close` or by the fail-safe path of
        :meth:`start`. The path is recorded **before** the write so even a
        partially written file is cleaned up.
        """
        binary = Path(self.config.exported_binary or "")
        path = binary.expanduser().resolve().parent / "override.cfg"
        self._override_file = path
        self._override_backup = None
        try:
            self._override_backup = path.read_bytes()
        except FileNotFoundError:
            self._override_backup = None
        path.write_text(f'[application]\nrun/main_scene="{self.config.scene}"\n', encoding="utf-8")

    def _start_readers(self) -> None:
        process = self._process
        assert process is not None
        assert process.stdout is not None
        assert process.stderr is not None
        readers = [
            threading.Thread(target=self._pump, args=(process.stdout, self._stdout), daemon=True),
            threading.Thread(target=self._pump, args=(process.stderr, self._stderr), daemon=True),
        ]
        for reader in readers:
            reader.start()
        self._readers = readers

    @staticmethod
    def _pump(stream: Any, sink: collections.deque[str]) -> None:
        try:
            for line in stream:
                sink.append(line.rstrip("\n"))
        except (ValueError, OSError):
            return

    def _wait_ready(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.config.startup_timeout_sec
        while time.monotonic() < deadline:
            process = self._process
            if process is not None and process.poll() is not None:
                # Reap the pumps first: the exit reason lives on stderr and the
                # scene-override retry keys off that tail.
                self._drain_readers()
                raise GodotProcessError(
                    "godot exited before becoming ready",
                    returncode=process.returncode,
                    stderr_tail=self.stderr_tail,
                    command=self.command,
                )
            payload = self._read_port_file()
            if payload is not None:
                return payload
            payload = self._read_ready_line()
            if payload is not None:
                return payload
            time.sleep(0.02)
        raise GodotProcessError(
            f"godot did not become ready within {self.config.startup_timeout_sec:.0f} s",
            returncode=self.returncode,
            stderr_tail=self.stderr_tail,
            command=self.command,
        )

    def _read_port_file(self) -> dict[str, Any] | None:
        path = self.config.port_file
        if path is None or not Path(path).is_file():
            return None
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if self._valid_ready(payload) else None

    def _read_ready_line(self) -> dict[str, Any] | None:
        for line in reversed(self._stdout):
            if not line.startswith("RL_READY "):
                continue
            try:
                payload = json.loads(line[len("RL_READY ") :])
            except json.JSONDecodeError:
                continue
            if self._valid_ready(payload):
                return payload
        return None

    def _valid_ready(self, payload: Any) -> bool:
        if not isinstance(payload, dict) or "port" not in payload:
            return False
        try:
            return int(payload["port"]) > 0
        except (TypeError, ValueError):
            return False


def launch_client(
    config: LaunchConfig | None = None, timeout: float | None = None
) -> tuple[GodotProcess, AshCrownClient]:
    """Start a process and return ``(process, connected client)``.

    The caller owns both and must ``process.close()`` (which also closes the
    client).
    """
    process = GodotProcess(config)
    process.start()
    try:
        client = process.connect(timeout)
    except BaseException:
        process.close()
        raise
    return process, client


__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_SHUTDOWN_TIMEOUT",
    "DEFAULT_STARTUP_TIMEOUT",
    "ENGINE_FLAGS",
    "GodotProcess",
    "GodotProcessError",
    "GodotUnavailable",
    "LaunchConfig",
    "OBS_SCHEMA",
    "PROTOCOL",
    "RL_SCENE",
    "default_game_project",
    "find_godot",
    "launch_client",
]

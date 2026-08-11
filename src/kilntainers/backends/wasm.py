"""WASM backend implementation."""

import argparse
import asyncio
import os
import platform
import shutil
import sys
import sysconfig
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING


def _ensure_windows_processor_architecture() -> None:
    """Provide Wasmtime's fallback architecture when Windows omits it.

    Wasmtime first queries WMI and falls back to PROCESSOR_ARCHITECTURE.
    Parallel test workers can overwhelm WMI, while some environments omit the
    fallback variable entirely. Python's build platform is deterministic and
    identifies the same architecture without another system query.
    """
    if sys.platform != "win32":
        return
    if os.environ.get("PROCESSOR_ARCHITEW6432") or os.environ.get(
        "PROCESSOR_ARCHITECTURE"
    ):
        return

    python_platform = sysconfig.get_platform().lower()
    architecture = {
        "win-amd64": "AMD64",
        "win-arm64": "ARM64",
    }.get(python_platform)
    if architecture is not None:
        os.environ["PROCESSOR_ARCHITECTURE"] = architecture
        # A failed WMI query may already have cached an empty machine value.
        # Clear it so Wasmtime's platform.machine() call uses the fallback.
        platform._uname_cache = None  # ty: ignore[unresolved-attribute]


_ensure_windows_processor_architecture()

if TYPE_CHECKING:
    import wasmtime  # type: ignore[import-not-found]
else:
    try:
        import wasmtime
    except ImportError:
        wasmtime = None

import importlib.resources

from kilntainers.backends.base import Backend, ExecRequest, ExecResult, Sandbox
from kilntainers.config import BackendConfig
from kilntainers.errors import BackendError, SandboxDiedError


class _WasmTimeout(Exception):
    """Internal signal: WASM execution hit epoch deadline."""

    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class WasmBackendConfig(BackendConfig):
    """Configuration for the WASM backend.

    Populated from CLI args by WasmBackend.config_from_args().
    Consumed by WasmBackend.
    """

    # Module
    wasm_path: str  # --wasm-path (required for WasmBackend)

    # Shell for command mode
    shell: str | None = None  # --shell if supported (e.g., "sh")

    # Network
    network_enabled: bool = False  # --network

    # Resource limits
    max_memory_mib: int = 256  # --wasm-max-memory (MiB)
    fuel: int | None = None  # --wasm-fuel (instruction limit)

    # Epoch ticker (internal, not exposed as CLI arg)
    epoch_tick_interval_ms: int = 1000


class WasmBackend(Backend):
    """WASM backend implementation.

    Manages WASM module compilation and execution through the
    wasmtime Python package. All execution happens in-process.
    """

    @classmethod
    def add_cli_arguments(cls, group: argparse._ArgumentGroup) -> None:
        """Register WASM-specific CLI arguments."""
        group.add_argument(
            "--wasm-path",
            default=None,
            help="Path to the .wasm file to execute (required for wasm backend)",
        )
        group.add_argument(
            "--wasm-max-memory",
            type=int,
            default=256,
            help="Max WASM memory in MiB (default: 256)",
        )
        group.add_argument(
            "--wasm-fuel",
            type=int,
            default=None,
            help="WASM instruction fuel limit (default: unlimited)",
        )

    @classmethod
    def config_from_args(cls, args: argparse.Namespace) -> BackendConfig:
        """Build WasmBackendConfig from parsed CLI arguments."""
        # For WASM, we want shell=None by default (no command mode)
        # But the CLI defaults to "/bin/bash", so we explicitly check if it was changed
        shell = args.shell if args.shell != "/bin/bash" else None
        return WasmBackendConfig(
            wasm_path=args.wasm_path,
            shell=shell,  # None means no command mode
            network_enabled=args.network,
            max_memory_mib=args.wasm_max_memory,
            fuel=args.wasm_fuel,
            default_timeout=args.timeout,
        )

    def __init__(self, config: WasmBackendConfig) -> None:
        super().__init__(config)
        # Override parent's _config with more specific type for type checker
        self._config: WasmBackendConfig = config

        # Wasmtime components (created during validation)
        self._engine: wasmtime.Engine | None = None
        self._module: wasmtime.Module | None = None
        self._linker: wasmtime.Linker | None = None

        # Epoch ticker thread
        self._epoch_ticker_stop: threading.Event | None = None
        self._epoch_ticker_thread: threading.Thread | None = None

    def _epoch_ticker_loop(self) -> None:
        """
        Background thread that increments the engine epoch at regular intervals.

        See specs/architecture/wasm_backend.md for more details.
        """
        interval_s = self._config.epoch_tick_interval_ms / 1000.0
        while self._epoch_ticker_stop and not self._epoch_ticker_stop.is_set():
            self._epoch_ticker_stop.wait(interval_s)
            if not self._epoch_ticker_stop.is_set() and self._engine is not None:
                self._engine.increment_epoch()

    async def _validate(self) -> None:
        """Validate WASM prerequisites.

        Checks that wasmtime is installed, the .wasm file exists,
        and compiles the module.
        """
        if wasmtime is None:
            raise BackendError(
                "WASM backend requires the 'wasmtime' package. "
                "Install with: uv add kilntainers[wasm]"
            )

        # Check .wasm file exists
        if not os.path.isfile(self._config.wasm_path):
            raise BackendError(f"WASM file not found: '{self._config.wasm_path}'")

        # Create engine with epoch interruption for timeout enforcement
        wt_config = wasmtime.Config()
        wt_config.epoch_interruption = True
        self._engine = wasmtime.Engine(wt_config)

        # Compile module (expensive — cached for all future execs)
        try:
            self._module = wasmtime.Module.from_file(
                self._engine, self._config.wasm_path
            )
        except wasmtime.WasmtimeError as e:
            raise BackendError(
                f"Failed to compile WASM module '{self._config.wasm_path}': {e}"
            )

        # Create linker with WASI
        self._linker = wasmtime.Linker(self._engine)
        self._linker.define_wasi()

        # Start epoch ticker thread
        self._epoch_ticker_stop = threading.Event()
        self._epoch_ticker_thread = threading.Thread(
            target=self._epoch_ticker_loop,
            daemon=True,
        )
        self._epoch_ticker_thread.start()

    async def _create_sandbox(self) -> "WasmSandbox":
        """Create a WASM sandbox.

        Performs the full startup sequence:
        1. Create base temporary directory
        2. Create sandbox and internal subdirectories
        3. Create sandbox object
        """
        # 1. Create base temporary directory
        base_dir = tempfile.mkdtemp(prefix="kilntainers-wasm-")
        sandbox_dir = os.path.join(base_dir, "sandbox")
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(sandbox_dir)
        os.makedirs(internal_dir)

        # 2. Create sandbox object
        assert self._engine is not None
        assert self._module is not None
        assert self._linker is not None
        sandbox = WasmSandbox(
            engine=self._engine,
            module=self._module,
            linker=self._linker,
            base_dir=base_dir,
            sandbox_dir=sandbox_dir,
            internal_dir=internal_dir,
            config=self._config,
        )

        return sandbox

    def tool_instructions(self) -> str | None:
        """Return tool description for WASM backend.

        The base WasmBackend cannot describe an arbitrary WASM module.
        Returns None — requires --tool-instruction-override.
        """
        return None


class WasmSandbox(Sandbox):
    """WASM sandbox implementation.

    Wraps a temporary directory that serves as the sandbox filesystem.
    Handles command execution with epoch-based timeout enforcement.
    """

    def __init__(
        self,
        *,
        engine: "wasmtime.Engine",
        module: "wasmtime.Module",
        linker: "wasmtime.Linker",
        base_dir: str,
        sandbox_dir: str,
        internal_dir: str,
        config: WasmBackendConfig,
    ) -> None:
        self._engine = engine
        self._module = module
        self._linker = linker
        self._base_dir = base_dir
        self._sandbox_dir = sandbox_dir
        self._internal_dir = internal_dir
        self._config = config
        self._stopped = False
        self._exec_lock = asyncio.Lock()
        self._sandbox_id = os.path.basename(base_dir)

    @property
    def sandbox_id(self) -> str:
        """Return the base directory basename as the sandbox ID."""
        return self._sandbox_id

    def _build_argv(self, request: ExecRequest) -> list[str]:
        """Build the WASI argv for this exec call."""
        # argv[0] is the module name (derived from filename)
        module_name = os.path.splitext(os.path.basename(self._config.wasm_path))[0]

        if request.command is not None:
            # Command mode — requires shell
            if self._config.shell is None:
                raise BackendError(
                    "This WASM backend does not support command mode "
                    "(no shell configured). Use args mode instead, which "
                    "passes arguments directly to the WASM module. "
                    "Example: args=['ls', '-la'] instead of command='ls -la'. "
                    "To enable command mode, run with the "
                    "shell available in the WASM module (e.g., kilntainers --shell bash ...)."
                )
            return [module_name, self._config.shell, "-c", request.command]
        else:
            # Args mode — pass directly
            assert request.args is not None
            return [module_name, *request.args]

    def _run_wasm_sync(
        self,
        *,
        argv: list[str],
        stdout_path: str,
        stderr_path: str,
        stdin_path: str | None,
        timeout: int,
        working_directory: str | None,
    ) -> int:
        """Run the WASM module synchronously. Called from a thread.

        Returns the exit code. Raises _WasmTimeout on epoch deadline.
        """
        # 1. Configure WASI
        assert wasmtime is not None
        wasi_config = wasmtime.WasiConfig()
        wasi_config.argv = tuple(argv)
        wasi_config.stdout_file = stdout_path
        wasi_config.stderr_file = stderr_path
        if stdin_path is not None:
            wasi_config.stdin_file = stdin_path

        # 2. Preopen the sandbox directory
        if working_directory is not None:
            # Resolve working_directory relative to sandbox_dir
            host_dir = os.path.join(self._sandbox_dir, working_directory.lstrip("/"))
            os.makedirs(host_dir, exist_ok=True)
            wasi_config.preopen_dir(host_dir, ".")
        else:
            wasi_config.preopen_dir(self._sandbox_dir, ".")

        # Always preopen full sandbox_dir so the module can access all sandbox files
        wasi_config.preopen_dir(self._sandbox_dir, "/")

        # 3. Create store with epoch deadline and resource limits
        store = wasmtime.Store(self._engine)
        store.set_wasi(wasi_config)

        # Epoch deadline: timeout seconds * (1000ms / tick_interval_ms) ticks
        ticks_per_second = 1000 // self._config.epoch_tick_interval_ms
        store.set_epoch_deadline(timeout * ticks_per_second)

        # Fuel limit (optional)
        if self._config.fuel is not None:
            store.set_fuel(self._config.fuel)

        # Memory limit
        store.set_limits(memory_size=self._config.max_memory_mib * 1024 * 1024)

        # 4. Instantiate and run
        try:
            instance = self._linker.instantiate(store, self._module)
            exports = instance.exports(store)
            start_func = exports["_start"]
            if not isinstance(start_func, wasmtime.Func):
                raise BackendError(
                    "WASM module does not export a '_start' function. "
                    "Only WASI command modules are supported."
                )
            start_func(store)
            return 0  # Normal exit (no proc_exit call)

        except wasmtime.ExitTrap as e:
            return e.code  # WASI proc_exit() was called

        except wasmtime.Trap as e:
            trap_msg = str(e)
            if "epoch" in trap_msg.lower() or "interrupt" in trap_msg.lower():
                raise _WasmTimeout()
            if "fuel" in trap_msg.lower():
                # Fuel exhaustion — treat as timeout-like condition
                raise _WasmTimeout()
            # Other traps (unreachable, memory access, etc.)
            # Write trap info to stderr file and return exit code 1
            with open(stderr_path, "a", encoding="utf-8") as f:
                f.write(f"[kilntainers: WASM trap: {trap_msg}]\n")
            return 1

    def _read_output_file(self, path: str) -> str:
        """Read an output file and return its contents as a string."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except FileNotFoundError:
            return ""
        finally:
            # Clean up I/O temp files after reading
            try:
                os.unlink(path)
            except OSError:
                pass

    async def _do_exec(self, request: ExecRequest) -> ExecResult:
        """Core exec implementation."""
        argv = self._build_argv(request)

        # Create temp files for I/O capture in the internal directory
        stdout_path = os.path.join(self._internal_dir, ".kilntainers_stdout")
        stderr_path = os.path.join(self._internal_dir, ".kilntainers_stderr")
        stdin_path = None

        if request.stdin is not None:
            stdin_path = os.path.join(self._internal_dir, ".kilntainers_stdin")
            with open(stdin_path, "w", encoding="utf-8") as f:
                f.write(request.stdin)

        start_time = time.monotonic()

        try:
            # Dual timeout: epoch interruption (primary, in-thread) +
            # asyncio.wait_for (safety net for blocking WASI calls).
            # See specs/architecture/wasm_backend.md for more details.
            exit_code = await asyncio.wait_for(
                asyncio.to_thread(
                    self._run_wasm_sync,
                    argv=argv,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    stdin_path=stdin_path,
                    timeout=request.timeout,
                    working_directory=request.working_directory,
                ),
                timeout=request.timeout
                + 1,  # safety net: 1s after epoch timeout should fire
            )
        except (_WasmTimeout, asyncio.TimeoutError):
            # _WasmTimeout: epoch fired inside WASM (normal timeout path)
            # asyncio.TimeoutError: epoch didn't fire in time (blocking WASI call)
            elapsed_ms = int((time.monotonic() - start_time) * 1000)
            return ExecResult(
                stdout="",
                stderr=f"[kilntainers: command timed out after {request.timeout}s]",
                exit_code=124,
                exec_duration_ms=elapsed_ms,
            )

        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # Read output files
        stdout_str = self._read_output_file(stdout_path)
        stderr_str = self._read_output_file(stderr_path)

        # Check output limit (post-execution)
        combined_bytes = len(stdout_str.encode("utf-8")) + len(
            stderr_str.encode("utf-8")
        )
        if combined_bytes > request.output_limit:
            return ExecResult(
                stdout="",
                stderr=(
                    f"[kilntainers: output limit exceeded "
                    f"({request.output_limit} bytes). Command terminated. "
                    f"No output returned. Re-run requesting a smaller output size."
                ),
                exit_code=1,
                exec_duration_ms=elapsed_ms,
            )

        return ExecResult(
            stdout=stdout_str,
            stderr=stderr_str,
            exit_code=exit_code,
            exec_duration_ms=elapsed_ms,
        )

    async def exec(self, request: ExecRequest) -> ExecResult:
        """Execute a command in the sandbox.

        Uses a lock to serialize exec calls within this sandbox.
        """
        if self._stopped:
            raise SandboxDiedError("Sandbox has been stopped")

        async with self._exec_lock:
            return await self._do_exec(request)

    async def stop(self) -> None:
        """Stop the sandbox and release all resources.

        Idempotent — safe to call on an already-stopped sandbox.
        """
        if self._stopped:
            return
        self._stopped = True

        # Clean up the base directory (includes sandbox and internal)
        try:
            shutil.rmtree(self._base_dir, ignore_errors=True)
        except Exception:
            pass  # Best-effort cleanup

    async def wait_for_death(self) -> None:
        """Block forever — WASM sandbox cannot die unexpectedly.

        There is no long-running process that could crash, no container
        that could be OOM-killed, and no remote service that could
        terminate. The sandbox is "alive" as long as the temp directory
        exists and stop() hasn't been called.
        """
        try:
            await asyncio.Future()  # Never completes
        except asyncio.CancelledError:
            return  # Normal shutdown — MCP layer cancelled this task


def _get_bundled_busybox_path() -> str:
    """Return the path to the bundled busybox.wasm binary."""
    ref = importlib.resources.files("kilntainers.backends.wasm_data") / "busybox.wasm"
    # as_posix() works for file-system-backed resources
    return str(ref)


class GoBusyBoxBackend(WasmBackend):
    """WASM backend with bundled go-busybox multicall binary.

    Provides a zero-config POSIX-like sandbox with common shell
    utilities (ls, cat, grep, sed, awk, etc.) via go-busybox
    compiled to WebAssembly.

    Usage: kilntainers --backend go_busybox
    """

    def __init__(self, config: WasmBackendConfig) -> None:
        super().__init__(config)

    @classmethod
    def add_cli_arguments(cls, group: argparse._ArgumentGroup) -> None:
        """Register GoBusyBox-specific CLI arguments (none).

        Note: GoBusyBox inherits WASM resource limit args from WasmBackend,
        so we don't re-register --wasm-max-memory and --wasm-fuel here.
        The CLI will call WasmBackend.add_cli_arguments separately.
        """
        # GoBusyBox doesn't need --wasm-path (it's bundled)
        # Resource limits are registered by WasmBackend.add_cli_arguments

    @classmethod
    def config_from_args(cls, args: argparse.Namespace) -> BackendConfig:
        """Build WasmBackendConfig from parsed CLI arguments."""
        # GoBusyBox uses args mode only (no shell)
        return WasmBackendConfig(
            wasm_path=_get_bundled_busybox_path(),
            shell=None,  # No command mode - args only
            network_enabled=args.network,
            max_memory_mib=args.wasm_max_memory,
            fuel=args.wasm_fuel,
            default_timeout=args.timeout,
        )

    def tool_instructions(self) -> str | None:
        """Return tool description for GoBusyBox backend."""
        return (
            "Execute a command in a lightweight sandbox (WebAssembly, Busybox style). "
            "This is a multicall binary that provides common Unix utilities directly.\n\n"
            "Use the 'args' parameter to specify commands. The first arg is "
            "the command name, followed by its arguments. Examples:\n"
            "  - args=['ls', '-la']\n"
            "  - args=['grep', '-r', 'pattern', '.']\n"
            "  - args=['cat', 'file.txt']\n\n"
            "Available commands: ls, cat, grep, sed, awk, sort, uniq, wc, cut, tr, diff, "
            "head, tail, find, mkdir, rmdir, rm, cp, mv, echo, sh, xargs, sleep, "
            "time, pwd, nc, wget, ss, dig, and more. Run args=['--list'] for the full list.\n\n"
            "This is not a full Linux environment: many commands may be missing or simplified."
            "You should not assume package managers, daemons, or full GNU behavior."
            "There is no interactive shell (no bash, no pipes |, no redirection >, no shell globbing). "
            "Filesystem will persist between calls within a session, but not long term. "
        )

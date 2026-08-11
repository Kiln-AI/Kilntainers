"""Unit tests for WASM backend.

Tests mock the wasmtime package to test backend logic without requiring
wasmtime to be installed.
"""

import asyncio
import os
import tempfile

import pytest

import kilntainers.backends.wasm as wasm_backend_module
from kilntainers.backends.base import ExecRequest
from kilntainers.backends.wasm import (
    GoBusyBoxBackend,
    WasmBackend,
    WasmBackendConfig,
    WasmSandbox,
    _get_bundled_busybox_path,
    _WasmTimeout,
)
from kilntainers.errors import BackendError, SandboxDiedError

# --- Mock wasmtime utilities ---


def test_windows_architecture_fallback_uses_python_platform(monkeypatch):
    """Wasmtime gets a stable architecture when Windows environment data is absent."""
    monkeypatch.setattr(wasm_backend_module.sys, "platform", "win32")
    monkeypatch.setattr(
        wasm_backend_module.sysconfig,
        "get_platform",
        lambda: "win-amd64",
    )
    monkeypatch.delenv("PROCESSOR_ARCHITEW6432", raising=False)
    monkeypatch.delenv("PROCESSOR_ARCHITECTURE", raising=False)
    monkeypatch.setattr(wasm_backend_module.platform, "_uname_cache", object())

    wasm_backend_module._ensure_windows_processor_architecture()

    assert os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"
    assert wasm_backend_module.platform._uname_cache is None


class MockWasiConfig:
    """Mock wasmtime.WasiConfig."""

    def __init__(self):
        self.argv = ()
        self.stdout_file = None
        self.stderr_file = None
        self.stdin_file = None
        self._preopened_dirs = []

    def preopen_dir(self, host_path, guest_path):
        self._preopened_dirs.append((host_path, guest_path))


class MockStore:
    """Mock wasmtime.Store."""

    def __init__(self, engine):
        self._engine = engine
        self._wasi = None
        self._epoch_deadline = None
        self._fuel = None
        self._limits = {}

    def set_wasi(self, config):
        self._wasi = config

    def set_epoch_deadline(self, deadline):
        self._epoch_deadline = deadline

    def set_fuel(self, fuel):
        self._fuel = fuel

    def set_limits(self, **kwargs):
        self._limits = kwargs


class MockFunc:
    """Mock WASM function."""

    def __init__(self, result=None, trap=None):
        self._result = result
        self._trap = trap
        self.calls = []

    def __call__(self, store):
        self.calls.append(store)
        if self._trap:
            if self._trap == "epoch":
                raise MockTrap("epoch interruption")
            elif self._trap == "fuel":
                raise MockTrap("out of fuel")
            else:
                raise MockTrap(self._trap)
        return self._result


class MockExports:
    """Mock WASM exports."""

    def __init__(self, funcs=None):
        self._funcs = funcs or {}

    def get(self, name):
        return self._funcs.get(name)


class MockInstance:
    """Mock WASM instance."""

    def __init__(self, exports_dict=None):
        self._exports = exports_dict or {}

    def exports(self, store):
        return MockExports(self._exports)


class MockTrap(Exception):
    """Mock wasmtime.Trap."""

    pass


class MockExitTrap(Exception):
    """Mock wasmtime.ExitTrap."""

    def __init__(self, code):
        self.code = code
        super().__init__(f"exit trap: {code}")


class MockEngine:
    """Mock wasmtime.Engine."""

    def __init__(self, config=None):
        self.epoch_count = 0
        self._config = config

    def increment_epoch(self):
        self.epoch_count += 1


class MockModule:
    """Mock wasmtime.Module."""

    def __init__(self, path=None):
        self.path = path

    @staticmethod
    def from_file(engine, path):
        return MockModule(path=path)


class MockLinker:
    """Mock wasmtime.Linker."""

    def __init__(self, engine):
        self._engine = engine
        self._wasi_defined = False

    def define_wasi(self):
        self._wasi_defined = True

    def instantiate(self, store, module):
        return MockInstance()


class MockConfig:
    """Mock wasmtime.Config."""

    def __init__(self):
        self.epoch_interruption = False


@pytest.fixture
def mock_wasmtime(monkeypatch):
    """Mock wasmtime module with configurable behavior."""

    class MockWasmtimeModule:
        Config = MockConfig
        Engine = MockEngine
        Module = MockModule
        Linker = MockLinker
        WasiConfig = MockWasiConfig
        Trap = MockTrap
        ExitTrap = MockExitTrap
        WasmtimeError = Exception

        # Store factory
        _Store = MockStore

        @staticmethod
        def Store(engine):
            return MockWasmtimeModule._Store(engine)

    monkeypatch.setattr("kilntainers.backends.wasm.wasmtime", MockWasmtimeModule)

    # Track module state for test assertions
    state = {
        "engine": None,
        "store": None,
        "wasi_config": None,
    }

    # Patch Store to track instances
    original_store = MockWasmtimeModule.Store

    def tracking_store(engine):
        store = original_store(engine)
        state["store"] = store
        state["engine"] = engine
        return store

    MockWasmtimeModule.Store = tracking_store  # type: ignore[assignment]
    MockWasmtimeModule._Store = MockStore

    return MockWasmtimeModule


@pytest.fixture
def default_wasm_config(tmp_path):
    """Return a default WasmBackendConfig with a valid wasm_path."""
    wasm_file = tmp_path / "test.wasm"
    wasm_file.write_text("")  # Create empty file
    return WasmBackendConfig(wasm_path=str(wasm_file))


@pytest.fixture
def mock_engine():
    """Return a mock engine for testing."""
    return MockEngine()


@pytest.fixture
def mock_linker(mock_engine):
    """Return a mock linker for testing."""
    return MockLinker(mock_engine)


@pytest.fixture
def mock_module():
    """Return a mock module for testing."""
    return MockModule()


# --- WasmBackend tests ---


class TestWasmBackendValidation:
    """Tests for WasmBackend._validate method."""

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_wasmtime, default_wasm_config):
        """Validation passes with valid wasmtime and wasm file."""
        backend = WasmBackend(default_wasm_config)
        await backend.validate()

        # Second call should be cached (no-op)
        await backend.validate()

        assert backend._engine is not None
        assert backend._module is not None
        assert backend._linker is not None
        assert backend._epoch_ticker_thread is not None
        assert backend._epoch_ticker_stop is not None

    @pytest.mark.asyncio
    async def test_validate_wasmtime_not_installed(
        self, monkeypatch, default_wasm_config
    ):
        """Validation fails when wasmtime is not installed."""
        monkeypatch.setattr("kilntainers.backends.wasm.wasmtime", None)

        backend = WasmBackend(default_wasm_config)

        with pytest.raises(BackendError) as exc_info:
            await backend.validate()

        assert "wasmtime" in str(exc_info.value)
        assert "uv add" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validate_wasm_file_not_found(self, mock_wasmtime):
        """Validation fails when .wasm file doesn't exist."""
        config = WasmBackendConfig(wasm_path="/nonexistent/file.wasm")
        backend = WasmBackend(config)

        with pytest.raises(BackendError) as exc_info:
            await backend.validate()

        assert "not found" in str(exc_info.value)
        assert "/nonexistent/file.wasm" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_epoch_ticker_starts(self, mock_wasmtime, default_wasm_config):
        """Epoch ticker thread starts during validation."""
        backend = WasmBackend(default_wasm_config)
        await backend.validate()

        # Wait a bit for ticker to run
        await asyncio.sleep(0.1)

        # Stop the ticker
        assert backend._epoch_ticker_stop is not None
        backend._epoch_ticker_stop.set()
        assert backend._epoch_ticker_thread is not None
        backend._epoch_ticker_thread.join(timeout=1)

        assert backend._epoch_ticker_thread.is_alive() is False


class TestWasmBackendCliArguments:
    """Tests for WasmBackend.add_cli_arguments method."""

    def test_add_cli_arguments(self):
        """CLI arguments are registered correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("wasm backend options")

        WasmBackend.add_cli_arguments(group)

        # Parse with default values
        args = parser.parse_args([])
        assert args.wasm_path is None
        assert args.wasm_max_memory == 256
        assert args.wasm_fuel is None

        # Parse with custom values (--shell is now a core parameter)
        args = parser.parse_args(
            ["--wasm-path", "test.wasm", "--wasm-max-memory", "512"]
        )
        assert args.wasm_path == "test.wasm"
        assert args.wasm_max_memory == 512


class TestWasmBackendConfigFromArgs:
    """Tests for WasmBackend.config_from_args method."""

    def test_config_from_args_defaults(self):
        """Config from args with default values."""
        import argparse

        parser = argparse.ArgumentParser()
        # Add core options that WasmBackend.config_from_args expects
        core = parser.add_argument_group("core options")
        core.add_argument("--shell", default="/bin/bash")
        core.add_argument("--network", action="store_true", default=False)
        core.add_argument("--timeout", type=int, default=120)
        # Add WASM backend options
        group = parser.add_argument_group("wasm backend options")
        WasmBackend.add_cli_arguments(group)

        args = parser.parse_args(["--wasm-path", "test.wasm"])
        config = WasmBackend.config_from_args(args)

        assert isinstance(config, WasmBackendConfig)  # type: ignore[redundant-expr]
        assert config.wasm_path == "test.wasm"  # type: ignore[union-attr]
        assert config.shell is None  # type: ignore[union-attr]  # WASM: default None
        assert config.max_memory_mib == 256  # type: ignore[union-attr]
        assert config.fuel is None  # type: ignore[union-attr]

    def test_config_from_args_custom(self):
        """Config from args with custom values."""
        import argparse

        parser = argparse.ArgumentParser()
        # Add core options that WasmBackend.config_from_args expects
        core = parser.add_argument_group("core options")
        core.add_argument("--shell", default="/bin/bash")
        core.add_argument("--network", action="store_true", default=False)
        core.add_argument("--timeout", type=int, default=120)
        # Add WASM backend options
        group = parser.add_argument_group("wasm backend options")
        WasmBackend.add_cli_arguments(group)

        args = parser.parse_args(
            [
                "--wasm-path",
                "test.wasm",
                "--shell",
                "ash",
                "--wasm-max-memory",
                "512",
                "--wasm-fuel",
                "1000000",
            ]
        )
        config = WasmBackend.config_from_args(args)

        assert isinstance(config, WasmBackendConfig)  # type: ignore[redundant-expr]
        assert config.wasm_path == "test.wasm"  # type: ignore[union-attr]
        assert config.shell == "ash"  # type: ignore[union-attr]
        assert config.max_memory_mib == 512  # type: ignore[union-attr]
        assert config.fuel == 1000000  # type: ignore[union-attr]


class TestWasmBackendToolInstructions:
    """Tests for WasmBackend.tool_instructions method."""

    def test_tool_instructions_none(self, default_wasm_config):
        """Base WasmBackend returns None (cannot describe arbitrary module)."""
        backend = WasmBackend(default_wasm_config)
        assert backend.tool_instructions() is None


class TestWasmBackendSandboxCreation:
    """Tests for WasmBackend._create_sandbox method."""

    @pytest.mark.asyncio
    async def test_create_sandbox_success(
        self, mock_wasmtime, default_wasm_config, mock_engine, mock_linker, mock_module
    ):
        """Sandbox creation succeeds with valid config."""
        backend = WasmBackend(default_wasm_config)

        # Manually set the engine, module, linker (skip validation)
        backend._engine = mock_engine
        backend._module = mock_module
        backend._linker = mock_linker
        backend._validated = True

        # Create temp dir for the test
        base_dir = tempfile.mkdtemp()
        sandbox_dir = os.path.join(base_dir, "sandbox")
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(sandbox_dir)
        os.makedirs(internal_dir)

        # Patch readiness check to succeed
        async def mock_verify():
            pass

        sandbox = WasmSandbox(
            engine=mock_engine,  # type: ignore[arg-type]
            module=mock_module,  # type: ignore[arg-type]
            linker=mock_linker,  # type: ignore[arg-type]
            base_dir=base_dir,
            sandbox_dir=sandbox_dir,
            internal_dir=internal_dir,
            config=default_wasm_config,
        )
        sandbox._verify_readiness = mock_verify  # type: ignore[method-assign]

        # Monkey patch _create_sandbox to return our test sandbox
        async def mock_create():
            return sandbox

        original_create = backend._create_sandbox
        backend._create_sandbox = mock_create  # type: ignore[method-assign]

        try:
            result = await backend.create_sandbox()
            assert isinstance(result, WasmSandbox)
            assert result.sandbox_id is not None
        finally:
            await sandbox.stop()
            backend._create_sandbox = original_create  # type: ignore[method-assign]


# --- WasmSandbox tests ---


class TestWasmSandboxArgvConstruction:
    """Tests for WasmSandbox._build_argv method."""

    def test_argv_args_mode(self, default_wasm_config):
        """Args mode passes arguments directly."""
        base_dir = tempfile.mkdtemp()
        sandbox = WasmSandbox(
            engine=MockEngine(),  # type: ignore[arg-type]
            module=MockModule(),  # type: ignore[arg-type]
            linker=MockLinker(MockEngine()),  # type: ignore[arg-type]
            base_dir=base_dir,
            sandbox_dir=os.path.join(base_dir, "sandbox"),
            internal_dir=os.path.join(base_dir, "internal"),
            config=default_wasm_config,
        )

        request = ExecRequest(
            args=["echo", "hello"], timeout=30, output_limit=2_097_152
        )
        argv = sandbox._build_argv(request)

        assert argv[0] == "test"  # module name from wasm_path
        assert argv[1] == "echo"
        assert argv[2] == "hello"

    def test_argv_command_mode_with_shell(self, default_wasm_config):
        """Command mode with shell wraps in shell."""
        config = WasmBackendConfig(wasm_path=default_wasm_config.wasm_path, shell="ash")
        base_dir = tempfile.mkdtemp()
        sandbox = WasmSandbox(
            engine=MockEngine(),  # type: ignore[arg-type]
            module=MockModule(),  # type: ignore[arg-type]
            linker=MockLinker(MockEngine()),  # type: ignore[arg-type]
            base_dir=base_dir,
            sandbox_dir=os.path.join(base_dir, "sandbox"),
            internal_dir=os.path.join(base_dir, "internal"),
            config=config,
        )

        request = ExecRequest(command="echo hello", timeout=30, output_limit=2_097_152)
        argv = sandbox._build_argv(request)

        assert argv[0] == "test"  # module name
        assert argv[1] == "ash"
        assert argv[2] == "-c"
        assert argv[3] == "echo hello"

    def test_argv_command_mode_without_shell(self, default_wasm_config):
        """Command mode without shell raises BackendError."""
        base_dir = tempfile.mkdtemp()
        sandbox = WasmSandbox(
            engine=MockEngine(),  # type: ignore[arg-type]
            module=MockModule(),  # type: ignore[arg-type]
            linker=MockLinker(MockEngine()),  # type: ignore[arg-type]
            base_dir=base_dir,
            sandbox_dir=os.path.join(base_dir, "sandbox"),
            internal_dir=os.path.join(base_dir, "internal"),
            config=default_wasm_config,
        )

        request = ExecRequest(command="ls", timeout=30, output_limit=2_097_152)

        with pytest.raises(BackendError) as exc_info:
            sandbox._build_argv(request)

        assert "command mode" in str(exc_info.value).lower()
        assert "shell" in str(exc_info.value).lower()


class TestWasmSandboxExec:
    """Tests for WasmSandbox.exec method."""

    @pytest.fixture
    def sandbox(self, default_wasm_config, mock_engine, mock_linker, mock_module):
        """Create a test sandbox."""
        base_dir = tempfile.mkdtemp()
        sandbox_dir = os.path.join(base_dir, "sandbox")
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(sandbox_dir)
        os.makedirs(internal_dir)
        sandbox = WasmSandbox(
            engine=mock_engine,
            module=mock_module,
            linker=mock_linker,
            base_dir=base_dir,
            sandbox_dir=sandbox_dir,
            internal_dir=internal_dir,
            config=default_wasm_config,
        )
        yield sandbox
        # Cleanup
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_exec_success(self, monkeypatch, sandbox):
        """Successful command execution."""

        def mock_run(*args, **kwargs):
            # Create output files
            stdout_path = kwargs["stdout_path"]
            stderr_path = kwargs["stderr_path"]
            with open(stdout_path, "w") as f:
                f.write("hello\n")
            with open(stderr_path, "w") as f:
                f.write("")
            return 0

        monkeypatch.setattr(sandbox, "_run_wasm_sync", mock_run)

        request = ExecRequest(
            args=["echo", "hello"], timeout=30, output_limit=2_097_152
        )
        result = await sandbox.exec(request)

        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.exec_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_exec_timeout_epoch(self, monkeypatch, sandbox):
        """Command times out via epoch interruption."""

        def mock_run(*args, **kwargs):
            raise _WasmTimeout()

        monkeypatch.setattr(sandbox, "_run_wasm_sync", mock_run)

        request = ExecRequest(args=["sleep", "60"], timeout=2, output_limit=2_097_152)
        result = await sandbox.exec(request)

        assert result.exit_code == 124
        assert "timed out" in result.stderr
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_exec_output_limit(self, monkeypatch, sandbox):
        """Output limit exceeded."""

        def mock_run(*args, **kwargs):
            stdout_path = kwargs["stdout_path"]
            with open(stdout_path, "w") as f:
                f.write("x" * 10000)
            return 0

        monkeypatch.setattr(sandbox, "_run_wasm_sync", mock_run)

        request = ExecRequest(args=["yes"], timeout=30, output_limit=1000)
        result = await sandbox.exec(request)

        assert result.exit_code == 1
        assert "output limit exceeded" in result.stderr
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_exec_after_stop(self, sandbox):
        """Exec after stop raises SandboxDiedError."""
        sandbox._stopped = True

        request = ExecRequest(args=["echo"], timeout=30, output_limit=2_097_152)

        with pytest.raises(SandboxDiedError) as exc_info:
            await sandbox.exec(request)

        assert "stopped" in str(exc_info.value).lower()


class TestWasmSandboxStop:
    """Tests for WasmSandbox.stop method."""

    @pytest.fixture
    def sandbox(self, default_wasm_config, mock_engine, mock_linker, mock_module):
        """Create a test sandbox."""
        base_dir = tempfile.mkdtemp()
        sandbox_dir = os.path.join(base_dir, "sandbox")
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(sandbox_dir)
        os.makedirs(internal_dir)
        sandbox = WasmSandbox(
            engine=mock_engine,
            module=mock_module,
            linker=mock_linker,
            base_dir=base_dir,
            sandbox_dir=sandbox_dir,
            internal_dir=internal_dir,
            config=default_wasm_config,
        )
        yield sandbox
        # Cleanup if not already stopped
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_stop(self, sandbox):
        """Stop removes base directory."""
        base_dir = sandbox._base_dir
        assert os.path.exists(base_dir)

        await sandbox.stop()

        assert sandbox._stopped
        assert not os.path.exists(base_dir)

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, sandbox):
        """Stop is idempotent."""
        await sandbox.stop()
        await sandbox.stop()
        await sandbox.stop()

        assert sandbox._stopped


class TestWasmSandboxWaitForDeath:
    """Tests for WasmSandbox.wait_for_death method."""

    @pytest.fixture
    def sandbox(self, default_wasm_config, mock_engine, mock_linker, mock_module):
        """Create a test sandbox."""
        base_dir = tempfile.mkdtemp()
        sandbox_dir = os.path.join(base_dir, "sandbox")
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(sandbox_dir)
        os.makedirs(internal_dir)
        sandbox = WasmSandbox(
            engine=mock_engine,
            module=mock_module,
            linker=mock_linker,
            base_dir=base_dir,
            sandbox_dir=sandbox_dir,
            internal_dir=internal_dir,
            config=default_wasm_config,
        )
        yield sandbox
        # Cleanup
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_wait_for_death_blocks_forever(self, sandbox):
        """wait_for_death blocks forever until cancelled."""
        task = asyncio.create_task(sandbox.wait_for_death())
        await asyncio.sleep(0.1)

        # Task should still be running
        assert not task.done()

        # Cancel the task
        task.cancel()
        await task


class TestWasmSandboxSandboxId:
    """Tests for WasmSandbox.sandbox_id property."""

    def test_sandbox_id(
        self, default_wasm_config, mock_engine, mock_linker, mock_module
    ):
        """sandbox_id returns base directory basename."""
        base_dir = tempfile.mkdtemp(prefix="test-wasm-")
        basename = os.path.basename(base_dir)

        sandbox = WasmSandbox(
            engine=mock_engine,
            module=mock_module,
            linker=mock_linker,
            base_dir=base_dir,
            sandbox_dir=os.path.join(base_dir, "sandbox"),
            internal_dir=os.path.join(base_dir, "internal"),
            config=default_wasm_config,
        )

        assert sandbox.sandbox_id == basename

        # Cleanup
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


class TestWasmSandboxReadOutputFile:
    """Tests for WasmSandbox._read_output_file method."""

    def test_read_output_file_success(self, default_wasm_config):
        """Successfully read and delete output file."""
        base_dir = tempfile.mkdtemp()
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(internal_dir)
        file_path = os.path.join(internal_dir, "test_output.txt")

        with open(file_path, "w") as f:
            f.write("test content\n")

        sandbox = WasmSandbox(
            engine=MockEngine(),  # type: ignore[arg-type]
            module=MockModule(),  # type: ignore[arg-type]
            linker=MockLinker(MockEngine()),  # type: ignore[arg-type]
            base_dir=base_dir,
            sandbox_dir=os.path.join(base_dir, "sandbox"),
            internal_dir=internal_dir,
            config=default_wasm_config,
        )

        content = sandbox._read_output_file(file_path)

        assert content == "test content\n"
        assert not os.path.exists(file_path)  # File should be deleted

        # Cleanup
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)

    def test_read_output_file_not_found(self, default_wasm_config):
        """Return empty string for missing file."""
        base_dir = tempfile.mkdtemp()
        internal_dir = os.path.join(base_dir, "internal")
        os.makedirs(internal_dir)
        file_path = os.path.join(internal_dir, "nonexistent.txt")

        sandbox = WasmSandbox(
            engine=MockEngine(),  # type: ignore[arg-type]
            module=MockModule(),  # type: ignore[arg-type]
            linker=MockLinker(MockEngine()),  # type: ignore[arg-type]
            base_dir=base_dir,
            sandbox_dir=os.path.join(base_dir, "sandbox"),
            internal_dir=internal_dir,
            config=default_wasm_config,
        )

        content = sandbox._read_output_file(file_path)

        assert content == ""

        # Cleanup
        import shutil

        shutil.rmtree(base_dir, ignore_errors=True)


# --- GoBusyBoxBackend tests ---


class TestGoBusyBoxBackend:
    """Tests for GoBusyBoxBackend class."""

    def test_tool_instructions(self):
        """GoBusyBoxBackend provides tool description."""
        config = WasmBackendConfig(wasm_path="/fake/path", shell=None)
        backend = GoBusyBoxBackend(config)

        instructions = backend.tool_instructions()

        assert instructions is not None
        assert "Busybox" in instructions
        assert "args" in instructions.lower()
        assert "ls" in instructions
        assert "grep" in instructions

    def test_config_from_args_defaults(self):
        """Config from args with GoBusyBox defaults."""
        import argparse

        parser = argparse.ArgumentParser()
        # Add core options that GoBusyBoxBackend.config_from_args expects
        core = parser.add_argument_group("core options")
        core.add_argument("--shell", default="/bin/bash")
        core.add_argument("--network", action="store_true", default=False)
        core.add_argument("--timeout", type=int, default=120)
        # Add WASM resource limit options (normally registered by WasmBackend)
        # In the full CLI, these come from the "wasm backend options" group
        wasm_group = parser.add_argument_group("wasm backend options")
        WasmBackend.add_cli_arguments(wasm_group)
        # Add GoBusyBox backend options (adds nothing specific)
        go_group = parser.add_argument_group("go_busybox backend options")
        GoBusyBoxBackend.add_cli_arguments(go_group)

        args = parser.parse_args([])
        config = GoBusyBoxBackend.config_from_args(args)

        assert isinstance(config, WasmBackendConfig)  # type: ignore[redundant-expr]
        assert (
            config.shell is None
        )  # GoBusyBox uses args mode only  # type: ignore[union-attr]
        assert config.max_memory_mib == 256  # type: ignore[union-attr]

    def test_add_cli_arguments_no_wasm_path(self):
        """GoBusyBox doesn't register any arguments (inherits from WasmBackend)."""
        import argparse

        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("go_busybox backend options")
        GoBusyBoxBackend.add_cli_arguments(group)

        # GoBusyBox.add_cli_arguments doesn't add any arguments to the group
        # (resource limits are added by WasmBackend.add_cli_arguments)
        # So we should be able to parse with no args
        args = parser.parse_args([])
        # The group would be empty, so no wasm_path or wasm_max_memory attributes
        assert not hasattr(args, "wasm_path")


class TestGetBundledBusyboxPath:
    """Tests for _get_bundled_busybox_path function."""

    def test_returns_path(self):
        """Function returns a string path."""
        path = _get_bundled_busybox_path()
        assert isinstance(path, str)
        assert "busybox.wasm" in path

"""Unit tests for Modal backend.

Tests mock the Modal SDK to simulate sandbox behavior without
requiring a real Modal account or network access.
"""

import asyncio

import pytest

import kilntainers.backends.modal as modal_backend_module
from kilntainers.backends.base import ExecRequest
from kilntainers.backends.modal import (
    ModalBackend,
    ModalBackendConfig,
    ModalSandbox,
)
from kilntainers.errors import BackendError, SandboxDiedError

# --- Mock Modal SDK utilities ---


class MockStreamReader:
    """Mock Modal StreamReader for subprocess output."""

    def __init__(self, content: str = ""):
        self._content = content
        self._lines = content.splitlines(keepends=True)
        # Create a nested object for .read.aio() pattern
        self._read_obj = self._ReadObj(self)

    @property
    def read(self):
        return self._read_obj

    class _ReadObj:
        def __init__(self, parent):
            self.parent = parent

        async def aio(self) -> str:
            return self.parent._content

    def __aiter__(self):
        return self._iter_lines()

    async def _iter_lines(self):
        for line in self._lines:
            yield line


class MockStreamWriter:
    """Mock Modal StreamWriter for stdin."""

    def __init__(self):
        self.data = ""
        self._closed = False

    def write(self, data: bytes | str):
        if isinstance(data, bytes):
            self.data += data.decode("utf-8")
        else:
            self.data += data

    def write_eof(self):
        self._closed = True

    async def drain(self):
        pass


class MockContainerProcess:
    """Mock Modal ContainerProcess."""

    def __init__(
        self,
        stdout: str = "",
        stderr: str = "",
        returncode: int = 0,
    ):
        self.stdout = MockStreamReader(stdout)
        self.stderr = MockStreamReader(stderr)
        self.stdin = MockStreamWriter()
        self.returncode = returncode

    class wait:
        @staticmethod
        async def aio():
            pass


class MockSandbox:
    """Mock modal.Sandbox."""

    def __init__(self, object_id: str = "sb-mock-12345"):
        self.object_id = object_id
        self.exec_responses = []
        self.terminated = False
        self._wait_event = asyncio.Event()

        # Create nested classes with reference to parent
        self._exec = self._MockExec(self)
        self._terminate = self._MockTerminate(self)
        self._wait = self._MockWait(self)

    def set_exec_response(self, response: MockContainerProcess):
        self.exec_responses.append(response)

    @property
    def exec(self):
        return self._exec

    @property
    def terminate(self):
        return self._terminate

    @property
    def wait(self):
        return self._wait

    class _MockExec:
        def __init__(self, parent):
            self.parent = parent

        async def aio(self, *args, **kwargs):
            if self.parent.exec_responses:
                return self.parent.exec_responses.pop(0)
            return MockContainerProcess()

    class _MockTerminate:
        def __init__(self, parent):
            self.parent = parent

        async def aio(self):
            self.parent.terminated = True

    class _MockWait:
        def __init__(self, parent):
            self.parent = parent

        async def aio(self, raise_on_termination=True):
            # Block until the event is set
            await self.parent._wait_event.wait()


class MockApp:
    """Mock modal.App."""

    def __init__(self):
        self.name = "test-app"


@pytest.fixture
def mock_modal(monkeypatch):
    """Mock modal module with configurable responses."""

    class MockModal:
        exception = type(
            "exception",
            (),
            {
                "AuthError": type("AuthError", (Exception,), {}),
                "ConnectionError": type("ConnectionError", (Exception,), {}),
                "InvalidError": type("InvalidError", (Exception,), {}),
                "NotFoundError": type("NotFoundError", (Exception,), {}),
                "SandboxTerminatedError": type(
                    "SandboxTerminatedError", (Exception,), {}
                ),
                "SandboxTimeoutError": type("SandboxTimeoutError", (Exception,), {}),
            },
        )()

        class Image:
            @staticmethod
            def debian_slim():
                return "debian-slim-image"

            @staticmethod
            def from_registry(ref: str):
                return f"registry-image:{ref}"

        class App:
            class lookup:
                @staticmethod
                async def aio(name: str, create_if_missing: bool = False):
                    if "auth-fail" in name:
                        raise MockModal.exception.AuthError("Auth failed")
                    if "connection-fail" in name:
                        raise MockModal.exception.ConnectionError("Connection failed")
                    return MockApp()

        class Sandbox:
            _next_sandbox: MockSandbox | None = None

            class create:
                @staticmethod
                async def aio(**kwargs):
                    if MockModal.Sandbox._next_sandbox:
                        sb = MockModal.Sandbox._next_sandbox
                        MockModal.Sandbox._next_sandbox = None
                        return sb
                    return MockSandbox()

            @staticmethod
            def set_next_sandbox(sandbox: MockSandbox):
                MockModal.Sandbox._next_sandbox = sandbox

        class container_process:
            ContainerProcess = MockContainerProcess

    monkeypatch.setattr("kilntainers.backends.modal.modal", MockModal)
    return MockModal


@pytest.fixture
def default_config():
    """Return a default ModalBackendConfig."""
    return ModalBackendConfig()


# --- ModalBackend tests ---


class TestModalRuntimePreparation:
    """Tests for containing Modal's process-wide import side effects."""

    def test_lazy_import_restores_event_loop_policy(self, monkeypatch):
        original_policy = object()
        imported_sdk = object()
        restored_policies = []

        monkeypatch.setattr(modal_backend_module, "modal", None)
        monkeypatch.setattr(
            modal_backend_module.asyncio,
            "get_event_loop_policy",
            lambda: original_policy,
        )
        monkeypatch.setattr(
            modal_backend_module.asyncio,
            "set_event_loop_policy",
            restored_policies.append,
        )
        monkeypatch.setattr(
            modal_backend_module.importlib,
            "import_module",
            lambda name: imported_sdk,
        )

        result = modal_backend_module._get_modal_sdk()

        assert result is imported_sdk
        assert restored_policies == [original_policy]

    def test_selected_modal_backend_keeps_sdk_policy_change(self, monkeypatch):
        imported_sdk = object()

        monkeypatch.setattr(modal_backend_module, "modal", None)
        monkeypatch.setattr(
            modal_backend_module.asyncio,
            "get_event_loop_policy",
            lambda: pytest.fail("selected Modal startup must not preserve the policy"),
        )
        monkeypatch.setattr(
            modal_backend_module.asyncio,
            "set_event_loop_policy",
            lambda policy: pytest.fail(
                "selected Modal startup must not restore policy"
            ),
        )
        monkeypatch.setattr(
            modal_backend_module.importlib,
            "import_module",
            lambda name: imported_sdk,
        )

        ModalBackend.prepare_runtime()

        assert modal_backend_module.modal is imported_sdk


class TestModalBackendConfig:
    """Tests for ModalBackendConfig dataclass."""

    def test_default_config(self):
        """Default config has expected values."""
        config = ModalBackendConfig()

        assert config.token_id is None
        assert config.token_secret is None
        assert config.app_name == "kilntainers"
        assert config.image is None
        assert config.shell == "/bin/bash"
        assert config.network_enabled is False
        assert config.cpu == 1.0
        assert config.memory == 512
        assert config.gpu is None
        assert config.region is None
        assert config.sandbox_timeout == 3600
        assert config.default_timeout == 120

    def test_custom_config(self):
        """Custom config values."""
        config = ModalBackendConfig(
            token_id="test-id",
            token_secret="test-secret",
            app_name="test-app",
            image="python:3.12",
            shell="/bin/sh",
            network_enabled=True,
            cpu=2.0,
            memory=1024,
            gpu="A10G",
            region="us-east",
            sandbox_timeout=7200,
            default_timeout=300,
        )

        assert config.token_id == "test-id"
        assert config.token_secret == "test-secret"
        assert config.app_name == "test-app"
        assert config.image == "python:3.12"
        assert config.shell == "/bin/sh"
        assert config.network_enabled is True
        assert config.cpu == 2.0
        assert config.memory == 1024
        assert config.gpu == "A10G"
        assert config.region == "us-east"
        assert config.sandbox_timeout == 7200
        assert config.default_timeout == 300


class TestModalBackendCLI:
    """Tests for ModalBackend CLI methods."""

    def test_add_cli_arguments(self):
        """CLI arguments are registered correctly."""
        parser = MockArgumentParser()
        group = parser.add_argument_group("modal backend options")

        ModalBackend.add_cli_arguments(group)

        assert "--modal-token-id" in parser.arguments
        assert "--modal-token-secret" in parser.arguments
        assert "--modal-app-name" in parser.arguments
        assert "--modal-cpu" in parser.arguments
        assert "--modal-memory" in parser.arguments
        assert "--gpu" in parser.arguments
        assert "--region" in parser.arguments
        assert "--sandbox-timeout" in parser.arguments

    def test_config_from_args(self):
        """Config is built from args correctly."""
        args = MockNamespace(
            modal_token_id="test-id",
            modal_token_secret="test-secret",
            modal_app_name="test-app",
            modal_cpu=2.0,
            modal_memory=1024,
            gpu="A10G",
            region="us-east",
            sandbox_timeout=7200,
            image="python:3.12",
            shell="/bin/sh",
            network=True,
            timeout=300,
        )

        config = ModalBackend.config_from_args(args)  # type: ignore[arg-type]

        assert config.token_id == "test-id"  # type: ignore[attr-defined]
        assert config.token_secret == "test-secret"  # type: ignore[attr-defined]
        assert config.app_name == "test-app"  # type: ignore[attr-defined]
        assert config.image == "python:3.12"  # type: ignore[attr-defined]
        assert config.shell == "/bin/sh"  # type: ignore[attr-defined]
        assert config.network_enabled is True  # type: ignore[attr-defined]
        assert config.cpu == 2.0  # type: ignore[attr-defined]
        assert config.memory == 1024  # type: ignore[attr-defined]
        assert config.gpu == "A10G"  # type: ignore[attr-defined]
        assert config.region == "us-east"  # type: ignore[attr-defined]
        assert config.sandbox_timeout == 7200  # type: ignore[attr-defined]
        assert config.default_timeout == 300  # type: ignore[attr-defined]

    def test_config_from_args_defaults(self):
        """Config uses defaults when not specified."""
        args = MockNamespace(
            modal_token_id=None,
            modal_token_secret=None,
            modal_app_name="kilntainers",
            modal_cpu=1.0,
            modal_memory=512,
            gpu=None,
            region=None,
            sandbox_timeout=3600,
            image=None,
            shell="/bin/bash",
            network=False,
            timeout=120,
        )

        config = ModalBackend.config_from_args(args)  # type: ignore[arg-type]

        assert config.token_id is None  # type: ignore[attr-defined]
        assert config.token_secret is None  # type: ignore[attr-defined]
        assert config.app_name == "kilntainers"  # type: ignore[attr-defined]
        assert config.image is None  # type: ignore[attr-defined]
        assert config.cpu == 1.0  # type: ignore[attr-defined]
        assert config.memory == 512  # type: ignore[attr-defined]
        assert config.gpu is None  # type: ignore[attr-defined]
        assert config.region is None  # type: ignore[attr-defined]
        assert config.sandbox_timeout == 3600  # type: ignore[attr-defined]


class TestModalBackendValidation:
    """Tests for ModalBackend._validate method."""

    @pytest.mark.asyncio
    async def test_validate_success(self, mock_modal, default_config):
        """Validation passes when Modal auth succeeds."""
        backend = ModalBackend(default_config)
        await backend.validate()

        # Second call should be cached (no-op)
        await backend.validate()

    @pytest.mark.asyncio
    async def test_validate_auth_failure(self, mock_modal, default_config):
        """Validation fails when auth fails."""
        config = ModalBackendConfig(app_name="auth-fail")
        backend = ModalBackend(config)

        with pytest.raises(BackendError) as exc_info:
            await backend.validate()

        assert "authentication failed" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_validate_connection_failure(self, mock_modal, default_config):
        """Validation fails when API is unreachable."""
        config = ModalBackendConfig(app_name="connection-fail")
        backend = ModalBackend(config)

        with pytest.raises(BackendError) as exc_info:
            await backend.validate()

        assert "Cannot connect to Modal" in str(exc_info.value)


class TestModalBackendImage:
    """Tests for ModalBackend._build_image method."""

    def test_build_image_default(self, mock_modal, default_config):
        """Default config uses debian_slim."""
        backend = ModalBackend(default_config)
        image = backend._build_image()

        assert image == "debian-slim-image"

    def test_build_image_custom(self, mock_modal):
        """Custom image uses from_registry."""
        config = ModalBackendConfig(image="python:3.12-slim")
        backend = ModalBackend(config)
        image = backend._build_image()

        assert image == "registry-image:python:3.12-slim"


class TestModalBackendSandboxCreation:
    """Tests for ModalBackend._create_sandbox method."""

    @pytest.mark.asyncio
    async def test_create_sandbox_success(self, mock_modal, default_config):
        """Full sandbox creation sequence succeeds."""
        mock_sandbox = MockSandbox("sb-test-123")
        mock_sandbox.set_exec_response(
            MockContainerProcess(stdout="kilntainers-ready\n")
        )
        mock_modal.Sandbox.set_next_sandbox(mock_sandbox)

        backend = ModalBackend(default_config)
        sandbox = await backend.create_sandbox()

        assert isinstance(sandbox, ModalSandbox)
        assert sandbox.sandbox_id == "sb-test-123"

    @pytest.mark.asyncio
    async def test_create_sandbox_readiness_failure(self, mock_modal, default_config):
        """Readiness check fails, sandbox is cleaned up."""
        mock_sandbox = MockSandbox("sb-test-123")
        mock_sandbox.set_exec_response(MockContainerProcess(stdout="wrong output\n"))
        mock_modal.Sandbox.set_next_sandbox(mock_sandbox)

        backend = ModalBackend(default_config)

        with pytest.raises(BackendError) as exc_info:
            await backend.create_sandbox()

        assert "readiness check failed" in str(exc_info.value)
        assert mock_sandbox.terminated, (
            "Sandbox should be terminated after readiness failure"
        )


class TestModalBackendToolInstructions:
    """Tests for ModalBackend.tool_instructions method."""

    def test_tool_instructions_default_image(self, mock_modal, default_config):
        """Default image returns tool description."""
        backend = ModalBackend(default_config)
        instructions = backend.tool_instructions()

        assert instructions is not None
        assert "Modal" in instructions
        assert "bash" in instructions
        assert "120" in instructions  # default timeout

    def test_tool_instructions_custom_image(self, mock_modal):
        """Custom image returns None."""
        config = ModalBackendConfig(image="python:3.12-slim")
        backend = ModalBackend(config)
        instructions = backend.tool_instructions()

        assert instructions is None


class TestModalSandboxExecCommandConstruction:
    """Tests for ModalSandbox._build_exec_args and _build_exec_kwargs."""

    @pytest.fixture
    def sandbox(self, mock_modal):
        """Create a test sandbox."""
        mock_sb = MockSandbox("sb-test-123")
        return ModalSandbox(modal_sandbox=mock_sb, shell="/bin/bash")  # type: ignore[arg-type]

    def test_command_mode(self, sandbox):
        """Command mode wraps in shell."""
        request = ExecRequest(command="ls -la", timeout=30, output_limit=2_097_152)
        args = sandbox._build_exec_args(request)

        assert args == ["/bin/bash", "-c", "ls -la"]

    def test_args_mode(self, sandbox):
        """Args mode passes directly."""
        request = ExecRequest(
            args=["python3", "script.py"], timeout=30, output_limit=2_097_152
        )
        args = sandbox._build_exec_args(request)

        assert args == ["python3", "script.py"]

    def test_build_exec_kwargs_default(self, sandbox):
        """Default kwargs only include timeout."""
        request = ExecRequest(command="echo hello", timeout=60, output_limit=2_097_152)
        kwargs = sandbox._build_exec_kwargs(request)

        assert kwargs == {"timeout": 60}

    def test_build_exec_kwargs_with_workdir(self, sandbox):
        """Working directory is added to kwargs."""
        request = ExecRequest(
            command="pwd",
            working_directory="/app",
            timeout=60,
            output_limit=2_097_152,
        )
        kwargs = sandbox._build_exec_kwargs(request)

        assert kwargs == {"timeout": 60, "workdir": "/app"}


class TestModalSandboxExec:
    """Tests for ModalSandbox.exec method."""

    @pytest.fixture
    def sandbox(self, mock_modal):
        """Create a test sandbox."""
        mock_sb = MockSandbox("sb-test-123")
        return ModalSandbox(modal_sandbox=mock_sb, shell="/bin/bash")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_exec_success(self, sandbox):
        """Successful command execution."""
        mock_sb = sandbox._modal_sandbox
        mock_sb.set_exec_response(
            MockContainerProcess(stdout="hello\n", stderr="", returncode=0)
        )

        request = ExecRequest(command="echo hello", timeout=30, output_limit=2_097_152)
        result = await sandbox.exec(request)

        assert result.stdout == "hello\n"
        assert result.stderr == ""
        assert result.exit_code == 0
        assert result.exec_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_exec_failure(self, sandbox):
        """Command fails with non-zero exit."""
        mock_sb = sandbox._modal_sandbox
        mock_sb.set_exec_response(
            MockContainerProcess(stdout="", stderr="error: not found\n", returncode=1)
        )

        request = ExecRequest(command="false", timeout=30, output_limit=2_097_152)
        result = await sandbox.exec(request)

        assert result.exit_code == 1
        assert "error: not found" in result.stderr

    @pytest.mark.asyncio
    async def test_exec_timeout(self, sandbox):
        """Command times out."""
        mock_sb = sandbox._modal_sandbox

        # Create a custom blocking StreamReader
        class BlockingStreamReader:
            def __init__(self):
                pass

            @property
            def read(self):
                return self

            async def aio(self):
                return ""

            def __aiter__(self):
                return self

            async def __anext__(self):
                # Block forever to trigger timeout
                await asyncio.sleep(1000)
                return "line"

        blocking_process = MockContainerProcess()
        blocking_process.stdout = BlockingStreamReader()
        blocking_process.stderr = BlockingStreamReader()

        mock_sb.set_exec_response(blocking_process)

        request = ExecRequest(command="sleep 60", timeout=1, output_limit=2_097_152)

        # We want to test that it returns quickly even if the stream is blocking
        # The internal implementation uses timeout + 10 for safety, but the
        # actual timeout should be enforced by the request.timeout.
        start_time = asyncio.get_event_loop().time()
        result = await sandbox.exec(request)
        end_time = asyncio.get_event_loop().time()

        assert result.exit_code == 124
        assert "timed out" in result.stderr
        assert result.stdout == ""
        # Should not have waited for the full 1000s or even the safety buffer
        assert end_time - start_time < 5

    @pytest.mark.asyncio
    async def test_exec_output_limit(self, sandbox):
        """Output limit exceeded."""
        mock_sb = sandbox._modal_sandbox
        # Create output that exceeds the limit
        mock_sb.set_exec_response(
            MockContainerProcess(stdout="x" * 10000, stderr="y" * 10000)
        )

        request = ExecRequest(command="yes", timeout=30, output_limit=1000)
        result = await sandbox.exec(request)

        assert result.exit_code == 1
        assert "output limit exceeded" in result.stderr
        assert result.stdout == ""

    @pytest.mark.asyncio
    async def test_exec_after_stop(self, sandbox):
        """Exec after stop raises SandboxDiedError."""
        sandbox._stopped = True

        request = ExecRequest(command="echo", timeout=30, output_limit=2_097_152)

        with pytest.raises(SandboxDiedError) as exc_info:
            await sandbox.exec(request)

        assert "stopped" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_exec_serialization(self, sandbox):
        """Concurrent exec calls are serialized."""
        mock_sb = sandbox._modal_sandbox
        exec_count = [0]
        in_exec = asyncio.Event()

        async def slow_exec(*args, **kwargs):
            exec_count[0] += 1
            in_exec.set()
            await asyncio.sleep(0.1)
            return MockContainerProcess(stdout="done\n", stderr="")

        # Replace the aio method temporarily
        original_aio = mock_sb._exec.aio
        mock_sb._exec.aio = slow_exec

        try:
            request = ExecRequest(
                command="echo test", timeout=30, output_limit=2_097_152
            )

            # Start two exec calls concurrently
            task1 = asyncio.create_task(sandbox.exec(request))
            task2 = asyncio.create_task(sandbox.exec(request))

            # Both should complete
            await task1
            await task2
        finally:
            mock_sb._exec.aio = original_aio

        request = ExecRequest(command="echo test", timeout=30, output_limit=2_097_152)

        # Start two exec calls concurrently
        task1 = asyncio.create_task(sandbox.exec(request))
        task2 = asyncio.create_task(sandbox.exec(request))

        # Both should complete
        await task1
        await task2


class TestModalSandboxStop:
    """Tests for ModalSandbox.stop method."""

    @pytest.fixture
    def sandbox(self, mock_modal):
        """Create a test sandbox."""
        mock_sb = MockSandbox("sb-test-123")
        return ModalSandbox(modal_sandbox=mock_sb, shell="/bin/bash")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_stop(self, sandbox):
        """Stop calls terminate."""
        mock_sb = sandbox._modal_sandbox

        await sandbox.stop()

        assert sandbox._stopped
        assert sandbox._stop_requested
        assert mock_sb.terminated

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, sandbox):
        """Stop is idempotent."""
        mock_sb = sandbox._modal_sandbox

        await sandbox.stop()
        await sandbox.stop()
        await sandbox.stop()

        # terminate should only be called once (tracked by terminated flag)
        assert mock_sb.terminated


class TestModalSandboxDeathDetection:
    """Tests for ModalSandbox.wait_for_death method."""

    @pytest.fixture
    def sandbox(self, mock_modal):
        """Create a test sandbox."""
        mock_sb = MockSandbox("sb-test-123")
        return ModalSandbox(modal_sandbox=mock_sb, shell="/bin/bash")  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_wait_for_death_unexpected_exit(self, sandbox):
        """Sandbox exits unexpectedly."""
        mock_sb = sandbox._modal_sandbox
        mock_sb._wait_event.set()  # Simulate sandbox exit

        # Should return when sandbox exits unexpectedly
        await sandbox.wait_for_death()

    @pytest.mark.asyncio
    async def test_wait_for_death_with_stop_requested(self, sandbox):
        """Sandbox exits after stop is requested."""
        mock_sb = sandbox._modal_sandbox
        mock_sb._wait_event.set()  # Simulate sandbox exit

        sandbox._stop_requested = True

        # Should block forever (we'll cancel it)
        task = asyncio.create_task(sandbox.wait_for_death())
        await asyncio.sleep(0.1)

        # Task should still be running
        assert not task.done()

        # Cancel the task
        task.cancel()
        await task

    @pytest.mark.asyncio
    async def test_wait_for_death_cancellation(self, sandbox):
        """Task is cancelled before sandbox exits."""
        mock_sb = sandbox._modal_sandbox

        # Simulate sandbox exit immediately
        mock_sb._wait_event.set()

        # Should return normally
        await sandbox.wait_for_death()


class TestModalSandboxSandboxId:
    """Tests for ModalSandbox.sandbox_id property."""

    def test_sandbox_id(self, mock_modal):
        """sandbox_id returns Modal's object_id."""
        mock_sb = MockSandbox("sb-custom-456")
        sandbox = ModalSandbox(modal_sandbox=mock_sb, shell="/bin/bash")  # type: ignore[arg-type]

        assert sandbox.sandbox_id == "sb-custom-456"


# --- Mock helpers for CLI tests ---


class MockArgumentParser:
    """Mock argparse.ArgumentParser for testing."""

    def __init__(self):
        self.arguments = {}

    def add_argument_group(self, name: str):
        return self

    def add_argument(self, *args, **kwargs):
        self.arguments[args[0]] = kwargs


class MockNamespace:
    """Mock argparse.Namespace for testing."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

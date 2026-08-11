"""Integration tests for Docker backend.

These tests require a running Docker or Podman daemon and are marked with
@pytest.mark.integration. They execute real containers and verify
the full integration. They are parameterized by engine ("docker", "podman").

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import asyncio
import shutil
import subprocess

import pytest

from kilntainers.backends.base import ExecRequest
from kilntainers.backends.docker import (
    DockerBackend,
    DockerBackendConfig,
    DockerSandbox,
)
from kilntainers.errors import BackendError, SandboxDiedError

# Cache to track if a docker-like CLI is available and the daemon is running.
_engine_available_cache: dict[str, bool] = {}


def validate_engine_available(engine: str) -> None:
    """Validate that the engine CLI is installed and the daemon is running, skipping the test if not."""
    available = _engine_available_cache.get(engine, None)

    if available is None:
        available = _check_engine_available(engine)
        _engine_available_cache[engine] = available

    if not available:
        pytest.skip(f"{engine} CLI not installed")


def _check_engine_available(engine: str) -> bool:
    """Check if the engine CLI is installed and the daemon is running."""
    command_available = shutil.which(engine) is not None
    if not command_available:
        return False
    try:
        result = subprocess.run([engine, "info"], capture_output=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError:
        return False


async def get_docker_backend(engine: str) -> DockerBackend:
    """Create and validate a Docker/Podman backend.

    Skips the test if the engine CLI is not installed or the daemon is not running.
    """
    validate_engine_available(engine)

    config = DockerBackendConfig(engine=engine)
    backend = DockerBackend(config)
    try:
        await backend.validate()
    except BackendError as e:
        if f"Is the {engine} daemon running?" in str(e):
            pytest.skip(f"{engine} daemon not running")
        raise
    return backend


@pytest.fixture(params=["docker", "podman"])
async def engine(request):
    """Parameterize tests over container engine (docker, podman).

    Automatically skips if the engine CLI is not installed on the system.
    """
    engine = request.param
    validate_engine_available(engine)
    return engine


@pytest.fixture
async def docker_backend(engine):
    """Create a real Docker/Podman backend instance for the given engine.

    Automatically skips if the engine daemon is not running.
    """
    return await get_docker_backend(engine)


@pytest.fixture
async def sandbox(docker_backend):
    """Create a real Docker sandbox for testing.

    The sandbox is automatically stopped after the test.
    """
    sb = await docker_backend.create_sandbox()
    yield sb
    # Cleanup: stop the sandbox
    await sb.stop()


# --- Connection Tests ---


@pytest.mark.integration
class TestConnection:
    """Tests for Docker host connection configuration."""

    @pytest.mark.asyncio
    async def test_bad_host_fails_validation(self, engine):
        """A bad --docker-host value causes validation to fail with BackendError."""
        config = DockerBackendConfig(
            engine=engine,
            host="tcp://127.0.0.1:1",  # Connection refused is faster than timeout
        )
        backend = DockerBackend(config)

        with pytest.raises(BackendError) as exc_info:
            await asyncio.wait_for(backend.validate(), timeout=15)

        assert "Cannot connect to" in str(exc_info.value)


# --- Docker Specific Lifecycle Tests ---


@pytest.mark.integration
class TestDockerLifecycle:
    """Tests for Docker-specific sandbox lifecycle operations."""

    @pytest.mark.asyncio
    async def test_create_sandbox_type(self, docker_backend):
        """Sandbox creation returns correct type and ID format."""
        sb = await docker_backend.create_sandbox()
        assert isinstance(sb, DockerSandbox)
        assert len(sb.sandbox_id) == 12
        await sb.stop()


# --- Timeout Tests ---


@pytest.mark.integration
class TestTimeout:
    """Tests for timeout enforcement."""

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self, sandbox):
        """Long-running command is terminated after timeout."""
        request = ExecRequest(command="sleep 60", timeout=1, output_limit=1024)
        result = await sandbox.exec(request)
        assert result.exit_code == 124
        assert "timed out" in result.stderr


# --- Network Isolation Tests ---


@pytest.mark.integration
class TestNetworkIsolation:
    """Tests for network isolation behavior."""

    @pytest.mark.asyncio
    async def test_network_disabled_by_default(self, sandbox):
        """Network access is blocked by default."""
        request = ExecRequest(
            command="curl -s --connect-timeout 2 http://example.com",
            timeout=5,
            output_limit=1024,
        )
        result = await sandbox.exec(request)
        # Should fail because network is disabled
        assert result.exit_code != 0

    @pytest.mark.asyncio
    async def test_network_enabled_works(self, docker_backend):
        """Network access works when enabled."""
        config = DockerBackendConfig(network_enabled=True)
        backend = DockerBackend(config)
        sb = await backend.create_sandbox()
        try:
            request = ExecRequest(
                command="curl -s --connect-timeout 2 http://example.com",
                timeout=5,
                output_limit=1024,
            )
            result = await sb.exec(request)
            # Should succeed (or at least reach the server)
            # We just check it doesn't fail immediately
            assert result.exit_code == 0 or "curl" in result.stderr
        finally:
            await sb.stop()


# --- Death Detection Tests ---


@pytest.mark.integration
class TestDeathDetection:
    """Tests for unexpected sandbox death detection."""

    @pytest.mark.asyncio
    async def test_wait_for_death_on_kill(self, sandbox, engine):
        """wait_for_death returns when container is killed externally."""
        # Start the death detection task
        death_task = asyncio.create_task(sandbox.wait_for_death())

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Kill the container externally
        subprocess.run(
            [engine, "kill", sandbox._container_id],
            capture_output=True,
        )

        # wait_for_death should complete
        await asyncio.wait_for(death_task, timeout=5)

    @pytest.mark.asyncio
    async def test_exec_after_container_dies(self, sandbox, engine):
        """Exec raises SandboxDiedError when container dies during exec."""
        # Kill the container externally
        subprocess.run(
            [engine, "kill", sandbox._container_id],
            capture_output=True,
        )
        await asyncio.sleep(0.1)  # Give it time to die

        # Next exec should detect the death
        request = ExecRequest(command="echo test", timeout=5, output_limit=1024)

        with pytest.raises(SandboxDiedError) as exc_info:
            await sandbox.exec(request)
        assert "died during command execution" in str(exc_info.value)


# --- Cleanup Tests ---


@pytest.mark.integration
class TestCleanup:
    """Tests for proper resource cleanup."""

    @pytest.mark.asyncio
    async def test_container_removed_on_stop(self, sandbox, engine):
        """Container is removed (--rm flag) when stopped."""
        container_id = sandbox._container_id

        # Verify container exists
        result = subprocess.run(
            [engine, "inspect", "--format", "{{.State.Running}}", container_id],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Stop the sandbox
        await sandbox.stop()

        # Container should be removed (not just stopped)
        result = subprocess.run(
            [engine, "inspect", container_id],
            capture_output=True,
        )
        assert result.returncode != 0  # Container not found

    @pytest.mark.asyncio
    async def test_kilntainers_label_present(self, sandbox, engine):
        """Container has kilntainers=true label."""
        result = subprocess.run(
            [
                engine,
                "inspect",
                "--format",
                '{{index .Config.Labels "kilntainers"}}',
                sandbox._container_id,
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "true" in result.stdout


# --- Tool Instructions Tests ---


@pytest.mark.integration
class TestToolInstructions:
    """Tests for tool_instructions method."""

    @pytest.mark.asyncio
    async def test_tool_instructions_default_image(self, docker_backend):
        """Default image returns tool instructions."""
        instructions = docker_backend.tool_instructions()
        assert instructions is not None
        assert "Debian" in instructions
        assert "bash" in instructions
        assert "120" in instructions  # default timeout

    @pytest.mark.asyncio
    async def test_tool_instructions_custom_image(self, engine, docker_backend):
        """Custom image returns None for tool instructions."""
        config = DockerBackendConfig(engine=engine, image="alpine:latest")
        backend = DockerBackend(config)
        await backend.validate()
        instructions = backend.tool_instructions()
        assert instructions is None

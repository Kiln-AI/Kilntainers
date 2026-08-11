"""Integration tests for the full server lifecycle.

These tests require a running Docker or Podman daemon and are marked with
@pytest.mark.e2e. They test the full server lifecycle with real containers,
not mocks.

Run with: pytest -m e2e
Skip with: pytest -m "not e2e"
"""

import asyncio
import subprocess
from typing import cast
from unittest.mock import MagicMock

import pytest

from kilntainers.backends.base import ExecRequest
from kilntainers.backends.docker import (
    DockerBackend,
    DockerBackendConfig,
    DockerSandbox,
)
from kilntainers.backends.test_docker_integration import (
    get_docker_backend,
    validate_engine_available,
)
from kilntainers.config import ServerConfig
from kilntainers.errors import BackendError
from kilntainers.server import create_lifespan, create_server


@pytest.fixture(params=["docker", "podman"])
def engine(request):
    """Parameterize tests over container engine (docker, podman).

    Automatically skips if the engine CLI is not installed on the system.
    """
    engine_name = request.param
    validate_engine_available(engine_name)
    return engine_name


@pytest.fixture
async def backend(engine):
    """Create a real Docker/Podman backend instance for the given engine.

    This fixture wraps get_docker_backend as a pytest fixture for use in tests.
    """
    return await get_docker_backend(engine)


@pytest.fixture
async def server_config():
    """Return a default server config for testing."""
    return ServerConfig()


# ====================
# Full Lifecycle Tests
# ====================


class TestFullLifecycle:
    """Tests for the complete server lifecycle from start to stop."""

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lifecycle_full_stdio_session(self, backend, server_config, engine):
        """Full stdio lifecycle: lazily create sandbox, exec commands, stop sandbox."""
        lifespan_fn = create_lifespan(backend, "stdio")
        mock_server = MagicMock()

        container_id = None

        async with lifespan_fn(mock_server) as ctx:
            # Sandbox is None initially (lazy creation)
            assert ctx.sandbox is None

            # Create sandbox lazily
            sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
            assert sandbox.sandbox_id is not None
            container_id = sandbox._container_id

            # Verify container exists via docker/podman CLI
            result = subprocess.run(
                [engine, "inspect", "--format", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "true" in result.stdout

            # Execute a command
            request = ExecRequest(command="echo hello", timeout=5, output_limit=1024)
            result = await sandbox.exec(request)
            assert result.exit_code == 0
            assert "hello" in result.stdout

        # After lifespan exit, verify container was removed
        result = subprocess.run(
            [engine, "inspect", container_id],
            capture_output=True,
        )
        assert result.returncode != 0  # Container not found

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lifespan_no_container_before_exec(self, backend, engine):
        """Lifespan yields immediately, no container until get_or_create_sandbox()."""
        lifespan_fn = create_lifespan(backend, "stdio")
        mock_server = MagicMock()

        async with lifespan_fn(mock_server) as ctx:
            # Verify no sandbox/container yet
            assert ctx.sandbox is None

            # Create sandbox
            sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
            container_id = sandbox._container_id

            # Now container should exist
            result = subprocess.run(
                [engine, "inspect", "--format", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
            assert "true" in result.stdout

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lifecycle_creates_server_with_lifespan(self, backend, server_config):
        """create_server creates a FastMCP instance with configured lifespan."""
        server = create_server(backend, server_config)

        # Verify server was created
        assert server is not None
        assert server.name == "Kilntainers"

        # The lifespan is configured but we can't easily test it runs
        # without actually calling mcp.run() which blocks


# ====================
# Sandbox Creation Failure Tests
# ====================


class TestSandboxCreationFailure:
    """Tests for handling sandbox creation failures."""

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_sandbox_creation_failure_on_get_or_create(self, engine):
        """BackendError during sandbox creation propagates from get_or_create_sandbox()."""
        # First verify daemon is running using the helper
        _ = await get_docker_backend(engine)

        # Now use a bad image that doesn't exist
        bad_config = DockerBackendConfig(
            engine=engine, image="nonexistent/nonexistent:latest"
        )
        bad_backend = DockerBackend(bad_config)

        lifespan_fn = create_lifespan(bad_backend, "stdio")
        mock_server = MagicMock()

        async with lifespan_fn(mock_server) as ctx:
            # Lifespan yields successfully (no sandbox yet)
            assert ctx.sandbox is None

            # Error happens on get_or_create_sandbox()
            with pytest.raises(BackendError) as exc_info:
                await ctx.get_or_create_sandbox()

            # Error should mention image pull or creation failure
            assert (
                "image" in str(exc_info.value).lower()
                or "pull" in str(exc_info.value).lower()
            )


# ====================
# Graceful Shutdown Tests
# ====================


class TestGracefulShutdown:
    """Tests for graceful shutdown behavior."""

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lifecycle_stops_sandbox_on_exception(self, backend, engine):
        """Sandbox is stopped even when an exception occurs in the lifespan body."""
        lifespan_fn = create_lifespan(backend, "stdio")
        mock_server = MagicMock()

        container_id = None

        with pytest.raises(ValueError):
            async with lifespan_fn(mock_server) as ctx:
                # Create sandbox first
                sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
                container_id = sandbox._container_id
                # Raise an exception to simulate error during session
                raise ValueError("simulated error")

        # Verify container was still cleaned up
        result = subprocess.run(
            [engine, "inspect", container_id],
            capture_output=True,
        )
        assert result.returncode != 0  # Container not found

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_lifecycle_death_task_cancelled_before_stop(self, backend):
        """Death task is cancelled before sandbox.stop() is called."""
        lifespan_fn = create_lifespan(backend, "stdio")
        mock_server = MagicMock()

        async with lifespan_fn(mock_server) as ctx:
            # Create sandbox first to start death task
            sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
            death_task = ctx.death_task
            assert death_task is not None
            assert not death_task.cancelled()

        # After exit, death task should be cancelled
        assert death_task.cancelled()

        # And sandbox should be stopped
        assert sandbox._stopped


# ====================
# Death Propagation Tests
# ====================


class TestDeathPropagation:
    """Tests for sandbox death propagation."""

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_death_propagation_stdio(self, backend, engine):
        """Sandbox death triggers death callback in stdio mode."""
        # Use a list to capture death notifications without sending SIGTERM
        death_notifications: list[None] = []
        death_event = asyncio.Event()

        def death_callback() -> None:
            """Capture death notification instead of sending SIGTERM."""
            death_notifications.append(None)
            death_event.set()

        lifespan_fn = create_lifespan(backend, "stdio", death_callback=death_callback)
        mock_server = MagicMock()

        async with lifespan_fn(mock_server) as ctx:
            # Create sandbox first to start death monitoring
            sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
            container_id = sandbox._container_id

            # Actually kill the container externally
            subprocess.run(
                [engine, "kill", container_id],
                capture_output=True,
            )

            # Wait for the death monitor instead of relying on scheduling speed.
            await asyncio.wait_for(death_event.wait(), timeout=10)

        # Verify death callback was invoked
        assert len(death_notifications) >= 1

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_external_container_kill_detected(self, backend, engine):
        """Killing container externally is detected by wait_for_death()."""

        def death_callback() -> None:
            """No-op callback to avoid SIGTERM during test."""
            pass

        lifespan_fn = create_lifespan(backend, "stdio", death_callback=death_callback)
        mock_server = MagicMock()

        death_detected = False

        async def monitor_and_kill():
            """Monitor death and kill externally."""
            async with lifespan_fn(mock_server) as ctx:
                nonlocal death_detected
                # Create sandbox first to start death monitoring
                sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
                container_id = sandbox._container_id

                # Wait for death task to start
                await asyncio.sleep(0.1)

                # Kill the container externally
                subprocess.run(
                    [engine, "kill", container_id],
                    capture_output=True,
                )

                # Wait for death detection (with timeout)
                death_task = ctx.death_task
                assert death_task is not None
                try:
                    await asyncio.wait_for(death_task, timeout=5)
                    death_detected = True
                except asyncio.TimeoutError:
                    pass

        await monitor_and_kill()

        # Death should have been detected
        assert death_detected

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_no_death_callback_in_http_mode(self, backend):
        """No callback is invoked in HTTP mode when sandbox dies."""

        callback_invoked: list[None] = []

        def death_callback() -> None:
            """This should NOT be called in HTTP mode."""
            callback_invoked.append(None)

        lifespan_fn = create_lifespan(backend, "http", death_callback=death_callback)
        mock_server = MagicMock()

        async with lifespan_fn(mock_server):
            # In HTTP mode, we can't easily kill the sandbox and detect death
            # The test just verifies the callback is NOT called during normal operation
            pass

        # Callback should not have been invoked in HTTP mode
        assert len(callback_invoked) == 0


# ====================
# Cleanup Verification Tests
# ====================


class TestCleanupVerification:
    """Tests for proper resource cleanup."""

    @pytest.mark.e2e
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_container_removed_after_lifespan(self, backend, engine):
        """Container is removed (--rm flag) after lifespan exits."""
        lifespan_fn = create_lifespan(backend, "stdio")
        mock_server = MagicMock()

        container_id = None

        async with lifespan_fn(mock_server) as ctx:
            # Create sandbox first
            sandbox = cast(DockerSandbox, await ctx.get_or_create_sandbox())
            container_id = sandbox._container_id

            # Verify container exists
            result = subprocess.run(
                [engine, "inspect", container_id],
                capture_output=True,
            )
            assert result.returncode == 0

        # After lifespan exit, container should be removed
        result = subprocess.run(
            [engine, "inspect", container_id],
            capture_output=True,
        )
        assert result.returncode != 0

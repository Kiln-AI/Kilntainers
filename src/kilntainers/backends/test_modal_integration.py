"""Integration tests for Modal backend.

These tests require Modal authentication and are marked with
@pytest.mark.integration. They execute real Modal sandboxes and verify
the full integration. Run with: pytest -m integration
Skip with: pytest -m "not integration"

NOTE: These tests will incur actual Modal costs when run.
"""

import asyncio
import os

import pytest

from kilntainers.backends.base import ExecRequest
from kilntainers.backends.modal import (
    ModalBackend,
    ModalBackendConfig,
    ModalSandbox,
    _get_modal_sdk,
)
from kilntainers.errors import BackendError, SandboxDiedError


def _modal_auth_available() -> bool:
    """Check if Modal authentication is configured."""
    # Check environment variables
    if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
        return True

    # Try to validate with Modal client (may fail if no auth)
    try:
        modal = _get_modal_sdk()

        # Check if there's a default profile or active token
        # modal.config.Config() will load the configuration
        config = modal.config.Config()
        if config.get("token_id") and config.get("token_secret"):
            return True
        return False
    except Exception:
        return False


skip_without_modal = pytest.mark.skipif(
    not _modal_auth_available(),
    reason="Modal credentials not configured. Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET, or run 'modal token set'",
)


@pytest.fixture
async def modal_backend():
    """Create a real Modal backend instance for testing.

    Automatically skips if Modal authentication is not configured.
    """
    config = ModalBackendConfig()
    backend = ModalBackend(config)
    try:
        await backend.validate()
    except BackendError as e:
        if "authentication failed" in str(e).lower():
            pytest.skip("Modal authentication failed")
        if "Cannot connect to Modal" in str(e):
            pytest.skip("Cannot connect to Modal API")
        raise
    return backend


@pytest.fixture
async def sandbox(modal_backend):
    """Create a real Modal sandbox for testing.

    The sandbox is automatically stopped after the test.
    """
    sb = await modal_backend.create_sandbox()
    yield sb
    # Cleanup: stop the sandbox
    await sb.stop()


# --- Connection Tests ---


class TestConnection:
    """Tests for Modal connection configuration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_backend_validation_succeeds(self):
        """Backend validation succeeds with valid auth."""
        config = ModalBackendConfig()
        backend = ModalBackend(config)
        # Should not raise
        await backend.validate()

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_invalid_token_fails(self):
        """Invalid token causes validation to fail with BackendError.

        NOTE: This test is skipped when default Modal auth is configured,
        as Modal SDK falls back to default auth when explicit tokens are invalid.
        """
        # Skip if user has default Modal auth configured (e.g., via modal token new)
        # The SDK falls back to default auth, making this test unreliable
        if not os.getenv("MODAL_TOKEN_ID") and not os.getenv("MODAL_TOKEN_SECRET"):
            pytest.skip(
                "Cannot test invalid tokens when default Modal auth is configured. "
                "Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET to test explicit auth."
            )

        # Temporarily clear and set invalid tokens
        original_token_id = os.environ.get("MODAL_TOKEN_ID")
        original_token_secret = os.environ.get("MODAL_TOKEN_SECRET")

        try:
            os.environ["MODAL_TOKEN_ID"] = "invalid_token_id"
            os.environ["MODAL_TOKEN_SECRET"] = "invalid_token_secret"

            # Create new backend with invalid auth
            config = ModalBackendConfig(
                token_id="invalid_token_id",
                token_secret="invalid_token_secret",
            )
            backend = ModalBackend(config)

            with pytest.raises(BackendError) as exc_info:
                await backend.validate()

            assert "authentication failed" in str(exc_info.value).lower()
        finally:
            # Restore original values
            if original_token_id is not None:
                os.environ["MODAL_TOKEN_ID"] = original_token_id
            else:
                os.environ.pop("MODAL_TOKEN_ID", None)
            if original_token_secret is not None:
                os.environ["MODAL_TOKEN_SECRET"] = original_token_secret
            else:
                os.environ.pop("MODAL_TOKEN_SECRET", None)


# --- Modal Specific Lifecycle Tests ---


@pytest.mark.integration
class TestModalLifecycle:
    """Tests for Modal-specific sandbox lifecycle operations."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_create_sandbox_type(self, modal_backend):
        """Sandbox creation returns correct type."""
        sb = await modal_backend.create_sandbox()
        assert isinstance(sb, ModalSandbox)
        assert sb.sandbox_id is not None
        await sb.stop()


# --- Timeout Tests ---


@pytest.mark.integration
class TestTimeout:
    """Tests for timeout enforcement."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_timeout_exceeded(self, sandbox):
        """Long-running command is terminated after timeout."""
        request = ExecRequest(command="sleep 60", timeout=1, output_limit=1024)
        result = await sandbox.exec(request)
        # Modal returns -1 for timeout (different from Docker's 124)
        # Note: Modal's native timeout doesn't populate stderr, so we only check exit code
        assert result.exit_code == -1


# --- Network Isolation Tests ---


@pytest.mark.integration
class TestNetworkIsolation:
    """Tests for network isolation behavior."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_network_disabled_by_default(self, sandbox):
        """Network access is blocked by default."""
        # Use Python to test network (no curl required)
        request = ExecRequest(
            command="python3 -c 'import urllib.request; urllib.request.urlopen(\"http://example.com\", timeout=2)'",
            timeout=5,
            output_limit=1024,
        )
        result = await sandbox.exec(request)
        # Should fail because network is disabled
        assert result.exit_code != 0

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_network_enabled_works(self):
        """Network access works when enabled."""
        config = ModalBackendConfig(network_enabled=True)
        backend = ModalBackend(config)
        sb = await backend.create_sandbox()
        try:
            # Use Python to test network (no curl required)
            request = ExecRequest(
                command="python3 -c 'import sys, urllib.request; sys.stdout.write(urllib.request.urlopen(\"http://example.com\", timeout=5).read().decode()[:100])'",
                timeout=10,
                output_limit=1024,
            )
            result = await sb.exec(request)
            # Should succeed (or at least reach the server)
            assert result.exit_code == 0
            assert (
                "Example Domain" in result.stdout or "example" in result.stdout.lower()
            )
        finally:
            await sb.stop()


# --- Exec Lock Tests ---


@pytest.mark.integration
class TestExecLock:
    """Tests for exec serialization via lock."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_concurrent_execs_are_serialized(self, sandbox):
        """Concurrent exec calls are properly serialized."""
        # Fire multiple exec requests concurrently
        requests = [
            ExecRequest(command=f"echo {i}", timeout=5, output_limit=1024)
            for i in range(5)
        ]

        async def run_exec(req):
            return await sandbox.exec(req)

        results = await asyncio.gather(*[run_exec(req) for req in requests])

        # All should succeed
        assert len(results) == 5
        for result in results:
            assert result.exit_code == 0


# --- Death Detection Tests ---


@pytest.mark.integration
class TestDeathDetection:
    """Tests for unexpected sandbox death detection."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_wait_for_death_blocks_on_normal_stop(self, sandbox):
        """wait_for_death blocks when stop() is called (expected death)."""
        # Start the death detection task
        death_task = asyncio.create_task(sandbox.wait_for_death())

        # Give it a moment to start
        await asyncio.sleep(0.1)

        # Stop the sandbox
        await sandbox.stop()

        # The task should still be running (waiting forever)
        # Cancel it to clean up
        death_task.cancel()
        try:
            await death_task
        except asyncio.CancelledError:
            pass  # Expected

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_exec_after_stop_raises_error(self, sandbox):
        """Exec raises SandboxDiedError when called after stop()."""
        await sandbox.stop()

        request = ExecRequest(command="echo test", timeout=5, output_limit=1024)
        with pytest.raises(SandboxDiedError) as exc_info:
            await sandbox.exec(request)
        assert "stopped" in str(exc_info.value).lower()


# --- Tool Instructions Tests ---


@pytest.mark.integration
class TestToolInstructions:
    """Tests for tool_instructions method."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_tool_instructions_default_image(self):
        """Default image returns tool instructions."""
        config = ModalBackendConfig()
        backend = ModalBackend(config)
        await backend.validate()
        instructions = backend.tool_instructions()
        assert instructions is not None
        assert "bash" in instructions
        assert "120" in instructions  # default timeout

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_tool_instructions_custom_image(self):
        """Custom image returns None for tool instructions."""
        config = ModalBackendConfig(image="alpine:latest")
        backend = ModalBackend(config)
        await backend.validate()
        instructions = backend.tool_instructions()
        assert instructions is None


# --- Custom Shell Tests ---


@pytest.mark.integration
class TestCustomShell:
    """Tests for custom shell configuration."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    @skip_without_modal
    async def test_custom_sh_shell(self):
        """Sandbox works with /bin/sh shell."""
        config = ModalBackendConfig(shell="/bin/sh")
        backend = ModalBackend(config)
        sb = await backend.create_sandbox()
        try:
            request = ExecRequest(command="echo hello", timeout=5, output_limit=1024)
            result = await sb.exec(request)
            assert result.exit_code == 0
            assert "hello" in result.stdout
        finally:
            await sb.stop()
